#!/usr/bin/env python3
"""Small compile/cache/resident-pointer gate; not a configuration sweep.

Checks five formats, four routes, ragged/empty experts and changed routers.
Uses official GGUF dequantization as the numeric oracle. Online compilation
is isolated in compile_only; inference cache queries must never invoke it.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler
from quactlize.runtime.native import NativeBackend, SDK
from quactlize.runtime.tuning import Request, Tactic, Tuner, TuningCache, ROUTES


class GateBackend(NativeBackend):
    def check(self, handle):
        # Every candidate must overwrite every real output cell, not inherit a
        # previous candidate's correct values. This fill is outside timing.
        self.sdk.fill(self.buffers["output"], 0x7B, handle.call.m * handle.call.n * 2)
        super().check(handle)


def parents(formats, routes):
    result = []
    for q in formats:
        tile_k = {10: 128, 11: 256, 12: 64, 13: 256, 14: 128}[q]
        for route in routes:
            for tile_m in (8, 16):
                # Grouped FQ persistent/nonpersistent are separate parents.
                for p in ((0, 1) if route == "fq-grouped" else (-1,)):
                    symbol = f"qk_gate_q{q}_{route.replace('-','_')}_tm{tile_m}_p{p+1}"
                    result.append(
                        dict(
                            symbol=symbol,
                            route=route,
                            qtype=q,
                            tm=tile_m,
                            tn=64,
                            tk=tile_k,
                            wm=tile_m,
                            wn=16,
                            stages=2,
                            ap=0,
                            dn=16,
                            persistent=p,
                        )
                    )
    return result


def fixture(q, request):
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    from reference import gguf_kpack as reference

    spec = reference.SPECS[q]
    experts = len(request.rows) if request.grouped else 1
    rng = np.random.default_rng(95000 + q)
    raw = rng.integers(
        0,
        256,
        size=(experts * request.n * (request.k // 256), spec.raw_bytes),
        dtype=np.uint8,
    )
    for offset in (spec.d_offset, spec.dmin_offset):
        if offset >= 0:
            value = (rng.random(raw.shape[0]) * 0.025 + 0.005).astype(np.float16)
            raw[:, offset : offset + 2] = value.view(np.uint8).reshape(-1, 2)
    artifact = (
        reference.prepare_grouped(
            torch.from_numpy(raw), request.n, request.k, q, experts
        )
        if request.grouped
        else reference.prepare_dense(torch.from_numpy(raw), request.n, request.k, q)
    )
    if not torch.equal(reference.recover_raw_blocks(artifact), torch.from_numpy(raw)):
        raise AssertionError("offline roundtrip failed")
    weights = (
        dequantize(raw.reshape(-1), GGMLQuantizationType(q))
        .astype(np.float64)
        .reshape(experts, request.n, request.k)
    )
    a = (rng.standard_normal((request.m, request.k)) * 0.2).astype(np.float16)
    rows = request.rows or (request.m,)
    golden = []
    denom = []
    offset = 0
    for e, m in enumerate(rows):
        act = a[offset : offset + m].astype(np.float64)
        golden.append(act @ weights[e].T)
        denom.append(np.abs(act) @ np.abs(weights[e]).T)
        offset += m
    golden = np.concatenate(golden)
    denom = np.concatenate(denom)
    if not np.isfinite(golden).all() or not np.any(golden):
        raise AssertionError("degenerate numeric oracle")
    scale = np.empty(
        (experts, request.k // spec.group_size, request.n), dtype=np.float16
    )
    zero = np.empty_like(scale)
    zmul = {10: 0, 11: -4, 12: 8, 13: 8, 14: -24}[q]
    blob = raw.tobytes()
    for e in range(experts):
        for n in range(request.n):
            for sb in range(request.k // 256):
                base = ((e * request.n + n) * (request.k // 256) + sb) * spec.raw_bytes
                d = struct.unpack_from("<e", blob, base + spec.d_offset)[0]
                dmin = (
                    struct.unpack_from("<e", blob, base + spec.dmin_offset)[0]
                    if spec.has_min
                    else 0
                )
                for g in range(spec.groups):
                    sc, mn = reference._metadata_codes(blob, base, spec, g)
                    if q == 11:
                        sc -= 32
                    if q == 14 and sc >= 128:
                        sc -= 256
                    s = np.float16(d * sc)
                    z = np.float16(-np.float16(dmin * mn))
                    # Half multiply/add, as in the canonical prepass.
                    z = np.float16(np.float16(zmul * s) + z) if zmul else z
                    scale[e, sb * spec.groups + g, n] = s
                    zero[e, sb * spec.groups + g, n] = z
    return (
        dict(
            a=a.tobytes(),
            low=artifact.low.numpy().tobytes(),
            high=artifact.high.numpy().tobytes() if artifact.high.numel() else b"",
            metadata=(
                artifact.units.numpy().tobytes()
                if request.route.startswith("fq")
                else scale.tobytes()
            ),
            zero=b"" if request.route.startswith("fq") else zero.tobytes(),
        ),
        golden,
        denom,
    )


def run_case(sdk_path, records, request, output, cache_name):
    import numpy as np

    inputs, golden, denom = fixture(request.qtype, request)
    sdk = SDK(sdk_path)
    allocated = []
    backend = None

    def upload(data):
        if not data:
            return 0
        ptr = sdk.upload(data)
        allocated.append(ptr)
        return ptr

    try:
        buffers = {name: upload(data) for name, data in inputs.items()}
        size = request.m * request.n * 2
        buffers["output"] = sdk.allocate(size)
        allocated.append(buffers["output"])
        if request.grouped:
            offsets = [0]
            for row in request.rows:
                offsets.append(offsets[-1] + row)
            buffers["rows_device"] = upload(
                struct.pack("<" + "i" * len(request.rows), *request.rows)
            )
            buffers["offsets_device"] = upload(
                struct.pack("<" + "i" * len(offsets), *offsets)
            )
        errors = []

        def check(b):
            got = (
                np.frombuffer(sdk.download(buffers["output"], size), dtype=np.float16)
                .astype(np.float64)
                .reshape(golden.shape)
            )
            err = float(
                np.max(
                    np.abs(got - golden) / np.maximum(denom, np.finfo(np.float64).tiny)
                )
            )
            errors.append(err)
            return np.isfinite(got).all() and np.isfinite(err) and err < 5e-3

        backend = GateBackend(sdk_path, records, buffers, check)
        if (
            backend.identity["device"] != "PPU-ZW810"
            or backend.identity["compute_units"] != 72
        ):
            raise ValueError(f"gate requires ZW810/72 CUs, got {backend.identity}")
        cache = TuningCache(backend.identity, output / f"{cache_name}.json")
        tuner = Tuner(cache, budget_ms=5000, warmups=2, repeats=5, samples=3)
        candidates = []
        for record in records:
            p = record["parent"]
            if request.route == "fq-dense":
                candidates.extend(Tactic(p["symbol"], f"TC_S{s}", s) for s in (1, 2))
            else:
                prefix = "GROUPED_" if request.grouped else ""
                if p["persistent"] != 1:
                    candidates.append(Tactic(p["symbol"], prefix + "NONPERSISTENT"))
                if p["persistent"] != 0:
                    candidates.append(
                        Tactic(p["symbol"], prefix + "PERSISTENT", 1, "capacity", 1)
                    )
        previous = tuner.select(request, backend)
        if previous["status"] == "BUCKET_HINT":
            cached_handle = backend.prepare(request, previous["tactic"])
            try:
                backend.check(cached_handle)
            finally:
                backend.synchronize()
                backend.close(cached_handle)
        result = tuner.warmup(request, candidates, backend, force=True)
        if result["status"] != "MEASURED_EXACT":
            raise AssertionError(f"warmup did not produce a measured choice: {result}")
        # A reloaded cache and inference query must use the same choice. The
        # changed-row case below can reuse its bucket but is never called exact.
        selected = Tuner(
            TuningCache(backend.identity, output / f"{cache_name}.json")
        ).select(request, backend)
        if selected["tactic"] != result["tactic"]:
            raise AssertionError("cache replay differs")
        handle = backend.prepare(request, selected["tactic"])
        try:
            backend.check(handle)
        finally:
            backend.synchronize()
            backend.close(handle)
        # An altered code plane must fail the same numerical check. A launch
        # failure is not accepted as the expected negative.
        zero_low = sdk.allocate(len(inputs["low"]))
        allocated.append(zero_low)
        sdk.fill(zero_low, 0, len(inputs["low"]))
        original_low = backend.buffers["low"]
        backend.buffers["low"] = zero_low
        handle = backend.prepare(request, selected["tactic"])
        negative = False
        try:
            try:
                backend.check(handle)
            except RuntimeError as error:
                if str(error) != "candidate correctness check failed":
                    raise
                negative = True
        finally:
            backend.synchronize()
            backend.close(handle)
            backend.buffers["low"] = original_low
        if not negative:
            raise AssertionError("zero-low negative was not detected")
        planted_error = errors.pop()
        result["tactic"] = asdict(result["tactic"])
        return dict(
            request=asdict(request),
            **result,
            max_error=max(errors),
            checks=len(errors),
            identity=backend.identity,
            previous_cache_status=previous["status"],
            planted_error=planted_error,
            zero_low_rejected=negative,
        )
    finally:
        if backend:
            backend.release()
        for pointer in reversed(allocated):
            sdk.free(pointer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--formats", type=int, nargs="+", default=list(range(10, 15)))
    parser.add_argument("--routes", choices=ROUTES, nargs="+", default=list(ROUTES))
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if any(q not in range(10, 15) for q in args.formats):
        parser.error("formats must be Q2..Q6 (10..14)")
    if len(set(args.formats)) != len(args.formats) or len(set(args.routes)) != len(
        args.routes
    ):
        parser.error("formats/routes must be unique")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(
            "output must be new/empty; the separate compiled cache remains reusable"
        )
    if not args.compile_only:
        # Fail before a cold compile if this host cannot construct the oracle.
        try:
            import gguf, numpy, torch
        except ImportError as error:
            parser.error(f"device gate needs gguf, numpy and torch: {error}")
    args.output.mkdir(parents=True, exist_ok=True)
    compiler = Compiler(args.sdk, args.cache, args.jobs)
    ps = parents(args.formats, args.routes)
    started = time.monotonic()
    print(
        f"KPACK_WARMUP_COMPILE parents={len(ps)} jobs={args.jobs} device_execution=0",
        flush=True,
    )
    records = compiler.compile_only(
        ps,
        lambda done, total: print(
            f"KPACK_WARMUP_COMPILE_PROGRESS completed={done}/{total}", flush=True
        ),
    )
    build_seconds = time.monotonic() - started
    replay = compiler.compile_only(ps)
    if not all(r["cache_hit"] for r in replay):
        raise AssertionError("compile-only cache replay missed")
    print(
        f"KPACK_WARMUP_COMPILE status=PASS parents={len(records)} seconds={build_seconds:.3f} cache_replay=HIT",
        flush=True,
    )
    (args.output / "modules.json").write_text(json.dumps(records, indent=2))
    if args.compile_only:
        return 0
    results = []
    for q in args.formats:
        for route in args.routes:
            selected = [
                r
                for r in records
                if r["parent"]["qtype"] == q and r["parent"]["route"] == route
            ]
            contexts = (
                ((9, 7, 0, 0), (8, 8, 0, 0), (2, 0, 3, 1))
                if route.endswith("grouped")
                else (1, 9)
            )
            for index, ctx in enumerate(contexts):
                request = (
                    Request(route, q, 256, 512, sum(ctx), ctx)
                    if isinstance(ctx, tuple)
                    else Request(route, q, 256, 512, ctx)
                )
                print(
                    f"KPACK_WARMUP_DEVICE q={q} route={route} m={request.m} case={index}",
                    flush=True,
                )
                result = run_case(
                    args.sdk, selected, request, args.output, f"q{q}-{route}"
                )
                results.append(result)
                (args.output / "results.json").write_text(
                    json.dumps(results, indent=2, allow_nan=False)
                )
                print(
                    f"KPACK_WARMUP_CASE status=PASS q={q} route={route} case={index} candidates={result['candidates']} max_error={result['max_error']:.6g} us={result['us']:.3f}",
                    flush=True,
                )
    receipt = dict(
        status="PASS",
        cases=len(results),
        parents=len(records),
        build_seconds=build_seconds,
        wall_seconds=time.monotonic() - started,
        scope="SMALL_RUNTIME_GATE_NOT_PERFORMANCE_ADMISSION",
    )
    (args.output / "summary.json").write_text(json.dumps(receipt, indent=2))
    print("KPACK_WARMUP_GATE " + json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, AssertionError) as error:
        print(
            f"KPACK_WARMUP_GATE status=FAIL error={error}", file=sys.stderr, flush=True
        )
        raise SystemExit(1)
