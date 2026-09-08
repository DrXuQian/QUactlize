#!/usr/bin/env python3
"""Bounded canonical GEMV/GEMM comparison; execute prebuilt libraries only.

Also measures the all-expert scale/zero prepass, separately from resident
compute. No heuristic or production route is changed by this tool.
"""

import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.execution.native import (
    Call,
    Config,
    Sizes,
    Arrangement,
    arrangement,
    bind,
)
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK, checked
from tools.kpack_warmup_fixture import activation_values
from tools.kpack_execution_fixture import IndexedWeights
from tools.run_kpack_pack_gate import device_identity
from quactlize.dispatch.native import Dispatch, receipt as dispatch_receipt
from quactlize.runtime.native import Call as GemmCall
from tools.verify_kpack_dispatch import verify as verify_dispatch

CONFIGS = [(c, w, s) for c in (16, 32) for w in (4, 8) for s in (1, 4)]


def plan(real, dense_real=False):
    # Every format: small M, empty experts, indexed broadcast and per-slot A.
    weights = [
        dict(
            q=q,
            n=256,
            k=512,
            e=4,
            cases=[dict(mode=0, rows=m, channels=1, topk=1) for m in (1, 7)]
            + [
                dict(mode=1, rows=6, channels=1, topk=1),
                dict(mode=2, rows=8, channels=1, topk=2),
                dict(mode=2, rows=8, channels=2, topk=2),
            ],
        )
        for q in range(10, 15)
    ]
    if real:
        weights += [
            dict(
                q=q,
                n=n,
                k=k,
                e=256,
                cases=[
                    dict(mode=2, rows=t * 8, channels=ch, topk=8)
                    for t in (1, 4, 16)
                    for ch in (1, 8)
                ],
            )
            for q, n, k in ((12, 512, 2048), (13, 2048, 512))
        ]
    if dense_real:
        weights += [
            dict(
                q=q,
                n=n,
                k=k,
                e=1,
                cases=[dict(mode=0, rows=m, channels=1, topk=1) for m in (1, 4)],
            )
            for q in range(10, 15)
            for n, k in (
                (1024, 5120),
                (5120, 8192),
                (5120, 25600),
                (8192, 5120),
                (25600, 5120),
            )
        ]
    return weights


class Resources:
    def __init__(self, sdk):
        self.sdk = sdk
        self.allocations = []
        self.events = []
        self.stream = C.c_void_p()
        bindings = {
            "hggcStreamCreateWithFlags": [C.POINTER(C.c_void_p), C.c_uint],
            "hggcStreamDestroy": [C.c_void_p],
            "hggcEventCreate": [C.POINTER(C.c_void_p)],
            "hggcEventRecord": [C.c_void_p, C.c_void_p],
            "hggcEventSynchronize": [C.c_void_p],
            "hggcEventElapsedTime": [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
            "hggcEventDestroy": [C.c_void_p],
        }
        for name, args in bindings.items():
            fn = getattr(sdk.lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        checked(
            sdk.lib.hggcStreamCreateWithFlags(C.byref(self.stream), 1), "stream create"
        )
        for _ in range(2):
            p = C.c_void_p()
            checked(sdk.lib.hggcEventCreate(C.byref(p)), "event create")
            self.events.append(p)

    def alloc(self, bytes_):
        p = self.sdk.allocate(bytes_)
        self.allocations.append(p)
        return p

    def upload(self, a):
        p = self.sdk.upload(a.tobytes())
        self.allocations.append(p)
        return p

    def samples(self, fn, count):
        samples = []
        for _ in range(count):
            checked(
                self.sdk.lib.hggcEventRecord(self.events[0], self.stream),
                "record begin",
            )
            checked(fn(), "enqueue")
            checked(
                self.sdk.lib.hggcEventRecord(self.events[1], self.stream), "record end"
            )
            checked(
                self.sdk.lib.hggcEventSynchronize(self.events[1]), "timing completion"
            )
            elapsed = C.c_float()
            checked(
                self.sdk.lib.hggcEventElapsedTime(C.byref(elapsed), *self.events),
                "elapsed",
            )
            if not math.isfinite(elapsed.value) or elapsed.value <= 0:
                raise ValueError("invalid timing")
            samples.append(elapsed.value * 1000)
        return samples

    def close(self):
        self.sdk.synchronize(self.stream)
        for p in reversed(self.allocations):
            self.sdk.free(p)
        for p in self.events:
            checked(self.sdk.lib.hggcEventDestroy(p), "destroy event")
        checked(self.sdk.lib.hggcStreamDestroy(self.stream), "destroy stream")


def fixture(weights, case):
    m, mode = case["rows"], case["mode"]
    n, k, e = weights.n, weights.k, weights.experts
    topk, channels = case["topk"], case["channels"]
    ids = offsets = None
    if mode == 0:
        expert = np.zeros(m, dtype=np.int32)
        arows = np.arange(m)
        acount = m
    elif mode == 1:
        rows = np.array([2, 0, 3, 1], dtype=np.int32)
        offsets = np.r_[0, rows.cumsum()].astype(np.int32)
        expert = np.repeat(np.arange(e), rows)
        arows = np.arange(m)
        acount = m
    else:
        tokens = m // topk
        ids = np.stack(
            [(np.arange(topk) * 17 + t * 13) % e for t in range(tokens)]
        ).astype(np.int32)
        expert = ids.reshape(-1)
        arows = np.arange(m) // topk * channels + np.arange(m) % topk % channels
        acount = tokens * channels
    values = activation_values(np.arange(acount))
    a = values[:, weights.categories[0]].astype("<f2")
    golden = np.stack([values[arows[r]] @ weights.sums[expert[r]] for r in range(m)])
    denom = np.stack(
        [np.abs(values[arows[r]]) @ weights.abs_sums[expert[r]] for r in range(m)]
    )
    return dict(
        a=a,
        expert=expert,
        arows=arows,
        ids=ids,
        offsets=offsets,
        golden=golden,
        denom=denom,
    )


def compare(got, data):
    if not np.isfinite(got).all():
        raise ValueError("nonfinite output / unwritten poison")
    error = float(
        np.max(
            np.abs(got.astype("f8") - data["golden"]) / np.maximum(data["denom"], 1e-30)
        )
    )
    if error >= 0.005:
        raise ValueError(f"independent GGUF dot oracle failed: {error:.6g}")
    return error


def legacy(lib, r, weights, data, case, planes, arr):
    """Existing GEMM incumbent, pre-gathered inputs; explicitly core-only."""
    q, n, k, e = weights.q, weights.n, weights.k, weights.experts
    m = case["rows"]
    dense = case["mode"] == 0
    order = np.argsort(data["expert"], kind="stable")
    rows = np.bincount(data["expert"], minlength=e).astype("i4")
    offsets = np.r_[0, rows.cumsum()].astype("i4")
    a = r.upload(data["a"][data["arows"][order]])
    out = r.alloc(m * n * 2)
    bounds = r.upload(offsets)
    suffix = "dense" if dense else "grouped"
    query = getattr(
        lib,
        f"quactlize_ppu_{suffix}_fully_quantized_workspace_bytes_for_arrangement_v2",
    )
    query.argtypes = [C.c_int] * (4 if dense else 6) + [C.POINTER(Arrangement)]
    query.restype = C.c_int64
    args = (m, n, k, q) if dense else (m, int(rows.max()), n, k, e, q)
    size = query(*args, C.byref(arr))
    if size < 0:
        raise ValueError(f"GEMM incumbent workspace declined: {size}")
    workspace = r.alloc(max(size, 1))
    fn = getattr(lib, f"quactlize_ppu_{suffix}_fully_quantized_dev_for_arrangement_v2")
    fn.argtypes = (
        [C.c_void_p] * (5 if dense else 6)
        + [C.c_int] * (4 if dense else 6)
        + [C.c_void_p, C.c_int64, C.c_void_p, C.c_char_p, C.POINTER(Arrangement)]
    )
    fn.restype = C.c_int
    pointers = (
        (a, planes["low"], planes["high"], planes["units"], out)
        if dense
        else (a, planes["low"], planes["high"], planes["units"], bounds, out)
    )

    def launch():
        return fn(*pointers, *args, workspace, size, r.stream, None, C.byref(arr))

    checked(launch(), "GEMM incumbent correctness")
    r.sdk.synchronize(r.stream)
    got = np.frombuffer(r.sdk.download(out, m * n * 2), dtype="<f2").reshape(m, n)
    inverse = np.argsort(order)
    error = compare(got[inverse], data)
    return launch, error


def selected_gemm(dispatch, r, weights, data, case, planes, arr):
    dense = case["mode"] == 0
    m, n, k = case["rows"], weights.n, weights.k
    e = 1 if dense else weights.experts
    bound = m if dense else m // case["topk"]
    choice = dispatch.query(
        weights.q, 0 if dense else 2, m, n, k, e, bound, arr.mapping_id
    )
    if choice is None:
        return None
    order = np.argsort(data["expert"], kind="stable")
    rows = np.bincount(data["expert"], minlength=e).astype("i4")
    bounds = r.upload(np.r_[0, rows.cumsum()].astype("i4")) if not dense else None
    out = r.alloc(m * n * 2)
    c = GemmCall(
        version=1,
        size=C.sizeof(GemmCall),
        m=m,
        n=n,
        k=k,
        experts=e,
        group_size=arr.group_size,
        device=choice.device,
        compute_units=choice.compute_units,
        mapping_id=arr.mapping_id,
        a=r.upload(data["a"][data["arows"][order]]),
        low=planes["low"],
        high=planes["high"],
        metadata=planes["units"],
        output=out,
        offsets_device=bounds,
        workspace=r.alloc(max(1, choice.workspace_bytes)),
        workspace_bytes=choice.workspace_bytes,
        stream=r.stream.value,
    )
    launch = dispatch.prepare(choice, c)
    r.sdk.fill(out, 0xA5, m * n * 2)
    checked(launch(), "selected FQ correctness")
    r.sdk.synchronize(r.stream)
    got = np.frombuffer(r.sdk.download(out, m * n * 2), dtype="<f2").reshape(m, n)
    error = compare(got[np.argsort(order)], data)
    return launch, error, dispatch_receipt(choice)


def run_weight(args, sdk, functions, item):
    query, run, sf = functions
    q, n, k, e = (item[x] for x in ("q", "n", "k", "e"))
    start = time.monotonic()
    w = IndexedWeights(
        q,
        n,
        k,
        e,
        progress=lambda done, total: print(
            f"KPACK_GEMV_FIXTURE q={q} experts={done}/{total}", flush=True
        ),
    )
    fixture_seconds = time.monotonic() - start
    r = Resources(sdk)
    records = []
    arr = arrangement(q)
    dispatch = Dispatch(args.native_bundle) if args.native_bundle else None
    try:
        planes = {
            name: r.upload(w.planes[name]) if w.planes[name].size else None
            for name in ("low", "high", "units")
        }
        # Both metadata channels remain in this contract even for Q3/Q6.
        shape = (e, k // arr.group_size, n)
        plane_bytes = int(np.prod(shape)) * 2
        scale, zero = r.alloc(plane_bytes), r.alloc(plane_bytes)
        sdk.fill(scale, 0x7B, plane_bytes)
        sdk.fill(zero, 0x7B, plane_bytes)

        def prepass():
            return sf(
                q,
                n,
                k,
                e,
                planes["units"],
                w.planes["units"].nbytes,
                scale,
                zero,
                plane_bytes,
                C.byref(arr),
                r.stream,
            )

        first = r.samples(prepass, 1)[0]
        # Scale is bit-exact; the existing historical zero fixture rounds an
        # intermediate product. Reconstruct canonical zero from the actual
        # packed units for the separate metadata oracle below.
        wanted_scale, wanted_zero = metadata_oracle(w.planes["units"], q, n, k, e)
        for p, want, name in (
            (scale, wanted_scale, "scale"),
            (zero, wanted_zero, "zero"),
        ):
            got = np.frombuffer(sdk.download(p, plane_bytes), dtype="<u2").reshape(
                shape
            )
            if not np.array_equal(got, want.view("<u2")):
                raise ValueError(
                    f'{name} prepass differs: {np.count_nonzero(got!=want.view("<u2"))}'
                )
        meta_samples = r.samples(prepass, 3)
        print(
            f"KPACK_SF_PREPASS q={q} N={n} K={k} E={e} output_bytes={2*plane_bytes} first_us={first:.3f} median_us={statistics.median(meta_samples):.3f}",
            flush=True,
        )
        library_path = (
            args.gemm_bundle
            / f"libquactlize_ppu_fmt{ {10:2,11:3,12:0,13:1,14:4}[q] }.so"
        )
        lib = C.CDLL(str(library_path.resolve()), mode=C.RTLD_LOCAL)
        for ci, case in enumerate(item["cases"]):
            data = fixture(w, case)
            m = case["rows"]
            ap = r.upload(data["a"].astype("f4"))
            output = r.alloc(m * n * 4 + 32)
            workspace = r.alloc(m * n * 4 * 4)
            c = Call(
                version=1,
                size=C.sizeof(Call),
                qtype=q,
                n=n,
                k=k,
                experts=1 if case["mode"] == 0 else e,
                rows=m,
                mode=case["mode"],
                input_type=1,
                channels=case["channels"],
                topk=case["topk"],
                a_row_stride=k,
                a_token_stride=k * case["channels"],
                ids_stride=case["topk"],
                out_row_stride=n,
                a=ap,
                low=planes["low"],
                high=planes["high"],
                units=planes["units"],
                output=output + 16,
                ids=r.upload(data["ids"]) if data["ids"] is not None else None,
                offsets=(
                    r.upload(data["offsets"]) if data["offsets"] is not None else None
                ),
                workspace=workspace,
                workspace_bytes=m * n * 16,
                stream=r.stream.value,
            )
            native = (
                selected_gemm(dispatch, r, w, data, case, planes, arr)
                if dispatch
                else None
            )
            if native:
                incumbent, inc_error, inc_selection = native
                inc_selection["source"] = "NATIVE_HEURISTIC"
            else:
                incumbent, inc_error = legacy(lib, r, w, data, case, planes, arr)
                inc_selection = dict(source="LEGACY_FQ_POLICY_FALLBACK")
            configs = []
            errors = []
            samples = {f"{col}-{warp}-{split}": [] for col, warp, split in CONFIGS}
            bsamples = []
            for col, warp, split in CONFIGS:
                f = Config(col, warp, split)
                s = Sizes()
                checked(
                    query(C.byref(c), C.byref(f), C.byref(arr), C.byref(s)),
                    "GEMV query",
                )
                sdk.fill(output, 0xA5, m * n * 4 + 32)
                checked(run(C.byref(c), C.byref(f), C.byref(arr)), "GEMV correctness")
                sdk.synchronize(r.stream)
                raw = sdk.download(output, m * n * 4 + 32)
                if raw[:16] != b"\xa5" * 16 or raw[-16:] != b"\xa5" * 16:
                    raise ValueError("GEMV output guard")
                got = np.frombuffer(raw[16:-16], dtype="<f4").reshape(m, n)
                errors.append(compare(got, data))
                configs.append(f)
            # Finite planted fault must change the complete output, not only
            # fail a metadata/shape guard. It is never a timed candidate.
            low_save = c.low
            badlow = r.alloc(w.planes["low"].nbytes)
            sdk.fill(badlow, 0, w.planes["low"].nbytes)
            c.low = badlow
            checked(
                run(C.byref(c), C.byref(configs[0]), C.byref(arr)), "GEMV planted fault"
            )
            sdk.synchronize(r.stream)
            planted = np.frombuffer(
                sdk.download(c.output, m * n * 4), dtype="<f4"
            ).reshape(m, n)
            c.low = low_save
            if (
                not np.isfinite(planted).all()
                or np.max(np.abs(planted - data["golden"]) / data["denom"]) <= 0.005
            ):
                raise ValueError("GEMV planted zero-low fault was not detected")
            for round_ in range(3):
                order = list(range(len(configs)))
                if round_ % 2:
                    order.reverse()
                if round_ % 2 == 0:
                    bsamples.append(r.samples(incumbent, args.samples))
                for idx in order:
                    f = configs[idx]
                    key = f"{f.columns}-{f.warps}-{f.split}"
                    samples[key].append(
                        r.samples(
                            lambda: run(C.byref(c), C.byref(f), C.byref(arr)),
                            args.samples,
                        )
                    )
                if round_ % 2:
                    bsamples.append(r.samples(incumbent, args.samples))
            result = dict(
                q=q,
                n=n,
                k=k,
                experts=c.experts,
                case=case,
                oracle="OFFICIAL_GGUF_CONDITION_SCALED",
                errors=errors,
                incumbent_error=inc_error,
                incumbent_selection=inc_selection,
                gemv_samples_us=samples,
                gemm_samples_us=bsamples,
                incumbent_scope="CORE_PRE_GATHERED_GEMM_INCLUDING_DEVICE_DIRECTORY_NO_ADAPTERS",
                gemv_scope="FULL_OUTPUT_INCLUDING_IDS_AND_SPLIT_REDUCER",
                heuristic_admitted=False,
                input_type=c.input_type,
                correctness="PASS",
                zero_low_negative="DETECTED_FINITE",
            )
            medians = {
                key: statistics.median([v for row in values for v in row])
                for key, values in samples.items()
            }
            winner = min(medians, key=lambda key: (medians[key], key))
            baseline = statistics.median([v for row in bsamples for v in row])
            result.update(
                winner=winner,
                gemv_median_us=medians[winner],
                gemm_median_us=baseline,
                delta_pct=100 * (medians[winner] / baseline - 1),
            )
            records.append(result)
            (args.output / f"q{q}-n{n}-k{k}-e{e}-case{ci}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(
                f'KPACK_GEMV_CASE q={q} mode={c.mode} rows={m} N={n} K={k} E={c.experts} configs={len(configs)} correctness=PASS winner={winner} gemv_us={medians[winner]:.3f} gemm_us={baseline:.3f} delta_pct={result["delta_pct"]:.3f}',
                flush=True,
            )
        return dict(
            q=q,
            n=n,
            k=k,
            e=e,
            fixture_seconds=fixture_seconds,
            prepass_first_us=first,
            prepass_samples_us=meta_samples,
            metadata_output_bytes=2 * plane_bytes,
            records=records,
        )
    finally:
        if dispatch:
            sdk.synchronize(r.stream)
            dispatch.close()
        r.close()


def metadata_oracle(units, q, n, k, e):
    from reference import gguf_kpack as ref

    spec = ref.SPECS[q]
    sb = k // 256
    packed = units.reshape(
        e, sb // spec.superblocks_per_unit, n, spec.superblocks_per_unit, spec.sb_bytes
    )
    packed = packed.transpose(0, 1, 3, 2, 4).reshape(e, sb, n, spec.sb_bytes)
    d = packed[..., :2].copy().view("<f2")[..., 0]
    dm = (
        packed[..., 2:4].copy().view("<f2")[..., 0]
        if spec.has_min
        else np.zeros_like(d)
    )
    scale = np.empty((e, sb * spec.groups, n), dtype="<f2")
    zero = np.empty_like(scale)
    for g in range(spec.groups):
        codes = []
        for which, bits in ((0, spec.scale_bits), (1, spec.min_bits)):
            if not bits:
                codes.append(np.zeros_like(d, dtype="i2"))
                continue
            bit = ref._unit_bit(spec, g, which)
            byte, shift = divmod(bit, 8)
            word = packed[..., byte].astype("u2")
            if shift + bits > 8:
                word |= packed[..., byte + 1].astype("u2") << 8
            codes.append(((word >> shift) & ((1 << bits) - 1)).astype("i2"))
        sc, mn = codes
        if q == 11:
            sc = sc - 32
        if q == 14:
            sc = np.where(sc >= 128, sc - 256, sc)
        s = (d.astype("f4") * sc).astype("<f2")
        z = (
            (-(dm.astype("f4") * mn)).astype("<f2")
            if spec.has_min
            else np.zeros_like(s)
        )
        zmul = {10: 0, 11: -4, 12: 8, 13: 8, 14: -24}[q]
        if zmul:
            z = (z.astype("f4") + np.float32(zmul) * s.astype("f4")).astype("<f2")
        scale[:, g :: spec.groups] = s
        zero[:, g :: spec.groups] = z
    return scale, zero


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--gemm-bundle", type=Path, required=True)
    p.add_argument(
        "--native-bundle",
        type=Path,
        help="compare GEMV with the native heuristic-selected FQ recipe when covered",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--real-model-shapes", action="store_true")
    p.add_argument(
        "--model-decode-only",
        action="store_true",
        help="Q4/Q5 E256 experts plus the model's Q6 248320x2048 output head",
    )
    p.add_argument("--samples", type=int, default=11)
    p.add_argument(
        "--dense-real-shapes",
        action="store_true",
        help="also compare five dense N/K families, M1/M4, five formats",
    )
    p.add_argument(
        "--qtypes",
        default="10,11,12,13,14",
        help="comma-separated format subset for a focused rerun",
    )
    args = p.parse_args()
    if args.native_bundle:
        verify_dispatch(args.native_bundle)
    if args.samples < 3:
        p.error("--samples must be at least 3")
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    path = args.bundle / manifest["library"]
    if (
        manifest["schema"] != "quactlize.kpack-execution-build.v1"
        or sha(path) != manifest["sha256"]
    ):
        raise ValueError("execution payload identity differs")
    if manifest["gemv_configs"] != [
        dict(columns=c, warps=w, split=s) for c, w, s in CONFIGS
    ]:
        raise ValueError("candidate inventory differs")
    for name, value in manifest["runtime"].items():
        if sha(args.sdk / "lib" / name) != value:
            raise ValueError(f"SDK runtime differs: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    sdk = SDK(args.sdk)
    identity = device_identity(sdk)
    functions = bind(C.CDLL(str(path.resolve()), mode=C.RTLD_LOCAL))
    selected = {int(x) for x in args.qtypes.split(",")}
    if not selected or not selected <= set(range(10, 15)):
        p.error("invalid --qtypes")
    requested = [
        x
        for x in plan(args.real_model_shapes, args.dense_real_shapes)
        if x["q"] in selected
    ]
    if args.model_decode_only:
        requested = [x for x in plan(True) if x["e"] == 256]
        requested.append(
            dict(
                q=14,
                n=248320,
                k=2048,
                e=1,
                cases=[dict(mode=0, rows=m, channels=1, topk=1) for m in (1, 4)],
            )
        )
        requested = [x for x in requested if x["q"] in selected]
    summary = dict(
        status="INCOMPLETE",
        device=identity,
        manifest_sha256=sha(args.bundle / "manifest.json"),
        execution_sha256=manifest["sha256"],
        plan=requested,
        results=[],
        failures=[],
        heuristic_admitted=False,
        source=subprocess_source(),
        baseline_libraries={},
        native_manifest_sha256=(
            sha(args.native_bundle / "manifest.json") if args.native_bundle else None
        ),
    )
    for q in selected:
        name = f"libquactlize_ppu_fmt{ {10:2,11:3,12:0,13:1,14:4}[q] }.so"
        summary["baseline_libraries"][name] = sha(args.gemm_bundle / name)
    start = time.monotonic()
    try:
        for item in requested:
            try:
                summary["results"].append(run_weight(args, sdk, functions, item))
            except Exception as exc:
                traceback.print_exc()
                summary["failures"].append(dict(weight=item, error=str(exc)))
                print(
                    f'KPACK_GEMV_WEIGHT status=FAIL q={item["q"]} N={item["n"]} K={item["k"]} error={exc}',
                    flush=True,
                )
        summary["status"] = "FAIL" if summary["failures"] else "PASS"
    finally:
        summary["wall_seconds"] = time.monotonic() - start
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f'KPACK_GEMV_GATE status={summary["status"]} weights={len(requested)} results={args.output}',
        flush=True,
    )
    return 0 if summary["status"] == "PASS" else 1


def subprocess_source():
    # Freeze the host oracle/runner used, not merely a possibly dirty HEAD.
    return {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            Path(__file__).resolve(),
            ROOT / "quactlize/execution/native.py",
            ROOT / "tools/kpack_warmup_fixture.py",
            ROOT / "tools/kpack_execution_fixture.py",
            ROOT / "reference/gguf_kpack.py",
        )
    }


if __name__ == "__main__":
    raise SystemExit(main())
