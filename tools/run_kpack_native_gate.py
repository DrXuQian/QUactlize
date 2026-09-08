#!/usr/bin/env python3
"""Exercise the C++ selected recipe, not a manually chosen same-size kernel."""

import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.dispatch.native import Dispatch, receipt
from quactlize.execution.native import arrangement, bind
from quactlize.runtime.native import SDK, Call, checked
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import Request
from tools.run_kpack_gemv_gate import Resources, compare
from tools.run_kpack_grouped_device_gate import graph_bind, profiles
from tools.run_kpack_pack_gate import device_identity
from tools.kpack_warmup_fixture import Weights
from tools.verify_kpack_dispatch import verify


def run(sdk, bundle, w, route, tokens, samples):
    grouped = route >= 2
    e = w.experts
    if grouped:
        if tokens == 128:
            vectors = profiles(e)
        else:
            rows = np.zeros(e, dtype="i4")
            rows[np.arange(8) * 17] = 1
            vectors = [tuple(map(int, np.roll(rows, s))) for s in (0, 3, 17)]
    else:
        vectors = [(tokens,)] * 3
    m = sum(vectors[0])
    arr = arrangement(w.q)
    d = Dispatch(bundle)
    r = Resources(sdk)
    graph, instance = C.c_void_p(), C.c_void_p()
    try:
        choice = d.query(w.q, route, m, w.n, w.k, e, tokens, arr.mapping_id)
        if choice is None:
            raise ValueError("required native context missed its selected module")
        again = d.query(w.q, route, m, w.n, w.k, e, tokens, arr.mapping_id)
        if bytes(choice) != bytes(again):
            raise ValueError("cached selection changed")
        planes = {
            x: r.upload(w.planes[x]) if w.planes[x].size else None
            for x in ("low", "high", "units")
        }
        sf = route in (1, 3)
        prepass_samples = []
        scale = zero = None
        if sf:
            _, _, prepare = bind(
                C.CDLL(str(bundle / "libquactlize_ppu_execution.so"), mode=C.RTLD_LOCAL)
            )
            size = e * (w.k // arr.group_size) * w.n * 2
            scale, zero = r.alloc(size), r.alloc(size)
            prepass = lambda: prepare(
                w.q,
                w.n,
                w.k,
                e,
                planes["units"],
                w.planes["units"].nbytes,
                scale,
                zero,
                size,
                C.byref(arr),
                r.stream,
            )
            prepass_samples = r.samples(prepass, 3)
            for ptr, name in ((scale, "scale"), (zero, "zero")):
                if sdk.download(ptr, size) != w.planes[name].tobytes():
                    raise ValueError(
                        "native SF prepass differs from packed-unit oracle: " + name
                    )
        ap = r.alloc(m * w.k * 2)
        output = r.alloc(m * w.n * 2 + 32)
        bounds = r.alloc((e + 1) * 4) if grouped else None
        c = Call(
            version=1,
            size=C.sizeof(Call),
            m=m,
            n=w.n,
            k=w.k,
            experts=e,
            group_size=arr.group_size,
            device=choice.device,
            compute_units=choice.compute_units,
            mapping_id=arr.mapping_id,
            a=ap,
            low=planes["low"],
            high=planes["high"],
            metadata=scale if sf else planes["units"],
            zero=zero,
            output=output + 16,
            offsets_device=bounds,
            workspace=r.alloc(max(1, choice.workspace_bytes)),
            workspace_bytes=choice.workspace_bytes,
            stream=r.stream.value,
        )
        launch = d.prepare(choice, c)
        records = []
        for index, rows in enumerate(vectors):
            req = Request(
                ("fq-dense", "sf-dense", "fq-grouped", "sf-grouped")[route],
                w.q,
                w.n,
                w.k,
                m,
                rows if grouped else (),
            )
            a, gold, denom = w.activation(req)
            checked(
                sdk.lib.hggcMemcpy(ap, a.ctypes.data, a.nbytes, 1), "A fixture upload"
            )
            if bounds:
                offsets = np.r_[0, np.cumsum(rows)].astype("i4")
                checked(
                    sdk.lib.hggcMemcpy(bounds, offsets.ctypes.data, offsets.nbytes, 1),
                    "GPU router fixture upload",
                )
            sdk.fill(output, 0xA5, m * w.n * 2 + 32)
            if index == 0:
                checked(launch(), "selected eager run")
                sdk.synchronize(r.stream)
                checked(sdk.lib.hggcStreamBeginCapture(r.stream, 0), "begin capture")
                checked(launch(), "selected capture")
                checked(
                    sdk.lib.hggcStreamEndCapture(r.stream, C.byref(graph)),
                    "end capture",
                )
                checked(
                    sdk.lib.hggcGraphInstantiateWithFlags(C.byref(instance), graph, 0),
                    "instantiate",
                )
            checked(sdk.lib.hggcGraphLaunch(instance, r.stream), "selected replay")
            sdk.synchronize(r.stream)
            raw = sdk.download(output, m * w.n * 2 + 32)
            if raw[:16] != b"\xa5" * 16 or raw[-16:] != b"\xa5" * 16:
                raise ValueError("selected output guard changed")
            got = np.frombuffer(raw[16:-16], dtype="<f2").reshape(m, w.n)
            err = compare(got, dict(golden=gold, denom=denom))
            # Finite arithmetic corruption must be visible to this oracle.
            if np.max(np.abs(gold) / np.maximum(denom, 1e-30)) <= 0.005:
                raise ValueError("zero-output negative is not discriminating")
            values = r.samples(launch, samples)
            records.append(
                dict(profile=index, rows=list(rows), error=err, samples_us=values)
            )
        result = dict(
            status="PASS",
            q=w.q,
            route=route,
            m=m,
            n=w.n,
            k=w.k,
            experts=e,
            max_rows=tokens,
            selection=receipt(choice),
            profiles=records,
            prepass_samples_us=prepass_samples,
            graph_replays=3,
            rows_host=False,
            rows_device=False,
            scope="RESIDENT_SELECTED_GEMM_INCLUDING_GPU_DIRECTORY_NOT_LLAMA_ADAPTERS",
        )
        print(
            "KPACK_NATIVE_CONTEXT "
            + json.dumps({k: v for k, v in result.items() if k != "profiles"}),
            flush=True,
        )
        return result
    finally:
        sdk.synchronize(r.stream)
        if instance:
            sdk.lib.hggcGraphExecDestroy(instance)
        if graph:
            sdk.lib.hggcGraphDestroy(graph)
        d.close()
        r.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=5)
    a = p.parse_args()
    if a.samples < 3:
        p.error("at least three samples required")
    a.bundle = a.bundle.resolve(strict=True)
    verify(a.bundle)
    a.output.mkdir(parents=True, exist_ok=False)
    sdk = SDK(a.sdk)
    graph_bind(sdk)
    summary = dict(
        status="INCOMPLETE",
        device=device_identity(sdk),
        manifest_sha256=sha(a.bundle / "manifest.json"),
        results=[],
        failures=[],
        expected_contexts=28,
    )
    items = [(q, 1024, 5120, 1) for q in range(10, 15)] + [
        (12, 512, 2048, 256),
        (13, 2048, 512, 256),
    ]
    try:
        for q, n, k, e in items:
            print(f"KPACK_NATIVE_FIXTURE q={q} N={n} K={k} E={e}", flush=True)
            w = Weights(q, n, k, e)
            for route in ((0, 1) if e == 1 else (2, 3)):
                for tokens in (1, 128):
                    try:
                        summary["results"].append(
                            run(sdk, a.bundle, w, route, tokens, a.samples)
                        )
                    except Exception as error:
                        traceback.print_exc()
                        summary["failures"].append(
                            dict(
                                q=q,
                                n=n,
                                k=k,
                                route=route,
                                tokens=tokens,
                                error=str(error),
                            )
                        )
                    (a.output / "summary.json").write_text(
                        json.dumps(summary, indent=2) + "\n"
                    )
        summary["status"] = (
            "PASS"
            if len(summary["results"]) == 28 and not summary["failures"]
            else "FAIL"
        )
    finally:
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"KPACK_NATIVE_GATE {summary['status']} contexts={len(summary['results'])}/28 output={a.output}"
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
