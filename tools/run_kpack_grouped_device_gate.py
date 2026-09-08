#!/usr/bin/env python3
"""Check device-only grouped handles against v1, including mutable graph replay.

The same measured parent is used on both sides. Only metadata preparation
changes. ScaleFirst additionally records prepass+GEMM, not just resident GEMM.
"""

import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.native import (
    SDK,
    Module,
    Call,
    Recipe,
    Resources as QueryResources,
    checked,
)
from quactlize.runtime.tuning import Request
from quactlize.runtime.compiler import sha
from quactlize.execution.native import bind, arrangement
from tools.run_kpack_gemv_gate import Resources, compare
from tools.run_kpack_pack_gate import device_identity
from tools.kpack_warmup_fixture import Weights


class DeviceCall(C.Structure):
    _fields_ = [
        ("version", C.c_uint32),
        ("size", C.c_uint32),
        ("call", Call),
        ("max_rows", C.c_int32),
        ("reserved", C.c_int32),
    ]


def graph_bind(sdk):
    for name, args in {
        "hggcStreamBeginCapture": [C.c_void_p, C.c_int],
        "hggcStreamEndCapture": [C.c_void_p, C.POINTER(C.c_void_p)],
        "hggcGraphInstantiateWithFlags": [
            C.POINTER(C.c_void_p),
            C.c_void_p,
            C.c_uint64,
        ],
        "hggcGraphLaunch": [C.c_void_p, C.c_void_p],
        "hggcGraphExecDestroy": [C.c_void_p],
        "hggcGraphDestroy": [C.c_void_p],
    }.items():
        fn = getattr(sdk.lib, name)
        fn.argtypes, fn.restype = args, C.c_int


def profiles(experts):
    if experts == 4:
        return [(129, 0, 1, 1), (1, 129, 0, 1), (0, 1, 1, 129)]
    # 128 tokens, top8, all 256 experts, then replay with permuted rows.
    rows = np.bincount(
        np.array([(np.arange(8) * 17 + t * 13) % experts for t in range(128)]).reshape(
            -1
        ),
        minlength=experts,
    )
    return [tuple(int(x) for x in np.roll(rows, shift)) for shift in (0, 3, 17)]


def run_parent(sdk, record, config, w, sf):
    module = Module(record)
    lib = module.lib
    v2query = lib.quactlize_kpack_grouped_query_v2
    v2query.argtypes, v2query.restype = [
        C.POINTER(DeviceCall),
        C.POINTER(Recipe),
        C.POINTER(QueryResources),
    ], C.c_int
    v2prepare = lib.quactlize_kpack_grouped_prepare_v2
    v2prepare.argtypes, v2prepare.restype = [
        C.POINTER(DeviceCall),
        C.POINTER(Recipe),
        C.POINTER(C.c_void_p),
    ], C.c_int
    name = C.create_string_buffer(256)
    ordinal = C.c_int()
    cu = C.c_int()
    checked(
        module.device(name, len(name), C.byref(ordinal), C.byref(cu)), "module device"
    )
    r = Resources(sdk)
    handle = C.c_void_p()
    graph = C.c_void_p()
    executable = C.c_void_p()
    q, n, k, e = w.q, w.n, w.k, w.experts
    vectors = profiles(e)
    m = sum(vectors[0])
    # Model integration knows the token count, not the actual maximum expert
    # rows. Do not quietly price a tighter host-oracle bound than it can use.
    maximum = 128 if e == 256 else max(max(rows) for rows in vectors)
    packed = record["parent"]["route"] == "fq-grouped"
    try:
        planes = {
            x: r.upload(w.planes[x]) if w.planes[x].size else None
            for x in ("low", "high", "units")
        }
        plane_bytes = w.planes["scale"].nbytes
        scale, zero = r.alloc(plane_bytes), r.alloc(plane_bytes)
        arr = arrangement(q)

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

        first_prepass = r.samples(prepass, 1)[0] if not packed else None
        ap = r.alloc(m * k * 2)
        bounds = r.alloc((e + 1) * 4)
        device_rows = r.alloc(e * 4)
        output = r.alloc(m * n * 2 + 32)
        c = Call(
            version=1,
            size=C.sizeof(Call),
            m=m,
            n=n,
            k=k,
            experts=e,
            group_size=arr.group_size,
            device=ordinal.value,
            compute_units=cu.value,
            mapping_id=arr.mapping_id,
            a=ap,
            low=planes["low"],
            high=planes["high"],
            metadata=planes["units"] if packed else scale,
            zero=None if packed else zero,
            output=output + 16,
            offsets_device=bounds,
            stream=r.stream.value,
        )
        d = DeviceCall(2, C.sizeof(DeviceCall), c, maximum, 0)
        persistent = config["algorithm"] == "GROUPED_PERSISTENT"
        recipe = Recipe(1, C.sizeof(Recipe), int(persistent), 1, 1 if persistent else 0)
        resources = QueryResources()
        checked(v2query(C.byref(d), C.byref(recipe), C.byref(resources)), "v2 query")
        if persistent:
            upper = (
                e
                * ((maximum + config["tm"] - 1) // config["tm"])
                * ((n + config["tn"] - 1) // config["tn"])
            )
            recipe.grid = min(cu.value * resources.occupancy, upper)
        d.call.workspace = r.alloc(resources.workspace_bytes)
        d.call.workspace_bytes = resources.workspace_bytes
        checked(v2prepare(C.byref(d), C.byref(recipe), C.byref(handle)), "v2 prepare")
        # The old ABI remains a separate owned handle. No synthetic host row
        # vector is passed to v2 to make its validation appear to succeed.
        values = []
        for index, rows in enumerate(vectors):
            request = Request(record["parent"]["route"], q, n, k, m, rows)
            a, gold, denom = w.activation(request)
            offsets = np.r_[0, np.cumsum(rows)].astype("i4")
            rows_np = np.array(rows, dtype="i4")
            for dest, array in ((ap, a), (bounds, offsets), (device_rows, rows_np)):
                checked(
                    sdk.lib.hggcMemcpy(dest, array.ctypes.data, array.nbytes, 1),
                    "fixture upload",
                )
            sdk.fill(output, 0xA5, m * n * 2 + 32)
            if index == 0:
                checked(module.run(handle, r.stream), "v2 eager")
                sdk.synchronize(r.stream)
                checked(sdk.lib.hggcStreamBeginCapture(r.stream, 0), "begin capture")
                checked(module.run(handle, r.stream), "v2 capture")
                checked(
                    sdk.lib.hggcStreamEndCapture(r.stream, C.byref(graph)),
                    "end capture",
                )
                checked(
                    sdk.lib.hggcGraphInstantiateWithFlags(
                        C.byref(executable), graph, 0
                    ),
                    "instantiate",
                )
            checked(
                sdk.lib.hggcGraphLaunch(executable, r.stream), "mutable routing replay"
            )
            sdk.synchronize(r.stream)
            raw = sdk.download(output, m * n * 2 + 32)
            if raw[:16] != b"\xa5" * 16 or raw[-16:] != b"\xa5" * 16:
                raise ValueError("v2 output guards")
            got = np.frombuffer(raw[16:-16], dtype="<f2").reshape(m, n)
            error = compare(got, dict(golden=gold, denom=denom))
            host_rows = (C.c_int32 * e)(*rows)
            old = Call.from_buffer_copy(d.call)
            old.rows_host = C.cast(host_rows, C.c_void_p)
            old.rows_device = device_rows
            old.output = r.alloc(m * n * 2)
            oldresources = QueryResources()
            checked(
                module.query(C.byref(old), C.byref(recipe), C.byref(oldresources)),
                "v1 query",
            )
            old.workspace = r.alloc(oldresources.workspace_bytes)
            old.workspace_bytes = oldresources.workspace_bytes
            oldhandle = C.c_void_p()
            checked(
                module.prepare(C.byref(old), C.byref(recipe), C.byref(oldhandle)),
                "v1 prepare",
            )
            try:
                checked(module.run(oldhandle, r.stream), "v1 run")
                sdk.synchronize(r.stream)
                baseline = sdk.download(old.output, m * n * 2)
                if baseline != raw[16:-16]:
                    raise ValueError("v2 output differs from same-parent v1")
                old_samples = r.samples(lambda: module.run(oldhandle, r.stream), 5)
            finally:
                sdk.synchronize(r.stream)
                module.destroy(oldhandle)
            samples = r.samples(lambda: module.run(handle, r.stream), 5)
            combined = None
            if not packed:

                def both():
                    status = prepass()
                    return status if status else module.run(handle, r.stream)

                combined = r.samples(both, 5)
            values.append(
                dict(
                    profile=index,
                    rows=list(rows),
                    error=error,
                    v1_raw_equal=True,
                    graph_replay=True,
                    resident_samples_us=samples,
                    v1_resident_samples_us=old_samples,
                    prepass_plus_sf_samples_us=combined,
                )
            )
        return dict(
            parent=record["parent"],
            n=n,
            k=k,
            e=e,
            m=m,
            max_rows=maximum,
            recipe=dict(algorithm=recipe.algorithm, grid=recipe.grid, split=1),
            first_prepass_us=first_prepass,
            metadata_output_bytes=2 * plane_bytes if not packed else 0,
            values=values,
            status="PASS",
        )
    finally:
        sdk.synchronize(r.stream)
        if executable:
            checked(sdk.lib.hggcGraphExecDestroy(executable), "destroy graph exec")
        if graph:
            checked(sdk.lib.hggcGraphDestroy(graph), "destroy graph")
        if handle:
            module.destroy(handle)
        r.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--execution-bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--real-model-shapes", action="store_true")
    args = p.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if manifest["schema"] != "quactlize.kpack-grouped-device-build.v2":
        raise ValueError("grouped bundle schema")
    records = {x["parent"]["symbol"]: x for x in manifest["modules"]}
    if len(records) != 10 or len(manifest["selected"]) != 10:
        raise ValueError("grouped parent denominator")
    execution = json.loads((args.execution_bundle / "manifest.json").read_text())
    path = args.execution_bundle / execution["library"]
    if sha(path) != execution["sha256"]:
        raise ValueError("execution payload differs")
    for name, value in execution["runtime"].items():
        if sha(args.sdk / "lib" / name) != value:
            raise ValueError(f"SDK runtime differs: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    sdk = SDK(args.sdk)
    identity = device_identity(sdk)
    graph_bind(sdk)
    _, _, sf = bind(C.CDLL(str(path.resolve()), mode=C.RTLD_LOCAL))
    summary = dict(
        status="INCOMPLETE",
        device=identity,
        results=[],
        failures=[],
        device_only=True,
        manifest_sha256=sha(args.bundle / "manifest.json"),
        heuristic_admitted=False,
    )
    weights = {}
    work = []
    for entry in manifest["selected"]:
        q = entry["parent"]["qtype"]
        shapes = [(256, 512, 4)]
        if args.real_model_shapes and q in (12, 13):
            shapes.append((512, 2048, 256) if q == 12 else (2048, 512, 256))
        for n, k, e in shapes:
            work.append(((q, n, k, e), entry))
    work.sort(key=lambda item: (item[0], item[1]["parent"]["route"]))
    try:
        for key, entry in work:
            q, n, k, e = key
            if key not in weights:
                # Keep at most one large host fixture live.
                weights = {
                    key: Weights(
                        q,
                        n,
                        k,
                        e,
                        progress=lambda done, total: print(
                            f"KPACK_GROUPED_FIXTURE q={q} experts={done}/{total}",
                            flush=True,
                        ),
                    )
                }
            record = dict(records[entry["parent"]["symbol"]])
            record["path"] = str((args.bundle / record["path"]).resolve())
            try:
                result = run_parent(sdk, record, entry["config"], weights[key], sf)
                summary["results"].append(result)
                print(
                    f'KPACK_GROUPED_DEVICE q={q} route={entry["parent"]["route"]} N={n} K={k} E={e} profiles=3 status=PASS',
                    flush=True,
                )
            except Exception as exc:
                traceback.print_exc()
                summary["failures"].append(
                    dict(parent=entry["parent"], shape=list(key), error=str(exc))
                )
        summary["status"] = "FAIL" if summary["failures"] else "PASS"
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f'KPACK_GROUPED_DEVICE_GATE status={summary["status"]} results={args.output}',
        flush=True,
    )
    return int(summary["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
