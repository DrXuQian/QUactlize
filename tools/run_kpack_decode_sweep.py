#!/usr/bin/env python3
"""Execute the bounded grouped Split-K / SIMT experiment, without compilation.

Each parent runs in a fresh process. A failed parent cannot discard another
parent's result. --resume retries only failed/missing jobs with identical inputs.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import (
    SDK,
    Call,
    Module,
    Recipe,
    Resources as Query,
    checked,
    sdk_identity,
)
from quactlize.execution.native import Call as VecCall, Config, Sizes, arrangement, bind
from tools.kpack_execution_fixture import IndexedWeights
from tools.kpack_warmup_fixture import activation_values
from tools.run_kpack_gemv_gate import Resources, fixture, compare
from tools.run_kpack_grouped_device_gate import DeviceCall, graph_bind
from tools.run_kpack_grouped_decode_probe import Replay, timing_summary
from tools.run_kpack_pack_gate import device_identity

SCHEMA = "quactlize.kpack-decode-sweep.v1"
CASES = {12: (512, 2048), 13: (2048, 512)}


def verify(bundle):
    m = json.loads((bundle / "manifest.json").read_text())
    if m.get("schema") != SCHEMA or m.get("production_selection_changed") is not False:
        raise ValueError("wrong experimental bundle manifest")
    paths = set()
    for r in m["modules"]:
        path = (bundle / r["path"]).resolve(strict=True)
        if not path.is_relative_to(bundle) or path in paths or sha(path) != r["sha256"]:
            raise ValueError("module path/hash/uniqueness differs")
        paths.add(path)
    if sha(bundle / "libquactlize_ppu_execution.so") != m["execution"]["sha256"]:
        raise ValueError("SIMT payload differs")
    return m


def grouped_data(w, rows):
    rows = np.asarray(rows, dtype="i4")
    expert = np.repeat(np.arange(w.experts), rows)
    values = activation_values(np.arange(len(expert)))
    return dict(
        a=values[:, w.categories[0]].astype("<f2"),
        arows=np.arange(len(expert)),
        expert=expert,
        values=values,
        golden=np.stack([values[i] @ w.sums[e] for i, e in enumerate(expert)]),
        denom=np.stack(
            [np.abs(values[i]) @ w.abs_sums[e] for i, e in enumerate(expert)]
        ),
    )


def partial_gold(w, data, tk, split):
    entries = w.partial_sums[tk, split]
    sums = np.empty((split, len(data["expert"]), w.n), dtype="f8")
    absolute = np.empty_like(sums)
    for row, e in enumerate(data["expert"]):
        values = data["values"][data["arows"][row]]
        for s in range(split):
            sums[s, row] = values @ entries[int(e)][0][s]
            absolute[s, row] = np.abs(values) @ entries[int(e)][1][s]
    return dict(golden=sums, denom=absolute)


def check_partials(raw, output, w, data, tk, split, order):
    got = np.frombuffer(raw, dtype="<f4").reshape(split, len(order), w.n)
    reference = partial_gold(w, data, tk, split)
    err = admit(got[:, np.argsort(order)], reference, "partial [split,row,N]")
    total = np.zeros_like(got[0])
    for plane in got:
        np.add(total, plane, out=total)
    if not np.array_equal(total.astype("<f2").view("u2"), output.view("u2")):
        raise ValueError("ordered reducer differs from downloaded FP32 partials")
    return err


def admit(got, data, seam):
    try:
        return compare(got, data)
    except ValueError as error:
        bad = (~np.isfinite(got)) | (
            np.abs(got.astype("f8") - data["golden"]) / np.maximum(data["denom"], 1e-30)
            >= 0.005
        )
        first = tuple(int(x) for x in np.argwhere(bad)[0])
        raise ValueError(
            f"{seam}: bad={int(bad.sum())}/{bad.size} first={first} "
            f"want={float(data['golden'][first]):.9g} got={float(got[first]):.9g}; {error}"
        ) from error


def upload_into(sdk, p, a):
    a = np.ascontiguousarray(a)
    checked(sdk.lib.hggcMemcpy(p, a.ctypes.data, a.nbytes, 1), "fixture H2D")


def bind_group(module):
    query = module.lib.quactlize_kpack_grouped_query_v2
    prepare = module.lib.quactlize_kpack_grouped_prepare_v2
    query.argtypes, query.restype = [
        C.POINTER(DeviceCall),
        C.POINTER(Recipe),
        C.POINTER(Query),
    ], C.c_int
    prepare.argtypes, prepare.restype = [
        C.POINTER(DeviceCall),
        C.POINTER(Recipe),
        C.POINTER(C.c_void_p),
    ], C.c_int
    return query, prepare


def gemm_cell(
    args, sdk, module, w, profiles, split, arm, *, grid_b=0, gpu_directory=False
):
    data = profiles[0]
    order = np.argsort(data["expert"], kind="stable")
    rows = np.bincount(data["expert"], minlength=w.experts).astype("i4")
    offsets = np.r_[0, rows.cumsum()].astype("i4")
    m = len(order)
    rec = module.record["parent"]
    maximum = max(
        int(np.bincount(x["expert"], minlength=w.experts).max()) for x in profiles
    )
    tile_m = rec["tm"]
    active_bound = min(m, w.experts)
    directory_capacity = min(
        (m + active_bound * (tile_m - 1)) // tile_m,
        w.experts * ((maximum + tile_m - 1) // tile_m),
    )
    work_upper = directory_capacity * ((w.n + rec["tn"] - 1) // rec["tn"]) * split
    r = Resources(sdk)
    handle = C.c_void_p()
    graph = None
    result = dict(
        kind="tc",
        parent=rec["symbol"],
        split=split,
        arm=arm,
        q=w.q,
        shape=[m, w.n, w.k],
        experts=w.experts,
        active_experts=int((rows > 0).sum()),
        max_rows=int(rows.max()),
        status="FAIL",
        partial_error=0.0,
        algorithm="PERSISTENT" if grid_b else "ORDINARY",
        grid_b=grid_b,
        gpu_directory=gpu_directory,
    )
    try:
        dev = module.device_identity()
        planes = {
            k: r.upload(w.planes[k]) if w.planes[k].size else None
            for k in ("low", "high", "units")
        }
        sf = rec.get("route") == "sf-grouped"
        if sf:
            from tools.run_kpack_gemv_gate import metadata_oracle

            scale, zero = metadata_oracle(w.planes["units"], w.q, w.n, w.k, w.experts)
            planes["scale"], planes["zero"] = r.upload(scale), r.upload(zero)
        outbytes = m * w.n * 2
        output = r.alloc(outbytes + 32)
        sdk.fill(output, 0xA5, outbytes + 32)
        c = Call(
            version=1,
            size=C.sizeof(Call),
            m=m,
            n=w.n,
            k=w.k,
            experts=w.experts,
            group_size=arrangement(w.q).group_size,
            device=dev["ordinal"],
            compute_units=dev["compute_units"],
            mapping_id=arrangement(w.q).mapping_id,
            a=r.upload(data["a"][data["arows"][order]]),
            low=planes["low"],
            high=planes["high"],
            metadata=planes["scale"] if sf else planes["units"],
            zero=planes["zero"] if sf else None,
            output=output + 16,
            offsets_device=r.upload(offsets),
            stream=r.stream.value,
        )
        recipe = Recipe(1, C.sizeof(Recipe), int(grid_b > 0), split, 1 if grid_b else 0)
        query = Query()
        v2query, v2prepare = bind_group(module)
        if arm == "device-only":
            d = DeviceCall(
                2,
                C.sizeof(DeviceCall),
                c,
                maximum,
                0,
            )
            checked(
                v2query(C.byref(d), C.byref(recipe), C.byref(query)),
                "Split-K device query",
            )
        else:
            c.rows_host, c.rows_device = rows.ctypes.data, r.upload(rows)
            checked(
                module.query(C.byref(c), C.byref(recipe), C.byref(query)),
                "Split-K compact query",
            )
        if grid_b:
            recipe.grid = min(
                work_upper, dev["compute_units"] * min(grid_b, query.occupancy)
            )
            if recipe.grid < 1:
                raise ValueError("persistent grid has no admitted residency")
            checked(
                (
                    v2query(C.byref(d), C.byref(recipe), C.byref(query))
                    if arm == "device-only"
                    else module.query(C.byref(c), C.byref(recipe), C.byref(query))
                ),
                "persistent grid query",
            )
        workspace = r.alloc(query.workspace_bytes + 32)
        sdk.fill(workspace, 0xA5, query.workspace_bytes + 32)
        c.workspace, c.workspace_bytes = workspace + 16, query.workspace_bytes
        if arm == "device-only":
            d.call = c
            checked(
                v2prepare(C.byref(d), C.byref(recipe), C.byref(handle)),
                "Split-K device prepare",
            )
        else:
            checked(
                module.prepare(C.byref(c), C.byref(recipe), C.byref(handle)),
                "Split-K compact prepare",
            )
        sdk.synchronize(r.stream)
        launch = lambda: module.run(handle, r.stream)
        graph = Replay(sdk, r.stream, launch, args.graph_repeats)
        errors = []
        partialbytes = split * m * w.n * 4 if split > 1 else 0
        partialptr = c.workspace + query.workspace_bytes - partialbytes
        for index, profile in enumerate(
            profiles if arm == "device-only" else profiles[:1]
        ):
            permutation = np.argsort(profile["expert"], kind="stable")
            rowcounts = np.bincount(profile["expert"], minlength=w.experts).astype("i4")
            bounds = np.r_[0, rowcounts.cumsum()].astype("i4")
            upload_into(sdk, c.offsets_device, bounds)
            upload_into(sdk, c.a, profile["a"][profile["arows"][permutation]])
            for repeat in range(args.correctness_repeats):
                sdk.fill(output, 0xA5, outbytes + 32)
                if partialbytes:
                    sdk.fill(partialptr, 0xFF, partialbytes)
                if gpu_directory:
                    sdk.fill(c.workspace, 0xA5, 16 + 16 * directory_capacity)
                checked(launch() if repeat == 0 else graph(), "Split-K correctness")
                sdk.synchronize(r.stream)
                raw = sdk.download(output, outbytes + 32)
                if (
                    raw[:16] != b"\xa5" * 16
                    or raw[-16:] != b"\xa5" * 16
                    or sdk.download(workspace, 16) != b"\xa5" * 16
                    or sdk.download(c.workspace + query.workspace_bytes, 16)
                    != b"\xa5" * 16
                ):
                    raise ValueError("output/workspace guard changed")
                got = np.frombuffer(raw[16:-16], dtype="<f2").reshape(m, w.n)
                if gpu_directory:
                    prefix = np.r_[0, np.cumsum((rowcounts + tile_m - 1) // tile_m)]
                    header = np.frombuffer(sdk.download(c.workspace, 16), dtype="<i4")
                    if not np.array_equal(header, [prefix[-1], 0, tile_m, w.experts]):
                        raise ValueError(
                            f"device directory header differs: {header.tolist()}"
                        )
                    entries = np.frombuffer(
                        sdk.download(c.workspace + 16, int(prefix[-1]) * 16),
                        dtype="<i4",
                    ).reshape(-1, 4)
                    expected = np.array(
                        [
                            [expert, rowcounts[expert], prefix[expert], bounds[expert]]
                            for expert in range(w.experts)
                            for _ in range(int(prefix[expert + 1] - prefix[expert]))
                        ],
                        dtype="<i4",
                    )
                    if not np.array_equal(entries, expected):
                        raise ValueError("device directory entry/owner differs")
                    used = int(prefix[-1])
                    if used > directory_capacity or sdk.download(
                        c.workspace + 16 + used * 16,
                        (directory_capacity - used) * 16,
                    ) != b"\xa5" * ((directory_capacity - used) * 16):
                        raise ValueError("device directory wrote unused capacity")
                errors.append(
                    admit(
                        got[np.argsort(permutation)], profile, "grouped output [row,N]"
                    )
                )
                if partialbytes:
                    error = check_partials(
                        sdk.download(partialptr, partialbytes),
                        got,
                        w,
                        profile,
                        rec["tk"],
                        split,
                        permutation,
                    )
                    result["partial_error"] = max(result["partial_error"], error)
        # Restore the timed router/input. Timings include metadata, producer,
        # and (S>1) reducer, but no fixture H2D/D2H or Python per-kernel gap.
        upload_into(sdk, c.offsets_device, offsets)
        upload_into(sdk, c.a, data["a"][data["arows"][order]])
        for _ in range(args.warmups):
            checked(graph(), "warmup")
        sdk.synchronize(r.stream)
        timing = [r.samples(graph, args.samples) for _ in range(args.rounds)]
        weights = sum(w.planes[k].nbytes for k in ("low", "high"))
        weights += scale.nbytes + zero.nbytes if sf else w.planes["units"].nbytes
        weights = weights // w.experts * int((rows > 0).sum())
        result.update(timing_summary(timing, args.graph_repeats, weights))
        result.update(
            status="PASS",
            error=max(errors),
            shared_bytes=query.shared_bytes,
            workspace_bytes=query.workspace_bytes,
            partial_bytes=partialbytes,
            partial_format="FP32",
            correctness_checks=len(errors),
            profiles_checked=len(profiles) if arm == "device-only" else 1,
            timing_scope=(
                "METADATA_PLUS_" if arm == "device-only" else "HOST_PREPARED_"
            )
            + ("DIRECTORY_PLUS_" if gpu_directory else "")
            + "GEMM_PLUS_REDUCER",
            valid_ctas=sum((int(x) + rec["tm"] - 1) // rec["tm"] for x in rows)
            * ((w.n + rec["tn"] - 1) // rec["tn"])
            * split,
            grid=(
                [1, (w.n + rec["tn"] - 1) // rec["tn"], w.experts * split]
                if arm == "device-only" and rows.max() == 1
                else "SOURCE_GRID_DEPENDS_ON_ROW_TILES"
            ),
        )
        if gpu_directory:
            result["grid"] = [recipe.grid if grid_b else work_upper, 1, 1]
            result["directory_capacity"] = directory_capacity
        return result
    finally:
        sdk.synchronize(r.stream)
        if graph is not None:
            graph.close()
        if handle:
            module.destroy(handle)
        r.close()


def parent_job(args, sdk, manifest, index):
    record = manifest["modules"][index]
    module = Module(record | {"path": str(args.bundle / record["path"])})
    p = record["parent"]
    cells = []
    for shape in ("control", "model"):
        n, k = (256, CASES[p["qtype"]][1]) if shape == "control" else CASES[p["qtype"]]
        e = 4 if shape == "control" else 256
        splits = [
            s
            for s in (1, 2, 4, 8)
            if k % (p["tk"] * s) == 0 and k // (p["tk"] * s) >= p["stages"] - 1
        ]
        ids = np.arange(8) * 17
        selected = range(4) if e == 4 else np.r_[ids, (ids + 1) % e]
        w = IndexedWeights(
            p["qtype"],
            n,
            k,
            e,
            partial_specs=[(p["tk"], s) for s in splits if s > 1],
            partial_experts=selected,
        )
        if e == 4:
            profiles = [grouped_data(w, x) for x in ([9, 0, 3, 1], [0, 3, 1, 9])]
        else:
            first = fixture(
                w, dict(mode=2, rows=8, topk=8, channels=1 if w.q == 12 else 8)
            )
            first["values"] = activation_values(np.arange(len(first["a"])))
            second = {**first, "expert": (first["expert"] + 1) % e}
            second["golden"] = np.stack(
                [
                    first["values"][first["arows"][i]] @ w.sums[expert]
                    for i, expert in enumerate(second["expert"])
                ]
            )
            second["denom"] = np.stack(
                [
                    np.abs(first["values"][first["arows"][i]]) @ w.abs_sums[expert]
                    for i, expert in enumerate(second["expert"])
                ]
            )
            profiles = [first, second]
        for s in splits:
            if s > 1:
                negative = partial_gold(w, profiles[0], p["tk"], s)
                try:
                    compare(np.zeros_like(negative["golden"]), negative)
                except ValueError:
                    pass
                else:
                    raise ValueError("missing partial plane is not detected by fixture")
            for arm in ("device-only", "host-compact"):
                print(
                    f"KPACK_DECODE_PROGRESS parent={index} shape={shape} split={s} arm={arm}",
                    flush=True,
                )
                result = gemm_cell(args, sdk, module, w, profiles, s, arm)
                cells.append(result)
                print("KPACK_DECODE_CELL " + json.dumps(result), flush=True)
    return cells


def pair_bind(lib):
    query = lib.quactlize_kpack_gemv_pair_query_v1
    run = lib.quactlize_kpack_gemv_pair_run_v1
    old_query, old_run, _ = bind(lib)
    query.argtypes, query.restype = old_query.argtypes, old_query.restype
    run.argtypes, run.restype = old_run.argtypes, old_run.restype
    return {"scalar": (old_query, old_run), "pair": (query, run)}


def simt_cell(args, sdk, functions, w, case, config, variant):
    data = fixture(w, case)
    r = Resources(sdk)
    graph = None
    try:
        query, run = functions[variant]
        arr = arrangement(w.q)
        config = Config(*config)
        outbytes = case["rows"] * w.n * 4
        output = r.alloc(outbytes + 32)
        planes = {
            k: r.upload(w.planes[k]) if w.planes[k].size else None
            for k in ("low", "high", "units")
        }
        c = VecCall(
            version=1,
            size=C.sizeof(VecCall),
            qtype=w.q,
            n=w.n,
            k=w.k,
            experts=w.experts,
            rows=case["rows"],
            mode=case["mode"],
            input_type=1,
            channels=case["channels"],
            topk=case["topk"],
            a_row_stride=w.k,
            a_token_stride=case["channels"] * w.k,
            ids_stride=case["topk"],
            out_row_stride=w.n,
            a=r.upload(data["a"].astype("<f4")),
            low=planes["low"],
            high=planes["high"],
            units=planes["units"],
            offsets=r.upload(data["offsets"]) if data["offsets"] is not None else None,
            ids=r.upload(data["ids"]) if data["ids"] is not None else None,
            output=output + 16,
            stream=r.stream.value,
        )
        sizes = Sizes()
        checked(
            query(C.byref(c), C.byref(config), C.byref(arr), C.byref(sizes)),
            "SIMT query",
        )
        workspace = r.alloc(sizes.workspace_bytes + 32)
        sdk.fill(workspace, 0xA5, sizes.workspace_bytes + 32)
        c.workspace, c.workspace_bytes = workspace + 16, sizes.workspace_bytes
        launch = lambda: run(C.byref(c), C.byref(config), C.byref(arr))
        graph = Replay(sdk, r.stream, launch, args.graph_repeats)
        errors = []
        for repeat in range(args.correctness_repeats):
            sdk.fill(output, 0xA5, outbytes + 32)
            if sizes.workspace_bytes:
                sdk.fill(c.workspace, 0xFF, sizes.workspace_bytes)
            checked(launch() if repeat == 0 else graph(), "SIMT correctness")
            sdk.synchronize(r.stream)
            raw = sdk.download(output, outbytes + 32)
            if (
                raw[:16] != b"\xa5" * 16
                or raw[-16:] != b"\xa5" * 16
                or sdk.download(workspace, 16) != b"\xa5" * 16
                or sdk.download(c.workspace + sizes.workspace_bytes, 16) != b"\xa5" * 16
            ):
                raise ValueError("SIMT output/workspace guard changed")
            got = np.frombuffer(raw[16:-16], dtype="<f4").reshape(case["rows"], w.n)
            errors.append(admit(got, data, "SIMT output [row,N]"))
        for _ in range(args.warmups):
            checked(graph(), "SIMT warmup")
        timing = [r.samples(graph, args.samples) for _ in range(args.rounds)]
        weightbytes = sum(
            w.planes[k].nbytes // w.experts * len(set(data["expert"])) for k in planes
        )
        return dict(
            kind="simt",
            variant=variant,
            q=w.q,
            shape=[case["rows"], w.n, w.k],
            experts=w.experts,
            case=case,
            columns=config.columns,
            warps=config.warps,
            split=config.split,
            error=max(errors),
            status="PASS",
            workspace_bytes=sizes.workspace_bytes,
            timing_scope="SIMT_F32_INPUT_OUTPUT_INCLUDING_INDEXING_AND_REDUCER",
            **timing_summary(timing, args.graph_repeats, weightbytes),
        )
    finally:
        sdk.synchronize(r.stream)
        if graph is not None:
            graph.close()
        r.close()


def simt_job(args, sdk, q):
    lib = C.CDLL(str(args.bundle / "libquactlize_ppu_execution.so"), mode=C.RTLD_LOCAL)
    functions = pair_bind(lib)
    cells = []
    geometries = [(256, 512, 1), (256, 512, 4)]
    if q in CASES:
        geometries.append((*CASES[q], 256))
    for n, k, e in geometries:
        model = e == 256
        w = IndexedWeights(q, n, k, e)
        cases = (
            [dict(mode=2, rows=8, channels=1 if q == 12 else 8, topk=8)]
            if model
            else (
                [dict(mode=0, rows=1, channels=1, topk=1)]
                if e == 1
                else [
                    dict(mode=1, rows=6, channels=1, topk=1),
                    dict(mode=2, rows=8, channels=1, topk=2),
                    dict(mode=2, rows=8, channels=2, topk=2),
                ]
            )
        )
        for case in cases:
            for variant in ("scalar", "pair"):
                configs = (
                    [
                        (c, w, s)
                        for c in (16, 32)
                        for w in ((4, 8) if variant == "scalar" else (2, 4, 8))
                        for s in ((1, 4) if variant == "scalar" else (1, 2, 4, 8))
                    ]
                    if model
                    else (
                        [(16, 8, 1)]
                        if variant == "scalar"
                        else [(16, 2, 2), (32, 8, 4)]
                    )
                )
                for config in configs:
                    print(
                        f"KPACK_DECODE_PROGRESS simt=q{q} model={int(model)} variant={variant} config={config}",
                        flush=True,
                    )
                    result = simt_cell(args, sdk, functions, w, case, config, variant)
                    cells.append(result)
                    print("KPACK_DECODE_CELL " + json.dumps(result), flush=True)
    return cells


def jobs(manifest):
    return [f"tc-{i}" for i in range(len(manifest["modules"]))] + [
        f"simt-{q}" for q in range(10, 15)
    ]


def expected_cells(job, manifest):
    if job.startswith("simt-"):
        return 44 if int(job[5:]) in CASES else 12
    p = manifest["modules"][int(job[3:])]["parent"]
    k = CASES[p["qtype"]][1]
    return 4 * sum(
        k % (p["tk"] * s) == 0 and k // (p["tk"] * s) >= p["stages"] - 1
        for s in (1, 2, 4, 8)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument(
        "--bundle", type=Path, default=ROOT / "prebuilt/ppu0010/kpack-decode-sweep-v1"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--job")
    p.add_argument("--samples", type=int, default=11)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--graph-repeats", type=int, default=16)
    p.add_argument("--correctness-repeats", type=int, default=3)
    args = p.parse_args()
    if (
        min(
            args.samples,
            args.rounds,
            args.warmups,
            args.graph_repeats,
            args.correctness_repeats,
        )
        < 1
    ):
        p.error("counts must be positive")
    args.bundle = args.bundle.resolve(strict=True)
    manifest = verify(args.bundle)
    counts = {
        k: getattr(args, k)
        for k in (
            "samples",
            "rounds",
            "warmups",
            "graph_repeats",
            "correctness_repeats",
        )
    }
    authority = dict(
        manifest=sha(args.bundle / "manifest.json"),
        runner=sha(Path(__file__)),
        counts=counts,
        sdk=sdk_identity(args.sdk),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device=device_identity(SDK(args.sdk)),
        source={
            str(path.relative_to(ROOT)): sha(path)
            for path in [
                ROOT / "tools/kpack_execution_fixture.py",
                ROOT / "tools/kpack_warmup_fixture.py",
                ROOT / "tools/run_kpack_gemv_gate.py",
                ROOT / "tools/run_kpack_grouped_decode_probe.py",
                ROOT / "tools/run_kpack_grouped_device_gate.py",
                ROOT / "quactlize/runtime/native.py",
                ROOT / "reference/gguf_kpack.py",
            ]
        },
    )
    if args.job:
        if args.job not in jobs(manifest):
            p.error("unknown job")
        args.output.mkdir(parents=True, exist_ok=True)
        result = dict(
            status="FAIL",
            job=args.job,
            authority=authority,
            cells=[],
            device_admission="PENDING",
        )
        start = time.monotonic()
        try:
            sdk = SDK(args.sdk)
            graph_bind(sdk)
            result["device"] = device_identity(sdk)
            result["cells"] = (
                parent_job(args, sdk, manifest, int(args.job[3:]))
                if args.job.startswith("tc-")
                else simt_job(args, sdk, int(args.job[5:]))
            )
            if len(result["cells"]) != expected_cells(args.job, manifest) or any(
                x["status"] != "PASS" for x in result["cells"]
            ):
                raise ValueError("job correctness/timing denominator differs")
            result["status"] = "PASS"
            result["device_admission"] = "CORRECTNESS_PASS_PERFORMANCE_PENDING_REVIEW"
        except Exception as exc:
            traceback.print_exc()
            result["error"] = str(exc)
        result["seconds"] = time.monotonic() - start
        (args.output / (args.job + ".json")).write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            f"KPACK_DECODE_JOB job={args.job} status={result['status']} seconds={result['seconds']:.1f}",
            flush=True,
        )
        return int(result["status"] != "PASS")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    receipt = args.output / "authority.json"
    if receipt.exists() and json.loads(receipt.read_text()) != authority:
        raise ValueError("resume inputs/counts differ")
    receipt.write_text(json.dumps(authority, indent=2) + "\n")
    start = time.monotonic()
    results = []
    for i, job in enumerate(jobs(manifest)):
        target = args.output / (job + ".json")
        old = json.loads(target.read_text()) if target.exists() else None
        if (
            args.resume
            and old
            and old.get("status") == "PASS"
            and old.get("authority") == authority
        ):
            print(f"KPACK_DECODE_RESUME job={job} status=PASS", flush=True)
        else:
            command = [
                sys.executable,
                str(Path(__file__)),
                "--sdk",
                str(args.sdk),
                "--bundle",
                str(args.bundle),
                "--output",
                str(args.output),
                "--job",
                job,
            ]
            for key, value in counts.items():
                command += ["--" + key.replace("_", "-"), str(value)]
            with (args.output / (job + ".log")).open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT
                )
                print(f"KPACK_DECODE_START job={job} log={log.name}", flush=True)
                while True:
                    try:
                        rc = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        print(
                            f"KPACK_DECODE_ALIVE job={job} completed={i}/{len(jobs(manifest))} elapsed_minutes={(time.monotonic()-start)/60:.1f}",
                            flush=True,
                        )
            if not target.exists():
                target.write_text(
                    json.dumps(
                        dict(
                            status="FAIL",
                            job=job,
                            process_rc=rc,
                            authority=authority,
                            cells=[],
                        )
                    )
                    + "\n"
                )
        result = json.loads(target.read_text())
        results.append(result)
        elapsed = time.monotonic() - start
        print(
            f"KPACK_DECODE_PROGRESS completed={i+1}/{len(jobs(manifest))} status={result['status']} elapsed_minutes={elapsed/60:.1f}",
            flush=True,
        )
    failed = [x["job"] for x in results if x["status"] != "PASS"]
    summary = dict(
        status="FAIL" if failed else "PASS",
        failed_jobs=failed,
        authority=authority,
        elapsed_seconds=time.monotonic() - start,
        measured_cells=sum(len(x["cells"]) for x in results if x["status"] == "PASS"),
        expected_cells=sum(expected_cells(job, manifest) for job in jobs(manifest)),
        jobs=results,
        production_selection_changed=False,
        performance_admission="PENDING_REVIEW",
        excluded="LLAMA_ADAPTERS_AND_REFERENCE_MMVQ_FUSION",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"KPACK_DECODE_DONE status={summary['status']} cells={summary['measured_cells']}/{summary['expected_cells']} failed={failed} results={args.output}",
        flush=True,
    )
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
