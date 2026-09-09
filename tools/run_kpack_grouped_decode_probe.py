#!/usr/bin/env python3
"""Isolate a model-shaped grouped decode call using existing native modules.

device-only is the production v2 call (GPU metadata + rectangular GEMM).
host-compact is a diagnostic v1 call to the SAME parent, with resident host-
prepared metadata and a compact grid. It is not a deployable dynamic-router
replacement. Neither arm includes llama.cpp IDs/gather/scatter or MMVQ fusion.
"""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.dispatch.native import Dispatch, receipt
from quactlize.execution.native import arrangement
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import (
    SDK,
    Call,
    Module,
    Recipe,
    Resources as QueryResources,
    checked,
)
from tools.kpack_execution_fixture import IndexedWeights
from tools.run_kpack_gemv_gate import Resources, compare, fixture
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_pack_gate import device_identity
from tools.verify_kpack_dispatch import verify

CASES = {"q4-up": (12, 512, 2048), "q5-down": (13, 2048, 512)}
ARMS = ("device-only", "host-compact")


def timing_summary(rounds, repeats, weight_bytes):
    values = [float(x) / repeats for row in rounds for x in row]
    if not values or not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("missing/nonfinite/nonpositive graph timing")
    median = statistics.median(values)
    return dict(
        median_us=median,
        min_us=min(values),
        max_us=max(values),
        graph_elapsed_samples_us=rounds,
        calls_per_graph=repeats,
        effective_weight_GB_per_s=weight_bytes / (median * 1000),
        bandwidth_scope="DISTINCT_ACTIVE_WEIGHT_BYTES_OVER_CALL_TIME_NOT_MEASURED_DRAM",
    )


def grid_model(parent, rows, n):
    if parent["persistent"] or max(rows) != 1 or sum(rows) != 8:
        raise ValueError("this probe requires nonpersistent single-token top8 decode")
    nt = (n + parent["tn"] - 1) // parent["tn"]
    return {"device-only": [1, nt, len(rows)], "host-compact": [8, nt, 1]}


class Replay:
    def __init__(self, sdk, stream, launch, repeats):
        self.sdk, self.stream = sdk, stream
        self.graph, self.instance = C.c_void_p(), C.c_void_p()
        checked(sdk.lib.hggcStreamBeginCapture(stream, 0), "begin capture")
        try:
            for _ in range(repeats):
                checked(launch(), "capture selected call")
        except BaseException:
            sdk.lib.hggcStreamEndCapture(stream, C.byref(self.graph))
            self.close()
            raise
        checked(
            sdk.lib.hggcStreamEndCapture(stream, C.byref(self.graph)), "end capture"
        )
        try:
            checked(
                sdk.lib.hggcGraphInstantiateWithFlags(
                    C.byref(self.instance), self.graph, 0
                ),
                "instantiate",
            )
        except BaseException:
            self.close()
            raise

    def __call__(self):
        return self.sdk.lib.hggcGraphLaunch(self.instance, self.stream)

    def close(self):
        if self.instance:
            checked(
                self.sdk.lib.hggcGraphExecDestroy(self.instance),
                "destroy graph instance",
            )
            self.instance = C.c_void_p()
        if self.graph:
            checked(self.sdk.lib.hggcGraphDestroy(self.graph), "destroy graph")
            self.graph = C.c_void_p()


def run(args, sdk, manifest):
    q, n, k = CASES[args.case]
    e, m, topk = 256, 8, 8
    print(
        f"KPACK_SINGLE_OP phase=fixture q={q} N={n} K={k} E={e} tokens=1 topk=8",
        flush=True,
    )
    start = time.monotonic()
    w = IndexedWeights(
        q,
        n,
        k,
        e,
        progress=lambda done, total: print(
            f"KPACK_SINGLE_OP fixture_experts={done}/{total}", flush=True
        ),
    )
    data = fixture(w, dict(mode=2, rows=m, channels=1, topk=topk))
    order = np.argsort(data["expert"], kind="stable")
    rows = np.bincount(data["expert"], minlength=e).astype("i4")
    offsets = np.r_[0, rows.cumsum()].astype("i4")
    a = np.ascontiguousarray(data["a"][data["arows"][order]])
    r, d = Resources(sdk), None
    module, oldhandle = None, C.c_void_p()
    graphs = {}
    try:
        d = Dispatch(args.bundle)
        arr = arrangement(q)
        choice = d.query(q, 2, m, n, k, e, 1, arr.mapping_id)
        if choice is None or choice.algorithm != 0 or choice.split != 1:
            raise ValueError("selected route is not the nonpersistent S1 subject")
        record = next(
            x for x in manifest["modules"] if x["key"] == choice.build_key.decode()
        )
        module = Module({**record, "path": str(args.bundle / record["path"])})
        grids = grid_model(record["parent"], rows, n)
        planes = {
            x: r.upload(w.planes[x]) if w.planes[x].size else None
            for x in ("low", "high", "units")
        }
        weight_bytes = sum(w.planes[x].nbytes // e * topk for x in planes)
        output_bytes = m * n * 2
        outputs, launches = {}, {}
        base = Call(
            version=1,
            size=C.sizeof(Call),
            m=m,
            n=n,
            k=k,
            experts=e,
            group_size=arr.group_size,
            device=choice.device,
            compute_units=choice.compute_units,
            mapping_id=arr.mapping_id,
            a=r.upload(a),
            low=planes["low"],
            high=planes["high"],
            metadata=planes["units"],
            offsets_device=r.upload(offsets),
            stream=r.stream.value,
        )
        selected_arms = ARMS if args.arm == "both" else (args.arm,)
        for arm in selected_arms:
            c = Call.from_buffer_copy(base)
            outputs[arm] = r.alloc(output_bytes + 32)
            c.output = outputs[arm] + 16
            if arm == "device-only":
                c.workspace = r.alloc(max(1, choice.workspace_bytes))
                c.workspace_bytes = choice.workspace_bytes
                launches[arm] = d.prepare(choice, c)
            else:
                c.rows_host, c.rows_device = rows.ctypes.data, r.upload(rows)
                recipe = Recipe(
                    1, C.sizeof(Recipe), choice.algorithm, choice.split, choice.grid
                )
                query = QueryResources()
                checked(
                    module.query(C.byref(c), C.byref(recipe), C.byref(query)),
                    "compact query",
                )
                if query.shared_bytes != choice.shared_bytes:
                    raise ValueError("same-parent shared memory differs")
                c.workspace = r.alloc(max(1, query.workspace_bytes))
                c.workspace_bytes = query.workspace_bytes
                checked(
                    module.prepare(C.byref(c), C.byref(recipe), C.byref(oldhandle)),
                    "compact prepare",
                )
                launches[arm] = lambda: module.run(oldhandle, r.stream)
        sdk.synchronize(r.stream)
        result = dict(
            status="INCOMPLETE",
            case=args.case,
            q=q,
            n=n,
            k=k,
            experts=e,
            tokens=1,
            topk=topk,
            total_rows=m,
            max_rows=1,
            active_experts=8,
            empty_experts=248,
            expert_ids=data["expert"].tolist(),
            rows=rows.tolist(),
            fixture="SYNTHETIC_OFFICIAL_GGUF_BROADCAST_A_NOT_MODEL_TENSOR_DUMP",
            fixture_seconds=time.monotonic() - start,
            selection=receipt(choice),
            module_sha256=record["sha256"],
            plane_sha256={
                x: hashlib.sha256(w.planes[x].tobytes()).hexdigest() for x in planes
            },
            active_weight_bytes=weight_bytes,
            grid_model=grids,
            grid_scope="SOURCE_DERIVED_VERIFY_ACTUAL_LAUNCH_WITH_PROFILER",
            excluded="LLAMA_IDS_GATHER_SCATTER_AND_REFERENCE_MMVQ_FUSION",
            cache_scope="RESIDENT_REPEATED_FIXED_IDS_NO_CACHE_FLUSH",
            comparison_scope="SAME_PARENT_METADATA_PLUS_GRID_CHANGE_NOT_GRID_ONLY_AB",
            scopes={
                "device-only": "GPU_METADATA_PLUS_RECTANGULAR_GEMM",
                "host-compact": "COMPACT_GEMM_HOST_METADATA_PREPARE_EXCLUDED",
            },
            raw_pair_equal=None,
            arms={},
        )
        try:
            compare(np.zeros((m, n), dtype="f2"), data)
        except ValueError:
            result["zero_output_negative"] = "DETECTED"
        else:
            raise ValueError("zero-output negative is not discriminating")
        raw_by_arm = {}
        for arm, launch in launches.items():
            print(
                f"KPACK_SINGLE_OP phase=correctness arm={arm} parent={choice.parent.decode()}",
                flush=True,
            )
            sdk.fill(outputs[arm], 0xA5, output_bytes + 32)
            checked(launch(), "eager correctness")
            sdk.synchronize(r.stream)
            eager = sdk.download(outputs[arm], output_bytes + 32)
            graphs[arm] = Replay(sdk, r.stream, launch, args.graph_repeats)
            sdk.fill(outputs[arm], 0xA5, output_bytes + 32)
            checked(graphs[arm](), "graph correctness")
            sdk.synchronize(r.stream)
            raw = sdk.download(outputs[arm], output_bytes + 32)
            if raw[:16] != b"\xa5" * 16 or raw[-16:] != b"\xa5" * 16 or raw != eager:
                raise ValueError(f"{arm}: output guards/eager/replay differ")
            got = np.frombuffer(raw[16:-16], dtype="<f2").reshape(m, n)
            error = compare(got[np.argsort(order)], data)
            raw_by_arm[arm] = raw
            result["arms"][arm] = dict(error=error, rounds=[])
            for _ in range(args.warmups):
                checked(graphs[arm](), "warmup")
            sdk.synchronize(r.stream)
        if len(raw_by_arm) == 2:
            result["raw_pair_equal"] = raw_by_arm[ARMS[0]] == raw_by_arm[ARMS[1]]
            if not result["raw_pair_equal"]:
                raise ValueError("same-parent compact/device-only output bits differ")
        for round_ in range(args.rounds):
            order_ = list(launches)
            if round_ % 2:
                order_.reverse()
            for arm in order_:
                print(
                    f"KPACK_SINGLE_OP phase=timing round={round_+1}/{args.rounds} arm={arm}",
                    flush=True,
                )
                result["arms"][arm]["rounds"].append(
                    r.samples(graphs[arm], args.samples)
                )
        for arm, row in result["arms"].items():
            row.update(
                timing_summary(row.pop("rounds"), args.graph_repeats, weight_bytes)
            )
            print(
                "KPACK_SINGLE_OP_RESULT "
                + json.dumps(
                    dict(
                        arm=arm,
                        median_us=row["median_us"],
                        error=row["error"],
                        scope=result["scopes"][arm],
                        effective_weight_GB_per_s=row["effective_weight_GB_per_s"],
                    )
                ),
                flush=True,
            )
        result["status"] = "PASS"
        return result
    finally:
        sdk.synchronize(r.stream)
        for graph in graphs.values():
            graph.close()
        if oldhandle:
            module.destroy(oldhandle)
        if d is not None:
            d.close()
        r.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument(
        "--bundle", type=Path, default=ROOT / "prebuilt/ppu0010/kpack-native-v1"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=CASES, default="q4-up")
    p.add_argument("--arm", choices=("both", *ARMS), default="both")
    p.add_argument("--samples", type=int, default=11)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--graph-repeats", type=int, default=16)
    args = p.parse_args()
    if min(args.samples, args.rounds, args.warmups, args.graph_repeats) <= 0:
        p.error("sample/round/warmup/repeat counts must be positive")
    args.bundle = args.bundle.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        status="FAIL",
        elapsed_seconds=0,
        argv=sys.argv,
        tool_sha256=sha(Path(__file__)),
        production_changes=False,
    )
    start = time.monotonic()
    try:
        manifest = verify(args.bundle)
        summary["manifest_sha256"] = sha(args.bundle / "manifest.json")
        sdk = SDK(args.sdk)
        graph_bind(sdk)
        summary["device"] = device_identity(sdk)
        summary.update(run(args, sdk, manifest))
    except Exception as error:
        traceback.print_exc()
        summary["error"] = str(error)
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"KPACK_SINGLE_OP {summary['status']} output={args.output}", flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
