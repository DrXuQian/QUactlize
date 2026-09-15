#!/usr/bin/env python3
"""Correctness-only selected Q4 BF16 gate; PPU and real-CUDA runtimes."""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.bf16_fastpath.gate_plan import SCHEMA, Library, plan
from dev.bf16_fastpath.gate_fixture import Weights, Buffer, storage, output
from dev.bf16_fastpath.build_gate import harness
from dev.bf16_compute.fixture import compare, digest
from dev.gemv_simt.native import Runtime, checked
from quactlize.execution.native import SimtCallV2
from quactlize.runtime.compiler import sha


def rejected(got, gold, denom):
    try:
        compare(got, gold, denom)
    except ValueError:
        return True
    raise ValueError("planted incorrect value was accepted by the independent oracle")


class Graph:
    def __init__(self, rt, launch):
        self.rt = rt
        self.graph, self.instance = C.c_void_p(), C.c_void_p()
        rt.sync()
        checked(rt.StreamBeginCapture(rt.stream, 0), "capture begin")
        try:
            checked(launch(), "captured typed Q4 run")
        except BaseException:
            rt.StreamEndCapture(rt.stream, C.byref(self.graph))
            self.close()
            raise
        checked(rt.StreamEndCapture(rt.stream, C.byref(self.graph)), "capture end")
        checked(rt.GraphInstantiateWithFlags(C.byref(self.instance), self.graph, 0), "instantiate")

    def launch(self):
        checked(self.rt.GraphLaunch(self.instance, self.rt.stream), "changed-input graph replay")

    def close(self):
        self.rt.sync()
        if self.instance:
            checked(self.rt.GraphExecDestroy(self.instance), "destroy instance")
        if self.graph:
            checked(self.rt.GraphDestroy(self.graph), "destroy graph")


def case(rt, lib, point, weights, buffers):
    baseline_count = len(rt.allocations)
    result = dict(id=point["id"], recipe=point["recipe"], status="FAIL", bf16=[], controls={})
    try:
        for kind in (1, 2):
            typed, config, sizes = lib.select(point, kind)
            image, coeff, ids = weights.inputs(point, 0)
            a, dest = Buffer(rt, image.size*(4 if kind == 1 else 2)), Buffer(rt, point["rows"]*(point["n"]+8)*4)
            id_buffer = Buffer(rt, ids.nbytes) if ids is not None else None
            typed.call.a, typed.call.output = a.ptr, dest.ptr
            typed.call.low, typed.call.units = buffers["low"].ptr, buffers["units"].ptr
            typed.call.ids, typed.call.stream = id_buffer.ptr if id_buffer else None, rt.stream.value
            run = lambda: lib.run2(C.byref(typed), C.byref(config), C.byref(lib.arr))

            def inputs(repeat, large=False):
                image, coeff, ids = weights.inputs(point, repeat, large)
                a.upload(storage(image, kind))
                if id_buffer:
                    id_buffer.upload(ids)
                dest.poison()
                gold, denom = weights.truth(point, coeff, ids, "bf16")
                return image, coeff, ids, gold, denom

            image, coeff, ids, gold, denom = inputs(0)
            checked(run(), "typed Q4 eager run")
            original = output(dest, point)
            proof = dict(storage="F32" if kind == 1 else "BF16", eager=compare(original, gold, denom), graph=[])
            proof["zero_output_negative"] = rejected(np.zeros_like(gold), gold, denom)
            a.guard()
            if id_buffer:
                id_buffer.guard()
            graph = Graph(rt, run)
            try:
                # Same addresses/graph, different A and routing values. Include
                # real-magnitude finite activations and restore the first input.
                for repeat, large in ((1, False), (2, True), (0, False)):
                    image, coeff, ids, gold, denom = inputs(repeat, large)
                    graph.launch()
                    got = output(dest, point)
                    row = dict(repeat=repeat, large=large, input_sha256=digest(storage(image, kind)),
                        ids_sha256=digest(ids) if ids is not None else None,
                        oracle=compare(got, gold, denom))
                    if large and ids is not None:
                        wrong_ids = ids.copy()
                        wrong_ids[:, :8] = np.roll(wrong_ids[:, :8], 1, axis=1)
                        wrong, wrong_denom = weights.truth(point, coeff, wrong_ids, "bf16")
                        row["wrong_expert_negative"] = rejected(got, wrong, wrong_denom)
                    if repeat == 0 and not np.array_equal(got.view("u4"), original.view("u4")):
                        raise ValueError("restored graph input differs from eager result")
                    proof["graph"].append(row)
                a.upload(np.zeros(image.size, dtype="f4" if kind == 1 else "u2"))
                dest.poison()
                graph.launch()
                zeros = output(dest, point)
                if not np.isfinite(zeros).all() or np.any(zeros):
                    raise ValueError("zero A graph replay retained old values")
                proof["zero_a"] = True
                if id_buffer:
                    image, coeff, ids, gold, denom = inputs(0)
                    ids[:, 0] = point["experts"]
                    id_buffer.upload(ids)
                    graph.launch()
                    invalid = output(dest, point)
                    if not np.isnan(invalid[::8]).all():
                        raise ValueError("invalid expert did not produce the specified NaN marker")
                    keep = np.arange(point["rows"]) % 8 != 0
                    compare(invalid[keep], gold[keep], denom[keep])
                    proof["invalid_ids"] = True
            finally:
                graph.close()
            a.guard()
            if id_buffer:
                id_buffer.guard()
            result["bf16"].append(proof)
            # F16 v1 remains a real device control, not another BF16 launch.
            if kind == 1:
                image, coeff, ids, _, _ = inputs(0)
                gold16, denom16 = weights.truth(point, coeff, ids, "f16")
                checked(lib.run1(C.byref(typed.call), C.byref(config), C.byref(lib.arr)), "F16 v1 control")
                old = output(dest, point)
                result["controls"]["f16_nominal"] = compare(old, gold16, denom16)
                f16 = SimtCallV2(typed.call, 0)
                dest.poison()
                checked(lib.run2(C.byref(f16), C.byref(config), C.byref(lib.arr)), "F16 v2 delegate")
                if not np.array_equal(output(dest, point).view("u4"), old.view("u4")):
                    raise ValueError("F16 v2 changed v1 result")
                result["controls"]["f16_v1_v2_equal"] = True
                image, coeff, ids, gold, denom = inputs(2, True)
                checked(lib.run1(C.byref(typed.call), C.byref(config), C.byref(lib.arr)), "F16 overflow negative")
                bad = output(dest, point)
                if np.isfinite(bad).all():
                    raise ValueError("F16 overflow negative did not exercise the original range loss")
                result["controls"]["f16_overflow_nonfinite"] = int((~np.isfinite(bad)).sum())
                result["controls"]["f16_overflow_rejected"] = rejected(bad, gold, denom)
            rt.release_after(baseline_count)
        for buffer in buffers.values():
            buffer.guard()
        result["status"] = "PASS"
    finally:
        rt.release_after(baseline_count)
    return result


def verified(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    path = (bundle / manifest["library"]).resolve(strict=True)
    if (manifest.get("schema") != SCHEMA or path.parent != bundle.resolve() or sha(path) != manifest["sha256"] or
            manifest["plan"] != plan() or manifest["harness"] != harness()):
        raise ValueError("typed Q4 gate receipt, image, harness or denominator differs")
    if any(sha(ROOT / p) != h for p, h in manifest["source_hashes"].items()):
        raise ValueError("typed Q4 source differs from gate package")
    return manifest, path


def child(args, manifest, path):
    selected = [r for r in manifest["plan"]["cases"] if f"{r['n']}x{r['k']}e{r['experts']}" == args.child]
    rt = Runtime(args.sdk, manifest["platform"])
    results = dict(shape=args.child, cases=[], status="FAIL", timing_valid=False)
    try:
        lib = Library(path)
        results["device"] = dict(platform=manifest["platform"], compute_units=rt.attribute(16), warp=rt.attribute(10))
        if results["device"]["warp"] != 32:
            raise ValueError("selected Q4 readers require 32-lane warps")
        active = [r for r in selected if r["expected"] == "SELECTED"]
        weights = Weights(active[0]["n"], active[0]["k"], active[0]["experts"]) if active else None
        if weights:
            results["fixture"] = weights.record()
            buffers = weights.upload(rt)
        for point in selected:
            if point["expected"] != "SELECTED":
                for storage_type in (1, 2):
                    lib.select(point, storage_type)
                results["cases"].append(dict(id=point["id"], status="EXPECTED_QKG_SHAPE"))
                continue
            result = case(rt, lib, point, weights, buffers)
            results["cases"].append(result)
            print(f"Q4_BF16_GATE_CASE id={point['id']} status=PASS", flush=True)
        results["status"] = "PASS"
    except Exception as exc:
        results["error"] = str(exc)
        traceback.print_exc()
    finally:
        try:
            rt.close()
        finally:
            (args.output / (args.child + ".json")).write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    return int(results["status"] != "PASS")


def summarize(manifest, results):
    expected = {r["id"]: r for r in manifest["plan"]["cases"]}
    records = [c for r in results for c in r.get("cases", [])]
    actual = {c["id"]: c for c in records}
    if len(records) != len(actual) or set(actual) != set(expected):
        raise ValueError("device case denominator is incomplete or duplicated")
    for name, point in expected.items():
        result = actual[name]
        if point["expected"] == "QKG_SHAPE":
            if result["status"] != "EXPECTED_QKG_SHAPE":
                raise ValueError("TC policy decline was reported as a SIMT result")
        elif (result["status"] != "PASS" or len(result["bf16"]) != 2 or
              {r["storage"] for r in result["bf16"]} != {"F32", "BF16"} or
              not result["controls"].get("f16_overflow_rejected") or
              not result["controls"].get("f16_v1_v2_equal") or
              result["controls"].get("f16_overflow_nonfinite", 0) <= 0 or
              result["controls"].get("f16_nominal", {}).get("bad", -1) != 0):
            raise ValueError("typed Q4 device coverage is incomplete")
        else:
            for proof in result["bf16"]:
                if (proof.get("eager", {}).get("bad", -1) != 0 or not proof.get("zero_a") or
                        not proof.get("zero_output_negative") or
                        [(r["repeat"], r["large"]) for r in proof["graph"]] != [(1, False), (2, True), (0, False)] or
                        any(r["oracle"].get("bad", -1) != 0 for r in proof["graph"])):
                    raise ValueError("typed Q4 graph/negative evidence is incomplete")
                if point["mode"] == 2 and (not proof.get("invalid_ids") or
                        not proof["graph"][1].get("wrong_expert_negative")):
                    raise ValueError("indexed Q4 changed/invalid-ID evidence is incomplete")
    if any(r["status"] != "PASS" for r in results):
        raise ValueError("one or more independent image processes failed")
    return dict(status="PASS", denominator=manifest["plan"]["denominator"],
                device_admission="NUMERICAL_ONLY", performance_admitted=False, timing_valid=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest, path = verified(args.bundle)
    if args.child:
        return child(args, manifest, path)
    args.output.mkdir(parents=True, exist_ok=False)
    started, results = time.monotonic(), []
    keys = sorted({f"{r['n']}x{r['k']}e{r['experts']}" for r in manifest["plan"]["cases"]})
    for index, key in enumerate(keys):
        command = [sys.executable, __file__, "--sdk", str(args.sdk), "--bundle", str(args.bundle),
                   "--output", str(args.output), "--child", key]
        with (args.output / (key+".log")).open("w") as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        receipt = args.output / (key+".json")
        results.append(json.loads(receipt.read_text()) if receipt.is_file() else dict(shape=key, status="FAIL", rc=rc))
        if rc:
            results[-1]["status"] = "FAIL"
        print(f"Q4_BF16_GATE_PROGRESS shapes={index+1}/{len(keys)} shape={key} rc={rc} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    try:
        summary = summarize(manifest, results)
    except ValueError as exc:
        summary = dict(status="FAIL", error=str(exc), timing_valid=False, performance_admitted=False)
    summary.update(manifest_sha256=sha(args.bundle / "manifest.json"), results=results,
                   runtime_seconds=time.monotonic()-started)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"Q4_BF16_GATE_DONE status={summary['status']} results={args.output}", flush=True)
    return int(summary["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
