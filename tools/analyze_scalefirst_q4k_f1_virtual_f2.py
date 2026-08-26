#!/usr/bin/env python3
"""Fail-close analyzer for the exact Q4 F2/F1/virtual-F2 three-arm closure."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
import tempfile


CELL = "SF_CELL "
SHARD_RE = re.compile(
    r"^SF_SHARD qtype=(?P<q>\d+) artifact_tile_k=(?P<a>\d+) bchunk=(?P<bc>\d+) "
    r"typed_rows=(?P<typed>\d+) selected_rows=(?P<selected>\d+) .*$")
COMPLETE_RE = re.compile(
    r"^SF_COMPLETE status=COMPLETE shape=(?P<shape>\d+x\d+x\d+) typed_rows=(?P<typed>\d+) "
    r"runtime_cells=(?P<runtime>\d+) measured_cells=(?P<measured>\d+) records=(?P<records>\d+) "
    r"iterations=(?P<iterations>\d+) .*$")


class Error(RuntimeError):
    pass


def parse(path: pathlib.Path) -> tuple[dict[str, int], list[dict], dict[str, int | str]]:
    text = path.read_text(encoding="utf-8")
    shards = [SHARD_RE.match(line) for line in text.splitlines()]
    shards = [m for m in shards if m]
    completes = [COMPLETE_RE.match(line) for line in text.splitlines()]
    completes = [m for m in completes if m]
    if len(shards) != 1 or len(completes) != 1:
        raise Error(f"{path}: expected one shard/complete marker, got {len(shards)}/{len(completes)}")
    shard = {k: int(v) for k, v in shards[0].groupdict().items()}
    complete: dict[str, int | str] = {
        k: (v if k == "shape" else int(v)) for k, v in completes[0].groupdict().items()
    }
    cells: list[dict] = []
    for line in text.splitlines():
        if line.startswith(CELL):
            try:
                cells.append(json.loads(line[len(CELL):]))
            except json.JSONDecodeError as exc:
                raise Error(f"{path}: malformed SF_CELL: {exc}") from exc
    if not cells:
        raise Error(f"{path}: no SF_CELL records")
    if shard != {"q": 12, "a": shard["a"], "bc": 0, "typed": 1, "selected": 1}:
        raise Error(f"{path}: wrong exact shard denominator: {shard}")
    if complete["typed"] != 1 or complete["records"] != len(cells):
        raise Error(f"{path}: complete denominator differs from records")
    return shard, cells, complete


def correctness(path: pathlib.Path) -> dict:
    shard, cells, complete = parse(path)
    if complete["iterations"] != 1 or complete["runtime"] != 1 or complete["measured"] != 1:
        raise Error(f"{path}: correctness arm must contain exactly one one-sample runtime cell")
    row = cells[0]
    if (row.get("algorithm"), row.get("metric_scope"), row.get("status"), row.get("raw_bad")) != (
            "NONPERSISTENT", "FULL_OUTPUT", "MEASURED", 0):
        raise Error(f"{path}: correctness cell is not raw-bit exact nonpersistent FULL_OUTPUT: {row}")
    return {"artifact_tile_k": shard["a"], "fingerprint": row["fingerprint"],
            "shape": complete["shape"], "config": row["config"]}


def performance(path: pathlib.Path) -> dict:
    shard, cells, complete = parse(path)
    iterations = int(complete["iterations"])
    groups: dict[tuple, list[dict]] = {}
    for row in cells:
        if row.get("metric_scope") != "FULL_OUTPUT" or row.get("status") != "MEASURED":
            continue
        if row.get("raw_bad") != 0:
            raise Error(f"{path}: measured performance row has raw_bad={row.get('raw_bad')}")
        key = tuple(row.get(k) for k in (
            "algorithm", "policy", "grid", "capacity_b_mask", "balanced_b_mask", "config"))
        groups.setdefault(key, []).append(row)
    if not groups:
        raise Error(f"{path}: no measured FULL_OUTPUT cells")
    fingerprints = {
        row.get("fingerprint")
        for rows in groups.values()
        for row in rows
    }
    if len(fingerprints) != 1 or None in fingerprints:
        raise Error(f"{path}: FULL_OUTPUT fingerprints differ within one performance shape: {fingerprints}")
    scored = []
    for key, rows in groups.items():
        samples = sorted(int(r["sample"]) for r in rows)
        if samples != list(range(iterations)):
            raise Error(f"{path}: sample denominator changed for {key}: {samples}")
        us = [float(r["sample_us"]) for r in rows]
        scored.append((statistics.median(us), min(us), max(us), key, rows[0]["fingerprint"]))
    scored.sort(key=lambda item: (item[0], item[3]))
    median, lo, hi, key, fingerprint = scored[0]
    return {"artifact_tile_k": shard["a"], "shape": complete["shape"],
            "median_us": median, "min_us": lo, "max_us": hi,
            "algorithm": key[0], "policy": key[1], "grid": key[2],
            "config": key[5], "fingerprint": fingerprint,
            "full_output_cells": len(groups), "iterations": iterations}


def analyze(arms: list[list[str]], output: pathlib.Path | None) -> dict:
    labels = [arm[0] for arm in arms]
    if sorted(labels) != ["f1", "native-f2", "virtual-f2"] or len(set(labels)) != 3:
        raise Error(f"exact arm set required, got {labels}")
    result = {"schema": "quactlize.q4_f1_virtual_f2_closure.v1", "arms": {}}
    for label, corr_s, perf_s in arms:
        result["arms"][label] = {
            "correctness": correctness(pathlib.Path(corr_s)),
            "performance": performance(pathlib.Path(perf_s)),
        }
    artifacts = {label: row["correctness"]["artifact_tile_k"]
                 for label, row in result["arms"].items()}
    if artifacts != {"native-f2": 32, "f1": 64, "virtual-f2": 64}:
        raise Error(f"physical artifact identities changed: {artifacts}")
    fingerprints = {row["correctness"]["fingerprint"] for row in result["arms"].values()}
    if len(fingerprints) != 1:
        raise Error(f"correctness fingerprints differ across physical layouts: {fingerprints}")
    correctness_shapes = {row["correctness"]["shape"] for row in result["arms"].values()}
    performance_shapes = {row["performance"]["shape"] for row in result["arms"].values()}
    if len(correctness_shapes) != 1 or len(performance_shapes) != 1:
        raise Error(
            "arm shapes differ within one phase: "
            f"correctness={correctness_shapes} performance={performance_shapes}")
    performance_fingerprints = {
        row["performance"]["fingerprint"] for row in result["arms"].values()
    }
    if len(performance_fingerprints) != 1:
        raise Error(
            "performance fingerprints differ across physical layouts: "
            f"{performance_fingerprints}")
    # Fingerprints hash the complete output, so they are comparable across
    # physical layouts at one shape, but not across different problem shapes.
    # If both phases intentionally use the same shape, retain the stronger
    # cross-phase closure as an additional guard.
    if correctness_shapes == performance_shapes and fingerprints != performance_fingerprints:
        raise Error("same-shape correctness/performance fingerprints differ")
    f1 = result["arms"]["f1"]["performance"]["median_us"]
    virtual = result["arms"]["virtual-f2"]["performance"]["median_us"]
    native = result["arms"]["native-f2"]["performance"]["median_us"]
    result["comparisons"] = {
        "virtual_vs_f1_delta_pct": (virtual / f1 - 1.0) * 100.0,
        "virtual_vs_native_f2_delta_pct": (virtual / native - 1.0) * 100.0,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Q4_F1_VIRTUAL_F2_CORRECTNESS PASS fingerprints=1 artifacts=native-f2:A32,f1:A64,virtual-f2:A64")
    for label in ("native-f2", "f1", "virtual-f2"):
        p = result["arms"][label]["performance"]
        print(f"Q4_F1_VIRTUAL_F2_PERF arm={label} shape={p['shape']} algorithm={p['algorithm']} "
              f"policy={p['policy']} grid={p['grid']} median_us={p['median_us']:.9f} "
              f"range=[{p['min_us']:.9f},{p['max_us']:.9f}] cells={p['full_output_cells']}")
    print("Q4_F1_VIRTUAL_F2_COMPARISON "
          f"virtual_vs_f1_delta_pct={result['comparisons']['virtual_vs_f1_delta_pct']:.6f} "
          f"virtual_vs_native_f2_delta_pct={result['comparisons']['virtual_vs_native_f2_delta_pct']:.6f}")
    return result


def self_test() -> None:
    def emit(path: pathlib.Path, artifact: int, perf: bool, raw_bad: int = 0,
             drop_sample: bool = False, fingerprint: str | None = None) -> None:
        iterations = 3 if perf else 1
        shape = "4096x5120x8192" if perf else "64x1024x5120"
        fingerprint = fingerprint or ("0x5678" if perf else "0x1234")
        rows = []
        algorithms = [("NONPERSISTENT", "ordinary", 8)]
        if perf:
            algorithms.append(("PERSISTENT", "capacity", 72))
        for algorithm, policy, grid in algorithms:
            for sample in range(iterations):
                if drop_sample and algorithm == "NONPERSISTENT" and sample == iterations - 1:
                    continue
                rows.append({"algorithm": algorithm, "metric_scope": "FULL_OUTPUT",
                             "status": "MEASURED", "raw_bad": raw_bad,
                             "sample": sample, "sample_us": 10.0 + sample + (grid == 72),
                             "policy": policy, "grid": grid, "capacity_b_mask": "0x0",
                             "balanced_b_mask": "0x0", "config": "64x128x128_w64x64_s3_bc0",
                             "fingerprint": fingerprint})
        lines = [f"SF_SHARD qtype=12 artifact_tile_k={artifact} bchunk=0 typed_rows=1 "
                 f"selected_rows=1 algorithm_mask=0x1 device=0 cu=72 iterations={iterations} "
                 "correctness_repeats=1 schedule_seed=0x1"]
        lines += [CELL + json.dumps(row, separators=(",", ":")) for row in rows]
        lines.append(f"SF_COMPLETE status=COMPLETE shape={shape} typed_rows=1 "
                     f"runtime_cells={len(algorithms)} measured_cells={len(algorithms)} records={len(rows)} "
                     f"iterations={iterations} fixture=ORDER-INDEPENDENT+FP16-EXACT")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="q4-f1-virtual-f2-analysis-") as td:
        root = pathlib.Path(td)
        arms = []
        for label, artifact in (("native-f2", 32), ("f1", 64), ("virtual-f2", 64)):
            corr, perf = root / f"{label}-corr.log", root / f"{label}-perf.log"
            emit(corr, artifact, False); emit(perf, artifact, True)
            arms.append([label, str(corr), str(perf)])
        analyze(arms, root / "summary.json")

        bad = root / "bad.log"
        emit(bad, 64, False, raw_bad=1)
        bad_arms = [list(arm) for arm in arms]
        bad_arms[1][1] = str(bad)
        try:
            analyze(bad_arms, None)
        except Error:
            pass
        else:
            raise Error("raw_bad plant escaped")

        emit(bad, 64, True, drop_sample=True)
        bad_arms = [list(arm) for arm in arms]
        bad_arms[1][2] = str(bad)
        try:
            analyze(bad_arms, None)
        except Error:
            pass
        else:
            raise Error("missing-sample plant escaped")

        emit(bad, 64, True, fingerprint="0x9999")
        bad_arms = [list(arm) for arm in arms]
        bad_arms[1][2] = str(bad)
        try:
            analyze(bad_arms, None)
        except Error:
            pass
        else:
            raise Error("cross-layout performance-fingerprint plant escaped")
    print("[q4-f1-virtual-f2-analysis:self-test] PASS cross-shape phase identity; "
          "raw_bad, missing-sample and cross-layout fingerprint plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", nargs=3,
                        metavar=("LABEL", "CORRECTNESS_LOG", "PERF_LOG"))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            if args.arm or args.output:
                raise Error("--self-test does not accept arm/output inputs")
            self_test()
            return 0
        if not args.arm:
            raise Error("at least one --arm is required")
        analyze(args.arm, args.output)
        return 0
    except (Error, OSError, ValueError) as exc:
        print(f"[q4-f1-virtual-f2-analysis] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
