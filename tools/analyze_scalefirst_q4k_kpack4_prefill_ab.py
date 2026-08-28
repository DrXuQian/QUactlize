#!/usr/bin/env python3
"""Adjudicate the persistent ScaleFirst Xplane/K-pack4 prefill A/B."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shlex
import statistics
import sys
import tempfile
from typing import Any


MAPPING_ID = "0x51344b5034540001"
SHAPES = ((2048, 1024, 5120), (4096, 1024, 5120))
CONFIGS = (
    "64x64x64_w64x32_s3_bc0",
    "64x128x64_w64x16_s6_bc0",
    "64x128x256_w64x16_s2_bc0",
)
ARMS = {
    "xplane": {"artifact": 64, "weight_layout": 0, "mapping": "0x0000000000000000"},
    "q4-kpack4": {"artifact": 0, "weight_layout": 1, "mapping": MAPPING_ID},
}


class AnalysisError(ValueError):
    pass


def parse_kv(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise AnalysisError(f"line lacks {prefix.strip()} prefix")
    result: dict[str, str] = {}
    for token in shlex.split(line[len(prefix):]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value
    return result


def parse_log(path: pathlib.Path, arm: str, shape: tuple[int, int, int],
              iterations: int) -> dict[str, Any]:
    text = path.read_text()
    lines = text.splitlines()
    shards = [parse_kv(line, "SF_SHARD ") for line in lines
              if line.startswith("SF_SHARD ")]
    completes = [parse_kv(line, "SF_COMPLETE ") for line in lines
                 if line.startswith("SF_COMPLETE ")]
    if len(shards) != 1 or len(completes) != 1:
        raise AnalysisError(f"{path}: shard/complete denominator differs")
    expected = ARMS[arm]
    shard = shards[0]
    checks = {
        "qtype": "12", "artifact_tile_k": str(expected["artifact"]),
        "bchunk": "0", "typed_rows": "3", "selected_rows": "3",
        "weight_layout": str(expected["weight_layout"]),
        "weight_mapping_id": expected["mapping"],
        "algorithm_mask": "0x2", "iterations": str(iterations),
    }
    for key, want in checks.items():
        if shard.get(key) != want:
            raise AnalysisError(
                f"{path}: SF_SHARD {key}={shard.get(key)} expected {want}")
    shape_text = "x".join(map(str, shape))
    complete = completes[0]
    if complete.get("status") != "COMPLETE" or \
            complete.get("shape") != shape_text or \
            complete.get("typed_rows") != "3" or \
            complete.get("iterations") != str(iterations) or \
            complete.get("roundtrip") != "PASS":
        raise AnalysisError(f"{path}: SF_COMPLETE identity differs")

    measured: dict[tuple[str, int], list[tuple[int, float]]] = {}
    fingerprints: set[str] = set()
    seen_configs: set[str] = set()
    for line in lines:
        if not line.startswith("SF_CELL "):
            continue
        try:
            cell = json.loads(line[len("SF_CELL "):])
        except json.JSONDecodeError as error:
            raise AnalysisError(f"{path}: malformed SF_CELL") from error
        if cell.get("shape") != shape_text or cell.get("qtype") != 12 or \
                cell.get("artifact_tile_k") != expected["artifact"] or \
                cell.get("bchunk") != 0 or \
                cell.get("algorithm") != "PERSISTENT" or \
                cell.get("metric_scope") != "FULL_OUTPUT" or \
                cell.get("split") != 1:
            raise AnalysisError(f"{path}: non-isomorphic runtime cell {cell}")
        config = cell.get("config")
        if config not in CONFIGS:
            raise AnalysisError(f"{path}: unexpected config {config}")
        seen_configs.add(config)
        if cell.get("status") != "MEASURED":
            continue
        if cell.get("raw_bad") != 0:
            raise AnalysisError(f"{path}: raw-bit mismatch {cell}")
        sample = cell.get("sample")
        us = cell.get("sample_us")
        grid = cell.get("grid")
        fingerprint = cell.get("fingerprint")
        if not isinstance(sample, int) or not isinstance(us, (int, float)) or \
                not isinstance(grid, int) or grid <= 0 or not fingerprint:
            raise AnalysisError(f"{path}: malformed measurement {cell}")
        measured.setdefault((config, grid), []).append((sample, float(us)))
        fingerprints.add(str(fingerprint))
    if seen_configs != set(CONFIGS):
        raise AnalysisError(f"{path}: config census differs {seen_configs}")
    if not measured or len(fingerprints) != 1:
        raise AnalysisError(f"{path}: measurement/fingerprint denominator differs")
    for key, samples in measured.items():
        if len(samples) != iterations or \
                sorted(sample for sample, _ in samples) != list(range(iterations)):
            raise AnalysisError(
                f"{path}: sample denominator differs for {key}: {len(samples)}")
    return {
        "cells": {key: [us for _, us in sorted(samples)]
                  for key, samples in measured.items()},
        "fingerprint": next(iter(fingerprints)),
    }


def verdict(delta_pct: float, threshold_pct: float) -> str:
    if delta_pct > threshold_pct:
        return "KPACK4_SLOWER"
    if delta_pct < -threshold_pct:
        return "KPACK4_FASTER"
    return "WITHIN_THRESHOLD"


def summarize(measurements: dict[tuple[str, tuple[int, int, int]],
                                 dict[tuple[str, int], list[float]]],
              threshold_pct: float) -> dict[str, Any]:
    rows = []
    shape_rows = []
    for shape in SHAPES:
        xcells = measurements[("xplane", shape)]
        kcells = measurements[("q4-kpack4", shape)]
        per_config = []
        for config in CONFIGS:
            xgrids = {grid for cfg, grid in xcells if cfg == config}
            kgrids = {grid for cfg, grid in kcells if cfg == config}
            common = sorted(xgrids & kgrids)
            if not common:
                raise AnalysisError(
                    f"shape={shape} config={config} has no common persistent grid")
            grid_scores = []
            for grid in common:
                xs = xcells[(config, grid)]
                ks = kcells[(config, grid)]
                if not xs or not ks:
                    raise AnalysisError("empty paired sample set")
                xus, kus = statistics.median(xs), statistics.median(ks)
                grid_scores.append({
                    "grid": grid, "xplane_us": xus, "kpack4_us": kus,
                    "delta_pct": (kus / xus - 1.0) * 100.0,
                    "xplane_range": [min(xs), max(xs)],
                    "kpack4_range": [min(ks), max(ks)],
                    "samples_per_arm": len(xs),
                })
            # The historical Xplane-best common grid is the causal reference;
            # K-pack4 may also choose its own best common grid, recorded
            # separately so scheduler policy and layout cost cannot be mixed.
            reference = min(grid_scores, key=lambda row: row["xplane_us"])
            kbest = min(grid_scores, key=lambda row: row["kpack4_us"])
            row = {
                "shape": list(shape), "config": config,
                "common_grids": len(common),
                "reference": reference,
                "kpack4_best_grid": kbest,
                "verdict": verdict(reference["delta_pct"], threshold_pct),
            }
            rows.append(row)
            per_config.append(row)
        xbest = min(per_config,
                    key=lambda row: row["reference"]["xplane_us"])
        kbest = min(per_config,
                    key=lambda row: row["kpack4_best_grid"]["kpack4_us"])
        xbest_us = xbest["reference"]["xplane_us"]
        ksame_us = xbest["reference"]["kpack4_us"]
        kbest_us = kbest["kpack4_best_grid"]["kpack4_us"]
        shape_rows.append({
            "shape": list(shape),
            "xplane_best_config": xbest["config"],
            "xplane_best_grid": xbest["reference"]["grid"],
            "xplane_best_us": xbest_us,
            "kpack4_same_config_grid_us": ksame_us,
            "same_config_grid_delta_pct": (ksame_us / xbest_us - 1) * 100,
            "kpack4_best_config": kbest["config"],
            "kpack4_best_grid": kbest["kpack4_best_grid"]["grid"],
            "kpack4_best_us": kbest_us,
            "best_vs_best_delta_pct": (kbest_us / xbest_us - 1) * 100,
            "verdict": verdict((kbest_us / xbest_us - 1) * 100,
                               threshold_pct),
        })
    return {
        "schema": "quactlize.scalefirst-q4k-kpack4-prefill-ab-result.v1",
        "metadata": "scalefirst-fp16-scale-zero",
        "algorithm": "PERSISTENT",
        "threshold_pct": threshold_pct,
        "config_comparisons": rows,
        "shape_comparisons": shape_rows,
    }


def analyze(runs: pathlib.Path, rounds: int, iterations: int,
            threshold_pct: float) -> dict[str, Any]:
    aggregate: dict[tuple[str, tuple[int, int, int]],
                    dict[tuple[str, int], list[float]]] = {}
    fingerprints: dict[tuple[int, int, int], set[str]] = {
        shape: set() for shape in SHAPES}
    for shape in SHAPES:
        shape_key = f"m{shape[0]}_n{shape[1]}_k{shape[2]}"
        for arm in ARMS:
            cells: dict[tuple[str, int], list[float]] = {}
            for round_index in range(1, rounds + 1):
                path = runs / shape_key / f"round-{round_index}-{arm}.log"
                parsed = parse_log(path, arm, shape, iterations)
                fingerprints[shape].add(parsed["fingerprint"])
                for key, samples in parsed["cells"].items():
                    cells.setdefault(key, []).extend(samples)
            aggregate[(arm, shape)] = cells
    for shape, values in fingerprints.items():
        if len(values) != 1:
            raise AnalysisError(
                f"shape={shape} raw-bit fingerprints differ across arms/rounds")
    result = summarize(aggregate, threshold_pct)
    result["rounds"] = rounds
    result["iterations_per_round"] = iterations
    return result


def emit(result: dict[str, Any], output_json: pathlib.Path,
         output_tsv: pathlib.Path) -> None:
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "M\tN\tK\txplane_config\txplane_grid\txplane_us\t"
        "kpack4_same_us\tsame_delta_pct\tkpack4_best_config\t"
        "kpack4_best_grid\tkpack4_best_us\tbest_delta_pct\tverdict"
    ]
    for row in result["shape_comparisons"]:
        m, n, k = row["shape"]
        lines.append("\t".join(map(str, (
            m, n, k, row["xplane_best_config"], row["xplane_best_grid"],
            row["xplane_best_us"], row["kpack4_same_config_grid_us"],
            row["same_config_grid_delta_pct"], row["kpack4_best_config"],
            row["kpack4_best_grid"], row["kpack4_best_us"],
            row["best_vs_best_delta_pct"], row["verdict"]))))
    output_tsv.write_text("\n".join(lines) + "\n")
    for row in result["config_comparisons"]:
        m, n, k = row["shape"]
        ref = row["reference"]
        best = row["kpack4_best_grid"]
        print("SF_KPACK4_SCALEFIRST_CONFIG "
              f"shape={m}x{n}x{k} config={row['config']} "
              f"reference_grid={ref['grid']} xplane_us={ref['xplane_us']:.9f} "
              f"kpack4_us={ref['kpack4_us']:.9f} "
              f"delta_pct={ref['delta_pct']:.6f} "
              f"kpack4_best_grid={best['grid']} "
              f"kpack4_best_us={best['kpack4_us']:.9f} "
              f"common_grids={row['common_grids']} verdict={row['verdict']}")
    for row in result["shape_comparisons"]:
        m, n, k = row["shape"]
        print("SF_KPACK4_SCALEFIRST_SHAPE "
              f"shape={m}x{n}x{k} xplane_config={row['xplane_best_config']} "
              f"xplane_grid={row['xplane_best_grid']} "
              f"xplane_us={row['xplane_best_us']:.9f} "
              f"kpack4_same_us={row['kpack4_same_config_grid_us']:.9f} "
              f"same_delta_pct={row['same_config_grid_delta_pct']:.6f} "
              f"kpack4_best_config={row['kpack4_best_config']} "
              f"kpack4_best_grid={row['kpack4_best_grid']} "
              f"kpack4_best_us={row['kpack4_best_us']:.9f} "
              f"best_delta_pct={row['best_vs_best_delta_pct']:.6f} "
              f"verdict={row['verdict']}")


def self_test() -> None:
    values: dict[tuple[str, tuple[int, int, int]],
                 dict[tuple[str, int], list[float]]] = {}
    for shape in SHAPES:
        for arm in ARMS:
            values[(arm, shape)] = {}
            for index, config in enumerate(CONFIGS):
                for grid in (72, 144):
                    base = 70.0 + index * 5 + (grid == 72)
                    if arm == "q4-kpack4":
                        base *= 1.01
                    values[(arm, shape)][(config, grid)] = [base, base + .2]
    result = summarize(values, 3.0)
    if len(result["shape_comparisons"]) != 2 or \
            any(row["verdict"] != "WITHIN_THRESHOLD"
                for row in result["shape_comparisons"]):
        raise AssertionError("positive summary differs")
    plants = []
    broken = copy.deepcopy(values)
    for key in list(broken[("q4-kpack4", SHAPES[0])]):
        if key[0] == CONFIGS[0]:
            del broken[("q4-kpack4", SHAPES[0])][key]
    plants.append(broken)
    broken = copy.deepcopy(values)
    del broken[("xplane", SHAPES[1])]
    plants.append(broken)
    for broken in plants:
        try:
            summarize(broken, 3.0)
        except (AnalysisError, KeyError):
            pass
        else:
            raise AssertionError("analysis negative stayed green")
    with tempfile.TemporaryDirectory(prefix="qz-sf-kpack4-analysis-") as temp:
        root = pathlib.Path(temp)
        emit(result, root / "summary.json", root / "summary.tsv")
        if not (root / "summary.json").is_file() or \
                len((root / "summary.tsv").read_text().splitlines()) != 3:
            raise AssertionError("output denominator differs")
        for arm, identity in ARMS.items():
            lines = [
                "SF_SHARD qtype=12 "
                f"artifact_tile_k={identity['artifact']} bchunk=0 typed_rows=3 "
                f"weight_layout={identity['weight_layout']} "
                f"weight_mapping_id={identity['mapping']} selected_rows=3 "
                "algorithm_mask=0x2 device=0 cu=72 iterations=2 "
                "correctness_repeats=2 schedule_seed=0x1"
            ]
            for config in CONFIGS:
                for sample, us in enumerate((70.0, 70.2)):
                    lines.append("SF_CELL " + json.dumps({
                        "shape": "2048x1024x5120", "qtype": 12,
                        "artifact_tile_k": identity["artifact"],
                        "bchunk": 0, "config": config,
                        "algorithm": "PERSISTENT", "metric_scope": "FULL_OUTPUT",
                        "split": 1, "grid": 144, "status": "MEASURED",
                        "sample": sample, "sample_us": us, "raw_bad": 0,
                        "fingerprint": "0x1234",
                    }, separators=(",", ":")))
            lines.append(
                "SF_COMPLETE status=COMPLETE shape=2048x1024x5120 "
                "typed_rows=3 runtime_cells=3 measured_cells=3 records=6 "
                "iterations=2 fixture=ORDER-INDEPENDENT+FP16-EXACT "
                "fixture_mode=exact roundtrip=PASS high_plane_coverage=PASS "
                "isolation_coverage=PASS")
            path = root / f"{arm}.log"
            path.write_text("\n".join(lines) + "\n")
            parsed = parse_log(path, arm, SHAPES[0], 2)
            if len(parsed["cells"]) != 3 or parsed["fingerprint"] != "0x1234":
                raise AssertionError("log parser positive differs")
        planted = (root / "q4-kpack4.log").read_text().replace(
            '"raw_bad":0', '"raw_bad":1', 1)
        planted_path = root / "planted.log"
        planted_path.write_text(planted)
        try:
            parse_log(planted_path, "q4-kpack4", SHAPES[0], 2)
        except AnalysisError:
            pass
        else:
            raise AssertionError("raw-bit parser plant stayed green")
    print("[sf-kpack4-prefill-analysis:self-test] PASS paired persistent "
          "common-grid comparison; raw-bit, missing-grid and missing-arm plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("analyze")
    run.add_argument("--runs", type=pathlib.Path, required=True)
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--iterations", type=int, required=True)
    run.add_argument("--threshold-pct", type=float, default=3.0)
    run.add_argument("--output-json", type=pathlib.Path, required=True)
    run.add_argument("--output-tsv", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            if args.rounds <= 0 or args.iterations <= 0 or \
                    not 0 < args.threshold_pct < 100:
                raise AnalysisError("rounds/iterations/threshold are invalid")
            result = analyze(args.runs, args.rounds, args.iterations,
                             args.threshold_pct)
            emit(result, args.output_json, args.output_tsv)
        return 0
    except (OSError, AnalysisError, AssertionError, KeyError) as error:
        print(f"[sf-kpack4-prefill-analysis] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
