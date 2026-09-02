#!/usr/bin/env python3
"""Adjudicate five Q4 K-pack real families without cross-family inference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
from typing import Any, Callable

import adjudicate_fq_kquant_policy_v2 as bounded
import plan_fq_kquant_policy_v2_real_families as planner


SCHEMA = "quactlize.fq-kquant-policy-v2-real-families-adjudication.v1"
ROW_KEYS = {
    "q", "round", "order", "layout", "mapping_id", "shape", "config",
    "provider", "iterations", "raw_bad", "median_us", "min_us", "max_us",
    "samples",
}
MARKER_KEYS = {
    "schema", "q", "round", "layout", "order", "iterations", "warmups",
    "all_configs", "dense_cases", "grouped_cases", "status",
}
WEIGHT_KEYS = {
    "schema", "q", "n", "k", "experts", "mapping_id", "low_bytes",
    "high_bytes", "unit_bytes", "low_hash", "high_hash", "unit_hash",
    "roundtrip",
}
HASH16 = re.compile(r"0x[0-9a-f]{16}")


class AnalysisError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def parse_fields(line: str, prefix: str) -> dict[str, str]:
    require(line.startswith(prefix), f"line does not start with {prefix!r}")
    result: dict[str, str] = {}
    for token in line.removeprefix(prefix).split():
        require("=" in token, f"malformed token in {prefix.strip()}: {token!r}")
        key, value = token.split("=", 1)
        require(key and value and key not in result, f"duplicate/empty field: {token!r}")
        result[key] = value
    return result


def unique_json(path: Path) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject(token: str) -> Any:
        raise AnalysisError(f"{path}: non-JSON numeric token {token}")

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=hook, parse_constant=reject)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot load {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: root must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sidecar(log: Path) -> dict[str, Any]:
    sidecar = log.with_suffix(log.suffix + ".sha256")
    require(sidecar.is_file() and not sidecar.is_symlink(), f"missing checksum for {log.name}")
    tokens = sidecar.read_text().split()
    require(
        len(tokens) == 2
        and tokens[0] == sha256(log)
        and Path(tokens[1]).name == log.name,
        f"{sidecar.name}: checksum authority differs",
    )
    return {
        "path": f"runs/{log.name}",
        "size": log.stat().st_size,
        "sha256": tokens[0],
        "sidecar_sha256": sha256(sidecar),
    }


def parse_samples(value: str) -> list[float]:
    try:
        samples = bounded.parse_samples(value)
    except bounded.AdjudicationError as error:
        raise AnalysisError(str(error)) from error
    return samples


def load_family(
    runs: Path, family: dict[str, Any]
) -> tuple[dict[int, dict[str, list[float]]], dict[str, str], list[dict[str, Any]]]:
    n, k = family["n"], family["k"]
    candidates = set(planner.CANDIDATES)
    collected = {
        m: {candidate: [] for candidate in planner.CANDIDATES}
        for m in planner.M_VALUES
    }
    authority: list[dict[str, Any]] = []
    weight_authority: dict[str, str] | None = None
    for round_id in range(1, planner.ROUNDS + 1):
        log = runs / f"q12-n{n}-k{k}-round{round_id}.log"
        require(log.is_file() and not log.is_symlink(), f"{log.name}: log is missing or symlinked")
        authority.append(validate_sidecar(log))
        text = log.read_text()
        require("FQ_KQUANT_LAYOUT_FAILURE " not in text, f"{log.name}: contains a device failure")
        lines = text.splitlines()

        markers = [
            parse_fields(line, "FQ_KQUANT_POLICY_RUN ")
            for line in lines if line.startswith("FQ_KQUANT_POLICY_RUN ")
        ]
        require(len(markers) == 1, f"{log.name}: completion marker count differs")
        marker = markers[0]
        require(set(marker) == MARKER_KEYS, f"{log.name}: completion fields differ")
        expected_marker = {
            "schema": "kpack-policy-v2",
            "q": "12",
            "round": str(round_id),
            "layout": "kpack",
            "order": "kpack-first",
            "iterations": str(planner.ITERATIONS),
            "warmups": str(planner.WARMUPS),
            "all_configs": "1",
            "dense_cases": str(len(planner.M_VALUES)),
            "grouped_cases": "0",
            "status": "PASS",
        }
        require(marker == expected_marker, f"{log.name}: completion identity differs")

        weights = [
            parse_fields(line, "FQ_KQUANT_POLICY_WEIGHT ")
            for line in lines if line.startswith("FQ_KQUANT_POLICY_WEIGHT ")
        ]
        require(len(weights) == 1, f"{log.name}: weight authority count differs")
        weight = weights[0]
        require(set(weight) == WEIGHT_KEYS, f"{log.name}: weight fields differ")
        expected_weight = {
            "schema": "kpack-policy-v2", "q": "12", "n": str(n), "k": str(k),
            "experts": "1", "mapping_id": planner.pilot.MAPPING_ID,
            "high_bytes": "0", "roundtrip": "PASS",
        }
        require(
            all(weight.get(key) == value for key, value in expected_weight.items()),
            f"{log.name}: weight identity/roundtrip differs",
        )
        for field in ("low_bytes", "unit_bytes"):
            require(weight[field].isdigit() and int(weight[field]) > 0, f"{log.name}: {field} differs")
        for field in ("low_hash", "high_hash", "unit_hash"):
            require(bool(HASH16.fullmatch(weight[field])), f"{log.name}: {field} is malformed")
        if weight_authority is None:
            weight_authority = weight
        else:
            require(weight == weight_authority, f"{log.name}: weight bytes differ across rounds")

        dense_lines = [line for line in lines if line.startswith("FQ_KQUANT_LAYOUT_DENSE ")]
        require(
            len(dense_lines) == len(planner.M_VALUES) * len(planner.CANDIDATES),
            f"{log.name}: dense row count differs",
        )
        seen: set[tuple[int, str]] = set()
        for line in dense_lines:
            row = parse_fields(line, "FQ_KQUANT_LAYOUT_DENSE ")
            require(set(row) == ROW_KEYS, f"{log.name}: dense row fields differ")
            identity = {
                "q": "12", "round": str(round_id), "order": "kpack-first",
                "layout": "kpack", "mapping_id": planner.pilot.MAPPING_ID,
                "provider": "standard-aiu", "iterations": str(planner.ITERATIONS),
                "raw_bad": "0",
            }
            require(
                all(row.get(key) == value for key, value in identity.items()),
                f"{log.name}: dense identity/raw-bit result differs",
            )
            try:
                shape = tuple(int(item) for item in row["shape"].split("x"))
            except ValueError as error:
                raise AnalysisError(f"{log.name}: malformed shape") from error
            require(
                len(shape) == 3 and shape[0] in planner.M_VALUES and shape[1:] == (n, k),
                f"{log.name}: shape is outside its exact family",
            )
            config = row["config"]
            require(config in candidates, f"{log.name}: unknown categorical config {config}")
            key = (shape[0], config)
            require(key not in seen, f"{log.name}: duplicate M/config {key}")
            samples = parse_samples(row["samples"])
            median = float(statistics.median(samples))
            try:
                reported = (float(row["median_us"]), float(row["min_us"]), float(row["max_us"]))
            except ValueError as error:
                raise AnalysisError(f"{log.name}: reported timing is not numeric") from error
            require(
                all(math.isfinite(item) for item in reported)
                and bounded.close(reported[0], median)
                and bounded.close(reported[1], samples[0])
                and bounded.close(reported[2], samples[-1]),
                f"{log.name}: reported timing differs from raw samples",
            )
            collected[shape[0]][config].append(median)
            seen.add(key)
        expected_seen = {
            (m, config) for m in planner.M_VALUES for config in planner.CANDIDATES
        }
        require(seen == expected_seen, f"{log.name}: M/config denominator differs")

    require(weight_authority is not None, "weight authority is absent")
    for m, rows in collected.items():
        require(set(rows) == candidates, f"M={m}: categorical denominator differs")
        require(
            all(len(round_medians) == planner.ROUNDS for round_medians in rows.values()),
            f"M={m}: round denominator differs",
        )
    return collected, weight_authority, authority


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def emit_tsv(path: Path, families: list[dict[str, Any]]) -> None:
    lines = [
        "N\tK\tkind\tm_min\tm_max\tconfig\tmax_regret_low\tmax_regret_high\t"
        "max_point_regret\tmean_point_regret\treason"
    ]
    for result in families:
        n, k = result["family"]["n"], result["family"]["k"]
        for leaf in result["leaves"]:
            lines.append(
                f"{n}\t{k}\tLEAF\t{leaf['m_min']}\t{leaf['m_max']}\t{leaf['config']}\t"
                f"{leaf['max_regret_interval'][0]:.12f}\t"
                f"{leaf['max_regret_interval'][1]:.12f}\t"
                f"{leaf['max_point_regret']:.12f}\t{leaf['mean_point_regret']:.12f}\tNONE"
            )
        for gap in result["unadjudicated_gaps"]:
            lines.append(
                f"{n}\t{k}\tGAP\t{gap['m_min']}\t{gap['m_max']}\tNONE\tNA\tNA\tNA\tNA\t"
                f"{gap['reason']}"
            )
    atomic_text(path, "\n".join(lines) + "\n")


def analyze(run_root: Path, output: Path) -> dict[str, Any]:
    run_root = run_root.resolve(strict=True)
    plan_path = run_root / "inputs/plan.json"
    require(plan_path.is_file() and not plan_path.is_symlink(), "inputs/plan.json is missing or symlinked")
    plan = unique_json(plan_path)
    try:
        planner.validate(plan)
    except (planner.PlanError, KeyError, TypeError, ValueError) as error:
        raise AnalysisError(f"plan authority differs: {error}") from error
    runs = run_root / "runs"
    require(runs.is_dir() and not runs.is_symlink(), "runs is missing or symlinked")
    expected_logs = {
        f"q12-n{n}-k{k}-round{round_id}.log"
        for n, k in planner.FAMILIES
        for round_id in range(1, planner.ROUNDS + 1)
    }
    observed_logs = {path.name for path in runs.glob("q12-*.log")}
    require(observed_logs == expected_logs, f"family/round log denominator differs: {sorted(observed_logs)}")

    results = []
    source_files = [{"path": "inputs/plan.json", "size": plan_path.stat().st_size, "sha256": sha256(plan_path)}]
    for family in plan["families"]:
        collected, weight, authority = load_family(runs, family)
        try:
            cells = bounded.score_cells(collected, planner.REGRET_THRESHOLD_PCT / 100.0)
            leaves, gaps = bounded.fit_leaves_and_gaps(cells, planner.REGRET_THRESHOLD_PCT / 100.0)
        except bounded.AdjudicationError as error:
            raise AnalysisError(str(error)) from error
        results.append({
            "family": {key: family[key] for key in ("identity", "n", "k")},
            "verdict": "UNADJUDICATED_GAPS" if gaps else "BOUNDED_REGRET_CATEGORICAL_LEAVES",
            "weight_authority": weight,
            "leaves": leaves,
            "unadjudicated_gaps": gaps,
            "cells": cells,
        })
        source_files.extend(authority)

    any_gap = any(row["unadjudicated_gaps"] for row in results)
    result = {
        "schema": SCHEMA,
        "plan_schema": planner.SCHEMA,
        "profile": "kpack-policy-v2",
        "qtype": 12,
        "layout": "q4-kpack4",
        "mapping_id": planner.pilot.MAPPING_ID,
        "verdict": "UNADJUDICATED_GAPS" if any_gap else "FIVE_REAL_FAMILY_BOUNDED_REGRET_TABLES",
        "measurement": {
            "families": len(planner.FAMILIES),
            "m_values_per_family": len(planner.M_VALUES),
            "categorical_candidates": len(planner.CANDIDATES),
            "rounds": planner.ROUNDS,
            "samples_per_round": planner.ITERATIONS,
            "warmups_per_round": planner.WARMUPS,
            "raw_bad_required": 0,
            "regret_threshold_pct": planner.REGRET_THRESHOLD_PCT,
        },
        "policy_contract": {
            "kind": "family-keyed-categorical-measured-leaves",
            "config_names_are_categorical": True,
            "compiled_default_is_measured_policy": False,
            "cross_family_extrapolation": False,
            "unknown_n_k": "NO_MEASURED_POLICY",
            "m_greater_than_64": "A07_SCALEFIRST",
            "interval_semantics": "inclusive contiguous integers; every M in every leaf was measured",
        },
        "families": results,
        "source_files": source_files,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "adjudication.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    emit_tsv(output / "adjudication.tsv", results)
    print(
        "FQ_KQUANT_POLICY_REAL_FAMILIES "
        f"verdict={result['verdict']} families=5 M=1..64 candidates=5 "
        "rounds=3 samples=11 warmups=3 threshold_pct=3.000000 "
        f"leaves={sum(len(row['leaves']) for row in results)} "
        f"gaps={sum(len(row['unadjudicated_gaps']) for row in results)}"
    )
    print(
        "FQ_KQUANT_POLICY_REAL_SCOPE verdict=FAMILY_KEYED_M1_TO_M64_ONLY "
        "cross_family=FORBIDDEN compiled_default=NOT_A_MEASURED_POLICY "
        "m_gt_64=A07_SCALEFIRST grouped=SEPARATE_GATE"
    )
    return result


def _base_timing(family: int, m: int, candidate: int, round_id: int) -> float:
    winner = (family + (0 if m <= 8 else 1 if m <= 32 else 4)) % len(planner.CANDIDATES)
    penalty = 1.0 if candidate == winner else 1.08 + candidate * 0.01
    return (10.0 + family + m / 100.0) * penalty * (1.0 + (round_id - 2) * 0.0005)


def _write_synthetic_run(
    root: Path, timing: Callable[[int, int, int, int], float] = _base_timing
) -> None:
    (root / "inputs").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "inputs/plan.json").write_text(json.dumps(planner.materialize(), sort_keys=True) + "\n")
    for family_index, (n, k) in enumerate(planner.FAMILIES):
        for round_id in range(1, planner.ROUNDS + 1):
            lines = [
                "FQ_KQUANT_POLICY_WEIGHT schema=kpack-policy-v2 q=12 "
                f"n={n} k={k} experts=1 mapping_id={planner.pilot.MAPPING_ID} "
                f"low_bytes={n * k // 2} high_bytes=0 unit_bytes={n * (k // 32) * 4} "
                "low_hash=0x1111111111111111 high_hash=0x2222222222222222 "
                "unit_hash=0x3333333333333333 roundtrip=PASS"
            ]
            for m in planner.M_VALUES:
                for candidate_index, config in enumerate(planner.CANDIDATES):
                    center = timing(family_index, m, candidate_index, round_id)
                    samples = [
                        center + (index - planner.ITERATIONS // 2) * 0.001
                        for index in range(planner.ITERATIONS)
                    ]
                    lines.append(
                        "FQ_KQUANT_LAYOUT_DENSE "
                        f"q=12 round={round_id} order=kpack-first layout=kpack "
                        f"mapping_id={planner.pilot.MAPPING_ID} shape={m}x{n}x{k} "
                        f"config={config} provider=standard-aiu iterations=11 raw_bad=0 "
                        f"median_us={samples[5]:.9f} min_us={samples[0]:.9f} "
                        f"max_us={samples[-1]:.9f} "
                        f"samples=[{','.join(f'{value:.9f}' for value in samples)}]"
                    )
            lines.append(
                "FQ_KQUANT_POLICY_RUN schema=kpack-policy-v2 q=12 "
                f"round={round_id} layout=kpack order=kpack-first iterations=11 warmups=3 "
                "all_configs=1 dense_cases=64 grouped_cases=0 status=PASS"
            )
            log = root / f"runs/q12-n{n}-k{k}-round{round_id}.log"
            log.write_text("\n".join(lines) + "\n")
            log.with_suffix(".log.sha256").write_text(f"{sha256(log)}  {log}\n")


def refresh_sidecar(log: Path) -> None:
    log.with_suffix(".log.sha256").write_text(f"{sha256(log)}  {log}\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qz-a04-real-") as temporary:
        root = Path(temporary) / "run"
        _write_synthetic_run(root)
        result = analyze(root, Path(temporary) / "out")
        require(
            result["verdict"] == "FIVE_REAL_FAMILY_BOUNDED_REGRET_TABLES"
            and len(result["families"]) == 5
            and all(not row["unadjudicated_gaps"] for row in result["families"]),
            "synthetic five-family result differs",
        )
        require(
            len({row["cells"][0]["point_winner"] for row in result["families"]}) == 5,
            "families were not adjudicated independently",
        )
        require(
            not result["policy_contract"]["cross_family_extrapolation"]
            and not result["policy_contract"]["compiled_default_is_measured_policy"]
            and result["policy_contract"]["m_greater_than_64"] == "A07_SCALEFIRST",
            "policy scope differs",
        )

        target = root / "runs/q12-n1024-k5120-round3.log"
        pristine = target.read_text()
        plants = (
            pristine.replace("raw_bad=0", "raw_bad=1", 1),
            pristine.replace(planner.pilot.MAPPING_ID, "0x0000000000000000", 1),
            pristine.replace("roundtrip=PASS", "roundtrip=FAIL", 1),
            pristine.replace("dense_cases=64", "dense_cases=63", 1),
            pristine.replace("provider=standard-aiu", "provider=packed-row", 1),
        )
        for index, broken in enumerate(plants):
            target.write_text(broken)
            refresh_sidecar(target)
            try:
                analyze(root, Path(temporary) / f"red-{index}")
            except AnalysisError:
                pass
            else:
                raise AssertionError(f"raw authority plant {index} stayed green")
        target.write_text(pristine)
        refresh_sidecar(target)

        gap_root = Path(temporary) / "gap"

        def gap_timing(family: int, m: int, candidate: int, round_id: int) -> float:
            if family == 2 and m == 9:
                return 10.0 + candidate * 0.01 + (0.5 if round_id == 2 else 0.0)
            return _base_timing(family, m, candidate, round_id)

        _write_synthetic_run(gap_root, gap_timing)
        gap = analyze(gap_root, Path(temporary) / "gap-out")
        require(
            gap["verdict"] == "UNADJUDICATED_GAPS"
            and not gap["families"][0]["unadjudicated_gaps"]
            and any(
                row["m_min"] <= 9 <= row["m_max"]
                for row in gap["families"][2]["unadjudicated_gaps"]
            ),
            "one-family overlap was not isolated as an explicit gap",
        )
    print(
        "[fq-kquant-policy-real-analysis:self-test] PASS exact 5x64x5x3x11 "
        "denominator, independent family-keyed 3% leaves/gaps, raw-bit/weight/"
        "mapping/provider/marker plants RED, no compiled default/cross-family inference, "
        "and M>64=A07_SCALEFIRST"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    run = commands.add_parser("analyze")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            analyze(args.input, args.output)
        return 0
    except (
        AnalysisError, AssertionError, KeyError, OSError, TypeError, ValueError
    ) as error:
        print(f"[fq-kquant-policy-real-analysis] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
