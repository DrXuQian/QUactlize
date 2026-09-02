#!/usr/bin/env python3
"""Adjudicate the single-family Q4 K-pack policy-v2 measurement.

This tool consumes only the exact plan and raw round logs produced by
``run_fq_kquant_policy_v2_box.sh``.  It deliberately does not trust the
pilot's earlier point-estimate summary and never turns a compiled default into
measured policy.  A categorical leaf is emitted only when one measured
candidate has at most the requested conservative regret at every measured M
in the leaf.  Otherwise the M value remains an explicit gap.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import plan_fq_kquant_policy_v2 as planner  # noqa: E402

SCHEMA = "quactlize.fq-kquant-policy-v2-adjudication.v1"
ROUNDS = 3
ITERATIONS = 11
WARMUPS = 3
DEFAULT_THRESHOLD_PCT = 3.0
MEASURED_FAMILY = {"n": 1024, "k": 5120}
UNPROVEN_REAL_FAMILIES = (
    {"n": 5120, "k": 8192},
    {"n": 5120, "k": 25600},
    {"n": 8192, "k": 5120},
    {"n": 25600, "k": 5120},
)

ROW_KEYS = {
    "q",
    "round",
    "order",
    "layout",
    "mapping_id",
    "shape",
    "config",
    "provider",
    "iterations",
    "raw_bad",
    "median_us",
    "min_us",
    "max_us",
    "samples",
}
MARKER_KEYS = {
    "schema",
    "q",
    "round",
    "layout",
    "order",
    "iterations",
    "warmups",
    "all_configs",
    "dense_cases",
    "grouped_cases",
    "status",
}


class AdjudicationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdjudicationError(message)


def parse_fields(line: str, prefix: str) -> dict[str, str]:
    require(line.startswith(prefix), f"line does not start with {prefix!r}")
    result: dict[str, str] = {}
    for token in line.removeprefix(prefix).split():
        require("=" in token, f"malformed token in {prefix.strip()}: {token!r}")
        key, value = token.split("=", 1)
        require(
            key and value and key not in result,
            f"duplicate/empty field in {prefix.strip()}: {token!r}",
        )
        result[key] = value
    return result


def load_unique_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise AdjudicationError(f"{path}: non-JSON numeric token {token}")

    try:
        value = json.loads(
            path.read_bytes(), object_pairs_hook=unique, parse_constant=reject_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdjudicationError(f"cannot load {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: root must be an object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_samples(value: str) -> list[float]:
    require(
        value.startswith("[") and value.endswith("]"), "sample vector syntax differs"
    )
    try:
        result = [float(item) for item in value[1:-1].split(",") if item]
    except ValueError as error:
        raise AdjudicationError("sample vector contains a non-number") from error
    require(
        len(result) == ITERATIONS,
        f"sample count differs: {len(result)} != {ITERATIONS}",
    )
    require(
        all(math.isfinite(item) and item > 0.0 for item in result),
        "samples must be finite and positive",
    )
    require(result == sorted(result), "samples are not sorted")
    return result


def close(lhs: float, rhs: float) -> bool:
    # Benchmark statistics are printed with nine digits after the decimal.
    return math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=2e-6)


def validate_sidecar(log: Path) -> None:
    sidecar = log.with_suffix(log.suffix + ".sha256")
    require(
        sidecar.is_file() and not sidecar.is_symlink(),
        f"missing regular checksum sidecar for {log.name}",
    )
    tokens = sidecar.read_text().split()
    require(
        len(tokens) == 2
        and tokens[0] == sha256(log)
        and Path(tokens[1]).name == log.name,
        f"{sidecar.name}: checksum authority differs",
    )


def load_round_medians(
    run_root: Path,
) -> tuple[dict[int, dict[str, list[float]]], list[dict[str, Any]]]:
    plan_path = run_root / "inputs/plan.json"
    require(
        plan_path.is_file() and not plan_path.is_symlink(),
        "inputs/plan.json is missing or symlinked",
    )
    plan = load_unique_json(plan_path)
    try:
        planner.validate(plan)
    except (planner.PlanError, KeyError, TypeError, ValueError) as error:
        raise AdjudicationError(f"plan authority differs: {error}") from error

    runs = run_root / "runs"
    require(
        runs.is_dir() and not runs.is_symlink(),
        "runs must be a regular non-symlink directory",
    )
    expected_logs = {f"q12-round{round_id}.log" for round_id in range(1, ROUNDS + 1)}
    observed_logs = {path.name for path in runs.glob("q12-round*.log")}
    require(
        observed_logs == expected_logs,
        f"raw round-log denominator differs: {sorted(observed_logs)}",
    )

    candidates = set(planner.CANDIDATES)
    collected: dict[int, dict[str, list[float]]] = {
        m: {candidate: [] for candidate in planner.CANDIDATES} for m in planner.M_VALUES
    }
    authority = [
        {
            "path": "inputs/plan.json",
            "sha256": sha256(plan_path),
            "size": plan_path.stat().st_size,
        }
    ]
    for round_id in range(1, ROUNDS + 1):
        log = runs / f"q12-round{round_id}.log"
        require(
            log.is_file() and not log.is_symlink(),
            f"{log.name}: raw log is missing or symlinked",
        )
        validate_sidecar(log)
        raw = log.read_text()
        require(
            "FQ_KQUANT_LAYOUT_FAILURE " not in raw,
            f"{log.name}: contains a device failure row",
        )
        lines = raw.splitlines()
        markers = [
            parse_fields(line, "FQ_KQUANT_POLICY_RUN ")
            for line in lines
            if line.startswith("FQ_KQUANT_POLICY_RUN ")
        ]
        require(len(markers) == 1, f"{log.name}: completion marker count differs")
        marker = markers[0]
        require(
            set(marker) == MARKER_KEYS, f"{log.name}: completion marker fields differ"
        )
        expected_marker = {
            "schema": "kpack-policy-v2",
            "q": "12",
            "round": str(round_id),
            "layout": "kpack",
            "order": "kpack-first",
            "iterations": str(ITERATIONS),
            "warmups": str(WARMUPS),
            "all_configs": "1",
            "dense_cases": "64",
            "grouped_cases": "0",
            "status": "PASS",
        }
        require(marker == expected_marker, f"{log.name}: completion identity differs")

        seen: set[tuple[int, str]] = set()
        dense_lines = [
            line for line in lines if line.startswith("FQ_KQUANT_LAYOUT_DENSE ")
        ]
        require(
            len(dense_lines) == len(planner.M_VALUES) * len(planner.CANDIDATES),
            f"{log.name}: dense row count differs",
        )
        for line in dense_lines:
            row = parse_fields(line, "FQ_KQUANT_LAYOUT_DENSE ")
            require(set(row) == ROW_KEYS, f"{log.name}: dense row fields differ")
            identity = {
                "q": "12",
                "round": str(round_id),
                "order": "kpack-first",
                "layout": "kpack",
                "mapping_id": planner.MAPPING_ID,
                "provider": "standard-aiu",
                "iterations": str(ITERATIONS),
                "raw_bad": "0",
            }
            require(
                all(row[key] == value for key, value in identity.items()),
                f"{log.name}: dense row identity/correctness differs",
            )
            try:
                shape = tuple(int(item) for item in row["shape"].split("x"))
            except ValueError as error:
                raise AdjudicationError(f"{log.name}: malformed dense shape") from error
            require(
                len(shape) == 3
                and shape[0] in planner.M_VALUES
                and shape[1:] == (planner.N, planner.K),
                f"{log.name}: shape is outside M=1..64/N=1024/K=5120",
            )
            config = row["config"]
            require(config in candidates, f"{log.name}: unknown candidate {config}")
            key = (shape[0], config)
            require(key not in seen, f"{log.name}: duplicate candidate {key}")
            values = parse_samples(row["samples"])
            median = float(statistics.median(values))
            try:
                reported = (
                    float(row["median_us"]),
                    float(row["min_us"]),
                    float(row["max_us"]),
                )
            except ValueError as error:
                raise AdjudicationError(
                    f"{log.name}: reported timing is not numeric"
                ) from error
            require(
                all(math.isfinite(item) for item in reported)
                and close(reported[0], median)
                and close(reported[1], values[0])
                and close(reported[2], values[-1]),
                f"{log.name}: reported statistics differ from raw samples",
            )
            collected[shape[0]][config].append(median)
            seen.add(key)
        expected_seen = {
            (m, candidate) for m in planner.M_VALUES for candidate in planner.CANDIDATES
        }
        require(seen == expected_seen, f"{log.name}: M/candidate denominator differs")
        authority.append(
            {
                "path": f"runs/{log.name}",
                "sha256": sha256(log),
                "size": log.stat().st_size,
            }
        )

    for m, configs in collected.items():
        require(set(configs) == candidates, f"M={m}: candidate denominator differs")
        require(
            all(len(medians) == ROUNDS for medians in configs.values()),
            f"M={m}: round-median denominator differs",
        )
    return collected, authority


def score_cells(
    collected: dict[int, dict[str, list[float]]], threshold: float
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for m in planner.M_VALUES:
        raw = collected[m]
        stats: dict[str, dict[str, Any]] = {}
        for config in planner.CANDIDATES:
            round_medians = raw[config]
            stats[config] = {
                "config": config,
                "round_medians_us": round_medians,
                "median_of_round_medians_us": float(statistics.median(round_medians)),
                "time_interval_us": [min(round_medians), max(round_medians)],
            }
        best_point = min(row["median_of_round_medians_us"] for row in stats.values())
        best_low = min(row["time_interval_us"][0] for row in stats.values())
        best_high = min(row["time_interval_us"][1] for row in stats.values())
        for row in stats.values():
            low, high = row["time_interval_us"]
            row["point_regret"] = row["median_of_round_medians_us"] / best_point - 1.0
            row["conservative_regret_interval"] = [
                max(0.0, low / best_high - 1.0),
                max(0.0, high / best_low - 1.0),
            ]
            row["proven_within_threshold"] = (
                row["conservative_regret_interval"][1] <= threshold + 1e-12
            )
        ranked = sorted(
            stats.values(),
            key=lambda row: (row["median_of_round_medians_us"], row["config"]),
        )
        point_winner = ranked[0]
        runner_low = min(row["time_interval_us"][0] for row in ranked[1:])
        resolution = (
            "RESOLVED"
            if point_winner["time_interval_us"][1] < runner_low
            else "OVERLAPPING_ENVELOPES"
        )
        admitted = sorted(
            row["config"] for row in stats.values() if row["proven_within_threshold"]
        )
        cells.append(
            {
                "m": m,
                "point_winner": point_winner["config"],
                "point_winner_resolution": resolution,
                "admissible_configs": admitted,
                "candidates": [stats[config] for config in planner.CANDIDATES],
            }
        )
    return cells


def candidate_for(cell: dict[str, Any], config: str) -> dict[str, Any]:
    return next(row for row in cell["candidates"] if row["config"] == config)


def interval_score(
    cells: list[dict[str, Any]], begin: int, end: int, threshold: float
) -> dict[str, Any] | None:
    selected = cells[begin : end + 1]
    common = set(planner.CANDIDATES)
    for cell in selected:
        common &= set(cell["admissible_configs"])
    if not common:
        return None
    scored = []
    for config in sorted(common):
        rows = [candidate_for(cell, config) for cell in selected]
        point = [row["point_regret"] for row in rows]
        lowers = [row["conservative_regret_interval"][0] for row in rows]
        uppers = [row["conservative_regret_interval"][1] for row in rows]
        score = {
            "config": config,
            "max_regret_interval": [max(lowers), max(uppers)],
            "max_point_regret": max(point),
            "mean_point_regret": float(statistics.fmean(point)),
        }
        require(
            score["max_regret_interval"][1] <= threshold + 1e-12,
            "internal leaf exceeded the bounded-regret threshold",
        )
        scored.append(score)
    return min(
        scored,
        key=lambda row: (
            row["max_regret_interval"][1],
            row["max_point_regret"],
            row["mean_point_regret"],
            row["config"],
        ),
    )


def fit_run(cells: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    count = len(cells)
    intervals: dict[tuple[int, int], dict[str, Any]] = {}
    for begin in range(count):
        for end in range(begin, count):
            score = interval_score(cells, begin, end, threshold)
            if score is not None:
                intervals[(begin, end)] = score
    # Minimal categorical table first; then minimize the conservative worst
    # regret and point-estimate regret.  Config names are labels, never values
    # to interpolate.
    best: list[tuple[int, float, float, tuple[Any, ...], list[Any]] | None] = [None] * (
        count + 1
    )
    best[0] = (0, 0.0, 0.0, (), [])
    for end in range(count):
        choices = []
        for begin in range(end + 1):
            previous = best[begin]
            score = intervals.get((begin, end))
            if previous is None or score is None:
                continue
            leaves, worst, weighted, tie, path = previous
            length = end - begin + 1
            choices.append(
                (
                    leaves + 1,
                    max(worst, score["max_regret_interval"][1]),
                    weighted + score["mean_point_regret"] * length,
                    tie + ((cells[begin]["m"], cells[end]["m"], score["config"]),),
                    path + [(begin, end, score)],
                )
            )
        require(choices, "adjudicable run could not be represented by leaves")
        best[end + 1] = min(choices, key=lambda row: row[:4])
    require(best[count] is not None, "leaf dynamic program did not close")
    leaves = []
    for begin, end, score in best[count][4]:
        selected = cells[begin : end + 1]
        leaves.append(
            {
                "m_min": selected[0]["m"],
                "m_max": selected[-1]["m"],
                "measured_m": [cell["m"] for cell in selected],
                **score,
                "exact_winner_cells": sum(
                    cell["point_winner_resolution"] == "RESOLVED"
                    and cell["point_winner"] == score["config"]
                    for cell in selected
                ),
                "overlapping_envelope_cells": sum(
                    cell["point_winner_resolution"] == "OVERLAPPING_ENVELOPES"
                    for cell in selected
                ),
            }
        )
    return leaves


def fit_leaves_and_gaps(
    cells: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    leaves: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    index = 0
    while index < len(cells):
        if cells[index]["admissible_configs"]:
            end = index
            while end + 1 < len(cells) and cells[end + 1]["admissible_configs"]:
                end += 1
            leaves.extend(fit_run(cells[index : end + 1], threshold))
            index = end + 1
            continue
        end = index
        while end + 1 < len(cells) and not cells[end + 1]["admissible_configs"]:
            end += 1
        details = []
        for cell in cells[index : end + 1]:
            best = min(
                cell["candidates"],
                key=lambda row: (
                    row["conservative_regret_interval"][1],
                    row["point_regret"],
                    row["config"],
                ),
            )
            details.append(
                {
                    "m": cell["m"],
                    "best_candidate": best["config"],
                    "best_conservative_regret_high": best[
                        "conservative_regret_interval"
                    ][1],
                    "threshold_excess": best["conservative_regret_interval"][1]
                    - threshold,
                    "point_winner_resolution": cell["point_winner_resolution"],
                }
            )
        gaps.append(
            {
                "m_min": cells[index]["m"],
                "m_max": cells[end]["m"],
                "reason": "NO_CANDIDATE_PROVEN_WITHIN_BOUNDED_REGRET",
                "cells": details,
            }
        )
        index = end + 1
    return leaves, gaps


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def emit_tsv(
    path: Path, leaves: list[dict[str, Any]], gaps: list[dict[str, Any]]
) -> None:
    lines = [
        "kind\tm_min\tm_max\tconfig\tmax_regret_low\tmax_regret_high\t"
        "max_point_regret\tmean_point_regret\texact_winner_cells\t"
        "overlapping_envelope_cells\treason"
    ]
    for leaf in leaves:
        lines.append(
            f"LEAF\t{leaf['m_min']}\t{leaf['m_max']}\t{leaf['config']}\t"
            f"{leaf['max_regret_interval'][0]:.12f}\t"
            f"{leaf['max_regret_interval'][1]:.12f}\t"
            f"{leaf['max_point_regret']:.12f}\t{leaf['mean_point_regret']:.12f}\t"
            f"{leaf['exact_winner_cells']}\t{leaf['overlapping_envelope_cells']}\tNONE"
        )
    for gap in gaps:
        lines.append(
            f"GAP\t{gap['m_min']}\t{gap['m_max']}\tNONE\tNA\tNA\tNA\tNA\t"
            f"0\t0\t{gap['reason']}"
        )
    atomic_text(path, "\n".join(lines) + "\n")


def adjudicate(
    run_root: Path, output: Path, threshold_pct: float = DEFAULT_THRESHOLD_PCT
) -> dict[str, Any]:
    require(
        math.isfinite(threshold_pct) and 0.0 < threshold_pct < 100.0,
        "regret threshold must be finite and in (0,100) percent",
    )
    run_root = run_root.resolve(strict=True)
    require(
        run_root.is_dir() and not run_root.is_symlink(),
        "input must be a regular non-symlink directory",
    )
    threshold = threshold_pct / 100.0
    collected, authority = load_round_medians(run_root)
    cells = score_cells(collected, threshold)
    leaves, gaps = fit_leaves_and_gaps(cells, threshold)
    verdict = "BOUNDED_REGRET_CATEGORICAL_LEAVES" if not gaps else "UNADJUDICATED_GAPS"
    resolution_counts = {
        name: sum(cell["point_winner_resolution"] == name for cell in cells)
        for name in ("RESOLVED", "OVERLAPPING_ENVELOPES")
    }
    result = {
        "schema": SCHEMA,
        "verdict": verdict,
        "profile": "kpack-policy-v2",
        "format": "Q4_K",
        "qtype": 12,
        "operator": "dense",
        "layout": "q4-kpack4",
        "mapping_id": planner.MAPPING_ID,
        "measurement": {
            "rounds": ROUNDS,
            "samples_per_round": ITERATIONS,
            "warmups": WARMUPS,
            "candidate_count": len(planner.CANDIDATES),
            "candidate_names": list(planner.CANDIDATES),
            "m_values": list(planner.M_VALUES),
            "family": MEASURED_FAMILY,
            "regret_threshold_pct": threshold_pct,
            "regret_interval": "candidate[min,max](round-medians) / best[max,min](round-medians)",
        },
        "policy_contract": {
            "kind": "categorical-measured-leaves",
            "config_names_are_categorical": True,
            "compiled_default_is_measured_policy": False,
            "outside_measured_scope": "NO_MEASURED_POLICY",
            "interval_semantics": "inclusive contiguous integers; every M in every leaf was measured",
        },
        "leaves": leaves,
        "unadjudicated_gaps": gaps,
        "point_winner_resolution_counts": resolution_counts,
        "cells": cells,
        "scope": {
            "verdict": "SINGLE_REAL_FAMILY_M1_TO_M64_ONLY",
            "proven_family": MEASURED_FAMILY,
            "proven_m_min": 1,
            "proven_m_max": 64,
            "other_real_families_proven": False,
            "unproven_real_families": list(UNPROVEN_REAL_FAMILIES),
            "m_greater_than_64_proven": False,
            "grouped_proven": False,
            "global_shipping_policy_proven": False,
        },
        "source_files": authority,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(
        output / "adjudication.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    emit_tsv(output / "adjudication.tsv", leaves, gaps)
    print(
        "FQ_KQUANT_POLICY_V2_ADJUDICATION "
        f"verdict={verdict} family=N1024-K5120 M=1..64 "
        f"candidates=5 rounds=3 samples=11 threshold_pct={threshold_pct:.6f} "
        f"leaves={len(leaves)} gaps={len(gaps)}"
    )
    print(
        "FQ_KQUANT_POLICY_V2_SCOPE "
        "verdict=SINGLE_REAL_FAMILY_M1_TO_M64_ONLY "
        "other_real_families=4 m_gt_64=UNPROVEN grouped=UNPROVEN "
        "compiled_default=NOT_A_MEASURED_POLICY"
    )
    return result


def _write_synthetic_run(root: Path, timing: Callable[[int, int, int], float]) -> None:
    (root / "inputs").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "inputs/plan.json").write_text(
        json.dumps(planner.materialize(), sort_keys=True) + "\n"
    )
    for round_id in range(1, ROUNDS + 1):
        lines = []
        for m in planner.M_VALUES:
            for candidate_index, config in enumerate(planner.CANDIDATES):
                center = timing(m, candidate_index, round_id)
                values = [
                    center + (index - ITERATIONS // 2) * 0.001
                    for index in range(ITERATIONS)
                ]
                lines.append(
                    "FQ_KQUANT_LAYOUT_DENSE "
                    f"q=12 round={round_id} order=kpack-first layout=kpack "
                    f"mapping_id={planner.MAPPING_ID} shape={m}x1024x5120 "
                    f"config={config} provider=standard-aiu iterations=11 raw_bad=0 "
                    f"median_us={values[5]:.9f} min_us={values[0]:.9f} "
                    f"max_us={values[-1]:.9f} "
                    f"samples=[{','.join(f'{value:.9f}' for value in values)}]"
                )
        lines.append(
            "FQ_KQUANT_POLICY_RUN schema=kpack-policy-v2 q=12 "
            f"round={round_id} layout=kpack order=kpack-first iterations=11 "
            "warmups=3 all_configs=1 dense_cases=64 grouped_cases=0 status=PASS"
        )
        log = root / f"runs/q12-round{round_id}.log"
        log.write_text("\n".join(lines) + "\n")
        log.with_suffix(".log.sha256").write_text(f"{sha256(log)}  {log}\n")


def _base_timing(m: int, candidate: int, round_id: int) -> float:
    winner = 0 if m <= 8 else 1 if m <= 32 else 4
    penalty = 1.0 if candidate == winner else 1.08 + candidate * 0.01
    return (10.0 + m / 100.0) * penalty * (1.0 + (round_id - 2) * 0.0005)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qz-a04-adjudication-") as temporary:
        root = Path(temporary) / "run"
        _write_synthetic_run(root, _base_timing)
        result = adjudicate(root, Path(temporary) / "out")
        require(
            result["verdict"] == "BOUNDED_REGRET_CATEGORICAL_LEAVES"
            and [(leaf["m_min"], leaf["m_max"]) for leaf in result["leaves"]]
            == [(1, 8), (9, 32), (33, 64)],
            "synthetic categorical leaves differ",
        )
        require(
            result["scope"]["unproven_real_families"] == list(UNPROVEN_REAL_FAMILIES)
            and not result["policy_contract"]["compiled_default_is_measured_policy"],
            "scope/default contract differs",
        )

        round3 = root / "runs/q12-round3.log"
        pristine = round3.read_text()
        plants = {
            "missing-m": "\n".join(
                line
                for line in pristine.splitlines()
                if " shape=64x1024x5120 " not in line
            )
            + "\n",
            "missing-candidate": pristine.replace(
                next(
                    line
                    for line in pristine.splitlines()
                    if " shape=1x1024x5120 " in line
                ),
                "",
                1,
            ),
            "raw-bad": pristine.replace("raw_bad=0", "raw_bad=1", 1),
        }
        for name, broken in plants.items():
            round3.write_text(broken)
            round3.with_suffix(".log.sha256").write_text(
                f"{sha256(round3)}  {round3}\n"
            )
            try:
                adjudicate(root, Path(temporary) / f"red-{name}")
            except AdjudicationError:
                pass
            else:
                raise AssertionError(f"{name} negative stayed green")
        round3.write_text(pristine)
        round3.with_suffix(".log.sha256").write_text(f"{sha256(round3)}  {round3}\n")

        overlap_root = Path(temporary) / "overlap"

        def overlap_timing(m: int, candidate: int, round_id: int) -> float:
            if m == 9:
                # The two closest candidates overlap and every candidate's
                # round-median high end is more than 3% above the best low end.
                base = 10.0 + (0.0 if candidate == 0 else 0.01 * candidate)
                return base + (0.5 if round_id == 2 else 0.0)
            return _base_timing(m, candidate, round_id)

        _write_synthetic_run(overlap_root, overlap_timing)
        overlap = adjudicate(overlap_root, Path(temporary) / "overlap-out")
        require(
            overlap["verdict"] == "UNADJUDICATED_GAPS"
            and any(
                gap["m_min"] <= 9 <= gap["m_max"]
                for gap in overlap["unadjudicated_gaps"]
            )
            and overlap["cells"][8]["point_winner_resolution"]
            == "OVERLAPPING_ENVELOPES",
            "near-neighbour overlap did not fail closed into a gap",
        )
    print(
        "[fq-kquant-policy-v2-adjudication:self-test] PASS exact "
        "64x5x3x11 denominator, categorical bounded-regret leaves, "
        "single-family scope, missing-M/missing-candidate/raw-bit plants RED, "
        "and near-neighbour overlap remains an unadjudicated gap"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    run = subparsers.add_parser("adjudicate")
    run.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A04 run root containing inputs/plan.json and runs/",
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--regret-threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT
    )
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            adjudicate(args.input, args.output, args.regret_threshold_pct)
        return 0
    except (
        AdjudicationError,
        AssertionError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"[fq-kquant-policy-v2-adjudication] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
