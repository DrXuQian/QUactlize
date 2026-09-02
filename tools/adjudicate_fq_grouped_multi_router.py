#!/usr/bin/env python3
"""Independently adjudicate the A05 grouped K-pack selector experiment.

The input authority is the A05 plan plus the raw per-round device logs.  This
module deliberately does not read the pilot ``summary.json``: local winners
are insufficient to prove that one tactic can serve profiles which expose the
same public route features.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import tempfile
from typing import Any, Iterable

import plan_fq_grouped_multi_router as planner

SCHEMA = "quactlize.fq-grouped-multi-router-adjudication.v1"
DEFAULT_THRESHOLD_PCT = 3.0
QTYPES = (10, 11, 12, 13, 14)
PROFILES = (
    "balanced",
    "hot-skewed",
    "sparse-empty",
    "tilem-boundary",
    "permutation-a",
    "permutation-b",
)
ROUTER_MEASUREMENT_FIELDS = (
    "total_rows",
    "max_rows",
    "active",
    "zero",
    "work_tm16",
    "work_tm32",
    "work_tm128",
)
# These are the histogram-dependent values which the current public selected-
# config entry can actually receive.  In particular, it does not receive the
# offsets, active/zero counts, or any work_tm* statistic.  N/K/qtype and the
# arrangement are also ABI inputs, but this pilot fixes them inside each
# qtype board, so only the fields below partition its six profiles.
SELECTOR_ABI_EQUIVALENCE_FIELDS = (
    "n",
    "k",
    "experts",
    "total_rows",
    "max_rows",
)
CELL_FIELDS = {
    "q",
    "round",
    "profile",
    "layout",
    "mapping_id",
    "n",
    "k",
    "experts",
    *ROUTER_MEASUREMENT_FIELDS,
    "rows_hash",
    "config",
    "provider",
    "iterations",
    "raw_bad",
    "median_us",
    "min_us",
    "max_us",
    "samples",
}
RUN_FIELDS = {
    "schema",
    "q",
    "round",
    "layout",
    "iterations",
    "warmups",
    "cells",
    "status",
}


class AdjudicationError(ValueError):
    """The raw measurement denominator or identity is not admissible."""


def _atomic_text(path: pathlib.Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)


def _fields(line: str, prefix: str, expected: set[str]) -> dict[str, str]:
    tokens = line.split()
    if not tokens or tokens[0] != prefix:
        raise AdjudicationError(f"{prefix} line prefix differs")
    result: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise AdjudicationError(f"{prefix} contains a non-field token")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            raise AdjudicationError(f"{prefix} contains an empty/duplicate field")
        result[key] = value
    if set(result) != expected:
        raise AdjudicationError(
            f"{prefix} field denominator differs: "
            f"missing={sorted(expected - set(result))} "
            f"extra={sorted(set(result) - expected)}"
        )
    return result


def _integer(value: str, what: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise AdjudicationError(f"{what} is not an integer") from error
    return parsed


def _positive_float(value: str, what: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise AdjudicationError(f"{what} is not numeric") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise AdjudicationError(f"{what} is not finite and positive")
    return parsed


def _samples(value: str, count: int) -> list[float]:
    if not (value.startswith("[") and value.endswith("]")):
        raise AdjudicationError("sample syntax differs")
    body = value[1:-1]
    values = (
        []
        if not body
        else [_positive_float(item, "sample") for item in body.split(",")]
    )
    if len(values) != count:
        raise AdjudicationError("sample denominator differs")
    if values != sorted(values):
        raise AdjudicationError("published samples are not ordered")
    return values


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-8, abs_tol=2e-6)


def _regret_interval(
    candidate_low: float,
    candidate_high: float,
    best_low: float,
    best_high: float,
) -> list[float]:
    # The lower bound gives the candidate its best observed round and the
    # oracle its worst; the upper bound does the converse.  This is the same
    # conservative envelope used by the real-shape ScaleFirst adjudicator.
    return [
        max(0.0, candidate_low / best_high - 1.0) * 100.0,
        max(0.0, candidate_high / best_low - 1.0) * 100.0,
    ]


def _score_common(
    configs: Iterable[str],
    profiles: Iterable[str],
    metrics: dict[tuple[int, str, str], dict[str, Any]],
    qtype: int,
) -> list[dict[str, Any]]:
    result = []
    profile_list = list(profiles)
    for config in sorted(configs):
        per_profile = []
        regrets = []
        lower = []
        upper = []
        for profile in profile_list:
            row = metrics[(qtype, profile, config)]
            regrets.append(float(row["regret_pct"]))
            lower.append(float(row["conservative_regret_interval_pct"][0]))
            upper.append(float(row["conservative_regret_interval_pct"][1]))
            per_profile.append(
                {
                    "profile": profile,
                    "aggregate_median_us": row["aggregate_median_us"],
                    "regret_pct": row["regret_pct"],
                    "conservative_regret_interval_pct": row[
                        "conservative_regret_interval_pct"
                    ],
                }
            )
        result.append(
            {
                "config": config,
                "max_regret_pct": max(regrets),
                "mean_regret_pct": statistics.mean(regrets),
                "conservative_max_regret_interval_pct": [
                    max(lower),
                    max(upper),
                ],
                "conservative_mean_regret_interval_pct": [
                    statistics.mean(lower),
                    statistics.mean(upper),
                ],
                "per_profile": per_profile,
            }
        )
    result.sort(
        key=lambda row: (
            row["max_regret_pct"],
            row["mean_regret_pct"],
            row["config"],
        )
    )
    return result


def _threshold_decision(
    scores: list[dict[str, Any]], threshold_pct: float
) -> dict[str, Any]:
    base = {
        "threshold_pct": threshold_pct,
        "common_candidate_count": len(scores),
        "point_minimax": None if not scores else scores[0],
        "selected": None,
    }
    if not scores:
        return {**base, "status": "NO_COMMON_TACTIC"}
    proven = [
        row
        for row in scores
        if row["conservative_max_regret_interval_pct"][1] <= threshold_pct + 1e-12
    ]
    if proven:
        proven.sort(
            key=lambda row: (
                row["max_regret_pct"],
                row["mean_regret_pct"],
                row["config"],
            )
        )
        return {
            **base,
            "status": "PROVEN_WITHIN_THRESHOLD",
            "selected": proven[0],
        }
    possible = [
        row
        for row in scores
        if row["conservative_max_regret_interval_pct"][0] <= threshold_pct + 1e-12
    ]
    return {
        **base,
        "status": (
            "UNRESOLVED_OVERLAPPING_INTERVAL" if possible else "PROVEN_OVER_THRESHOLD"
        ),
    }


def _load_measurements(
    plan_path: pathlib.Path,
    runs: pathlib.Path,
    rounds: int,
    iterations: int,
    warmups: int,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    if rounds <= 0 or iterations <= 0 or warmups <= 0:
        raise AdjudicationError("rounds/iterations/warmups must be positive")
    if not plan_path.is_file() or plan_path.is_symlink():
        raise AdjudicationError("plan must be a regular non-symlink file")
    if not runs.is_dir() or runs.is_symlink():
        raise AdjudicationError("runs must be a regular directory")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        planner.validate(plan)
    except (json.JSONDecodeError, OSError, planner.PlanError) as error:
        raise AdjudicationError(f"plan authority differs: {error}") from error
    if tuple(plan["qtypes"]) != QTYPES or tuple(plan["routers"]) != PROFILES:
        raise AdjudicationError("five-qtype/six-profile denominator differs")
    if len(plan["cells"]) != len(QTYPES) * len(PROFILES):
        raise AdjudicationError("plan cell denominator differs")
    authority = {(int(row["qtype"]), str(row["profile"])): row for row in plan["cells"]}
    if len(authority) != len(QTYPES) * len(PROFILES):
        raise AdjudicationError("plan qtype/profile identities are not unique")

    expected_names = {
        f"q{qtype}-round{round_index}.log"
        for qtype in QTYPES
        for round_index in range(1, rounds + 1)
    }
    actual_names = {path.name for path in runs.glob("q*-round*.log")}
    if actual_names != expected_names:
        raise AdjudicationError(
            "raw log denominator differs: "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )

    observations: dict[tuple[int, str, str], dict[int, list[float]]] = (
        collections.defaultdict(dict)
    )
    round_candidate_sets: dict[tuple[int, str], dict[int, set[str]]] = (
        collections.defaultdict(dict)
    )
    for qtype in QTYPES:
        for round_index in range(1, rounds + 1):
            path = runs / f"q{qtype}-round{round_index}.log"
            if not path.is_file() or path.is_symlink():
                raise AdjudicationError(f"raw log is missing/symlinked: {path}")
            lines = path.read_text(encoding="utf-8").splitlines()
            if any(line.startswith("FQ_GROUPED_ROUTER_FAILURE ") for line in lines):
                raise AdjudicationError("raw log contains a failure marker")
            markers = [
                _fields(line, "FQ_GROUPED_ROUTER_RUN", RUN_FIELDS)
                for line in lines
                if line.startswith("FQ_GROUPED_ROUTER_RUN ")
            ]
            if len(markers) != 1:
                raise AdjudicationError("run marker denominator differs")
            marker = markers[0]
            required_marker = {
                "schema": "grouped-kpack-multi-router-v1",
                "q": str(qtype),
                "round": str(round_index),
                "layout": "kpack",
                "iterations": str(iterations),
                "warmups": str(warmups),
                "cells": str(len(PROFILES)),
                "status": "PASS",
            }
            if marker != required_marker:
                raise AdjudicationError("run marker identity differs")

            rows = [
                _fields(line, "FQ_GROUPED_ROUTER_CELL", CELL_FIELDS)
                for line in lines
                if line.startswith("FQ_GROUPED_ROUTER_CELL ")
            ]
            if not rows:
                raise AdjudicationError("raw log has no candidate rows")
            seen: set[tuple[str, str]] = set()
            by_profile: dict[str, set[str]] = collections.defaultdict(set)
            for row in rows:
                profile = row["profile"]
                expected = authority.get((qtype, profile))
                if expected is None:
                    raise AdjudicationError("unknown qtype/profile cell")
                config = row["config"]
                if not config or config == "NONE":
                    raise AdjudicationError("candidate config is empty/NONE")
                identity = (profile, config)
                if identity in seen:
                    raise AdjudicationError("duplicate candidate in one round")
                seen.add(identity)

                exact_integers = {
                    "q": qtype,
                    "round": round_index,
                    "n": int(expected["n"]),
                    "k": int(expected["k"]),
                    "experts": int(expected["experts"]),
                    "iterations": iterations,
                    "raw_bad": 0,
                    **{name: int(expected[name]) for name in ROUTER_MEASUREMENT_FIELDS},
                }
                for name, wanted in exact_integers.items():
                    if _integer(row[name], f"cell.{name}") != wanted:
                        raise AdjudicationError(f"cell {name} identity differs")
                if (
                    row["layout"] != "kpack"
                    or row["mapping_id"].lower() != str(expected["mapping_id"]).lower()
                    or row["rows_hash"].lower() != str(expected["rows_hash"]).lower()
                    or row["provider"] != "standard-aiu"
                ):
                    raise AdjudicationError("cell layout/mapping/hash/provider differs")
                sample_values = _samples(row["samples"], iterations)
                published_median = _positive_float(row["median_us"], "median_us")
                published_minimum = _positive_float(row["min_us"], "min_us")
                published_maximum = _positive_float(row["max_us"], "max_us")
                if not (
                    _close(published_median, statistics.median(sample_values))
                    and _close(published_minimum, sample_values[0])
                    and _close(published_maximum, sample_values[-1])
                ):
                    raise AdjudicationError("published timing statistics differ")
                key = (qtype, profile, config)
                if round_index in observations[key]:
                    raise AdjudicationError("candidate round is duplicated")
                observations[key][round_index] = sample_values
                by_profile[profile].add(config)
            if set(by_profile) != set(PROFILES):
                raise AdjudicationError("one or more planned profiles are missing")
            for profile in PROFILES:
                if not by_profile[profile]:
                    raise AdjudicationError("profile candidate denominator is empty")
                round_candidate_sets[(qtype, profile)][round_index] = by_profile[
                    profile
                ]

    candidate_sets: dict[tuple[int, str], set[str]] = {}
    for qtype in QTYPES:
        for profile in PROFILES:
            by_round = round_candidate_sets[(qtype, profile)]
            if set(by_round) != set(range(1, rounds + 1)):
                raise AdjudicationError("candidate round denominator differs")
            first = by_round[1]
            if any(by_round[index] != first for index in range(2, rounds + 1)):
                raise AdjudicationError("candidate set differs across rounds")
            candidate_sets[(qtype, profile)] = set(first)

    metrics: dict[tuple[int, str, str], dict[str, Any]] = {}
    for qtype in QTYPES:
        for profile in PROFILES:
            configs = candidate_sets[(qtype, profile)]
            provisional = []
            for config in sorted(configs):
                by_round = observations[(qtype, profile, config)]
                if set(by_round) != set(range(1, rounds + 1)):
                    raise AdjudicationError("candidate sample rounds differ")
                if any(len(values) != iterations for values in by_round.values()):
                    raise AdjudicationError("round x iteration denominator differs")
                round_medians = [
                    statistics.median(by_round[index]) for index in range(1, rounds + 1)
                ]
                aggregate_samples = [
                    value for index in range(1, rounds + 1) for value in by_round[index]
                ]
                if len(aggregate_samples) != rounds * iterations:
                    raise AdjudicationError("aggregate sample denominator differs")
                provisional.append(
                    {
                        "qtype": qtype,
                        "format": authority[(qtype, profile)]["format"],
                        "profile": profile,
                        "config": config,
                        "sample_count": len(aggregate_samples),
                        "round_medians_us": round_medians,
                        "round_median_interval_us": [
                            min(round_medians),
                            max(round_medians),
                        ],
                        "aggregate_median_us": statistics.median(aggregate_samples),
                    }
                )
            best_median = min(row["aggregate_median_us"] for row in provisional)
            best_low = min(row["round_median_interval_us"][0] for row in provisional)
            best_high = min(row["round_median_interval_us"][1] for row in provisional)
            for row in provisional:
                row["best_aggregate_median_us"] = best_median
                row["regret_pct"] = (
                    max(0.0, row["aggregate_median_us"] / best_median - 1.0) * 100.0
                )
                row["conservative_regret_interval_pct"] = _regret_interval(
                    row["round_median_interval_us"][0],
                    row["round_median_interval_us"][1],
                    best_low,
                    best_high,
                )
                metrics[(qtype, profile, row["config"])] = row
    return plan, metrics


def adjudicate(
    plan_path: pathlib.Path,
    runs: pathlib.Path,
    output: pathlib.Path,
    rounds: int,
    iterations: int,
    warmups: int,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> dict[str, Any]:
    if not math.isfinite(threshold_pct) or threshold_pct < 0.0:
        raise AdjudicationError("threshold must be finite and nonnegative")
    plan, metrics = _load_measurements(plan_path, runs, rounds, iterations, warmups)
    qtype_results = []
    for qtype in QTYPES:
        permutation_a = next(
            row
            for row in plan["cells"]
            if row["qtype"] == qtype and row["profile"] == "permutation-a"
        )
        permutation_b = next(
            row
            for row in plan["cells"]
            if row["qtype"] == qtype and row["profile"] == "permutation-b"
        )
        if any(
            permutation_a[name] != permutation_b[name]
            for name in SELECTOR_ABI_EQUIVALENCE_FIELDS
        ):
            raise AdjudicationError("permutation public route features differ")
        if permutation_a["rows_hash"] == permutation_b["rows_hash"]:
            raise AdjudicationError("permutation controls are not distinct")

        profile_sets = {
            profile: {
                config
                for (measured_qtype, measured_profile, config) in metrics
                if measured_qtype == qtype and measured_profile == profile
            }
            for profile in PROFILES
        }
        all_common = set.intersection(*(profile_sets[name] for name in PROFILES))
        all_scores = _score_common(all_common, PROFILES, metrics, qtype)
        one_tactic = _threshold_decision(all_scores, threshold_pct)

        feature_groups: dict[tuple[int, ...], list[str]] = collections.defaultdict(list)
        for cell in (row for row in plan["cells"] if row["qtype"] == qtype):
            key = tuple(int(cell[name]) for name in SELECTOR_ABI_EQUIVALENCE_FIELDS)
            feature_groups[key].append(str(cell["profile"]))
        feature_classes = []
        for key, profiles in sorted(feature_groups.items()):
            ordered_profiles = sorted(profiles, key=PROFILES.index)
            common = set.intersection(
                *(profile_sets[profile] for profile in ordered_profiles)
            )
            scores = _score_common(common, ordered_profiles, metrics, qtype)
            feature_classes.append(
                {
                    "identity": dict(zip(SELECTOR_ABI_EQUIVALENCE_FIELDS, key)),
                    "profiles": ordered_profiles,
                    "decision": _threshold_decision(scores, threshold_pct),
                    "common_candidates": scores,
                }
            )

        permutation_common = (
            profile_sets["permutation-a"] & profile_sets["permutation-b"]
        )
        permutation_scores = _score_common(
            permutation_common,
            ("permutation-a", "permutation-b"),
            metrics,
            qtype,
        )
        permutation = _threshold_decision(permutation_scores, threshold_pct)
        class_statuses = {
            feature_class["decision"]["status"] for feature_class in feature_classes
        }
        if one_tactic["status"] == "PROVEN_WITHIN_THRESHOLD":
            verdict = "ONE_TACTIC_WITHIN_THRESHOLD"
        elif class_statuses <= {"PROVEN_WITHIN_THRESHOLD"}:
            verdict = "ROUTE_FEATURES_ABI_SUFFICIENT"
        elif class_statuses & {"NO_COMMON_TACTIC", "PROVEN_OVER_THRESHOLD"}:
            verdict = "ROUTE_FEATURES_INSUFFICIENT"
        else:
            verdict = "UNRESOLVED_OVERLAPPING_INTERVAL"
        qtype_results.append(
            {
                "qtype": qtype,
                "format": permutation_a["format"],
                "verdict": verdict,
                "one_tactic_decision": one_tactic,
                "one_tactic_common_candidates": all_scores,
                "route_feature_classes": feature_classes,
                "permutation_feature_identity": {
                    **{
                        name: permutation_a[name]
                        for name in SELECTOR_ABI_EQUIVALENCE_FIELDS
                    },
                    "group_size": 256,
                },
                "permutation_rows_hashes": [
                    permutation_a["rows_hash"],
                    permutation_b["rows_hash"],
                ],
                "permutation_decision": permutation,
                "permutation_common_candidates": permutation_scores,
            }
        )

    q_verdicts = {row["verdict"] for row in qtype_results}
    if "ROUTE_FEATURES_INSUFFICIENT" in q_verdicts:
        verdict = "ROUTE_FEATURES_INSUFFICIENT"
    elif "UNRESOLVED_OVERLAPPING_INTERVAL" in q_verdicts:
        verdict = "UNRESOLVED_OVERLAPPING_INTERVAL"
    elif q_verdicts == {"ONE_TACTIC_WITHIN_THRESHOLD"}:
        verdict = "ONE_TACTIC_WITHIN_THRESHOLD"
    else:
        verdict = "ROUTE_FEATURES_ABI_SUFFICIENT"

    candidate_rows = sorted(
        metrics.values(),
        key=lambda row: (row["qtype"], PROFILES.index(row["profile"]), row["config"]),
    )
    raw_log_sha256 = {
        f"q{qtype}-round{round_index}.log": hashlib.sha256(
            (runs / f"q{qtype}-round{round_index}.log").read_bytes()
        ).hexdigest()
        for qtype in QTYPES
        for round_index in range(1, rounds + 1)
    }
    result = {
        "schema": SCHEMA,
        "plan_schema": planner.SCHEMA,
        "measurement_status": "COMPLETE",
        "verdict": verdict,
        "threshold_pct": threshold_pct,
        "rounds": rounds,
        "iterations": iterations,
        "warmups": warmups,
        "qtypes": list(QTYPES),
        "profiles": list(PROFILES),
        "selector_abi": {
            "entry_inputs": [
                "qtype",
                "n",
                "k",
                "group_size",
                "experts",
                "total_rows",
                "max_rows",
                "arrangement",
            ],
            "measured_equivalence_fields": list(SELECTOR_ABI_EQUIVALENCE_FIELDS),
            "unavailable_histogram_fields": [
                "active",
                "zero",
                "work_tm16",
                "work_tm32",
                "work_tm128",
                "rows_hash",
            ],
        },
        "expected_profile_cells": len(QTYPES) * len(PROFILES),
        "candidate_measurements": len(candidate_rows),
        "input_authority": {
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "raw_log_sha256": raw_log_sha256,
        },
        "qtype_results": qtype_results,
        "profile_candidates": candidate_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output / "summary.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    summary_header = (
        "qtype\tformat\tverdict\tone_tactic_status\tone_tactic_config\t"
        "one_tactic_max_regret_pct\tone_tactic_interval_low_pct\t"
        "one_tactic_interval_high_pct\tpermutation_status\t"
        "permutation_config\tpermutation_max_regret_pct\t"
        "permutation_interval_low_pct\tpermutation_interval_high_pct\t"
        "route_feature_classes\troute_feature_statuses\n"
    )

    def selected_columns(decision: dict[str, Any]) -> list[str]:
        selected = decision["selected"]
        if selected is None:
            return ["NONE", "NA", "NA", "NA"]
        interval = selected["conservative_max_regret_interval_pct"]
        return [
            selected["config"],
            f"{selected['max_regret_pct']:.9f}",
            f"{interval[0]:.9f}",
            f"{interval[1]:.9f}",
        ]

    summary_lines = [summary_header.rstrip("\n")]
    for row in qtype_results:
        one = selected_columns(row["one_tactic_decision"])
        permutation = selected_columns(row["permutation_decision"])
        summary_lines.append(
            "\t".join(
                [
                    str(row["qtype"]),
                    row["format"],
                    row["verdict"],
                    row["one_tactic_decision"]["status"],
                    *one,
                    row["permutation_decision"]["status"],
                    *permutation,
                    str(len(row["route_feature_classes"])),
                    ",".join(
                        sorted(
                            {
                                feature_class["decision"]["status"]
                                for feature_class in row["route_feature_classes"]
                            }
                        )
                    ),
                ]
            )
        )
    _atomic_text(output / "summary.tsv", "\n".join(summary_lines) + "\n")

    candidate_lines = [
        "qtype\tformat\tprofile\tconfig\tsample_count\taggregate_median_us\t"
        "round_median_low_us\tround_median_high_us\tregret_pct\t"
        "regret_interval_low_pct\tregret_interval_high_pct"
    ]
    for row in candidate_rows:
        candidate_lines.append(
            "\t".join(
                [
                    str(row["qtype"]),
                    row["format"],
                    row["profile"],
                    row["config"],
                    str(row["sample_count"]),
                    f"{row['aggregate_median_us']:.9f}",
                    f"{row['round_median_interval_us'][0]:.9f}",
                    f"{row['round_median_interval_us'][1]:.9f}",
                    f"{row['regret_pct']:.9f}",
                    f"{row['conservative_regret_interval_pct'][0]:.9f}",
                    f"{row['conservative_regret_interval_pct'][1]:.9f}",
                ]
            )
        )
    _atomic_text(output / "candidates.tsv", "\n".join(candidate_lines) + "\n")
    common_lines = [
        "qtype\tformat\tscope\tconfig\tmax_regret_pct\tmean_regret_pct\t"
        "max_regret_interval_low_pct\tmax_regret_interval_high_pct\t"
        "mean_regret_interval_low_pct\tmean_regret_interval_high_pct"
    ]
    for qtype_row in qtype_results:
        for scope, scores in (
            ("all-profiles", qtype_row["one_tactic_common_candidates"]),
            ("permutation-pair", qtype_row["permutation_common_candidates"]),
        ):
            for score in scores:
                max_interval = score["conservative_max_regret_interval_pct"]
                mean_interval = score["conservative_mean_regret_interval_pct"]
                common_lines.append(
                    "\t".join(
                        [
                            str(qtype_row["qtype"]),
                            qtype_row["format"],
                            scope,
                            score["config"],
                            f"{score['max_regret_pct']:.9f}",
                            f"{score['mean_regret_pct']:.9f}",
                            f"{max_interval[0]:.9f}",
                            f"{max_interval[1]:.9f}",
                            f"{mean_interval[0]:.9f}",
                            f"{mean_interval[1]:.9f}",
                        ]
                    )
                )
    _atomic_text(output / "common-candidates.tsv", "\n".join(common_lines) + "\n")
    print(
        "FQ_GROUPED_MULTI_ROUTER_ADJUDICATION "
        f"verdict={verdict} measurement=COMPLETE qtypes={len(QTYPES)} "
        f"profiles={len(PROFILES)} candidates={len(candidate_rows)} "
        f"threshold_pct={threshold_pct:.6f}"
    )
    return result


def _synthetic_runs(
    root: pathlib.Path,
    plan: dict[str, Any],
    scenario: str,
    rounds: int = 2,
    iterations: int = 3,
    warmups: int = 1,
) -> None:
    root.mkdir(parents=True, exist_ok=True)

    def center(profile: str, config: str, round_index: int) -> float:
        if scenario == "one":
            return 10.00 + 0.01 * (round_index - 1) + (1.0 if config == "B" else 0.0)
        if scenario == "abi":
            if profile == "balanced":
                return 10.0 if config == "A" else 12.0
            if profile == "hot-skewed":
                return 12.0 if config == "A" else 10.0
            if profile.startswith("permutation"):
                return 10.0 if config == "A" else 10.1
            return 10.0 if config == "A" else 10.2
        if scenario == "insufficient":
            if profile == "permutation-a":
                return 10.0 if config == "A" else 12.0
            if profile == "permutation-b":
                return 12.0 if config == "A" else 10.0
            return 10.0 if config == "A" else 10.2
        if scenario == "unresolved":
            if profile == "permutation-a":
                base = 10.0 if config == "A" else 10.35
            elif profile == "permutation-b":
                base = 10.35 if config == "A" else 10.0
            else:
                base = 10.0 if config == "A" else 10.2
            return base + 0.10 * (round_index - 1)
        raise AssertionError("unknown synthetic scenario")

    for qtype in QTYPES:
        cells = [row for row in plan["cells"] if row["qtype"] == qtype]
        for round_index in range(1, rounds + 1):
            lines = []
            for cell in cells:
                for config in ("A", "B"):
                    middle = center(cell["profile"], config, round_index)
                    values = [middle - 0.01, middle, middle + 0.01]
                    features = " ".join(
                        f"{name}={cell[name]}" for name in ROUTER_MEASUREMENT_FIELDS
                    )
                    lines.append(
                        "FQ_GROUPED_ROUTER_CELL "
                        f"q={qtype} round={round_index} profile={cell['profile']} "
                        f"layout=kpack mapping_id={cell['mapping_id']} "
                        f"n={cell['n']} k={cell['k']} experts={cell['experts']} "
                        f"{features} rows_hash={cell['rows_hash']} config={config} "
                        f"provider=standard-aiu iterations={iterations} raw_bad=0 "
                        f"median_us={values[1]:.9f} min_us={values[0]:.9f} "
                        f"max_us={values[2]:.9f} "
                        f"samples=[{','.join(f'{value:.9f}' for value in values)}]"
                    )
            lines.append(
                "FQ_GROUPED_ROUTER_RUN "
                f"schema=grouped-kpack-multi-router-v1 q={qtype} "
                f"round={round_index} layout=kpack iterations={iterations} "
                f"warmups={warmups} cells=6 status=PASS"
            )
            (root / f"q{qtype}-round{round_index}.log").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        plan = planner.materialize()
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        expected = {
            "one": "ONE_TACTIC_WITHIN_THRESHOLD",
            "abi": "ROUTE_FEATURES_ABI_SUFFICIENT",
            "insufficient": "ROUTE_FEATURES_INSUFFICIENT",
            "unresolved": "UNRESOLVED_OVERLAPPING_INTERVAL",
        }
        for scenario, verdict in expected.items():
            runs = root / scenario
            _synthetic_runs(runs, plan, scenario)
            result = adjudicate(plan_path, runs, root / f"out-{scenario}", 2, 3, 1)
            if result["verdict"] != verdict:
                raise AssertionError(f"{scenario} verdict differs: {result['verdict']}")

        runs = root / "plants"
        _synthetic_runs(runs, plan, "one")
        target = runs / "q10-round2.log"
        original = target.read_text(encoding="utf-8")
        plants = (
            original.replace("raw_bad=0", "raw_bad=1", 1),
            original.replace("provider=standard-aiu", "provider=packed-row", 1),
            original.replace("work_tm16=64", "work_tm16=65", 1),
            original.replace("samples=[", "samples=[99,", 1),
            original.replace("config=B", "config=A", 1),
            original.replace(" status=PASS", " status=FAIL", 1),
        )
        for index, body in enumerate(plants):
            target.write_text(body, encoding="utf-8")
            try:
                adjudicate(plan_path, runs, root / f"red-{index}", 2, 3, 1)
            except AdjudicationError:
                pass
            else:
                raise AssertionError(f"planted negative {index} stayed green")
        target.write_text(original, encoding="utf-8")
        (runs / "q10-round3.log").write_text(original, encoding="utf-8")
        try:
            adjudicate(plan_path, runs, root / "red-extra", 2, 3, 1)
        except AdjudicationError:
            pass
        else:
            raise AssertionError("extra-log negative stayed green")
    print(
        "[fq-grouped-multi-router-adjudication:self-test] PASS "
        "5x6 denominator, aggregate/round envelopes, one/ABI/insufficient/"
        "unresolved verdicts; seven planted negatives RED"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    run = commands.add_parser("adjudicate")
    run.add_argument("--plan", type=pathlib.Path, required=True)
    run.add_argument("--runs", type=pathlib.Path, required=True)
    run.add_argument("--output", type=pathlib.Path, required=True)
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--iterations", type=int, required=True)
    run.add_argument("--warmups", type=int, required=True)
    run.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    arguments = parser.parse_args()
    try:
        if arguments.command == "self-test":
            self_test()
        else:
            adjudicate(
                arguments.plan,
                arguments.runs,
                arguments.output,
                arguments.rounds,
                arguments.iterations,
                arguments.warmups,
                arguments.threshold_pct,
            )
        return 0
    except (AdjudicationError, AssertionError, OSError, ValueError) as error:
        print(
            f"[fq-grouped-multi-router-adjudication] FAIL: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
