#!/usr/bin/env python3
"""Plan a conservative global K-pack screen shortlist.

The input is a normalized, execution-independent screen census.  A later
adapter may construct it from distributed worker logs, but this module knows
nothing about those logs or their paths.  It compares candidates only inside
one exact ``(qtype, operator, workload_key, comparison_key)`` group.

A measured candidate is excluded only when one stable witness has a strictly
better observed envelope by more than ten percent.  Noisy candidates,
historical anchors, candidates without a comparable end-to-end cost, and
axis/family sentinels are retained.  Selection is closed at symbol granularity
because the current binaries select parents by symbol, not individual runtime
variants.  There is deliberately no top-N limit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, NoReturn


INPUT_SCHEMA = "quactlize.kpack-screen-candidates.v1"
OUTPUT_SCHEMA = "quactlize.kpack-screen-retention.v1"
SCREEN_ITERATIONS = 2
CORRECTNESS_REPEATS = 1
EXCLUSION_MARGIN = 0.10
NOISY_SPREAD = 0.05
QTYPES = {10, 11, 12, 13, 14}
OPERATORS = {"dense", "grouped"}
ROUTES = {"scalefirst", "fully-quantized"}
CLASSIFICATIONS = {"MEASURED", "STRUCTURAL_UNAVAILABLE"}
COMPARISON_STATUSES = {"COMPARABLE", "NONCOMPARABLE"}
REDUCER_STATUSES = {"NOT_REQUIRED", "AVAILABLE", "MISSING"}
TOKEN = re.compile(r"[^\s\0]+\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RetentionError(ValueError):
    """The normalized census or resulting retention authority is unsafe."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as error:
        raise RetentionError(f"value is not canonical JSON: {error}") from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RetentionError(f"cannot hash {path}: {error}") from error


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or not TOKEN.fullmatch(value):
        raise RetentionError(f"{label} is not one nonempty token")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise RetentionError(f"{label} is not a SHA-256")
    return value


def _positive(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RetentionError(f"{label} is not numeric") from error
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
        raise RetentionError(f"{label} must be positive and finite")
    return result


def _identity(row: dict[str, Any]) -> tuple[int, str, str]:
    return int(row["qtype"]), str(row["operator"]), str(row["workload_key"])


def _validate_axes(value: Any, label: str) -> dict[str, int | str]:
    if not isinstance(value, dict) or not value:
        raise RetentionError(f"{label} axes must be a nonempty object")
    result: dict[str, int | str] = {}
    for name, axis_value in value.items():
        _token(name, f"{label} axis name")
        if (isinstance(axis_value, bool) or
                not isinstance(axis_value, (int, str)) or
                isinstance(axis_value, str) and not TOKEN.fullmatch(axis_value)):
            raise RetentionError(f"{label} axis {name} is not an int/token")
        result[name] = axis_value
    return result


def _validate_reducer(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RetentionError(f"{label} reducer is not an object")
    status = value.get("status")
    if status not in REDUCER_STATUSES:
        raise RetentionError(f"{label} reducer status is unknown")
    if status == "AVAILABLE":
        if set(value) != {"status", "lower_us", "upper_us", "authority_sha256"}:
            raise RetentionError(f"{label} available reducer fields differ")
        lower = _positive(value["lower_us"], f"{label} reducer lower")
        upper = _positive(value["upper_us"], f"{label} reducer upper")
        if lower > upper:
            raise RetentionError(f"{label} reducer envelope is reversed")
        _sha(value["authority_sha256"], f"{label} reducer authority")
    elif set(value) != {"status"}:
        raise RetentionError(f"{label} reducer fields differ")
    return value


def _candidate(row: Any, index: int) -> dict[str, Any]:
    label = f"candidate[{index}]"
    required = {
        "candidate_id", "work_item_id", "symbol", "qtype", "operator",
        "workload_key", "route", "algorithm", "family_key", "axes",
        "historical_anchor", "classification", "terminal_reason", "timing",
        "comparison", "reducer",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise RetentionError(f"{label} fields differ")
    _sha(row["candidate_id"], f"{label} ID")
    _sha(row["work_item_id"], f"{label} work-item ID")
    for field in ("symbol", "workload_key", "algorithm", "family_key"):
        _token(row[field], f"{label} {field}")
    if (isinstance(row["qtype"], bool) or row["qtype"] not in QTYPES or
            row["operator"] not in OPERATORS or row["route"] not in ROUTES or
            not isinstance(row["historical_anchor"], bool) or
            row["classification"] not in CLASSIFICATIONS):
        raise RetentionError(f"{label} identity is invalid")
    _validate_axes(row["axes"], label)
    timing, comparison = row["timing"], row["comparison"]
    if not isinstance(timing, dict) or set(timing) != {"samples_us"} or \
            not isinstance(comparison, dict) or set(comparison) != {"status", "key"}:
        raise RetentionError(f"{label} timing/comparison fields differ")
    if comparison["status"] not in COMPARISON_STATUSES:
        raise RetentionError(f"{label} comparison status is unknown")
    if comparison["status"] == "COMPARABLE":
        _token(comparison["key"], f"{label} comparison key")
    elif comparison["key"] is not None:
        raise RetentionError(f"{label} noncomparable key must be null")
    _validate_reducer(row["reducer"], label)
    samples = timing["samples_us"]
    if not isinstance(samples, list):
        raise RetentionError(f"{label} samples are not a list")
    if row["classification"] == "MEASURED":
        if (row["terminal_reason"] is not None or
                len(samples) != SCREEN_ITERATIONS):
            raise RetentionError(f"{label} measured denominator differs")
        for sample in samples:
            _positive(sample, f"{label} sample")
        if (comparison["status"] == "COMPARABLE" and
                row["reducer"]["status"] == "MISSING"):
            # This is a valid diagnostic state: it must be retained.
            pass
    else:
        if (not isinstance(row["terminal_reason"], str) or
                not TOKEN.fullmatch(row["terminal_reason"]) or samples or
                row["historical_anchor"] or
                comparison != {"status": "NONCOMPARABLE", "key": None} or
                row["reducer"] != {"status": "NOT_REQUIRED"}):
            raise RetentionError(f"{label} structural terminal is malformed")
    return copy.deepcopy(row)


def validate_input(value: Any) -> dict[str, Any]:
    required = {"schema", "authorities", "measurement", "workloads",
                "work_items", "candidates"}
    if not isinstance(value, dict) or set(value) != required or \
            value.get("schema") != INPUT_SCHEMA:
        raise RetentionError("screen census schema/fields differ")
    if not isinstance(value["authorities"], dict) or not value["authorities"]:
        raise RetentionError("screen authorities must be a nonempty object")
    canonical(value["authorities"])
    if value["measurement"] != {
            "screen_iterations": SCREEN_ITERATIONS,
            "correctness_repeats": CORRECTNESS_REPEATS}:
        raise RetentionError("screen measurement denominator differs")

    candidates = [_candidate(row, index)
                  for index, row in enumerate(value["candidates"])] \
        if isinstance(value["candidates"], list) else None
    if candidates is None or not candidates:
        raise RetentionError("candidate denominator is empty")
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    if len(candidate_by_id) != len(candidates):
        raise RetentionError("candidate ownership is duplicated")

    if not isinstance(value["work_items"], list) or not value["work_items"]:
        raise RetentionError("work-item denominator is empty")
    work_items: dict[str, dict[str, Any]] = {}
    owned_candidates: list[str] = []
    for index, row in enumerate(value["work_items"]):
        if not isinstance(row, dict) or set(row) != {
                "work_item_id", "qtype", "operator", "workload_key", "route",
                "candidate_ids"}:
            raise RetentionError(f"work_item[{index}] fields differ")
        work_id = _sha(row["work_item_id"], f"work_item[{index}] ID")
        if work_id in work_items:
            raise RetentionError("work-item ownership is duplicated")
        if (row["qtype"] not in QTYPES or row["operator"] not in OPERATORS or
                row["route"] not in ROUTES):
            raise RetentionError(f"work_item[{index}] identity is invalid")
        _token(row["workload_key"], f"work_item[{index}] workload")
        ids = row["candidate_ids"]
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise RetentionError(f"work_item[{index}] candidate ownership differs")
        for candidate_id in ids:
            _sha(candidate_id, f"work_item[{index}] candidate ID")
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                raise RetentionError("work item owns an unknown candidate")
            if (candidate["work_item_id"] != work_id or
                    _identity(candidate) != _identity(row) or
                    candidate["route"] != row["route"]):
                raise RetentionError(
                    "candidate ownership identity differs from its work-item owner")
        owned_candidates.extend(ids)
        work_items[work_id] = copy.deepcopy(row)
    if (len(owned_candidates) != len(set(owned_candidates)) or
            set(owned_candidates) != set(candidate_by_id)):
        raise RetentionError("candidate ownership is missing or duplicated")

    if not isinstance(value["workloads"], list) or not value["workloads"]:
        raise RetentionError("workload denominator is empty")
    workloads: dict[tuple[int, str, str], dict[str, Any]] = {}
    owned_work_items: list[str] = []
    for index, row in enumerate(value["workloads"]):
        if not isinstance(row, dict) or set(row) != {
                "qtype", "operator", "workload_key", "work_item_ids"}:
            raise RetentionError(f"workload[{index}] fields differ")
        if row["qtype"] not in QTYPES or row["operator"] not in OPERATORS:
            raise RetentionError(f"workload[{index}] identity is invalid")
        _token(row["workload_key"], f"workload[{index}] key")
        identity = _identity(row)
        if identity in workloads:
            raise RetentionError("workload identity is duplicated")
        ids = row["work_item_ids"]
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise RetentionError(f"workload[{index}] work-item ownership differs")
        for work_id in ids:
            _sha(work_id, f"workload[{index}] work-item ID")
            work = work_items.get(work_id)
            if work is None or _identity(work) != identity:
                raise RetentionError("work item is absent or owned by another workload")
        owned_work_items.extend(ids)
        workloads[identity] = copy.deepcopy(row)
    if (len(owned_work_items) != len(set(owned_work_items)) or
            set(owned_work_items) != set(work_items)):
        raise RetentionError("work-item ownership is missing or duplicated")
    for identity in workloads:
        if not any(_identity(row) == identity and
                   row["classification"] == "MEASURED" for row in candidates):
            raise RetentionError(f"workload {identity} has no measured candidate")
    return copy.deepcopy(value)


def _envelope(row: dict[str, Any]) -> tuple[float, float]:
    samples = [_positive(value, "screen sample")
               for value in row["timing"]["samples_us"]]
    lower, upper = min(samples), max(samples)
    reducer = row["reducer"]
    if reducer["status"] == "AVAILABLE":
        lower += float(reducer["lower_us"])
        upper += float(reducer["upper_us"])
    return lower, upper


def _spread(row: dict[str, Any]) -> float:
    samples = list(map(float, row["timing"]["samples_us"]))
    return (max(samples) - min(samples)) / min(samples)


def _rank(row: dict[str, Any]) -> tuple[float, float, str]:
    lower, upper = _envelope(row)
    return lower, upper, row["candidate_id"]


def _add(reasons: dict[str, set[str]], candidate_id: str, reason: str) -> None:
    reasons.setdefault(candidate_id, set()).add(reason)


def derive(value: dict[str, Any], *, input_sha256: str | None = None) -> dict[str, Any]:
    census = validate_input(value)
    candidates = census["candidates"]
    by_id = {row["candidate_id"]: row for row in candidates}
    measured = [row for row in candidates if row["classification"] == "MEASURED"]
    reasons: dict[str, set[str]] = {}

    for row in measured:
        candidate_id = row["candidate_id"]
        if row["historical_anchor"]:
            _add(reasons, candidate_id, "HISTORICAL_ANCHOR")
        if _spread(row) > NOISY_SPREAD:
            _add(reasons, candidate_id, "UNCERTAIN_SCREEN_SPREAD")
        if row["comparison"]["status"] == "NONCOMPARABLE":
            _add(reasons, candidate_id, "NONCOMPARABLE_E2E")
        if row["reducer"]["status"] == "MISSING":
            _add(reasons, candidate_id, "MISSING_EXACT_REDUCER")

    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in measured:
        grouped.setdefault(_identity(row), []).append(row)

    # Sentinels are global within one exact public workload, never shard-local.
    for identity, rows in grouped.items():
        families: dict[str, list[dict[str, Any]]] = {}
        axes: dict[tuple[str, bytes], list[dict[str, Any]]] = {}
        for row in rows:
            families.setdefault(row["family_key"], []).append(row)
            for name, axis_value in row["axes"].items():
                axes.setdefault((name, canonical(axis_value)), []).append(row)
        for family, values in families.items():
            winner = min(values, key=_rank)
            _add(reasons, winner["candidate_id"], f"FAMILY_SENTINEL:{family}")
        for (name, encoded), values in axes.items():
            winner = min(values, key=_rank)
            axis_value = json.loads(encoded.decode("ascii"))
            _add(reasons, winner["candidate_id"],
                 f"AXIS_SENTINEL:{name}={axis_value}")

    witnesses: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in measured:
        if (row["comparison"]["status"] != "COMPARABLE" or
                row["reducer"]["status"] == "MISSING" or
                _spread(row) > NOISY_SPREAD):
            continue
        key = (*_identity(row), row["comparison"]["key"])
        if key not in witnesses or _rank(row) < _rank(witnesses[key]):
            witnesses[key] = row
    for witness in witnesses.values():
        _add(reasons, witness["candidate_id"], "STABLE_COMPARISON_LEADER")

    possible_proofs: dict[str, dict[str, Any]] = {}
    for row in measured:
        candidate_id = row["candidate_id"]
        if candidate_id in reasons:
            continue
        if row["comparison"]["status"] != "COMPARABLE":
            _add(reasons, candidate_id, "NONCOMPARABLE_E2E")
            continue
        if row["reducer"]["status"] == "MISSING":
            _add(reasons, candidate_id, "MISSING_EXACT_REDUCER")
            continue
        key = (*_identity(row), row["comparison"]["key"])
        witness = witnesses.get(key)
        if witness is None:
            _add(reasons, candidate_id, "NO_STABLE_WITNESS")
            continue
        lower, _upper = _envelope(row)
        _wlower, witness_upper = _envelope(witness)
        threshold = witness_upper * (1.0 + EXCLUSION_MARGIN)
        if lower <= threshold:
            _add(reasons, candidate_id, "OVERLAPPING_OBSERVED_ENVELOPE")
            continue
        possible_proofs[candidate_id] = {
            "rule": "CANDIDATE_LOWER_GT_WITNESS_UPPER_TIMES_1P10",
            "witness_candidate_id": witness["candidate_id"],
            "candidate_lower_us": lower,
            "witness_upper_us": witness_upper,
            "margin_multiplier": 1.0 + EXCLUSION_MARGIN,
            "candidate_timing_sha256": digest(row["timing"]),
            "witness_timing_sha256": digest(witness["timing"]),
            "candidate_reducer_sha256": digest(row["reducer"]),
            "witness_reducer_sha256": digest(witness["reducer"]),
        }

    # The executable filter is a symbol list.  Close retention under that
    # physical granularity before publishing any exclusion claim.
    symbol_groups: dict[tuple[str, str], list[str]] = {}
    for row in measured:
        symbol_groups.setdefault(
            (row["work_item_id"], row["symbol"]), []).append(row["candidate_id"])
    for sibling_ids in symbol_groups.values():
        if any(candidate_id in reasons for candidate_id in sibling_ids):
            for candidate_id in sibling_ids:
                if candidate_id not in reasons:
                    _add(reasons, candidate_id, "SYMBOL_GRANULARITY_SIBLING")

    retained_ids = set(reasons)
    excluded_ids = set(possible_proofs) - retained_ids
    measured_ids = {row["candidate_id"] for row in measured}
    if retained_ids | excluded_ids != measured_ids or retained_ids & excluded_ids:
        raise RetentionError("measured retention/exclusion partition is incomplete")

    work_outputs = []
    for work in sorted(census["work_items"], key=lambda row: row["work_item_id"]):
        owned = [by_id[candidate_id] for candidate_id in work["candidate_ids"]]
        retained = sorted(row["candidate_id"] for row in owned
                          if row["candidate_id"] in retained_ids)
        excluded = sorted(row["candidate_id"] for row in owned
                          if row["candidate_id"] in excluded_ids)
        terminal = sorted(row["candidate_id"] for row in owned
                          if row["classification"] == "STRUCTURAL_UNAVAILABLE")
        selected_symbols = sorted({row["symbol"] for row in owned
                                   if row["candidate_id"] in retained_ids})
        work_outputs.append({
            "work_item_id": work["work_item_id"],
            "qtype": work["qtype"], "operator": work["operator"],
            "workload_key": work["workload_key"], "route": work["route"],
            "selected_symbols": selected_symbols,
            "retained_candidate_ids": retained,
            "excluded_candidate_ids": excluded,
            "structural_candidate_ids": terminal,
        })

    exclusion_rows = [{"candidate_id": candidate_id,
                       "proof": possible_proofs[candidate_id]}
                      for candidate_id in sorted(excluded_ids)]
    retained_rows = [{
        "candidate_id": candidate_id,
        "reasons": sorted(reasons[candidate_id]),
    } for candidate_id in sorted(retained_ids)]
    structural_ids = sorted(row["candidate_id"] for row in candidates
                            if row["classification"] == "STRUCTURAL_UNAVAILABLE")
    output = {
        "schema": OUTPUT_SCHEMA,
        "input": {
            "schema": INPUT_SCHEMA,
            "file_sha256": input_sha256,
            "canonical_sha256": digest(census),
            "authorities_sha256": digest(census["authorities"]),
        },
        "policy": {
            "scope": "GLOBAL_EXACT_QTYPE_OPERATOR_WORKLOAD",
            "screen_iterations": SCREEN_ITERATIONS,
            "correctness_repeats": CORRECTNESS_REPEATS,
            "exclusion_margin": EXCLUSION_MARGIN,
            "noisy_spread_threshold": NOISY_SPREAD,
            "candidate_cap": None,
            "top_n": None,
            "uncertain_action": "RETAIN",
            "missing_reducer_action": "RETAIN",
            "historical_anchor_action": "RETAIN_OR_FAIL_IF_NOT_MEASURED",
            "selection_granularity": "WORK_ITEM_SYMBOL",
        },
        "denominator": {
            "workloads": len(census["workloads"]),
            "work_items": len(census["work_items"]),
            "candidates": len(candidates),
            "measured_candidates": len(measured_ids),
            "structural_candidates": len(structural_ids),
            "retained_candidates": len(retained_ids),
            "excluded_candidates": len(excluded_ids),
            "selected_symbols": sum(len(row["selected_symbols"])
                                    for row in work_outputs),
            "empty_work_items": sum(not row["selected_symbols"]
                                    for row in work_outputs),
        },
        "work_items": work_outputs,
        "retained": retained_rows,
        "excluded": exclusion_rows,
        "structural": structural_ids,
    }
    output["payload_sha256"] = digest(output)
    return output


def validate_output(value: Any, census: dict[str, Any],
                    *, input_sha256: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != OUTPUT_SCHEMA:
        raise RetentionError("retention output schema differs")
    observed = value.get("payload_sha256")
    unsigned = copy.deepcopy(value)
    unsigned.pop("payload_sha256", None)
    if observed != digest(unsigned):
        raise RetentionError("retention payload hash differs")
    expected = derive(census, input_sha256=input_sha256)
    if value != expected:
        raise RetentionError("retention output differs from live screen census")
    return value


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetentionError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RetentionError(f"{label} is not an object")
    return value


def write_output(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise RetentionError(f"refusing to replace output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _fixture() -> dict[str, Any]:
    work_a, work_b = "a" * 64, "b" * 64

    def row(number: int, work: str, symbol: str, samples: list[float], **kw: Any
            ) -> dict[str, Any]:
        return {
            "candidate_id": f"{number:064x}", "work_item_id": work,
            "symbol": symbol, "qtype": 12, "operator": "dense",
            "workload_key": "m8-n5120-k8192", "route": kw.get("route", "scalefirst"),
            "algorithm": kw.get("algorithm", "S1"),
            "family_key": kw.get("family_key", "sf-s1"),
            "axes": kw.get("axes", {"tile_m": 8, "tile_n": 64}),
            "historical_anchor": kw.get("historical_anchor", False),
            "classification": "MEASURED", "terminal_reason": None,
            "timing": {"samples_us": samples},
            "comparison": kw.get(
                "comparison", {"status": "COMPARABLE", "key": "E2E-FP16"}),
            "reducer": kw.get("reducer", {"status": "NOT_REQUIRED"}),
        }

    rows = [
        row(1, work_a, "fast", [10.0, 10.1]),
        row(2, work_a, "slow", [12.0, 12.1]),
        row(3, work_a, "noisy", [12.0, 13.0]),
        row(4, work_a, "anchor", [14.0, 14.1], historical_anchor=True),
        row(5, work_b, "missing-reducer", [15.0, 15.1], route="fully-quantized",
            family_key="fq-split", reducer={"status": "MISSING"}),
        row(6, work_b, "noncomparable", [16.0, 16.1], route="fully-quantized",
            family_key="fq-other",
            comparison={"status": "NONCOMPARABLE", "key": None}),
    ]
    return {
        "schema": INPUT_SCHEMA,
        "authorities": {"screen": "f" * 64},
        "measurement": {"screen_iterations": 2, "correctness_repeats": 1},
        "workloads": [{"qtype": 12, "operator": "dense",
                       "workload_key": "m8-n5120-k8192",
                       "work_item_ids": [work_a, work_b]}],
        "work_items": [
            {"work_item_id": work_a, "qtype": 12, "operator": "dense",
             "workload_key": "m8-n5120-k8192", "route": "scalefirst",
             "candidate_ids": [row["candidate_id"] for row in rows[:4]]},
            {"work_item_id": work_b, "qtype": 12, "operator": "dense",
             "workload_key": "m8-n5120-k8192", "route": "fully-quantized",
             "candidate_ids": [row["candidate_id"] for row in rows[4:]]},
        ],
        "candidates": rows,
    }


def self_test() -> None:
    census = _fixture()
    result = derive(census)
    retained = {row["candidate_id"] for row in result["retained"]}
    if f"{2:064x}" not in {row["candidate_id"] for row in result["excluded"]}:
        raise AssertionError("strictly separated candidate was not excluded")
    if not {f"{value:064x}" for value in (1, 3, 4, 5, 6)} <= retained:
        raise AssertionError("mandatory retention rule was lost")
    if result["policy"]["top_n"] is not None:
        raise AssertionError("numeric candidate cap entered policy")
    validate_output(result, census)
    broken = copy.deepcopy(census)
    broken["work_items"][0]["candidate_ids"].pop()
    try:
        derive(broken)
    except RetentionError:
        pass
    else:
        raise AssertionError("missing candidate ownership stayed green")
    print("[kpack-screen-retention:self-test] PASS global-envelope=10pct "
          "noisy+anchor+reducer+noncomparable+axis+family=RETAIN "
          "top-n=NONE ownership-negative=RED")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"[kpack-screen-retention] FAIL: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    plan = commands.add_parser("plan")
    plan.add_argument("--input", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        census = load_json(args.input, "screen census")
        if args.command == "plan":
            result = derive(census, input_sha256=file_sha(args.input))
            write_output(args.output, result)
            print("KPACK_SCREEN_RETENTION PASS "
                  f"workloads={result['denominator']['workloads']} "
                  f"candidates={result['denominator']['candidates']} "
                  f"retained={result['denominator']['retained_candidates']} "
                  f"excluded={result['denominator']['excluded_candidates']} "
                  f"symbols={result['denominator']['selected_symbols']} "
                  f"output={args.output}")
            return 0
        result = load_json(args.plan, "retention plan")
        validate_output(result, census, input_sha256=file_sha(args.input))
        print(f"KPACK_SCREEN_RETENTION_VALID PASS plan={args.plan}")
        return 0
    except (OSError, RetentionError) as error:
        fail(str(error))


if __name__ == "__main__":
    main()
