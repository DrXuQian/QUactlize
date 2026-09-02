#!/usr/bin/env python3
"""Derive the fail-closed K-pack confidence set and follow-up denominator.

This planner is deliberately local and does not rank medians.  It consumes the
complete, hash-checked A09 steady-state census and the one shared, exact
FullyQuantized Split-K reducer lookup.  A candidate is removed only when one
strictly comparable candidate proves both of the following:

* the slow candidate's observed lower envelope is more than the product margin
  above the witness's observed upper envelope; and
* a distribution-free paired log-ratio interval is wholly above that margin.

Everything noisy, overlapping, or not exactly pairable stays in the confidence
set.  There is no numeric/top-N cap.  The output is the complete denominator
for the paired steady, cold-compute, prepass, and first-use follow-up campaign;
it is not a shipping heuristic.
"""

from __future__ import annotations

import argparse
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import aggregate_kpack_discovery_results as discovery_aggregate
import analyze_fq_splitk_reducer_lookup as reducer_analysis
import plan_fq_kpack_route_optimal as route_plan
import plan_fq_splitk_reducer_lookup as reducer_plan


SCHEMA = "quactlize.kpack-confidence-followup-plan.v1"
DENIAL_SCHEMA = "quactlize.kpack-confidence-followup-denial.v1"
CONFIDENCE = 0.99
MARGIN = 0.03
CONFIRM_ROUNDS = 3
SAMPLES_PER_ROUND = 11
SAMPLES_PER_CANDIDATE = CONFIRM_ROUNDS * SAMPLES_PER_ROUND
# One absolute comparison can contain producer+reducer on both sides.  Give
# each of those four component intervals at most one quarter of the 1% error
# budget.  The exact order-statistic construction rounds conservatively.
ABSOLUTE_COMPONENT_CONFIDENCE = 1.0 - (1.0 - CONFIDENCE) / 4.0
FOLLOWUP_BOARDS = ("steady", "cold_compute", "prepass", "first_use")
HEX64 = set("0123456789abcdef")


class PlannerError(ValueError):
    """An input authority or required measurement is absent or contradictory."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as error:
        raise PlannerError("NONCANONICAL_VALUE", str(error)) from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(1 << 20), b""):
                checksum.update(block)
        return checksum.hexdigest()
    except OSError as error:
        raise PlannerError("MISSING_AUTHORITY", f"cannot hash {path}: {error}") from error


def _plain_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PlannerError("MISSING_AUTHORITY", f"{label} is absent or symlinked: {path}")
    return path


def _json(path: Path, label: str) -> dict[str, Any]:
    _plain_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlannerError("MALFORMED_AUTHORITY", f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PlannerError("MALFORMED_AUTHORITY", f"{label} is not an object")
    return value


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in HEX64 for character in value)):
        raise PlannerError("MALFORMED_AUTHORITY", f"{label} is not a SHA-256")
    return value


def _positive(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PlannerError("MALFORMED_TIMING", f"{label} is not numeric") from error
    if not math.isfinite(result) or result <= 0:
        raise PlannerError("MALFORMED_TIMING", f"{label} is not positive and finite")
    return result


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PlannerError("MALFORMED_TIMING", f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise PlannerError("MALFORMED_TIMING", f"{label} is not finite")
    return result


def _pretty_sha(value: Any) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _expected_workloads(plan: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    result: dict[tuple[int, str, str], dict[str, Any]] = {}
    for cell in plan["cells"]:
        identity = (cell["qtype"], cell["operator"], cell["workload_key"])
        if identity in result:
            raise PlannerError("PLAN_DRIFT", f"duplicated public workload {identity}")
        result[identity] = {
            "public_problem": cell["public_problem"],
            "source_class": cell["source_class"],
        }
    return result


def load_discovery(aggregate_root: Path, plan: dict[str, Any]
                   ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the portable aggregate only after its own full validator passes."""
    try:
        summary = discovery_aggregate.validate_output(aggregate_root)
    except (discovery_aggregate.AggregateError, OSError, KeyError,
            TypeError, ValueError) as error:
        raise PlannerError("DISCOVERY_AUTHORITY_MISMATCH", str(error)) from error

    expected_plan_sha = _pretty_sha(plan)
    if summary.get("authorities", {}).get("workload_plan_sha256") != expected_plan_sha:
        raise PlannerError(
            "DISCOVERY_PLAN_MISMATCH",
            "steady census is not bound to the canonical A09 workload plan")
    contract = summary.get("run_contract", {})
    confirm = contract.get("confirm", {})
    if (confirm.get("timing_samples_per_runtime") != SAMPLES_PER_ROUND or
            len(confirm.get("rounds", [])) != CONFIRM_ROUNDS or
            contract.get("all_admissible_candidates") is not True or
            contract.get("top_n") is not None):
        raise PlannerError("DISCOVERY_CONTRACT_MISMATCH", "steady confirmation contract differs")

    census_path = aggregate_root / summary["census"]["path"]
    rows: list[dict[str, Any]] = []
    try:
        with census_path.open(encoding="utf-8") as stream:
            for line in stream:
                rows.append(json.loads(line))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlannerError("DISCOVERY_AUTHORITY_MISMATCH", str(error)) from error

    expected = _expected_workloads(plan)
    observed: dict[tuple[int, str, str], dict[str, Any]] = {}
    measured: dict[tuple[int, str, str], int] = {}
    for row in rows:
        identity = (row.get("qtype"), row.get("operator"), row.get("workload_key"))
        wanted = expected.get(identity)
        if wanted is None:
            raise PlannerError("DISCOVERY_DENOMINATOR_MISMATCH", f"unexpected workload {identity}")
        public = row.get("public_problem")
        source_class = row.get("source_class")
        if public != wanted["public_problem"] or source_class != wanted["source_class"]:
            raise PlannerError(
                "DISCOVERY_DENOMINATOR_MISMATCH",
                f"public workload differs: {identity}")
        previous = observed.setdefault(identity, public)
        if previous != public:
            raise PlannerError("DISCOVERY_DENOMINATOR_MISMATCH", f"ambiguous workload {identity}")
        if row.get("classification") == "MEASURED":
            measured[identity] = measured.get(identity, 0) + 1
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise PlannerError("DISCOVERY_DENOMINATOR_MISMATCH", f"missing workloads: {missing[:3]}")
    without_runtime = sorted(identity for identity in expected if not measured.get(identity))
    if without_runtime:
        raise PlannerError(
            "NO_PUBLIC_ADMISSIBLE_CANDIDATE",
            f"public workloads lack measured candidates: {without_runtime[:3]}")

    worker_rows = summary["authorities"]["worker_evidence"]
    device_workers = {
        row["device_identity_sha256"]: {
            "worker_id": row["worker_id"],
            "runtime_linkage_sha256": row["runtime_linkage_sha256"],
        }
        for row in worker_rows
    }
    if len(device_workers) != len(worker_rows):
        raise PlannerError(
            "DISCOVERY_AUTHORITY_MISMATCH",
            "steady census contains duplicate device identities")
    return rows, {
        "summary_sha256": file_sha(aggregate_root / "summary.json"),
        "census_sha256": file_sha(census_path),
        "device_workers": device_workers,
        "device_homogeneity_sha256":
            summary["authorities"]["device_homogeneity_sha256"],
        "source_sha": summary["authorities"]["source_sha"],
        "source_tree": summary["authorities"]["source_tree"],
        "workload_plan_sha256": expected_plan_sha,
        "run_contract_sha256": digest(contract),
    }


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise PlannerError("MALFORMED_REDUCER", f"{label} is not decimal") from error
    if not result.is_finite() or result <= 0:
        raise PlannerError("MALFORMED_REDUCER", f"{label} is not positive")
    return result


def load_reducer(reducer_root: Path, discovery_authority: dict[str, Any]
                ) -> tuple[dict[tuple[int, int, int, str, str, str], dict[str, Any]],
                           dict[str, Any]]:
    """Validate the exact 1,035-row reducer result and its output authority."""
    summary_path = _plain_file(reducer_root / "summary.json", "reducer summary")
    tsv_path = _plain_file(reducer_root / "summary.tsv", "reducer TSV")
    authority_path = _plain_file(
        reducer_root / "result-authority.json", "reducer result authority")
    summary = _json(summary_path, "reducer summary")
    authority = _json(authority_path, "reducer result authority")
    plan = reducer_plan.materialize()
    reducer_plan.validate_plan(plan, plan)

    summary_authority = summary.get("authorities")
    required_summary_authority = {
        "execution_authority_sha256", "source_commit", "source_tree",
        "sdk_authority_sha256", "binary_sha256", "build_authority_sha256",
        "device_identity_sha256", "device_homogeneity_sha256",
        "device_homogeneity_key", "runtime_linkage_sha256",
    }
    if (summary.get("schema") != reducer_analysis.SCHEMA or
            summary.get("verdict") != "EXACT_REDUCER_LOOKUP_MEASURED" or
            summary.get("plan_sha256") != reducer_plan.digest(plan) or
            summary.get("bundle_manifest_sha256") !=
                authority.get("bundle_manifest", {}).get("sha256") or
            not isinstance(summary_authority, dict) or
            set(summary_authority) != required_summary_authority):
        raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer summary identity differs")
    for field in required_summary_authority - {"source_commit", "source_tree"}:
        _sha(summary_authority[field], f"reducer summary authority {field}")
    if (summary_authority["source_commit"] != discovery_authority["source_sha"] or
            summary_authority["source_tree"] != discovery_authority["source_tree"]):
        raise PlannerError(
            "REDUCER_SOURCE_MISMATCH",
            "reducer and steady census source identities differ")
    expected_measurement = {
        "rounds": plan["measurement"]["rounds"],
        "warmups_per_round": 3,
        "samples_per_round": 11,
        "samples_per_case": 33,
        "case_order": plan["measurement"]["case_order"],
        "top_n": None,
        "point_estimate_pruning": False,
        "raw_bit_correctness": "EVERY_OUTPUT_ELEMENT_BEFORE_AND_AFTER_TIMING",
    }
    if (summary.get("measurement") != expected_measurement or
            summary.get("denominator") != {
                "cases": 1035, "unique_m_n": 345, "splits": [2, 4, 8]}):
        raise PlannerError("REDUCER_CONTRACT_MISMATCH", "reducer denominator differs")

    required_authority = {
        "schema", "execution_authority", "plan", "bundle_manifest",
        "binary", "device", "runs", "outputs", "analyzer",
    }
    if set(authority) != required_authority or \
            authority.get("schema") != reducer_analysis.AUTHORITY_SCHEMA:
        raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer authority schema differs")
    execution = authority.get("execution_authority")
    if not isinstance(execution, dict) or set(execution) != {"path", "sha256"}:
        raise PlannerError(
            "REDUCER_AUTHORITY_MISMATCH", "reducer execution authority differs")
    _sha(execution["sha256"], "reducer execution authority SHA-256")
    if execution["sha256"] != summary_authority["execution_authority_sha256"]:
        raise PlannerError(
            "REDUCER_AUTHORITY_MISMATCH", "reducer execution hash differs")
    device = authority.get("device")
    device_fields = {
        "worker_id", "device_identity_sha256", "device_homogeneity_sha256",
        "device_homogeneity_key", "runtime_linkage_sha256",
    }
    if not isinstance(device, dict) or set(device) != device_fields:
        raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer device record differs")
    for field in device_fields - {"worker_id"}:
        _sha(device[field], f"reducer device {field}")
        if device[field] != summary_authority[field]:
            raise PlannerError(
                "REDUCER_AUTHORITY_MISMATCH", f"reducer device {field} differs")
    reducer_device = device["device_identity_sha256"]
    campaign_worker = discovery_authority["device_workers"].get(reducer_device)
    if campaign_worker is None:
        raise PlannerError(
            "REDUCER_DEVICE_MISMATCH",
            "reducer was not measured on a device in the steady-census authority")
    if (device["device_homogeneity_sha256"] !=
            discovery_authority["device_homogeneity_sha256"] or
            device["worker_id"] != campaign_worker["worker_id"] or
            device["runtime_linkage_sha256"] !=
                campaign_worker["runtime_linkage_sha256"]):
        raise PlannerError(
            "REDUCER_DEVICE_MISMATCH",
            "reducer device homogeneity/worker/runtime linkage differs")

    binary = authority.get("binary")
    binary_fields = {
        "sha256", "build_authority_sha256", "sdk_authority_sha256"}
    if not isinstance(binary, dict) or set(binary) != binary_fields:
        raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer binary record differs")
    for field in binary_fields:
        _sha(binary[field], f"reducer binary {field}")
    if (binary["sha256"] != summary_authority["binary_sha256"] or
            binary["build_authority_sha256"] !=
                summary_authority["build_authority_sha256"] or
            binary["sdk_authority_sha256"] !=
                summary_authority["sdk_authority_sha256"]):
        raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer binary hashes differ")
    if authority.get("outputs") != {
            "summary.json": file_sha(summary_path),
            "summary.tsv": file_sha(tsv_path)}:
        raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer output hashes differ")
    analyzer = authority.get("analyzer")
    if analyzer != {
            "path": "tools/analyze_fq_splitk_reducer_lookup.py",
            "sha256": file_sha(ROOT / "tools/analyze_fq_splitk_reducer_lookup.py")}:
        raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer analyzer differs")
    plan_record = authority.get("plan")
    if (not isinstance(plan_record, dict) or
            set(plan_record) != {"path", "file_sha256", "canonical_sha256"} or
            plan_record.get("canonical_sha256") != reducer_plan.digest(plan)):
        raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer plan record differs")
    _sha(plan_record["file_sha256"], "reducer plan file SHA-256")
    manifest_record = authority.get("bundle_manifest")
    if (not isinstance(manifest_record, dict) or
            set(manifest_record) != {"path", "sha256"}):
        raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer manifest record differs")
    _sha(manifest_record["sha256"], "reducer manifest SHA-256")
    runs = authority.get("runs")
    if (not isinstance(runs, list) or len(runs) != 3 or
            [row.get("round") for row in runs if isinstance(row, dict)] != [1, 2, 3]):
        raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer run authority differs")
    for index, row in enumerate(runs):
        if set(row) != {"round", "path", "sha256"}:
            raise PlannerError("REDUCER_AUTHORITY_MISMATCH", f"reducer run {index} differs")
        _sha(row["sha256"], f"reducer run {index} SHA-256")

    raw_rows = summary.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(plan["cases"]):
        raise PlannerError("MISSING_REDUCER", "reducer case denominator is incomplete")
    result: dict[tuple[int, int, int, str, str, str], dict[str, Any]] = {}
    for expected, row in zip(plan["cases"], raw_rows):
        identity = {
            "ordinal": expected["ordinal"], "case_id": expected["case_id"],
            "m": expected["m"], "n": expected["n"], "split": expected["split"],
            "partial_dtype": "fp32", "output_dtype": "fp16",
            "implementation": expected["expected_implementation"],
            "workspace_bytes": expected["workspace_bytes"],
            "output_bytes": expected["output_bytes"],
        }
        if not isinstance(row, dict) or any(row.get(key) != value
                                            for key, value in identity.items()):
            raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer case identity differs")
        if row.get("raw_bit_correctness") != "PASS" or row.get("sample_count") != 33:
            raise PlannerError("MISSING_REDUCER", f"reducer case failed: {expected['case_id']}")
        samples_decimal = [_decimal(value, "reducer sample")
                           for value in row.get("samples_us", [])]
        if len(samples_decimal) != 33:
            raise PlannerError("MISSING_REDUCER", f"reducer samples missing: {expected['case_id']}")
        ordered = sorted(samples_decimal)
        round_medians = [statistics.median(samples_decimal[index:index + 11])
                         for index in range(0, 33, 11)]
        if ([str(value) for value in round_medians] !=
                [str(value) for value in row.get("round_medians_us", [])] or
                _decimal(row.get("median_us"), "reducer median") != ordered[16] or
                _decimal(row.get("min_us"), "reducer minimum") != ordered[0] or
                _decimal(row.get("max_us"), "reducer maximum") != ordered[-1]):
            raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "reducer sample statistics differ")
        for key in ("grid_ctas", "block_threads"):
            if isinstance(row.get(key), bool) or not isinstance(row.get(key), int) or row[key] <= 0:
                raise PlannerError("REDUCER_AUTHORITY_MISMATCH", f"reducer {key} differs")
        key = (expected["m"], expected["n"], expected["split"], "fp32", "fp16",
               expected["expected_implementation"])
        if key in result:
            raise PlannerError("REDUCER_AUTHORITY_MISMATCH", "duplicate reducer key")
        samples = [float(value) for value in samples_decimal]
        result[key] = {
            "case_id": expected["case_id"], "samples": samples,
            "samples_sha256": digest([str(value) for value in samples_decimal]),
            "min_us": float(ordered[0]),
            "max_us": float(ordered[-1]),
            "implementation": expected["expected_implementation"],
        }
    return result, {
        "summary_sha256": file_sha(summary_path),
        "summary_tsv_sha256": file_sha(tsv_path),
        "result_authority_sha256": file_sha(authority_path),
        "plan_sha256": reducer_plan.digest(plan),
        "bundle_manifest_sha256": summary["bundle_manifest_sha256"],
        "device_identity_sha256": reducer_device,
        "device_homogeneity_sha256": device["device_homogeneity_sha256"],
        "device_homogeneity_key": device["device_homogeneity_key"],
        "runtime_linkage_sha256": device["runtime_linkage_sha256"],
        "round_log_sha256": [row["sha256"] for row in runs],
    }


def _confirm_samples(row: dict[str, Any]) -> tuple[list[float], list[dict[str, Any]]]:
    timing = row.get("timing")
    runs = timing.get("confirm_runs") if isinstance(timing, dict) else None
    if not isinstance(runs, list) or len(runs) != 3:
        raise PlannerError("MALFORMED_TIMING", "candidate lacks three confirmation rounds")
    values: list[float] = []
    coordinates = []
    for round_index, run in enumerate(runs, 1):
        samples = run.get("samples") if isinstance(run, dict) else None
        if (run.get("round") != round_index or
                not isinstance(samples, list) or len(samples) != 11):
            raise PlannerError("MALFORMED_TIMING", "candidate confirmation denominator differs")
        coordinates.append({key: run[key] for key in (
            "round", "order", "schedule_seed", "source_log_sha256")})
        for sample_index, sample in enumerate(samples):
            if sample.get("sample_index") != sample_index:
                raise PlannerError("MALFORMED_TIMING", "candidate sample order differs")
            values.append(_positive(sample.get("us"), "candidate sample"))
    if len(values) != SAMPLES_PER_CANDIDATE:
        raise PlannerError("MALFORMED_TIMING", "candidate sample count differs")
    return values, coordinates


def _is_product_candidate(row: dict[str, Any]) -> bool:
    """ScaleFirst producer-only Split-K rows are diagnostics, not E2E timings."""
    return not (row["route"] == "scalefirst" and row["operator"] == "dense" and
                int(row.get("runtime", {}).get("split", 1)) > 1)


def order_statistic_interval(values: Iterable[float], confidence: float,
                             *, positive: bool = True
                            ) -> dict[str, float | int]:
    """Distribution-free two-sided confidence interval for a median."""
    parser = _positive if positive else _finite
    ordered = sorted(parser(value, "interval sample") for value in values)
    n = len(ordered)
    if n != SAMPLES_PER_CANDIDATE or not 0.0 < confidence < 1.0:
        raise PlannerError("MALFORMED_TIMING", "interval denominator differs")
    selected = None
    coverage = 0.0
    for rank in range(1, (n + 1) // 2 + 1):
        tail = sum(math.comb(n, index) for index in range(rank)) / (2 ** n)
        current = 1.0 - 2.0 * tail
        if current + 1e-15 >= confidence:
            selected, coverage = rank, current
        else:
            break
    if selected is None:
        raise PlannerError("MALFORMED_TIMING", "samples cannot support confidence")
    return {
        "sample_count": n, "order_rank": selected, "coverage": coverage,
        "lower": ordered[selected - 1], "upper": ordered[n - selected],
        "median": statistics.median(ordered),
    }


def _candidate(row: dict[str, Any], reducers: dict[tuple[Any, ...], dict[str, Any]]
              ) -> dict[str, Any]:
    producer, coordinates = _confirm_samples(row)
    public = row["public_problem"]
    reducer = None
    reducer_key = None
    if row["route"] == "fully-quantized" and row["operator"] == "dense":
        split = int(row["runtime"].get("split", 0))
        if split not in (1, 2, 4, 8):
            raise PlannerError("MALFORMED_TIMING", "FQ dense split identity differs")
        if split > 1:
            implementation = reducer_plan.expected_implementation(
                int(public["m"]), int(public["n"]))
            reducer_key = (int(public["m"]), int(public["n"]), split,
                           "fp32", "fp16", implementation)
            reducer = reducers.get(reducer_key)
            if reducer is None:
                raise PlannerError(
                    "MISSING_REDUCER",
                    f"no exact reducer for M={public['m']} N={public['n']} S={split}")
    component_id = reducer["case_id"] if reducer is not None else "NONE"
    identity = {
        "work_item_id": row["work_item_id"], "route": row["route"],
        "operator": row["operator"], "qtype": row["qtype"],
        "workload_key": row["workload_key"], "parent_id": row["parent_id"],
        "static_candidate_id": row["static_candidate_id"],
        "symbol": row["symbol"], "static": row["static"],
        "runtime": row["runtime"],
    }
    candidate_id = digest(identity)
    reducer_samples = reducer["samples"] if reducer is not None else None
    total = ([left + right for left, right in zip(producer, reducer_samples)]
             if reducer_samples is not None else list(producer))
    producer_interval = order_statistic_interval(
        producer, ABSOLUTE_COMPONENT_CONFIDENCE)
    reducer_interval = (order_statistic_interval(
        reducer_samples, ABSOLUTE_COMPONENT_CONFIDENCE)
        if reducer_samples is not None else None)
    lower = float(producer_interval["lower"]) + (
        float(reducer_interval["lower"]) if reducer_interval else 0.0)
    upper = float(producer_interval["upper"]) + (
        float(reducer_interval["upper"]) if reducer_interval else 0.0)
    return {
        "candidate_id": candidate_id, "identity": identity,
        "producer_samples": producer, "total_samples": total,
        "sample_coordinates": coordinates,
        "reducer_case_id": component_id, "reducer_key": reducer_key,
        "reducer_samples": reducer_samples,
        "envelope_lower_us": lower, "envelope_upper_us": upper,
        "producer_interval": producer_interval,
        "reducer_interval": reducer_interval,
        "median_total_us": statistics.median(total),
        "producer_samples_sha256": digest(producer),
        "reducer_samples_sha256": (reducer["samples_sha256"] if reducer else None),
        "pair_key": digest({
            "work_item_id": row["work_item_id"],
            "coordinates": coordinates, "reducer_case_id": component_id,
            "device_identity_sha256": row["authority"]["device_identity_sha256"],
            "runtime_linkage_sha256": row["authority"]["runtime_linkage_sha256"],
        }),
    }


def paired_log_ratio_interval(numerator: Iterable[float], denominator: Iterable[float],
                              confidence: float = CONFIDENCE) -> dict[str, float | int]:
    """Exact sign/order-statistic CI for the median paired log ratio.

    For 33 continuous paired observations the selected order interval has
    99.5448615868% coverage, conservatively exceeding the requested 99%.
    """
    left, right = list(numerator), list(denominator)
    if len(left) != len(right) or len(left) != SAMPLES_PER_CANDIDATE:
        raise PlannerError("UNPAIRED_TIMING", "paired denominator is not 33")
    logs = [math.log(_positive(a, "ratio numerator") /
                     _positive(b, "ratio denominator"))
            for a, b in zip(left, right)]
    interval = order_statistic_interval(logs, confidence, positive=False)
    selected = int(interval["order_rank"])
    return {
        "method": "EXACT_SIGN_ORDER_STATISTIC_MEDIAN_LOG_RATIO",
        "sample_count": len(logs), "order_rank": selected,
        "coverage": interval["coverage"],
        "point_ratio": math.exp(float(interval["median"])),
        "lower_ratio": math.exp(float(interval["lower"])),
        "upper_ratio": math.exp(float(interval["upper"])),
    }


def _exclusion_proof(candidate: dict[str, Any], witness: dict[str, Any]
                    ) -> dict[str, Any] | None:
    if candidate["candidate_id"] == witness["candidate_id"] or \
            candidate["pair_key"] != witness["pair_key"]:
        return None
    absolute_threshold = witness["envelope_upper_us"] * (1.0 + MARGIN)
    if not candidate["envelope_lower_us"] > absolute_threshold:
        return None
    ratio = paired_log_ratio_interval(
        candidate["total_samples"], witness["total_samples"])
    if not ratio["lower_ratio"] > 1.0 + MARGIN:
        return None
    return {
        "witness_candidate_id": witness["candidate_id"],
        "absolute_envelope": {
            "candidate_lower_us": candidate["envelope_lower_us"],
            "witness_upper_us": witness["envelope_upper_us"],
            "margin_multiplier": 1.0 + MARGIN,
            "proved_slower": True,
        },
        "paired_log_ratio": ratio,
        "pairing_authority_sha256": candidate["pair_key"],
    }


def _public_identity(row: dict[str, Any]) -> tuple[int, str, str]:
    return row["qtype"], row["operator"], row["workload_key"]


def _candidate_output(candidate: dict[str, Any]) -> dict[str, Any]:
    identity = candidate["identity"]
    route = identity["route"]
    return {
        "candidate_id": candidate["candidate_id"],
        "route": route, "parent_id": identity["parent_id"],
        "static_candidate_id": identity["static_candidate_id"],
        "symbol": identity["symbol"], "static": identity["static"],
        "runtime": identity["runtime"],
        "steady_evidence": {
            "samples": 33,
            "producer_samples_sha256": candidate["producer_samples_sha256"],
            "reducer_case_id": candidate["reducer_case_id"],
            "reducer_samples_sha256": candidate["reducer_samples_sha256"],
            "conservative_additive_99pct_envelope_us": [
                candidate["envelope_lower_us"], candidate["envelope_upper_us"]],
            "producer_component_interval": candidate["producer_interval"],
            "reducer_component_interval": candidate["reducer_interval"],
            "median_total_us_for_reporting_only": candidate["median_total_us"],
        },
        "requirements": {
            "steady": "REQUIRED_PAIRED_CONFIDENCE_SET",
            "cold_compute": "REQUIRED_EXPLICIT_L2_FLUSH",
            "prepass": ("REQUIRED_SHARED_RESIDENT_WEIGHT_COMPONENT"
                        if route == "scalefirst" else "NOT_APPLICABLE"),
            "first_use": ("REQUIRED_PREPASS_PLUS_COMPUTE"
                          if route == "scalefirst" else "REQUIRED_DIRECT_COMPUTE"),
        },
    }


def derive_followup(rows: list[dict[str, Any]], reducers: dict[tuple[Any, ...], dict[str, Any]],
                    expected: dict[tuple[int, str, str], dict[str, Any]],
                    authorities: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {
        identity: [] for identity in expected}
    diagnostic_count = 0
    for row in rows:
        if row.get("classification") != "MEASURED":
            continue
        identity = _public_identity(row)
        if identity not in grouped:
            raise PlannerError("DISCOVERY_DENOMINATOR_MISMATCH", f"unexpected row {identity}")
        if not _is_product_candidate(row):
            diagnostic_count += 1
            continue
        grouped[identity].append(_candidate(row, reducers))

    workloads = []
    jobs = []
    retained_total = excluded_total = candidate_total = 0
    for identity in sorted(grouped):
        candidates = sorted(grouped[identity], key=lambda row: row["candidate_id"])
        if not candidates:
            raise PlannerError(
                "NO_PUBLIC_ADMISSIBLE_CANDIDATE", f"no E2E candidate for {identity}")
        ids = [row["candidate_id"] for row in candidates]
        if len(ids) != len(set(ids)):
            raise PlannerError("DISCOVERY_DENOMINATOR_MISMATCH", f"duplicate candidate {identity}")
        retained, excluded = [], []
        for candidate in candidates:
            proofs = [proof for witness in candidates
                      if (proof := _exclusion_proof(candidate, witness)) is not None]
            if proofs:
                proofs.sort(key=lambda proof: proof["witness_candidate_id"])
                excluded.append({
                    "candidate_id": candidate["candidate_id"],
                    "proof": proofs[0],
                })
            else:
                retained.append(candidate)
        if not retained:
            raise PlannerError(
                "NO_PUBLIC_ADMISSIBLE_CANDIDATE",
                f"all candidates excluded for {identity}")

        qtype, operator, workload_key = identity
        public = expected[identity]["public_problem"]
        workload_id = digest({
            "qtype": qtype, "operator": operator,
            "workload_key": workload_key, "public_problem": public})
        retained_outputs = [_candidate_output(row) for row in retained]
        candidate_ids = [row["candidate_id"] for row in retained_outputs]
        sf_ids = [row["candidate_id"] for row in retained_outputs
                  if row["route"] == "scalefirst"]
        board_candidates = {
            "steady": candidate_ids,
            "cold_compute": candidate_ids,
            "prepass": sf_ids,
            "first_use": candidate_ids,
        }
        for board in FOLLOWUP_BOARDS:
            selected = board_candidates[board]
            if not selected:
                continue
            job_contract = {
                "steady": {
                    "rounds": 3, "samples_per_round": 11,
                    "warmups_per_round": 3,
                    "cache": "CACHE_READY_COMPUTE_ONLY",
                },
                "cold_compute": {
                    "rounds": 3, "samples_per_round": 11,
                    "warmups_per_round": 0,
                    "cache": "EXPLICIT_L2_FLUSH_BEFORE_EVERY_SAMPLE",
                },
                "prepass": {
                    "rounds": 3, "samples_per_round": 11,
                    "warmups_per_round": 0,
                    "scope": "ONE_TIME_PER_RESIDENT_WEIGHT_TENSOR",
                },
                "first_use": {
                    "rounds": 3, "samples_per_round": 11,
                    "warmups_per_round": 0,
                    "scope": "PREPASS_PLUS_COMPUTE_FOR_SF_DIRECT_COMPUTE_FOR_FQ",
                },
            }[board]
            jobs.append({
                "job_id": digest({"workload_id": workload_id, "board": board,
                                  "candidate_ids": selected}),
                "workload_id": workload_id, "board": board,
                "candidate_ids": selected, "candidate_count": len(selected),
                "contract": job_contract,
                "candidate_order": "PAIRED_FORWARD_REVERSE_HASHED_NO_PRUNING",
            })
        workloads.append({
            "workload_id": workload_id, "qtype": qtype, "operator": operator,
            "workload_key": workload_key,
            "source_class": expected[identity]["source_class"],
            "public_problem": public,
            "verdict": "CONFIDENCE_SET_RETAINED",
            "input_candidates": len(candidates),
            "retained_candidates": retained_outputs,
            "retained_count": len(retained_outputs),
            "excluded": excluded, "excluded_count": len(excluded),
        })
        candidate_total += len(candidates)
        retained_total += len(retained)
        excluded_total += len(excluded)

    contract = {
        "confidence": CONFIDENCE, "regret_margin": MARGIN,
        "absolute_rule": "BONFERRONI_ADDITIVE_MEDIAN_CI_LOWER_GT_WITNESS_UPPER_TIMES_1P03",
        "absolute_component_confidence": ABSOLUTE_COMPONENT_CONFIDENCE,
        "paired_rule": "EXACT_SIGN_ORDER_STATISTIC_MEDIAN_LOG_RATIO_99PCT",
        "elimination": "BOTH_ABSOLUTE_AND_PAIRED_PROOFS_REQUIRED",
        "unpaired_action": "RETAIN", "overlap_action": "RETAIN",
        "top_n": None, "numeric_candidate_cap": None,
        "point_estimate_pruning": False,
        "screen_samples_used": False, "confirm_samples_per_candidate": 33,
        "fq_splitk_reducer": "EXACT_SHARED_LOOKUP_ADDED_WITH_UNCERTAINTY",
        "compiled_default_fallback": False,
    }
    payload = {
        "schema": SCHEMA, "verdict": "FOLLOWUP_MEASUREMENTS_REQUIRED",
        "authorities": authorities, "contract": contract,
        "denominator": {
            "public_workloads": len(workloads),
            "input_product_candidates": candidate_total,
            "retained_candidates": retained_total,
            "excluded_candidates": excluded_total,
            "non_product_diagnostic_runtimes": diagnostic_count,
            "measurement_jobs": len(jobs),
            "retention_is_uncapped": True,
        },
        "workloads": workloads, "measurement_jobs": jobs,
    }
    payload["payload_sha256"] = digest(payload)
    return payload


def validate_plan(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise PlannerError("OUTPUT_MISMATCH", "follow-up plan schema differs")
    observed = value.get("payload_sha256")
    copy_value = copy.deepcopy(value)
    copy_value.pop("payload_sha256", None)
    if observed != digest(copy_value):
        raise PlannerError("OUTPUT_MISMATCH", "follow-up payload hash differs")
    contract = value.get("contract", {})
    if (contract.get("top_n") is not None or
            contract.get("numeric_candidate_cap") is not None or
            contract.get("point_estimate_pruning") is not False or
            contract.get("compiled_default_fallback") is not False):
        raise PlannerError("OUTPUT_MISMATCH", "follow-up pruning/fallback contract differs")
    workloads = value.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise PlannerError("OUTPUT_MISMATCH", "follow-up workload denominator is empty")
    retained = sum(row.get("retained_count", -1) for row in workloads)
    if retained != value.get("denominator", {}).get("retained_candidates"):
        raise PlannerError("OUTPUT_MISMATCH", "follow-up retained denominator differs")
    for row in workloads:
        candidates = row.get("retained_candidates")
        if not isinstance(candidates, list) or len(candidates) != row.get("retained_count"):
            raise PlannerError("OUTPUT_MISMATCH", "workload confidence set differs")


def plan(aggregate_root: Path, reducer_root: Path) -> dict[str, Any]:
    canonical_plan = route_plan.materialize()
    route_plan.validate_plan(canonical_plan)
    rows, discovery_authority = load_discovery(aggregate_root, canonical_plan)
    reducers, reducer_authority = load_reducer(reducer_root, discovery_authority)
    authorities = {
        "route_plan_canonical_sha256": route_plan.digest(canonical_plan),
        "route_plan_generator_sha256": file_sha(
            ROOT / "tools/plan_fq_kpack_route_optimal.py"),
        "planner_sha256": file_sha(Path(__file__)),
        "discovery": discovery_authority,
        "reducer": reducer_authority,
    }
    result = derive_followup(
        rows, reducers, _expected_workloads(canonical_plan), authorities)
    validate_plan(result)
    return result


def write_frozen(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise PlannerError("OUTPUT_EXISTS", f"refusing to replace stale output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def denial(error: PlannerError, aggregate_root: Path, reducer_root: Path) -> dict[str, Any]:
    inputs = {}
    for label, path in (("discovery_summary", aggregate_root / "summary.json"),
                        ("reducer_summary", reducer_root / "summary.json"),
                        ("reducer_result_authority", reducer_root / "result-authority.json")):
        if path.is_file() and not path.is_symlink():
            inputs[label] = {"path": str(path), "sha256": file_sha(path)}
    result = {
        "schema": DENIAL_SCHEMA, "verdict": "NO_MEASURED_POLICY",
        "reason_code": error.code, "reason": str(error),
        "compiled_default_fallback": False, "measurement_jobs": [],
        "inputs": inputs,
    }
    result["payload_sha256"] = digest(result)
    return result


def self_test() -> None:
    interval = paired_log_ratio_interval([2.0] * 33, [1.0] * 33)
    if interval["sample_count"] != 33 or interval["order_rank"] != 9 or \
            interval["coverage"] < 0.99 or interval["lower_ratio"] != 2.0:
        raise AssertionError("paired interval contract differs")
    print("[kpack-confidence-followup:self-test] PASS confidence=0.99 "
          "margin=3pct samples=3x11 top_n=NONE uncertain=RETAIN")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    create = commands.add_parser("plan")
    create.add_argument("--aggregate", type=Path, required=True)
    create.add_argument("--reducer-results", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("validate-output")
    check.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "validate-output":
            validate_plan(_json(args.plan, "follow-up plan"))
            print(f"[kpack-confidence-followup] PASS validated={args.plan}")
        else:
            try:
                result = plan(args.aggregate, args.reducer_results)
            except PlannerError as error:
                write_frozen(args.output, denial(error, args.aggregate, args.reducer_results))
                print(f"[kpack-confidence-followup] NO_MEASURED_POLICY: {error}", file=sys.stderr)
                return 2
            write_frozen(args.output, result)
            print("[kpack-confidence-followup] PASS "
                  f"workloads={result['denominator']['public_workloads']} "
                  f"retained={result['denominator']['retained_candidates']} "
                  f"excluded={result['denominator']['excluded_candidates']} "
                  f"output={args.output}")
        return 0
    except (PlannerError, route_plan.PlanError, reducer_plan.PlanError,
            OSError, KeyError, TypeError, ValueError) as error:
        print(f"[kpack-confidence-followup] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
