#!/usr/bin/env python3
"""Fail-closed aggregation of complete K-pack discovery worker evidence.

The existing ``kpack-discovery-worker-result.v1`` is intentionally only an
exactly-once completion ledger.  It cannot by itself adjudicate measurements:
it has no log hashes, argv, round/order, retention, or result-authority links.
This tool therefore consumes one additional immutable *worker evidence*
document per worker.  That document is the minimum bridge from the completion
ledger to measured bytes; no filename convention is treated as authority.

Worker evidence schema (all paths are regular, non-symlink files below the
evidence document's directory)::

  {
    "schema": "quactlize.kpack-discovery-worker-evidence.v2",
    "worker_id": 0,
    "worker_count": 8,
    "bundle_sha256": "...", "workload_plan_sha256": "...",
    "workload_index_sha256": "...", "master_sha256": "...",
    "assignment_sha256": "...", "device_homogeneity_sha256": "...",
    "device_identity_sha256": "...",
    "execution_authority": {"path": "inputs/execution-authority.json",
                              "sha256": "..."},
    "completion_result": {"path": "worker-result.json", "sha256": "..."},
    "result_authorities": [
      {"route": "scalefirst", "path": "sf-authority.json", "sha256": "..."}
    ],
    "run_contract": {
      "schedule_seed": {"schema": "...schedule-seed.v1", ...},
      "grouped_warmups": 3,
      "screen": {"timing_samples_per_runtime": 5,
                 "correctness_repeats": 256},
      "confirm": {"timing_samples_per_runtime": 11,
                   "correctness_repeats": 256,
                   "rounds": [{"round": 1, "order": "FORWARD"},
                              {"round": 2, "order": "REVERSE"},
                              {"round": 3, "order": "HASHED"}]},
      "all_admissible_candidates": true, "top_n": null
    },
    "work_items": [{
      "work_item_id": "...", "result_authority_sha256": "...",
      "execution_inputs": {"artifact_id": "...", "shard_key": "...",
                           "binary": {"executed_path": "...", "sha256": "..."},
                           "manifest": {"snapshot": {"path": "...", ...}}, ...},
      "screen": {"argv": ["binary", "--iterations=5", ...],
                 "argv_sha256": "...",
                 "log": {"path": "screen.log", "sha256": "..."}},
      "retention": null,
      "confirm": [{"round": 1, "order": "FORWARD",
                   "schedule_seed": "0x...", "argv": [...],
                   "argv_sha256": "...",
                   "log": {"path": "confirm-1.log", "sha256": "..."}}, ...]
    }]
  }

For ScaleFirst ``retention`` is required and is
``{"symbols": file-record, "sidecar": file-record}``; the symbol set must be
exactly the set having at least one raw-bit-clean measured runtime in screen.
FullyQuantized currently retains every static parent and uses ``null``.
If and only if a ScaleFirst screen proves an empty retained symbol file, its
confirmation records instead have exact keys ``round``, ``order``,
``schedule_seed``, ``empty_structural`` (true), and ``log``.  Each hash-bound
log contains only::

  KPACK_DISCOVERY_EMPTY_CONFIRM work_item_id=<id> round=<n> order=<order> \
schedule_seed=<seed> screen_sha256=<sha> retention_symbols_sha256=<sha>

This records that no device launch occurred; it does not fabricate a device
completion.  The global workload must still have an admissible runtime in
another shard.

Every argv is checked for the declared iteration/correctness controls and the
workload identity.  Screen establishes the exact dynamic runtime denominator;
all three confirmation rounds must reproduce it (restricted only by the
proved ScaleFirst structural retention).  A measured row with nonzero
``raw_bad``, a missing parent/runtime, a stale hash, or a top-N contract fails
the whole aggregation.  No timing rank is used to eliminate candidates.

Output is a scalable directory rather than one giant JSON object:
``census.jsonl`` has one canonical record per static-candidate/runtime/workload,
and ``summary.json`` binds its SHA-256 and all input authorities.  Every
measured record contains its screen samples separately and all 33 raw confirm
samples grouped by round/order/seed, plus source-log, worker, device, binary,
artifact, manifest, receipt, and runtime-linkage authority.  Aggregation uses
portable manifest/rows snapshots; execution-host absolute paths are checked as
argv strings and need not remain live after the worker finishes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import statistics
import sys
import tempfile
from typing import Any, Iterable, NoReturn

import kpack_discovery_worker_plan as worker_plan
import kpack_discovery_build_partitions as partitions
import fq_dense_structural_proof as fq_structural
import run_kpack_discovery_worker as worker_runner


EVIDENCE_SCHEMA = worker_runner.EVIDENCE_SCHEMA
OUTPUT_SCHEMA = "quactlize.kpack-discovery-complete-census.v2"
FILE_SCHEMA = "quactlize.kpack-discovery-census-file.v2"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_STRUCTURAL = {
    "INADMISSIBLE", "INADMISSIBLE_SHARED_STORAGE",
    "INADMISSIBLE_OCCUPANCY", "INADMISSIBLE_SHIPPING_SMEM",
    "INADMISSIBLE_PERSISTENT_SMEM", "INADMISSIBLE_SPLIT_SMEM",
    "INADMISSIBLE_SPLIT_PARTITION",
    "INADMISSIBLE_K_TILE_DOES_NOT_DIVIDE", "INADMISSIBLE_M8_DECODE_ONLY",
    "INADMISSIBLE_CAN_IMPLEMENT", "SHIPPING_SHARED_STORAGE",
    "SPLIT_SHARED_STORAGE", "SPLIT_PARTITION",
    "INADMISSIBLE_PIPELINE_DEPTH", "M8_DECODE_ONLY_M_GE_8",
    "PACKED_A_DECODE_ONLY_M_NOT_1", "REAL_CAN_IMPLEMENT",
}


class AggregateError(ValueError):
    """The evidence is incomplete, stale, contradictory, or numerically bad."""


def fail(message: str) -> NoReturn:
    raise SystemExit(f"kpack discovery result aggregation: {message}")


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AggregateError(f"value is not canonical JSON: {exc}") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AggregateError(f"cannot hash {path}: {exc}") from exc


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregateError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregateError(f"{label} must be a JSON object")
    return value


def integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AggregateError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise AggregateError(f"{label} must be >= {minimum}")
    return value


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(
            mark in value for mark in ("\0", "\n", "\r")):
        raise AggregateError(f"{label} must be one nonempty string")
    return value


def sha(value: Any, label: str) -> str:
    value = text(value, label)
    if not SHA_RE.fullmatch(value):
        raise AggregateError(f"{label} must be one lowercase SHA-256")
    return value


def finite_positive(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AggregateError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise AggregateError(f"{label} must be finite and positive")
    return result


def kv_line(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise AggregateError(f"line does not start with {prefix!r}")
    fields: dict[str, str] = {}
    for token in shlex.split(line[len(prefix):]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in fields:
            raise AggregateError(f"duplicate log field {key}")
        fields[key] = value
    return fields


def parse_samples(value: Any, label: str) -> list[float]:
    if not isinstance(value, str) or len(value) < 2 or value[0] != "[" or \
            value[-1] != "]":
        raise AggregateError(f"{label} sample vector is malformed")
    if value == "[]":
        return []
    return [finite_positive(item, label) for item in value[1:-1].split(",")]


def resolve_file(root: Path, record: Any, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise AggregateError(f"{label} file record is malformed")
    raw = text(record["path"], f"{label}.path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise AggregateError(f"{label} path must be relative and normalized")
    candidate = root.joinpath(*pure.parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AggregateError(f"cannot resolve {label}: {exc}") from exc
    if candidate.is_symlink() or not candidate.is_file() or \
            (resolved != resolved_root and resolved_root not in resolved.parents):
        raise AggregateError(f"{label} must be a regular in-root file")
    expected = sha(record["sha256"], f"{label}.sha256")
    if file_sha(resolved) != expected:
        raise AggregateError(f"{label} SHA-256 differs")
    return resolved


def write_frozen(path: Path, payload: bytes) -> None:
    if not payload:
        raise AggregateError(f"refusing empty output {path}")
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise AggregateError(f"refusing to replace stale output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalized_workloads(plan: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    result: dict[tuple[int, str, str], dict[str, Any]] = {}
    cells = plan.get("cells")
    if isinstance(cells, list):
        for index, row in enumerate(cells):
            if not isinstance(row, dict):
                raise AggregateError(f"workload cell {index} is malformed")
            qtype = integer(row.get("qtype"), f"cell {index}.qtype")
            operator = text(row.get("operator"), f"cell {index}.operator")
            key = text(row.get("workload_key"), f"cell {index}.workload_key")
            public = row.get("public_problem")
            if not isinstance(public, dict):
                raise AggregateError(f"workload {key} has no public_problem")
            normalized = dict(public)
            normalized["workload_key"] = key
            normalized["source_class"] = text(
                row.get("source_class"), f"workload {key}.source_class")
            normalized["diagnostics"] = row.get("diagnostics", {})
            identity = (qtype, operator, key)
            if identity in result:
                raise AggregateError(f"duplicate workload identity {identity}")
            result[identity] = normalized
        return result

    # Legacy/shared operator plans remain useful for small conformance fixtures.
    for operator in sorted(worker_plan.OPERATORS):
        rows = plan.get(operator)
        if not isinstance(rows, list) or not rows:
            raise AggregateError(f"legacy plan {operator} rows are malformed")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise AggregateError(f"legacy {operator} row {index} is malformed")
            key = text(row.get("key"), f"legacy {operator} row {index}.key")
            normalized = dict(row)
            normalized["workload_key"] = key
            normalized.setdefault("source_class", "legacy-control")
            for qtype in sorted(worker_plan.QTYPES):
                result[(qtype, operator, key)] = normalized
    return result


def manifest_candidates(route: str, operator: str, manifest: dict[str, Any],
                        shard: dict[str, Any]) -> list[dict[str, Any]]:
    if route == "scalefirst":
        raw = manifest.get("compiled_parents")
        id_field = "parent_id"
    elif operator == "dense":
        raw = manifest.get("dense_tc_parents")
        id_field = "static_candidate_id"
    else:
        raw = manifest.get("grouped_parents")
        id_field = "static_candidate_id"
    if not isinstance(raw, list) or not raw:
        raise AggregateError(f"{shard['shard_key']} manifest candidate rows are empty")
    out: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise AggregateError(f"manifest candidate {index} is malformed")
        symbol = text(row.get("symbol"), f"manifest candidate {index}.symbol")
        static_id = row.get("static_candidate_id")
        parent_id = row.get(id_field)
        if isinstance(parent_id, bool) or not isinstance(parent_id, (int, str)):
            raise AggregateError(f"manifest candidate {index} parent ID is malformed")
        if not isinstance(static_id, str) or not static_id:
            raise AggregateError(f"manifest candidate {index} static ID is malformed")
        if symbol in seen_symbols:
            raise AggregateError(f"manifest contains duplicate symbol {symbol}")
        seen_symbols.add(symbol)
        out.append({"parent_id": parent_id, "static_candidate_id": static_id,
                    "symbol": symbol, "manifest_row": row})
    if [row["parent_id"] for row in out] != shard["parent_ids"]:
        raise AggregateError(f"{shard['shard_key']} manifest/composite parent IDs differ")
    return out


def runtime_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    runtime = row["runtime"]
    return tuple((key, runtime[key]) for key in sorted(runtime))


def check_state(state: str, raw_bad: int, samples: list[float], expected: int,
                label: str) -> str:
    if raw_bad != 0:
        raise AggregateError(f"{label} raw-bit mismatch: raw_bad={raw_bad}")
    if state == "MEASURED":
        if len(samples) != expected:
            raise AggregateError(
                f"{label} timing denominator differs: {len(samples)} != {expected}")
        return "MEASURED"
    if state not in SAFE_STRUCTURAL:
        raise AggregateError(f"{label} non-admissible runtime state {state}")
    if samples:
        raise AggregateError(f"{label} structural runtime carries timing samples")
    return "STRUCTURAL_UNAVAILABLE"


def _group_sf_dense_rows(rows: list[dict[str, Any]], expected_samples: int,
                         label: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        runtime = {
            "algorithm": text(row.get("algorithm"), f"{label}.algorithm"),
            "policy": text(row.get("policy"), f"{label}.policy"),
            "grid": integer(row.get("grid"), f"{label}.grid", minimum=0),
            "occupancy": integer(row.get("occupancy"), f"{label}.occupancy"),
            "split": integer(row.get("split"), f"{label}.split", minimum=0),
            "capacity_b_mask": text(row.get("capacity_b_mask"), f"{label}.capacity_b_mask"),
            "balanced_b_mask": text(row.get("balanced_b_mask"), f"{label}.balanced_b_mask"),
        }
        key = (text(row.get("symbol"), f"{label}.symbol"),
               tuple(sorted(runtime.items())))
        grouped.setdefault(key, []).append(row)
    out = []
    for (symbol, runtime_tuple), values in grouped.items():
        runtime = dict(runtime_tuple)
        status_values = {value.get("status") for value in values}
        reason_values = {value.get("reason") for value in values}
        raw_values = {integer(value.get("raw_bad"), f"{label}.raw_bad", minimum=0)
                      for value in values}
        if len(status_values) != 1 or len(reason_values) != 1 or len(raw_values) != 1:
            raise AggregateError(f"{label}/{symbol} repeated row state differs")
        status = text(next(iter(status_values)), f"{label}.status")
        reason = text(next(iter(reason_values)), f"{label}.reason")
        if status not in {"MEASURED", "INADMISSIBLE"}:
            raise AggregateError(f"{label}/{symbol} ScaleFirst status differs")
        state = "MEASURED" if status == "MEASURED" else reason
        raw_bad = next(iter(raw_values))
        if state == "MEASURED":
            samples_by_index: dict[int, float] = {}
            for value in values:
                index = integer(value.get("sample"), f"{label}.sample", minimum=0)
                if index in samples_by_index:
                    raise AggregateError(f"{label}/{symbol} duplicate sample index")
                samples_by_index[index] = finite_positive(
                    value.get("sample_us"), f"{label}.sample_us")
            if set(samples_by_index) != set(range(expected_samples)):
                raise AggregateError(f"{label}/{symbol} sample indices differ")
            samples = [samples_by_index[index] for index in range(expected_samples)]
        else:
            if len(values) != 1 or float(values[0].get("sample_us", 1)) != 0:
                raise AggregateError(f"{label}/{symbol} structural row denominator differs")
            samples = []
        classification = check_state(
            state, raw_bad, samples, expected_samples, f"{label}/{symbol}")
        out.append({"symbol": symbol, "runtime": runtime, "state": state,
                    "classification": classification, "raw_bad": raw_bad,
                    "samples": samples})
    return out


def _parse_kv_runtime(line: str, prefix: str, expected_samples: int,
                      label: str, runtime_fields: Iterable[str]) -> dict[str, Any]:
    row = kv_line(line, prefix)
    symbol = text(row.get("symbol"), f"{label}.symbol")
    runtime: dict[str, Any] = {}
    for field in runtime_fields:
        value = row.get(field)
        if field in {"grid", "occupancy", "S", "provider_capacity_rows",
                     "scalezero_fused"}:
            runtime[field] = integer(int(value) if isinstance(value, str) and
                                     re.fullmatch(r"-?[0-9]+", value) else value,
                                     f"{label}.{field}", minimum=0)
        else:
            runtime[field] = text(value, f"{label}.{field}")
    state = text(row.get("state"), f"{label}.state")
    raw_bad = integer(int(row.get("raw_bad", "-1")), f"{label}.raw_bad", minimum=0)
    samples = parse_samples(row.get("samples"), f"{label}.samples")
    classification = check_state(
        state, raw_bad, samples, expected_samples, f"{label}/{symbol}")
    return {"symbol": symbol, "runtime": runtime, "state": state,
            "classification": classification, "raw_bad": raw_bad,
            "samples": samples}


def int_field(row: dict[str, str], field: str, label: str,
              *, minimum: int | None = None) -> int:
    value = row.get(field)
    if not isinstance(value, str) or not re.fullmatch(r"-?[0-9]+", value):
        raise AggregateError(f"{label}.{field} must be a decimal integer")
    return integer(int(value), f"{label}.{field}", minimum=minimum)


def _header_u64(row: dict[str, str], field: str, label: str) -> int:
    value = row.get(field)
    try:
        parsed = int(value, 0) if value is not None else -1
    except ValueError as exc:
        raise AggregateError(f"{label}.{field} must be a u64") from exc
    if parsed < 0 or parsed >= 1 << 64:
        raise AggregateError(f"{label}.{field} must be a u64")
    return parsed


def one_kv(lines: list[str], prefix: str, label: str,
           predicate=lambda _row: True) -> dict[str, str]:
    rows = [kv_line(line, prefix) for line in lines if line.startswith(prefix)]
    rows = [row for row in rows if predicate(row)]
    if len(rows) != 1:
        raise AggregateError(f"{label} row denominator differs: {len(rows)} != 1")
    return rows[0]


def expected_shape(workload: dict[str, Any], label: str) -> tuple[int, int, int]:
    values = []
    for field in ("m", "n", "k"):
        values.append(integer(workload.get(field), f"{label}.{field}", minimum=1))
    return tuple(values)  # type: ignore[return-value]


def validate_candidate_runtime_denominator(
        route: str, operator: str, candidates: list[dict[str, Any]],
        records: list[dict[str, Any]], label: str) -> None:
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_symbol.setdefault(record["symbol"], []).append(record)
    expected_symbols = {row["symbol"] for row in candidates}
    if set(by_symbol) != expected_symbols:
        missing = sorted(expected_symbols - set(by_symbol))
        extra = sorted(set(by_symbol) - expected_symbols)
        raise AggregateError(
            f"{label} static candidate denominator differs: missing={missing[:3]} "
            f"extra={extra[:3]}")
    for candidate in candidates:
        symbol = candidate["symbol"]
        rows = by_symbol[symbol]
        identities = [runtime_identity(row) for row in rows]
        if len(identities) != len(set(identities)):
            raise AggregateError(f"{label}/{symbol} has duplicate runtime identities")
        algorithms = [row["runtime"]["algorithm"] for row in rows]
        if route == "scalefirst" and operator == "dense":
            if algorithms.count("NONPERSISTENT") != 1 or \
                    not algorithms.count("PERSISTENT") or \
                    set(algorithms) != {"NONPERSISTENT", "PERSISTENT"}:
                raise AggregateError(f"{label}/{symbol} ScaleFirst runtime denominator differs")
        elif route == "scalefirst":
            if algorithms.count("GROUPED_NONPERSISTENT") != 1 or \
                    not algorithms.count("GROUPED_PERSISTENT") or \
                    set(algorithms) != {
                        "GROUPED_NONPERSISTENT", "GROUPED_PERSISTENT"}:
                raise AggregateError(f"{label}/{symbol} grouped runtime denominator differs")
        elif operator == "dense":
            wanted = set(candidate["manifest_row"].get("runtime_variants", []))
            if wanted != {"TC_S1", "TC_S2", "TC_S4", "TC_S8"} or \
                    set(algorithms) != wanted or len(algorithms) != 4:
                raise AggregateError(f"{label}/{symbol} FQ split denominator differs")
        else:
            wanted = candidate["manifest_row"].get("algorithm")
            if not isinstance(wanted, str) or set(algorithms) != {wanted}:
                raise AggregateError(f"{label}/{symbol} FQ grouped algorithm differs")
            if wanted == "GROUPED_NONPERSISTENT" and len(rows) != 1:
                raise AggregateError(f"{label}/{symbol} nonpersistent runtime duplicated")


def parse_sf_dense(lines: list[str], workload: dict[str, Any],
                   candidates: list[dict[str, Any]], samples: int,
                   correctness_repeats: int, shard: dict[str, Any],
                   label: str, expected_seed: int) -> list[dict[str, Any]]:
    m, n, k = expected_shape(workload, label)
    shape = f"{m}x{n}x{k}"
    header = one_kv(lines, "SF_SHARD ", f"{label} SF_SHARD")
    if int_field(header, "qtype", label) != shard["qtype"] or \
            int_field(header, "weight_layout", label) != shard["layout"] or \
            header.get("weight_mapping_id") != shard["mapping_id"] or \
            int_field(header, "iterations", label) != samples or \
            int_field(header, "correctness_repeats", label) != correctness_repeats or \
            _header_u64(header, "schedule_seed", label) != expected_seed or \
            "warmups" in header:
        raise AggregateError(f"{label} ScaleFirst invocation controls differ")
    selected = int_field(header, "selected_rows", label, minimum=1)
    if selected != len(candidates):
        raise AggregateError(f"{label} selected parent denominator differs")
    raw_rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.startswith("SF_CELL "):
            continue
        try:
            row = json.loads(line[len("SF_CELL "):])
        except json.JSONDecodeError as exc:
            raise AggregateError(f"{label} malformed SF_CELL: {exc}") from exc
        if not isinstance(row, dict):
            raise AggregateError(f"{label} SF_CELL must be an object")
        if row.get("shape") == shape:
            if row.get("metric_scope") != "FULL_OUTPUT":
                raise AggregateError(f"{label} non-product ScaleFirst cell leaked")
            raw_rows.append(row)
    records = _group_sf_dense_rows(raw_rows, samples, label)
    completion = one_kv(
        lines, "SF_COMPLETE ", f"{label} SF_COMPLETE",
        lambda row: row.get("shape") == shape)
    if completion.get("status") != "COMPLETE" or \
            int_field(completion, "typed_rows", label) != len(candidates) or \
            int_field(completion, "runtime_cells", label) != len(records) or \
            int_field(completion, "iterations", label) != samples:
        raise AggregateError(f"{label} ScaleFirst completion denominator differs")
    if int_field(completion, "measured_cells", label) != sum(
            row["classification"] == "MEASURED" for row in records):
        raise AggregateError(f"{label} ScaleFirst measured-cell denominator differs")
    expected_records = sum(samples if row["classification"] == "MEASURED" else 1
                           for row in records)
    if int_field(completion, "records", label) != expected_records:
        raise AggregateError(f"{label} ScaleFirst record denominator differs")
    validate_candidate_runtime_denominator(
        "scalefirst", "dense", candidates, records, label)
    return records


def grouped_header(lines: list[str], prefix: str, workload: dict[str, Any],
                   candidates: list[dict[str, Any]], shard: dict[str, Any],
                   label: str, samples: int, correctness_repeats: int,
                   grouped_warmups: int, expected_seed: int) -> dict[str, str]:
    header = one_kv(lines, prefix, f"{label} grouped header")
    if int_field(header, "q", label) != shard["qtype"] or \
            int_field(header, "layout", label) != shard["layout"] or \
            header.get("mapping_id") != shard["mapping_id"] or \
            int_field(header, "selected_rows", label) != len(candidates) or \
            int_field(header, "iterations", label) != samples or \
            int_field(header, "warmups", label) != grouped_warmups or \
            int_field(header, "correctness_repeats", label) != correctness_repeats or \
            _header_u64(header, "schedule_seed", label) != expected_seed:
        raise AggregateError(f"{label} grouped shard identity differs")
    # Current grouped headers do not print N/K.  Those exact values are bound
    # and checked in the immutable argv evidence instead.
    for field in ("experts", "total_rows", "max_rows"):
        if int_field(header, field, label) != integer(
                workload.get(field), f"{label}.workload.{field}", minimum=1):
            raise AggregateError(f"{label} grouped workload {field} differs")
    if header.get("workload") != workload["workload_key"]:
        raise AggregateError(f"{label} grouped workload key differs")
    diagnostics = workload.get("diagnostics", {})
    expected_profile = diagnostics.get("profile", diagnostics.get("router"))
    if isinstance(expected_profile, str) and header.get("router_profile") != expected_profile:
        raise AggregateError(f"{label} grouped router profile differs")
    if isinstance(diagnostics.get("active"), int) and \
            int_field(header, "active", label) != diagnostics["active"]:
        raise AggregateError(f"{label} grouped active-expert count differs")
    if isinstance(diagnostics.get("zero"), int) and \
            int_field(header, "empty", label) != diagnostics["zero"]:
        raise AggregateError(f"{label} grouped empty-expert count differs")
    if isinstance(diagnostics.get("rows_hash"), str) and \
            header.get("rows_hash") != diagnostics["rows_hash"]:
        raise AggregateError(f"{label} grouped exact-row hash differs")
    return header


def parse_grouped(route: str, lines: list[str], workload: dict[str, Any],
                  candidates: list[dict[str, Any]], samples: int,
                  correctness_repeats: int, shard: dict[str, Any], label: str,
                  grouped_warmups: int, expected_seed: int) -> list[dict[str, Any]]:
    if route == "scalefirst":
        header_prefix = "SF_GROUPED_SHARD "
        cell_prefix = "SF_GROUPED_CELL "
        done_prefix = "SF_GROUPED_COMPLETE "
    else:
        header_prefix = "FQ_GROUPED_KPACK_SHARD "
        cell_prefix = "FQ_GROUPED_KPACK_CELL "
        done_prefix = "FQ_GROUPED_KPACK_COMPLETE "
    grouped_header(
        lines, header_prefix, workload, candidates, shard, label, samples,
        correctness_repeats, grouped_warmups, expected_seed)
    records = [_parse_kv_runtime(
        line, cell_prefix, samples, label,
        ("algorithm", "policy", "grid", "occupancy",
         "capacity_b_mask", "balanced_b_mask"))
        for line in lines if line.startswith(cell_prefix)]
    validate_candidate_runtime_denominator(
        route, "grouped", candidates, records, label)
    done = one_kv(lines, done_prefix, f"{label} grouped completion")
    measured = sum(row["classification"] == "MEASURED" for row in records)
    structural = len(records) - measured
    if done.get("status") != "PASS" or \
            int_field(done, "rows", label) != len(candidates) or \
            int_field(done, "cells", label) != len(records) or \
            int_field(done, "measured", label) != measured or \
            int_field(done, "structural", label) != structural:
        raise AggregateError(f"{label} grouped completion denominator differs")
    return records


def parse_fq_dense(lines: list[str], workload: dict[str, Any],
                   candidates: list[dict[str, Any]], samples: int,
                   correctness_repeats: int, shard: dict[str, Any],
                   label: str, expected_seed: int) -> list[dict[str, Any]]:
    m, n, k = expected_shape(workload, label)
    shape = f"{m}x{n}x{k}"
    header = one_kv(
        lines, "FQ_SHARD ", f"{label} FQ_SHARD",
        lambda row: row.get("shape") == shape)
    if int_field(header, "q", label) != shard["qtype"] or \
            int_field(header, "weight_layout", label) != shard["layout"] or \
            header.get("weight_mapping_id") != shard["mapping_id"] or \
            int_field(header, "typed_rows", label) != len(candidates) or \
            int_field(header, "selected_rows", label) != len(candidates) or \
            int_field(header, "iterations", label) != samples or \
            int_field(header, "correctness_repeats", label) != correctness_repeats or \
            header.get("bc_mode") != "skip" or \
            _header_u64(header, "schedule_seed", label) != expected_seed or \
            "warmups" in header:
        raise AggregateError(f"{label} FQ dense shard/invocation differs")
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line.startswith("FQ_TC_CELL "):
            continue
        raw = kv_line(line, "FQ_TC_CELL ")
        if raw.get("shape") != shape:
            continue
        record = _parse_kv_runtime(
            line, "FQ_TC_CELL ", samples, label,
            ("provider", "S", "provider_capacity_rows", "scalezero_fused"))
        split = record["runtime"].pop("S")
        record["runtime"]["split"] = split
        record["runtime"]["algorithm"] = f"TC_S{split}"
        scope = raw.get("scope")
        if (split == 1 and scope != "FULL_OUTPUT") or \
                (split > 1 and scope != "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS"):
            raise AggregateError(f"{label}/{record['symbol']} split scope differs")
        if record["classification"] == "MEASURED" and split > 1 and \
                raw.get("reducer_untimed") != "1":
            raise AggregateError(f"{label}/{record['symbol']} split reducer proof missing")
        records.append(record)
    validate_candidate_runtime_denominator(
        "fully-quantized", "dense", candidates, records, label)
    payload_kind = shard.get("payload_kind", worker_runner.DEVICE_KERNEL)
    if payload_kind == worker_runner.NO_DEVICE_KERNEL_STRUCTURAL:
        if (not records or len(records) != len(candidates) * 4 or
                any(record["state"] != "SHIPPING_SHARED_STORAGE" or
                    record["classification"] != "STRUCTURAL_UNAVAILABLE"
                    for record in records)):
            raise AggregateError(
                f"{label} structural no-kernel runtime census differs")
        by_symbol: dict[str, set[int]] = {}
        for record in records:
            by_symbol.setdefault(record["symbol"], set()).add(
                record["runtime"]["split"])
        if any(splits != {1, 2, 4, 8} for splits in by_symbol.values()):
            raise AggregateError(
                f"{label} structural no-kernel split census differs")
    elif payload_kind != worker_runner.DEVICE_KERNEL:
        raise AggregateError(f"{label} payload kind differs")
    done = one_kv(
        lines, "FQ_SHAPE_DONE ", f"{label} FQ_SHAPE_DONE",
        lambda row: row.get("shape") == shape)
    if done.get("status") != "PASS" or \
            int_field(done, "typed_rows", label) != len(candidates) or \
            int_field(done, "selected_rows", label) != len(candidates) or \
            int_field(done, "iterations", label) != samples:
        raise AggregateError(f"{label} FQ dense completion differs")
    return records


def parse_log(route: str, operator: str, path: Path,
              workload: dict[str, Any], candidates: list[dict[str, Any]],
              samples: int, correctness_repeats: int,
              shard: dict[str, Any], label: str, expected_seed: int,
              grouped_warmups: int) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AggregateError(f"cannot read {label}: {exc}") from exc
    if not lines:
        raise AggregateError(f"{label} is empty")
    if route == "scalefirst" and operator == "dense":
        return parse_sf_dense(
            lines, workload, candidates, samples, correctness_repeats,
            shard, label, expected_seed)
    if operator == "grouped":
        return parse_grouped(route, lines, workload, candidates,
                             samples, correctness_repeats, shard, label,
                             grouped_warmups, expected_seed)
    return parse_fq_dense(
        lines, workload, candidates, samples, correctness_repeats, shard, label,
        expected_seed)


def validate_run_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
            "schedule_seed", "grouped_warmups", "screen", "confirm",
            "all_admissible_candidates", "top_n"}:
        raise AggregateError("worker run_contract schema differs")
    if value["all_admissible_candidates"] is not True or value["top_n"] is not None:
        raise AggregateError("worker evidence permits candidate ranking/truncation")
    if value["schedule_seed"] != worker_runner.schedule_seed_contract():
        raise AggregateError("worker schedule-seed formula differs")
    integer(value["grouped_warmups"], "grouped warmups", minimum=1)
    screen = value["screen"]
    confirm = value["confirm"]
    if not isinstance(screen, dict) or set(screen) != {
            "timing_samples_per_runtime", "correctness_repeats"}:
        raise AggregateError("screen run contract differs")
    if not isinstance(confirm, dict) or set(confirm) != {
            "timing_samples_per_runtime", "correctness_repeats", "rounds"}:
        raise AggregateError("confirm run contract differs")
    integer(screen["timing_samples_per_runtime"], "screen samples", minimum=1)
    screen_correctness = integer(
        screen["correctness_repeats"], "screen correctness", minimum=1)
    if integer(confirm["timing_samples_per_runtime"], "confirm samples", minimum=1) != 11:
        raise AggregateError("final confirmation requires exactly 11 timing samples")
    if integer(confirm["correctness_repeats"], "confirm correctness",
               minimum=1) != screen_correctness:
        raise AggregateError("screen/confirm correctness denominator differs")
    rounds = confirm["rounds"]
    if not isinstance(rounds, list) or len(rounds) != 3:
        raise AggregateError("final confirmation requires exactly three rounds")
    normalized = []
    expected_rounds = ((1, "FORWARD"), (2, "REVERSE"), (3, "HASHED"))
    for index, row in enumerate(rounds, 1):
        if not isinstance(row, dict) or set(row) != {"round", "order"}:
            raise AggregateError("confirmation round schema differs")
        number = integer(row["round"], "confirmation round", minimum=1)
        order = text(row["order"], "confirmation order")
        if (number, order) != expected_rounds[index - 1]:
            raise AggregateError("confirmation round/order schedule differs")
        normalized.append((number, order))
    return value


def parse_seed(value: Any, label: str) -> int:
    if not isinstance(value, str):
        raise AggregateError(f"{label} must be a hexadecimal u64")
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise AggregateError(f"{label} must be a hexadecimal u64") from exc
    if parsed < 0 or parsed >= 1 << 64 or value != f"0x{parsed:016x}":
        raise AggregateError(f"{label} must be canonical hexadecimal u64")
    return parsed


def _pretty_json_sha(value: dict[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) +
               "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_runtime_linkage(value: Any, label: str) -> list[list[Any]]:
    if not isinstance(value, list) or not value:
        raise AggregateError(f"{label} runtime linkage is empty/malformed")
    normalized: list[list[Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 5:
            raise AggregateError(f"{label} runtime linkage row {index} differs")
        soname = text(row[0], f"{label}.runtime_linkage[{index}].soname")
        reported = text(row[1], f"{label}.runtime_linkage[{index}].reported")
        resolved = text(row[2], f"{label}.runtime_linkage[{index}].resolved")
        size = integer(row[3], f"{label}.runtime_linkage[{index}].size", minimum=1)
        checksum = sha(row[4], f"{label}.runtime_linkage[{index}].sha256")
        normalized.append([soname, reported, resolved, size, checksum])
    if normalized != sorted(normalized) or len({row[0] for row in normalized}) != len(normalized):
        raise AggregateError(f"{label} runtime linkage order/identity differs")
    return normalized


def validate_execution_authority(
        value: Any, *, worker_id: int, worker_count: int,
        bindings: dict[str, str], master: dict[str, Any],
        assignment: dict[str, Any], bundle: dict[str, Any],
        contract: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "worker_id", "worker_count", "bundle_sha256",
        "workload_plan_sha256", "master_sha256", "assignment_sha256",
        "selection_sha256", "device_identity_sha256",
        "device_homogeneity_sha256", "visible_device_ordinal",
        "workload_index_sha256", "executor_sha256", "work_items",
        "validated_shards", "validated_shard_keys_sha256",
        "validated_partition_artifacts", "candidate_policy", "measurement",
        "runtime_linkage",
    }
    if not isinstance(value, dict) or set(value) != required or \
            value.get("schema") != worker_runner.EXECUTION_SCHEMA:
        raise AggregateError(f"worker {worker_id} execution authority schema differs")
    if value.get("worker_id") != worker_id or value.get("worker_count") != worker_count:
        raise AggregateError(f"worker {worker_id} execution identity differs")
    for field in ("bundle_sha256", "workload_plan_sha256", "master_sha256",
                  "assignment_sha256", "device_homogeneity_sha256",
                  "workload_index_sha256"):
        if sha(value.get(field), f"execution.{field}") != bindings[field]:
            raise AggregateError(f"worker {worker_id} execution {field} differs")
    sha(value.get("device_identity_sha256"), "execution.device_identity_sha256")
    sha(value.get("executor_sha256"), "execution.executor_sha256")
    ordinal = text(value.get("visible_device_ordinal"), "visible_device_ordinal")
    if not ordinal.isdecimal():
        raise AggregateError(f"worker {worker_id} visible device ordinal differs")
    assigned = next(row for row in assignment["workers"]
                    if row["worker_id"] == worker_id)
    item_ids = assigned["work_item_ids"]
    item_id_set = set(item_ids)
    shard_keys = sorted({row["shard_key"] for row in master["work_items"]
                         if row["work_item_id"] in item_id_set})
    if value.get("work_items") != len(item_ids) or \
            value.get("validated_shards") != len(shard_keys) or \
            value.get("validated_shard_keys_sha256") != digest(shard_keys):
        raise AggregateError(f"worker {worker_id} execution denominator differs")
    selection = worker_plan.make_worker_selection(
        master, assignment, worker_id, master_sha256=bindings["master_sha256"],
        assignment_sha256=bindings["assignment_sha256"])
    if value.get("selection_sha256") != _pretty_json_sha(selection):
        raise AggregateError(f"worker {worker_id} selection authority differs")
    if value.get("candidate_policy") != \
            "SCREEN_ALL_COMPILED_CONFIRM_ALL_ADMISSIBLE_NO_TIMING_PRUNE":
        raise AggregateError(f"worker {worker_id} candidate policy differs")
    expected_measurement = {
        "screen_iterations": contract["screen"]["timing_samples_per_runtime"],
        "confirm_iterations": contract["confirm"]["timing_samples_per_runtime"],
        "confirm_rounds": len(contract["confirm"]["rounds"]),
        "correctness_repeats": contract["screen"]["correctness_repeats"],
        "grouped_warmups": contract["grouped_warmups"],
        "schedule_seed": contract["schedule_seed"],
        "outer_round_order": "ASSIGNMENT_REVERSE_THEN_HASHED_V1",
        "inner_candidate_order": "ROUND_SEED_VARIED_ALL_ROUTES_OPERATORS",
    }
    if contract["screen"]["correctness_repeats"] != \
            contract["confirm"]["correctness_repeats"] or \
            value.get("measurement") != expected_measurement:
        raise AggregateError(f"worker {worker_id} execution measurement differs")
    expected_artifacts = []
    if bundle.get("schema") == worker_plan.CATALOG_SCHEMA:
        by_artifact = {row["artifact_id"]: row for row in bundle["partitions"]}
        for artifact_id in sorted(assigned.get("artifact_ids", [])):
            record = by_artifact.get(artifact_id)
            if record is None:
                raise AggregateError(f"worker {worker_id} artifact identity differs")
            expected_artifacts.append({
                "artifact_id": artifact_id,
                "partition_manifest_sha256":
                    record["partition_manifest"]["sha256"],
            })
    if value.get("validated_partition_artifacts") != expected_artifacts:
        raise AggregateError(f"worker {worker_id} partition artifact authority differs")
    linkage = _validate_runtime_linkage(value.get("runtime_linkage"),
                                        f"worker {worker_id}")
    return {**value, "runtime_linkage": linkage,
            "runtime_linkage_sha256": digest(linkage)}


def validate_execution_inputs(value: Any, root: Path, *, shard: dict[str, Any],
                              workload: dict[str, Any], label: str
                              ) -> dict[str, Any]:
    base_required = {"artifact_id", "shard_key", "binary", "manifest",
                     "binary_receipt", "rows_file",
                     "retention_symbols_executed_path"}
    expected_kind = shard.get("payload_kind", worker_runner.DEVICE_KERNEL)
    required = (base_required | {"payload_kind", "structural_proof"}
                if expected_kind == worker_runner.NO_DEVICE_KERNEL_STRUCTURAL
                else base_required)
    if not isinstance(value, dict) or set(value) != required or \
            value.get("shard_key") != shard["shard_key"] or \
            value.get("artifact_id") != shard.get("artifact_id"):
        raise AggregateError(f"{label} execution input identity differs")
    expected_files = shard["files"]
    payload_kind = value.get("payload_kind", worker_runner.DEVICE_KERNEL)
    if payload_kind != expected_kind or payload_kind not in {
            worker_runner.DEVICE_KERNEL,
            worker_runner.NO_DEVICE_KERNEL_STRUCTURAL}:
        raise AggregateError(f"{label} payload kind differs")
    normalized: dict[str, Any] = {
        "artifact_id": value["artifact_id"], "shard_key": value["shard_key"],
        "payload_kind": payload_kind}
    for field in ("binary", "binary_receipt"):
        record = value.get(field)
        keys = {"executed_path", "size", "sha256"} if field == "binary" else {
            "size", "sha256"}
        if not isinstance(record, dict) or set(record) != keys:
            raise AggregateError(f"{label} {field} execution metadata differs")
        if field == "binary":
            executed = text(record["executed_path"], f"{label}.binary.executed_path")
            if not Path(executed).is_absolute():
                raise AggregateError(f"{label} binary execution path is not absolute")
        size = integer(record["size"], f"{label}.{field}.size", minimum=1)
        checksum = sha(record["sha256"], f"{label}.{field}.sha256")
        expected = expected_files[field]
        if checksum != expected["sha256"] or \
                ("size" in expected and size != expected["size"]):
            raise AggregateError(f"{label} {field} catalog hash/size differs")
        normalized[field] = dict(record)
    manifest = value.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != {
            "size", "sha256", "snapshot"}:
        raise AggregateError(f"{label} manifest execution metadata differs")
    manifest_path = resolve_file(root, manifest["snapshot"],
                                 f"{label} manifest snapshot")
    manifest_size = integer(manifest["size"], f"{label}.manifest.size", minimum=1)
    manifest_sha = sha(manifest["sha256"], f"{label}.manifest.sha256")
    expected_manifest = expected_files["manifest"]
    if (manifest_path.stat().st_size != manifest_size or
            file_sha(manifest_path) != manifest_sha or
            manifest_sha != expected_manifest["sha256"] or
            ("size" in expected_manifest and
             manifest_size != expected_manifest["size"])):
        raise AggregateError(f"{label} manifest catalog/snapshot differs")
    normalized["manifest"] = {**manifest, "snapshot_path": manifest_path}
    proof = value.get("structural_proof")
    if payload_kind == worker_runner.NO_DEVICE_KERNEL_STRUCTURAL:
        if (not isinstance(proof, dict) or set(proof) != {
                "size", "sha256", "snapshot"} or
                "structural_proof" not in expected_files):
            raise AggregateError(f"{label} structural proof metadata differs")
        proof_path = resolve_file(
            root, proof["snapshot"], f"{label} structural proof snapshot")
        proof_size = integer(
            proof["size"], f"{label}.structural_proof.size", minimum=1)
        proof_sha = sha(
            proof["sha256"], f"{label}.structural_proof.sha256")
        expected_proof = expected_files["structural_proof"]
        if (proof_path.stat().st_size != proof_size or
                file_sha(proof_path) != proof_sha or
                proof_sha != expected_proof["sha256"] or
                ("size" in expected_proof and
                 proof_size != expected_proof["size"])):
            raise AggregateError(f"{label} structural proof snapshot differs")
        try:
            proof_doc = load_json(proof_path, f"{label} structural proof")
            native = {
                "shard_key": shard.get("native_shard_key", shard["shard_key"]),
                "qtype": shard["qtype"], "operator": shard["operator"],
                "route": shard["route"],
                "parent_begin": shard["parent_begin"],
                "parent_end": shard["parent_end"],
                "parent_count": shard["parent_count"],
                "authority_count": shard["authority_count"],
                "parent_ids": shard["parent_ids"],
            }
            fq_structural.validate_structural_proof(
                proof_doc, native, manifest_sha,
                value["binary"]["sha256"])
            manifest_doc = load_json(
                manifest_path, f"{label} structural manifest")
            manifest_symbols = [row.get("symbol") for row in
                                manifest_doc.get("dense_tc_parents", [])]
            proof_symbols = [row.get("symbol") for row in proof_doc["rows"]]
            if proof_symbols != manifest_symbols:
                raise AggregateError(
                    f"{label} structural proof/manifest symbols differ")
        except (KeyError, TypeError, ValueError) as error:
            raise AggregateError(
                f"{label} structural proof semantics differ: {error}") from error
        normalized["structural_proof"] = {
            **proof, "snapshot_path": proof_path}
    elif proof is not None or "structural_proof" in expected_files:
        raise AggregateError(f"{label} unexpected structural proof")
    else:
        normalized["structural_proof"] = None
    rows = value.get("rows_file")
    source_class = workload.get("source_class")
    if source_class == "router-control":
        if not isinstance(rows, dict) or set(rows) != {"executed_path", "file"}:
            raise AggregateError(f"{label} exact rows execution metadata differs")
        rows_path = resolve_file(root, rows["file"], f"{label} exact rows snapshot")
        expected_sha = workload.get("diagnostics", {}).get("rows_sha256")
        if not isinstance(expected_sha, str) or file_sha(rows_path) != expected_sha:
            raise AggregateError(f"{label} exact rows hash differs")
        normalized["rows_file"] = {
            **rows, "executed_path": text(
                rows["executed_path"], f"{label}.rows_file.executed_path"),
            "snapshot_path": rows_path,
        }
    elif rows is not None:
        raise AggregateError(f"{label} unexpected exact rows execution metadata")
    else:
        normalized["rows_file"] = None
    retention_path = value.get("retention_symbols_executed_path")
    if retention_path is not None:
        retention_path = text(retention_path, f"{label}.retention executed path")
    normalized["retention_symbols_executed_path"] = retention_path
    return normalized


def argv_options(argv: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in argv[1:]:
        if not token.startswith("--") or "=" not in token:
            raise AggregateError(f"{label} argv contains a non-option token")
        name, value = token[2:].split("=", 1)
        if not name or not value or name in result:
            raise AggregateError(f"{label} argv option set is not exact")
        result[name] = value
    return result


def validate_argv(argv: Any, expected_sha: Any, *, item: dict[str, Any],
                  workload: dict[str, Any], samples: int,
                  correctness_repeats: int, screen: bool, expected_seed: int,
                  grouped_warmups: int,
                  execution_inputs: dict[str, Any]) -> list[str]:
    if not isinstance(argv, list) or len(argv) < 2 or any(
            not isinstance(value, str) or not value or "\0" in value
            for value in argv):
        raise AggregateError(f"{item['work_item_id']} argv is malformed")
    if digest(argv) != sha(expected_sha, "argv_sha256"):
        raise AggregateError(f"{item['work_item_id']} argv digest differs")
    binary = execution_inputs["binary"]
    if argv[0] != binary["executed_path"]:
        raise AggregateError(f"{item['work_item_id']} binary argv differs")
    options = argv_options(argv, item["work_item_id"])
    expected = [binary["executed_path"]]
    if item["operator"] == "dense":
        shape = "x".join(str(value) for value in expected_shape(
            workload, item["work_item_id"]))
        expected += [f"--shape={shape}", f"--iterations={samples}",
                     f"--correctness-repeats={correctness_repeats}"]
        expected += (["--algorithm=full-output"] if item["route"] == "scalefirst"
                     else ["--bc-mode=skip"])
        expected.append(f"--schedule-seed={expected_seed}")
    else:
        numeric = {field: integer(
            workload.get(field), f"{item['work_item_id']}.{field}", minimum=1)
                   for field in ("experts", "n", "k")}
        source_class = workload.get("source_class")
        diagnostics = workload.get("diagnostics", {})
        expected_profile = diagnostics.get(
            "profile", diagnostics.get("router", workload.get("profile")))
        expected_profile = text(
            expected_profile, f"{item['work_item_id']}.router-profile")
        expected += [f"--experts={numeric['experts']}", f"--n={numeric['n']}",
                     f"--k={numeric['k']}",
                     f"--workload-key={item['workload_key']}",
                     f"--router-profile={expected_profile}",
                     f"--iterations={samples}", f"--warmups={grouped_warmups}",
                     f"--correctness-repeats={correctness_repeats}",
                     f"--schedule-seed={expected_seed}"]
        if source_class == "real-inventory":
            for field in ("tokens", "topk"):
                integer(
                    diagnostics.get(field), f"{item['work_item_id']}.{field}",
                    minimum=1)
        elif source_class == "router-control":
            rows_file = execution_inputs["rows_file"]
            if rows_file is None:
                raise AggregateError(
                    f"{item['work_item_id']} router-control rows authority differs")
        elif source_class == "legacy-control":
            for field in ("tokens", "topk"):
                integer(workload.get(field), f"{item['work_item_id']}.{field}",
                        minimum=1)
        else:
            raise AggregateError(
                f"{item['work_item_id']} grouped source class differs")
    retention_path = execution_inputs["retention_symbols_executed_path"]
    if not screen and retention_path is not None:
        expected.append(f"--symbol-file={retention_path}")
    if item["operator"] == "grouped":
        source_class = workload["source_class"]
        if source_class == "router-control":
            expected.append(
                f"--rows-file={execution_inputs['rows_file']['executed_path']}")
        else:
            diagnostics = workload.get("diagnostics", {})
            tokens = diagnostics.get("tokens", workload.get("tokens"))
            topk = diagnostics.get("topk", workload.get("topk"))
            work_label = item["work_item_id"]
            expected += [
                f"--tokens={integer(tokens, f'{work_label}.tokens', minimum=1)}",
                f"--topk={integer(topk, f'{work_label}.topk', minimum=1)}",
            ]
    if argv != expected or len(options) != len(argv) - 1:
        raise AggregateError(f"{item['work_item_id']} exact argv options differ")
    return argv


def validate_run_record(value: Any, root: Path, *, item: dict[str, Any],
                        workload: dict[str, Any], samples: int,
                        correctness_repeats: int, screen: bool,
                        round_index: int, order: str, worker_id: int,
                        grouped_warmups: int,
                        execution_inputs: dict[str, Any],
                        ) -> dict[str, Any]:
    keys = {"schedule_seed", "argv", "argv_sha256", "log"} if screen else {
        "round", "order", "schedule_seed", "argv", "argv_sha256", "log"}
    if not isinstance(value, dict) or set(value) != keys:
        raise AggregateError(f"{item['work_item_id']} run record schema differs")
    if not screen and (value["round"], value["order"]) != (round_index, order):
        raise AggregateError(f"{item['work_item_id']} confirmation round differs")
    phase = "screen" if screen else "confirm"
    expected_seed = worker_runner.schedule_seed(
        item["work_item_id"], phase, round_index)
    if parse_seed(value["schedule_seed"], "run schedule_seed") != expected_seed:
        raise AggregateError(f"{item['work_item_id']} schedule seed differs")
    argv = validate_argv(
        value["argv"], value["argv_sha256"], item=item, workload=workload,
        samples=samples, correctness_repeats=correctness_repeats,
        screen=screen, expected_seed=expected_seed,
        grouped_warmups=grouped_warmups,
        execution_inputs=execution_inputs)
    log = resolve_file(root, value["log"], f"{item['work_item_id']} log")
    lines = log.read_text(encoding="utf-8").splitlines()
    atom = one_kv(lines, "KPACK_DISCOVERY_ATOM ",
                  f"{item['work_item_id']} atom")
    expected_atom = {
        "work_item_id": item["work_item_id"], "phase": phase,
        "round": str(round_index), "order": order, "worker": str(worker_id),
        "schedule_seed": worker_runner.schedule_seed_hex(expected_seed),
        "grouped_warmups": (str(grouped_warmups)
                            if item["operator"] == "grouped" else "NONE"),
        "argv_sha256": digest(argv),
    }
    if not lines or not lines[0].startswith("KPACK_DISCOVERY_ATOM ") or \
            atom != expected_atom:
        raise AggregateError(f"{item['work_item_id']} atom authority differs")
    return {"log": log, "log_sha256": file_sha(log),
            "argv_sha256": digest(argv), "schedule_seed": expected_seed,
            **({} if screen else {"round": round_index, "order": order})}


def validate_empty_confirm(value: Any, root: Path, *, item: dict[str, Any],
                           round_row: dict[str, Any], screen_sha256: str,
                           retention_sha256: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
            "round", "order", "schedule_seed", "empty_structural", "log"} or \
            value.get("empty_structural") is not True or \
            (value.get("round"), value.get("order")) != \
            (round_row["round"], round_row["order"]):
        raise AggregateError(f"{item['work_item_id']} empty-confirm schema differs")
    log = resolve_file(root, value["log"], f"{item['work_item_id']} empty confirm")
    lines = log.read_text(encoding="utf-8").splitlines()
    marker = one_kv(lines, "KPACK_DISCOVERY_EMPTY_CONFIRM ",
                    f"{item['work_item_id']} empty confirm")
    seed = worker_runner.schedule_seed(
        item["work_item_id"], "confirm", round_row["round"])
    if parse_seed(value["schedule_seed"], "empty-confirm schedule_seed") != seed:
        raise AggregateError(f"{item['work_item_id']} empty-confirm seed differs")
    exact = {
        "work_item_id": item["work_item_id"],
        "round": str(round_row["round"]), "order": round_row["order"],
        "schedule_seed": worker_runner.schedule_seed_hex(seed),
        "screen_sha256": screen_sha256,
        "retention_symbols_sha256": retention_sha256,
    }
    if marker != exact or len(lines) != 1:
        raise AggregateError(f"{item['work_item_id']} empty-confirm marker differs")
    return {"round": value["round"], "order": value["order"],
            "schedule_seed": seed,
            "empty_structural": True, "log": log,
            "log_sha256": file_sha(log)}


def workload_index_authority(plan_path: Path, index_path: Path) -> dict[str, Any]:
    index = load_json(index_path, "workload index")
    if index.get("schema") == "quactlize.kpack-discovery-workloads.v1":
        try:
            import materialize_kpack_discovery_workloads as materializer
            return materializer.validate(plan_path, index_path.parent)
        except (OSError, ValueError, KeyError) as exc:
            raise AggregateError(f"canonical workload index differs: {exc}") from exc
    if index.get("schema") != "quactlize.kpack-discovery-workloads-test.v1" or \
            index.get("plan_file_sha256") != file_sha(plan_path):
        raise AggregateError("workload index schema/plan binding differs")
    return index


def load_evidence(
        evidence_paths: list[Path], *, bundle_path: Path, plan_path: Path,
        master_path: Path, assignment_path: Path, workload_index_path: Path,
        device_path: Path) -> tuple[dict[str, Any], dict[str, Any],
                                    dict[str, Any], dict[str, Any],
                                    dict[str, Any], dict[str, Any],
                                    dict[str, dict[str, Any]], list[dict[str, Any]],
                                    dict[str, Any]]:
    try:
        bundle = worker_plan.load_bundle_authority(bundle_path)
        master = worker_plan.load_json(master_path, "master ledger")
        worker_plan.validate_master(master, bundle_path, plan_path)
        assignment = worker_plan.load_json(assignment_path, "assignment")
        worker_plan.validate_assignment(assignment, master, file_sha(master_path))
        device = worker_plan.load_json(device_path, "device authority")
        identities = worker_plan.validate_device_authority(
            device, assignment["worker_count"])
    except (OSError, ValueError, KeyError) as exc:
        raise AggregateError(f"worker authority differs: {exc}") from exc
    plan = load_json(plan_path, "workload plan")
    workload_index = workload_index_authority(plan_path, workload_index_path)
    workloads = normalized_workloads(plan)
    if len(evidence_paths) != assignment["worker_count"]:
        raise AggregateError("worker evidence denominator differs")

    expected_by_worker = {
        row["worker_id"]: row["work_item_ids"] for row in assignment["workers"]}
    item_by_id = {row["work_item_id"]: row for row in master["work_items"]}
    shard_by_key = {row["shard_key"]: row for row in bundle["shards"]}
    receipt_workers: set[int] = set()
    completion_results: list[dict[str, Any]] = []
    item_specs: dict[str, dict[str, Any]] = {}
    common_contract: dict[str, Any] | None = None
    evidence_records = []
    bindings = {
        "bundle_sha256": file_sha(bundle_path),
        "workload_plan_sha256": file_sha(plan_path),
        "workload_index_sha256": file_sha(workload_index_path),
        "master_sha256": file_sha(master_path),
        "assignment_sha256": file_sha(assignment_path),
        "device_homogeneity_sha256": file_sha(device_path),
    }
    for evidence_path in evidence_paths:
        document = load_json(evidence_path, "worker evidence")
        required = {
            "schema", "worker_id", "worker_count", *bindings,
            "device_identity_sha256", "execution_authority",
            "completion_result",
            "result_authorities", "run_contract", "work_items"}
        if set(document) != required or document.get("schema") != EVIDENCE_SCHEMA:
            raise AggregateError("worker evidence schema differs")
        worker = integer(document["worker_id"], "evidence worker_id", minimum=0)
        if worker not in expected_by_worker or worker in receipt_workers or \
                document["worker_count"] != assignment["worker_count"]:
            raise AggregateError("worker evidence identity is missing/duplicated")
        for field, expected in bindings.items():
            if sha(document.get(field), f"worker {worker}.{field}") != expected:
                raise AggregateError(f"worker {worker} has stale {field}")
        if sha(document.get("device_identity_sha256"),
               f"worker {worker}.device_identity") != identities[worker]:
            raise AggregateError(f"worker {worker} device identity differs")
        contract = validate_run_contract(document["run_contract"])
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise AggregateError("worker measurement contracts differ")
        root = evidence_path.parent.resolve(strict=True)
        execution_path = resolve_file(
            root, document["execution_authority"],
            f"worker {worker} execution authority")
        execution = validate_execution_authority(
            load_json(execution_path, f"worker {worker} execution authority"),
            worker_id=worker, worker_count=assignment["worker_count"],
            bindings=bindings, master=master, assignment=assignment,
            bundle=bundle, contract=contract)
        if execution["device_identity_sha256"] != \
                document["device_identity_sha256"]:
            raise AggregateError(f"worker {worker} execution device differs")
        execution_sha = file_sha(execution_path)
        completion_path = resolve_file(
            root, document["completion_result"], f"worker {worker} completion")
        completion_results.append(load_json(completion_path, "worker completion"))
        authorities = document["result_authorities"]
        assigned_routes = {item_by_id[item]["route"]
                           for item in expected_by_worker[worker]}
        if not isinstance(authorities, list) or len(authorities) != len(assigned_routes):
            raise AggregateError(f"worker {worker} result-authority denominator differs")
        authority_by_route: dict[str, str] = {}
        authority_path_by_route: dict[str, Path] = {}
        for index, record in enumerate(authorities):
            if not isinstance(record, dict) or set(record) != {"route", "path", "sha256"}:
                raise AggregateError(f"worker {worker} result authority {index} is malformed")
            route = text(record["route"], "result authority route")
            path = resolve_file(root, {"path": record["path"], "sha256": record["sha256"]},
                                f"worker {worker} {route} result authority")
            authority = load_json(path, f"worker {worker} {route} result authority")
            expected_authority = {
                "schema": "quactlize.kpack-discovery-worker-route-result-authority.v2",
                "route": route, "worker_id": worker,
                "execution_authority_sha256": execution_sha,
                "candidate_policy": "ALL_ADMISSIBLE_NO_TIMING_PRUNE",
                "work_item_ids": [item_id for item_id in expected_by_worker[worker]
                                  if item_by_id[item_id]["route"] == route],
                "screen_iterations": contract["screen"]["timing_samples_per_runtime"],
                "confirm_iterations": contract["confirm"]["timing_samples_per_runtime"],
                "confirm_rounds": len(contract["confirm"]["rounds"]),
                "correctness_repeats": contract["screen"]["correctness_repeats"],
                "grouped_warmups": contract["grouped_warmups"],
                "schedule_seed": contract["schedule_seed"],
            }
            if (contract["screen"]["correctness_repeats"] !=
                    contract["confirm"]["correctness_repeats"] or
                    authority != expected_authority):
                raise AggregateError(
                    f"worker {worker} {route} result authority differs")
            if route in authority_by_route:
                raise AggregateError(f"worker {worker} duplicated result authority")
            authority_by_route[route] = file_sha(path)
            authority_path_by_route[route] = path
        if set(authority_by_route) != assigned_routes:
            raise AggregateError(f"worker {worker} result-authority routes differ")
        raw_items = document["work_items"]
        if not isinstance(raw_items, list) or [row.get("work_item_id")
                if isinstance(row, dict) else None for row in raw_items] != \
                expected_by_worker[worker]:
            raise AggregateError(f"worker {worker} work-item evidence order differs")
        for raw in raw_items:
            if not isinstance(raw, dict) or set(raw) != {
                    "work_item_id", "result_authority_sha256",
                    "execution_inputs", "screen", "retention", "confirm"}:
                raise AggregateError(f"worker {worker} work-item schema differs")
            item_id = raw["work_item_id"]
            item = item_by_id[item_id]
            workload = workloads.get((item["qtype"], item["operator"],
                                      item["workload_key"]))
            if workload is None:
                raise AggregateError(f"{item_id} workload authority is missing")
            if sha(raw["result_authority_sha256"], "result_authority_sha256") != \
                    authority_by_route[item["route"]]:
                raise AggregateError(f"{item_id} result authority differs")
            shard = shard_by_key.get(item["shard_key"])
            if shard is None:
                raise AggregateError(f"{item_id} shard authority is missing")
            execution_inputs = validate_execution_inputs(
                raw["execution_inputs"], root, shard=shard, workload=workload,
                label=item_id)
            retention_raw = raw["retention"]
            has_retention = retention_raw is not None
            if item["route"] == "scalefirst" and not has_retention:
                raise AggregateError(f"{item_id} ScaleFirst retention is missing")
            if item["route"] == "fully-quantized" and has_retention:
                raise AggregateError(f"{item_id} unexpected FQ retention")
            if has_retention != (execution_inputs[
                    "retention_symbols_executed_path"] is not None):
                raise AggregateError(f"{item_id} retention execution path differs")
            retention = None
            if has_retention:
                if not isinstance(retention_raw, dict) or set(retention_raw) != {
                        "symbols", "sidecar"}:
                    raise AggregateError(f"{item_id} retention schema differs")
                retention = {
                    "symbols": resolve_file(root, retention_raw["symbols"],
                                            f"{item_id} retained symbols"),
                    "sidecar": resolve_file(root, retention_raw["sidecar"],
                                            f"{item_id} retention sidecar"),
                }
            screen_contract = contract["screen"]
            screen_run = validate_run_record(
                raw["screen"], root, item=item, workload=workload,
                samples=screen_contract["timing_samples_per_runtime"],
                correctness_repeats=screen_contract["correctness_repeats"],
                screen=True, round_index=0,
                order="SCREEN", worker_id=worker,
                grouped_warmups=contract["grouped_warmups"],
                execution_inputs=execution_inputs)
            confirm_raw = raw["confirm"]
            rounds = contract["confirm"]["rounds"]
            empty_sf_retention = bool(
                has_retention and retention is not None and
                retention["symbols"].stat().st_size == 0)
            if not isinstance(confirm_raw, list) or len(confirm_raw) != len(rounds):
                raise AggregateError(f"{item_id} confirmation run denominator differs")
            if empty_sf_retention:
                confirm_runs = [validate_empty_confirm(
                    run, root, item=item, round_row=round_row,
                    screen_sha256=screen_run["log_sha256"],
                    retention_sha256=file_sha(retention["symbols"]))
                    for run, round_row in zip(confirm_raw, rounds)]
            else:
                confirm_runs = [validate_run_record(
                    run, root, item=item, workload=workload,
                    samples=contract["confirm"]["timing_samples_per_runtime"],
                    correctness_repeats=contract["confirm"]["correctness_repeats"],
                    screen=False,
                    round_index=round_row["round"], order=round_row["order"],
                    worker_id=worker,
                    grouped_warmups=contract["grouped_warmups"],
                    execution_inputs=execution_inputs)
                    for run, round_row in zip(confirm_raw, rounds)]
            item_specs[item_id] = {
                "worker_id": worker, "evidence_sha256": file_sha(evidence_path),
                "result_authority_sha256": authority_by_route[item["route"]],
                "execution_authority_sha256": execution_sha,
                "device_identity_sha256": document["device_identity_sha256"],
                "device_homogeneity_sha256": bindings["device_homogeneity_sha256"],
                "runtime_linkage": execution["runtime_linkage"],
                "runtime_linkage_sha256": execution["runtime_linkage_sha256"],
                "executor_sha256": execution["executor_sha256"],
                "execution_inputs": execution_inputs,
                "workload": workload, "screen": screen_run,
                "retention": retention, "confirm": confirm_runs,
            }
        receipt_workers.add(worker)
        shard_authorities: dict[str, dict[str, Any]] = {}
        for item_id in expected_by_worker[worker]:
            item = item_by_id[item_id]
            inputs = item_specs[item_id]["execution_inputs"]
            record = {
                "shard_key": item["shard_key"],
                "artifact_id": inputs["artifact_id"],
                "payload_kind": inputs["payload_kind"],
                "binary_sha256": inputs["binary"]["sha256"],
                "manifest_sha256": inputs["manifest"]["sha256"],
                "binary_receipt_sha256": inputs["binary_receipt"]["sha256"],
                "structural_proof_sha256": (
                    None if inputs["structural_proof"] is None else
                    inputs["structural_proof"]["sha256"]),
            }
            previous = shard_authorities.setdefault(item["shard_key"], record)
            if previous != record:
                raise AggregateError(
                    f"worker {worker} shard execution authority differs")
        evidence_records.append({
            "worker_id": worker,
            "device_identity_sha256": document["device_identity_sha256"],
            "evidence_sha256": file_sha(evidence_path),
            "completion_result_sha256": file_sha(completion_path),
            "execution_authority_sha256": execution_sha,
            "executor_sha256": execution["executor_sha256"],
            "runtime_linkage_sha256": execution["runtime_linkage_sha256"],
            "shards": [shard_authorities[key]
                       for key in sorted(shard_authorities)],
            "result_authorities": [
                {"route": route, "sha256": authority_by_route[route]}
                for route in sorted(authority_by_route)],
            "_source_paths": {
                "evidence": evidence_path,
                "execution": execution_path,
                "completion": completion_path,
                "result_authorities": authority_path_by_route,
            },
        })
    if receipt_workers != set(range(assignment["worker_count"])) or \
            set(item_specs) != set(item_by_id):
        raise AggregateError("worker evidence union differs from master")
    try:
        worker_plan.validate_results(
            completion_results, assignment,
            master_sha256=file_sha(master_path),
            assignment_sha256=file_sha(assignment_path),
            device_authority_sha256=file_sha(device_path),
            device_identities=identities)
    except (ValueError, KeyError) as exc:
        raise AggregateError(f"completion result union differs: {exc}") from exc
    assert common_contract is not None
    return (bundle, plan, master, assignment, device, workload_index,
            item_specs, sorted(evidence_records, key=lambda row: row["worker_id"]),
            common_contract)


def retained_symbols(item: dict[str, Any], spec: dict[str, Any],
                     manifest_sha256: str,
                     screen_records: list[dict[str, Any]]) -> set[str]:
    measured = {row["symbol"] for row in screen_records
                if row["classification"] == "MEASURED"}
    retention = spec["retention"]
    if retention is None:
        return {row["symbol"] for row in screen_records}
    try:
        symbols = retention["symbols"].read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AggregateError(f"{item['work_item_id']} cannot read retention: {exc}") from exc
    if any(not symbol or symbol.strip() != symbol for symbol in symbols) or \
            len(symbols) != len(set(symbols)) or set(symbols) != measured:
        raise AggregateError(
            f"{item['work_item_id']} retention is not the full measured symbol set")
    sidecar = load_json(retention["sidecar"], "retention sidecar")
    if sidecar.get("manifest_sha256") != manifest_sha256 or \
            sidecar.get("screen_sha256") != spec["screen"]["log_sha256"] or \
            sidecar.get("retained_symbols") != symbols or \
            sidecar.get("retained_count") != len(symbols) or \
            sidecar.get("timing_rank_used_for_elimination") is not False:
        raise AggregateError(f"{item['work_item_id']} retention sidecar differs")
    if sidecar.get("operator") != item["operator"] or \
            sidecar.get("qtype") != item["qtype"]:
        raise AggregateError(f"{item['work_item_id']} retention identity differs")
    return set(symbols)


def record_map(records: list[dict[str, Any]], label: str
               ) -> dict[tuple[str, tuple[Any, ...]], dict[str, Any]]:
    result: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
    for row in records:
        key = (row["symbol"], runtime_identity(row))
        if key in result:
            raise AggregateError(f"{label} duplicate normalized runtime")
        result[key] = row
    return result


def stable_static_row(candidate: dict[str, Any]) -> dict[str, Any]:
    source = candidate["manifest_row"]
    fields = (
        "tile_m", "tile_n", "tactic_tile_k", "warp_m", "warp_n",
        "stages", "a_provider", "a_provider_name", "resolved_delivery_n",
        "algorithm", "persistent", "bchunk", "weight_layout")
    return {field: source[field] for field in fields if field in source}


def snapshot_worker_authorities(root: Path,
                                records: list[dict[str, Any]]
                                ) -> list[dict[str, Any]]:
    """Copy small worker authorities into the aggregate by content identity."""
    output: list[dict[str, Any]] = []
    for raw in records:
        sources = raw.get("_source_paths")
        if not isinstance(sources, dict):
            raise AggregateError("worker authority source paths are missing")
        worker = raw["worker_id"]
        prefix = f"authorities/worker-{worker:04d}-{raw['evidence_sha256'][:16]}"

        def copy_one(name: str, source: Path) -> dict[str, str]:
            destination = root / prefix / name
            try:
                payload = source.read_bytes()
            except OSError as exc:
                raise AggregateError(
                    f"cannot snapshot worker {worker} {name}: {exc}") from exc
            write_frozen(destination, payload)
            return {"path": destination.relative_to(root).as_posix(),
                    "sha256": file_sha(destination)}

        routes = []
        route_sources = sources.get("result_authorities")
        if not isinstance(route_sources, dict):
            raise AggregateError("worker result-authority sources differ")
        for row in raw["result_authorities"]:
            route = row["route"]
            file_record = copy_one(
                f"result-authority-{route}.json", route_sources[route])
            if file_record["sha256"] != row["sha256"]:
                raise AggregateError("worker result-authority snapshot differs")
            routes.append({"route": route, "file": file_record})
        evidence = copy_one("worker-evidence.json", sources["evidence"])
        execution = copy_one("execution-authority.json", sources["execution"])
        completion = copy_one("completion-result.json", sources["completion"])
        if (evidence["sha256"] != raw["evidence_sha256"] or
                execution["sha256"] != raw["execution_authority_sha256"] or
                completion["sha256"] != raw["completion_result_sha256"]):
            raise AggregateError("worker authority snapshot hash differs")
        output.append({
            "worker_id": worker,
            "device_identity_sha256": raw["device_identity_sha256"],
            "evidence": evidence,
            "execution_authority": execution,
            "completion_result": completion,
            "executor_sha256": raw["executor_sha256"],
            "runtime_linkage_sha256": raw["runtime_linkage_sha256"],
            "shards": raw["shards"],
            "result_authorities": routes,
        })
    return output


def aggregate(*, bundle_path: Path, plan_path: Path, master_path: Path,
              assignment_path: Path, workload_index_path: Path,
              device_path: Path, evidence_paths: list[Path],
              output_dir: Path) -> dict[str, Any]:
    (bundle, _plan, master, assignment, device, workload_index, item_specs,
     evidence_records, contract) = load_evidence(
         evidence_paths, bundle_path=bundle_path, plan_path=plan_path,
         master_path=master_path, assignment_path=assignment_path,
         workload_index_path=workload_index_path, device_path=device_path)
    shards = {row["shard_key"]: row for row in bundle["shards"]}
    if not {row["shard_key"] for row in master["work_items"]}.issubset(shards):
        raise AggregateError("master/bundle shard-key union differs")
    manifest_cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    counters = {
        "work_items": 0, "static_candidate_instances": 0,
        "runtime_instances": 0, "admissible_runtime_instances": 0,
        "structural_runtime_instances": 0, "confirm_log_references": 0,
        "timing_samples": 0,
    }
    by_route_operator: dict[str, dict[str, int]] = {}
    workload_admissible: dict[tuple[str, int, str, str], int] = {}

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.current.", dir=parent))
    census_path = temporary / "census.jsonl"
    try:
        authority_records = snapshot_worker_authorities(
            temporary, evidence_records)
        bundle_snapshot_path = temporary / "authorities/bundle-authority.json"
        write_frozen(bundle_snapshot_path, bundle_path.read_bytes())
        bundle_snapshot = {
            "path": bundle_snapshot_path.relative_to(temporary).as_posix(),
            "sha256": file_sha(bundle_snapshot_path),
        }
        with census_path.open("wb") as stream:
            for item in master["work_items"]:
                item_id = item["work_item_id"]
                spec = item_specs[item_id]
                shard = shards[item["shard_key"]]
                cached = manifest_cache.get(item["shard_key"])
                if cached is None:
                    manifest_path = spec["execution_inputs"]["manifest"][
                        "snapshot_path"]
                    manifest = load_json(manifest_path, "candidate manifest")
                    candidates = manifest_candidates(
                        item["route"], item["operator"], manifest, shard)
                    cached = (file_sha(manifest_path), candidates)
                    manifest_cache[item["shard_key"]] = cached
                manifest_sha256, candidates = cached
                workload = spec["workload"]
                screen_contract = contract["screen"]
                screen = parse_log(
                    item["route"], item["operator"], spec["screen"]["log"],
                    workload, candidates,
                    screen_contract["timing_samples_per_runtime"],
                    screen_contract["correctness_repeats"], shard,
                    f"{item_id}/screen", spec["screen"]["schedule_seed"],
                    contract["grouped_warmups"])
                retained = retained_symbols(
                    item, spec, manifest_sha256, screen)
                screen_by_key = record_map(screen, f"{item_id}/screen")
                wanted_confirm = {
                    key: row for key, row in screen_by_key.items()
                    if row["symbol"] in retained}
                if any(row["classification"] == "MEASURED" and key not in wanted_confirm
                       for key, row in screen_by_key.items()):
                    raise AggregateError(f"{item_id} dropped an admissible runtime")
                confirms = []
                confirm_contract = contract["confirm"]
                for run in spec["confirm"]:
                    if run.get("empty_structural") is True:
                        if retained or wanted_confirm:
                            raise AggregateError(
                                f"{item_id} used empty-confirm with retained runtimes")
                        current: dict[tuple[str, tuple[Any, ...]],
                                      dict[str, Any]] = {}
                    else:
                        parsed = parse_log(
                            item["route"], item["operator"], run["log"], workload,
                            [row for row in candidates if row["symbol"] in retained],
                            confirm_contract["timing_samples_per_runtime"],
                            confirm_contract["correctness_repeats"], shard,
                            f"{item_id}/confirm-{run['round']}",
                            run["schedule_seed"], contract["grouped_warmups"])
                        current = record_map(
                            parsed, f"{item_id}/confirm-{run['round']}")
                    if set(current) != set(wanted_confirm):
                        raise AggregateError(
                            f"{item_id} confirm runtime denominator differs from screen")
                    for key in current:
                        if current[key]["classification"] != \
                                wanted_confirm[key]["classification"]:
                            raise AggregateError(
                                f"{item_id} runtime admissibility changed across rounds")
                    confirms.append((run, current))

                candidate_by_symbol = {row["symbol"]: row for row in candidates}
                for key, screen_row in sorted(
                        screen_by_key.items(),
                        key=lambda pair: (pair[0][0], canonical(dict(pair[0][1])))):
                    symbol = screen_row["symbol"]
                    candidate = candidate_by_symbol[symbol]
                    is_retained = symbol in retained
                    if screen_row["classification"] == "MEASURED":
                        samples_by_round = [current[key]["samples"]
                                            for _run, current in confirms]
                        pooled = [value for values in samples_by_round for value in values]
                        if len(pooled) != 3 * 11:
                            raise AggregateError(f"{item_id}/{symbol} pooled timing differs")
                        provenance = {
                            "worker_id": spec["worker_id"],
                            "device_identity_sha256":
                                spec["device_identity_sha256"],
                            "device_homogeneity_sha256":
                                spec["device_homogeneity_sha256"],
                            "artifact_id":
                                spec["execution_inputs"]["artifact_id"],
                            "shard_key": item["shard_key"],
                            "binary_sha256":
                                spec["execution_inputs"]["binary"]["sha256"],
                            "binary_receipt_sha256":
                                spec["execution_inputs"]["binary_receipt"]["sha256"],
                            "manifest_sha256": manifest_sha256,
                            "execution_authority_sha256":
                                spec["execution_authority_sha256"],
                            "executor_sha256": spec["executor_sha256"],
                            "worker_evidence_sha256": spec["evidence_sha256"],
                            "runtime_linkage": spec["runtime_linkage"],
                            "runtime_linkage_sha256":
                                spec["runtime_linkage_sha256"],
                        }
                        provenance_sha256 = digest(provenance)
                        screen_samples = [{"sample_index": index, "us": value}
                                          for index, value in enumerate(
                                              screen_row["samples"])]
                        confirm_runs = []
                        for (run, _current), values in zip(confirms,
                                                           samples_by_round):
                            confirm_runs.append({
                                "round": run["round"], "order": run["order"],
                                "schedule_seed": worker_runner.schedule_seed_hex(
                                    run["schedule_seed"]),
                                "argv_sha256": run["argv_sha256"],
                                "source_log_sha256": run["log_sha256"],
                                "provenance_sha256": provenance_sha256,
                                "sample_count": len(values),
                                "samples": [
                                    {"sample_index": index, "us": value}
                                    for index, value in enumerate(values)],
                            })
                        timing: dict[str, Any] | None = {
                            "provenance": provenance,
                            "screen": {
                                "round": 0, "order": "SCREEN",
                                "schedule_seed": worker_runner.schedule_seed_hex(
                                    spec["screen"]["schedule_seed"]),
                                "argv_sha256": spec["screen"]["argv_sha256"],
                                "source_log_sha256":
                                    spec["screen"]["log_sha256"],
                                "provenance_sha256": provenance_sha256,
                                "sample_count": len(screen_samples),
                                "samples": screen_samples,
                            },
                            "confirm_runs": confirm_runs,
                            "round_median_us": [statistics.median(values)
                                                for values in samples_by_round],
                            "median_us": statistics.median(pooled),
                            "min_us": min(pooled), "max_us": max(pooled),
                            "sample_count": len(pooled),
                            "samples_sha256": digest(pooled),
                        }
                        counters["admissible_runtime_instances"] += 1
                        counters["timing_samples"] += len(pooled)
                        workload_key = (item["route"], item["qtype"],
                                        item["operator"], item["workload_key"])
                        workload_admissible[workload_key] = \
                            workload_admissible.get(workload_key, 0) + 1
                    else:
                        timing = None
                        counters["structural_runtime_instances"] += 1
                    output = {
                        "schema": FILE_SCHEMA,
                        "work_item_id": item_id, "worker_id": spec["worker_id"],
                        "route": item["route"], "operator": item["operator"],
                        "qtype": item["qtype"], "shard_key": item["shard_key"],
                        "workload_key": item["workload_key"],
                        "source_class": workload.get("source_class"),
                        "public_problem": {key: value for key, value in workload.items()
                                           if key not in {"workload_key", "source_class",
                                                          "diagnostics", "key"}},
                        "parent_id": candidate["parent_id"],
                        "static_candidate_id": candidate["static_candidate_id"],
                        "symbol": symbol, "static": stable_static_row(candidate),
                        "runtime": screen_row["runtime"],
                        "classification": screen_row["classification"],
                        "retained_for_confirmation": is_retained,
                        "raw_bad": 0, "timing": timing,
                        "authority": {
                            "payload_kind":
                                spec["execution_inputs"]["payload_kind"],
                            "manifest_sha256": manifest_sha256,
                            "binary_sha256":
                                spec["execution_inputs"]["binary"]["sha256"],
                            "binary_receipt_sha256":
                                spec["execution_inputs"]["binary_receipt"]["sha256"],
                            "structural_proof_sha256": (
                                None if spec["execution_inputs"][
                                    "structural_proof"] is None else
                                spec["execution_inputs"][
                                    "structural_proof"]["sha256"]),
                            "artifact_id":
                                spec["execution_inputs"]["artifact_id"],
                            "screen_log_sha256": spec["screen"]["log_sha256"],
                            "confirm_log_sha256": [run["log_sha256"]
                                                   for run in spec["confirm"]],
                            "result_authority_sha256":
                                spec["result_authority_sha256"],
                            "execution_authority_sha256":
                                spec["execution_authority_sha256"],
                            "executor_sha256": spec["executor_sha256"],
                            "worker_evidence_sha256": spec["evidence_sha256"],
                            "device_identity_sha256":
                                spec["device_identity_sha256"],
                            "device_homogeneity_sha256":
                                spec["device_homogeneity_sha256"],
                            "runtime_linkage_sha256":
                                spec["runtime_linkage_sha256"],
                        },
                    }
                    stream.write(canonical(output) + b"\n")
                counters["work_items"] += 1
                counters["static_candidate_instances"] += len(candidates)
                counters["runtime_instances"] += len(screen)
                counters["confirm_log_references"] += len(spec["confirm"])
                pair = f"{item['route']}/{item['operator']}"
                bucket = by_route_operator.setdefault(pair, {
                    "work_items": 0, "static_candidates": 0,
                    "runtime_instances": 0, "admissible_runtime_instances": 0})
                bucket["work_items"] += 1
                bucket["static_candidates"] += len(candidates)
                bucket["runtime_instances"] += len(screen)
                bucket["admissible_runtime_instances"] += sum(
                    row["classification"] == "MEASURED" for row in screen)

        expected_static = sum(row["parent_count"] for row in master["work_items"])
        if counters["work_items"] != master["denominator"]["work_items"] or \
                counters["static_candidate_instances"] != expected_static:
            raise AggregateError("aggregate static/work-item denominator differs")
        workload_keys = {(row["route"], row["qtype"], row["operator"],
                          row["workload_key"]) for row in master["work_items"]}
        missing_workloads = sorted(workload_keys - set(workload_admissible))
        if missing_workloads:
            raise AggregateError(
                f"workloads have no admissible runtime: {missing_workloads[:3]}")
        census_sha = file_sha(census_path)
        summary = {
            "schema": OUTPUT_SCHEMA,
            "verdict": "COMPLETE_RAW_BIT_CLEAN_CENSUS",
            "selection_performed": False,
            "top_n": None,
            "authorities": {
                "bundle": bundle_snapshot,
                "bundle_schema": bundle["schema"],
                "workload_plan_sha256": file_sha(plan_path),
                "workload_index_sha256": file_sha(workload_index_path),
                "master_sha256": file_sha(master_path),
                "assignment_sha256": file_sha(assignment_path),
                "device_homogeneity_sha256": file_sha(device_path),
                "source_sha": bundle["source_sha"],
                "source_tree": bundle["source_tree"],
                "worker_evidence": authority_records,
            },
            "run_contract": contract,
            "denominator": {
                **counters,
                "binary_shards": master["denominator"]["binary_shards"],
                "workload_keys": master["denominator"]["workload_keys"],
                "workload_route_instances": len(workload_keys),
                "workers": assignment["worker_count"],
                "by_route_operator": by_route_operator,
            },
            "census": {"path": "census.jsonl", "sha256": census_sha,
                       "records": counters["runtime_instances"]},
            "workload_index_schema": workload_index.get("schema"),
            "device_homogeneity_schema": device.get("schema"),
        }
        summary_bytes = json.dumps(
            summary, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        (temporary / "summary.json").write_bytes(summary_bytes)
        if output_dir.exists():
            staged_files = sorted(
                path.relative_to(temporary) for path in temporary.rglob("*")
                if path.is_file() and not path.is_symlink())
            observed_files = (sorted(
                path.relative_to(output_dir) for path in output_dir.rglob("*")
                if path.is_file() and not path.is_symlink())
                if output_dir.is_dir() and not output_dir.is_symlink() else [])
            if (output_dir.is_symlink() or not output_dir.is_dir() or
                    observed_files != staged_files or
                    any(not (output_dir / relative).is_file() or
                        (output_dir / relative).is_symlink() or
                        (output_dir / relative).read_bytes() !=
                        (temporary / relative).read_bytes()
                        for relative in staged_files)):
                raise AggregateError(f"refusing to replace stale output {output_dir}")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, output_dir)
        return summary
    except Exception:
        # The mkdtemp directory owns only files made by this invocation.
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def expected_contract() -> dict[str, Any]:
    return {
        "worker_evidence_schema": EVIDENCE_SCHEMA,
        "completion_schema": worker_plan.RESULT_SCHEMA,
        "output_schema": OUTPUT_SCHEMA,
        "minimum_missing_from_completion_v1": [
            "workload_index_sha256", "measurement_result_authority_sha256",
            "screen_log_path_and_sha256", "retention_path_and_sha256",
            "confirm_log_path_and_sha256_per_round", "exact_argv_per_run",
            "timing_sample_denominator", "correctness_repeat_denominator",
            "round_and_order_denominator",
            "versioned_schedule_seed_formula", "grouped_warmup_denominator",
            "raw_confirmation_samples", "device_and_binary_linkage",
        ],
        "final_measurement": {
            "confirmation_rounds": 3,
            "timing_samples_per_runtime_per_round": 11,
            "order_counterbalance_required": True,
            "all_admissible_candidates": True,
            "top_n": None,
        },
        "screen_is_dynamic_runtime_denominator_authority": True,
        "empty_scalefirst_shard_confirmation":
            "THREE_HASHED_NO_DEVICE_LAUNCH_MARKERS",
        "selection_performed": False,
    }


def expect_red(label: str, action, needle: str) -> None:
    try:
        action()
    except AggregateError as exc:
        if needle not in str(exc):
            raise AggregateError(f"{label}: wrong failure: {exc}") from exc
        return
    raise AggregateError(f"{label}: planted negative stayed green")


def self_test() -> None:
    contract = {
        "schedule_seed": worker_runner.schedule_seed_contract(),
        "grouped_warmups": 3,
        "screen": {"timing_samples_per_runtime": 5,
                   "correctness_repeats": 17},
        "confirm": {"timing_samples_per_runtime": 11,
                    "correctness_repeats": 17,
                    "rounds": [{"round": 1, "order": "FORWARD"},
                               {"round": 2, "order": "REVERSE"},
                               {"round": 3, "order": "HASHED"}]},
        "all_admissible_candidates": True, "top_n": None,
    }
    validate_run_contract(contract)
    planted = json.loads(json.dumps(contract)); planted["top_n"] = 8
    expect_red("top-N", lambda: validate_run_contract(planted), "ranking")
    planted = json.loads(json.dumps(contract)); planted["confirm"]["rounds"].pop()
    expect_red("round gap", lambda: validate_run_contract(planted), "three rounds")
    planted = json.loads(json.dumps(contract))
    for row in planted["confirm"]["rounds"]:
        row["order"] = "FORWARD"
    expect_red("order", lambda: validate_run_contract(planted), "schedule")
    planted = json.loads(json.dumps(contract)); planted["grouped_warmups"] = 0
    expect_red("warmups", lambda: validate_run_contract(planted), ">= 1")

    candidate = [{"parent_id": 0, "static_candidate_id": "static",
                  "symbol": "candidate", "manifest_row": {}}]
    workload = {"workload_key": "dense", "source_class": "test",
                "m": 1, "n": 16, "k": 128}
    sf_lines = [
        "SF_SHARD qtype=10 artifact_tile_k=0 bchunk=0 typed_rows=1 "
        "weight_layout=2 weight_mapping_id=0x514b504b54000001 "
        "selected_rows=1 algorithm_mask=0x3 device=0 cu=72 "
        "iterations=5 correctness_repeats=17 schedule_seed=0x1"]
    for algorithm, policy, grid in (("NONPERSISTENT", "ordinary", 1),
                                     ("PERSISTENT", "capacity", 72)):
        for sample in range(5):
            sf_lines.append("SF_CELL " + json.dumps({
                "shape": "1x16x128", "symbol": "candidate",
                "algorithm": algorithm, "metric_scope": "FULL_OUTPUT",
                "policy": policy, "grid": grid, "occupancy": 1, "split": 1,
                "capacity_b_mask": "0x1", "balanced_b_mask": "0x0",
                "status": "MEASURED", "reason": "MEASURED", "sample": sample,
                "sample_us": 1.0 + sample, "raw_bad": 0}))
    sf_lines.append(
        "SF_COMPLETE status=COMPLETE shape=1x16x128 typed_rows=1 "
        "runtime_cells=2 measured_cells=2 records=10 iterations=5 "
        "fixture=ORDER-INDEPENDENT+FP16-EXACT fixture_mode=ordinary roundtrip=PASS")
    shard = {"qtype": 10, "layout": 2,
             "mapping_id": "0x514b504b54000001"}
    parsed = parse_sf_dense(
        sf_lines, workload, candidate, 5, 17, shard, "self-test", 1)
    if len(parsed) != 2:
        raise AggregateError("ScaleFirst positive runtime denominator differs")
    bad_lines = list(sf_lines)
    for index in range(1, 6):
        row = json.loads(bad_lines[index][len("SF_CELL "):])
        row["raw_bad"] = 1
        bad_lines[index] = "SF_CELL " + json.dumps(row)
    expect_red("raw-bit", lambda: parse_sf_dense(
        bad_lines, workload, candidate, 5, 17, shard, "self-test", 1),
        "raw-bit mismatch")
    missing = [line for line in sf_lines if '"PERSISTENT"' not in line]
    missing[-1] = missing[-1].replace("runtime_cells=2", "runtime_cells=1").replace(
        "measured_cells=2", "measured_cells=1").replace("records=10", "records=5")
    expect_red("runtime gap", lambda: parse_sf_dense(
        missing, workload, candidate, 5, 17, shard, "self-test", 1),
        "runtime denominator")
    print("[kpack-discovery-result-aggregate:self-test] PASS "
          "completion-v1-gap=EXPLICIT evidence-links=HASHED "
          "raw-bit=FAIL-CLOSED candidate-runtime=EXACT top-n=FORBIDDEN "
          "confirm=3x11-order-counterbalanced output=JSONL")


def _validate_summary_authorities(summary: dict[str, Any], root: Path,
                                  contract: dict[str, Any]
                                  ) -> dict[int, dict[str, Any]]:
    authorities = summary.get("authorities")
    required = {
        "bundle", "bundle_schema", "workload_plan_sha256",
        "workload_index_sha256", "master_sha256", "assignment_sha256",
        "device_homogeneity_sha256", "source_sha", "source_tree",
        "worker_evidence",
    }
    if not isinstance(authorities, dict) or set(authorities) != required:
        raise AggregateError("aggregate summary authority schema differs")
    for field in (
            "workload_plan_sha256", "workload_index_sha256", "master_sha256",
            "assignment_sha256",
            "device_homogeneity_sha256"):
        sha(authorities[field], f"summary authority.{field}")
    if authorities["bundle_schema"] not in {
            worker_plan.COMPOSITE_BUNDLE_SCHEMA, worker_plan.CATALOG_SCHEMA}:
        raise AggregateError("aggregate summary bundle schema differs")
    bundle_path = resolve_file(root, authorities["bundle"],
                               "aggregate bundle authority snapshot")
    bundle = load_json(bundle_path, "aggregate bundle authority snapshot")
    if bundle.get("schema") == worker_plan.CATALOG_SCHEMA:
        try:
            partitions.validate_catalog_document(bundle)
        except (ValueError, KeyError) as exc:
            raise AggregateError(
                f"aggregate distributed catalog snapshot differs: {exc}") from exc
    source_sha = text(authorities["source_sha"], "summary authority.source_sha")
    source_tree = text(authorities["source_tree"], "summary authority.source_tree")
    if (bundle.get("schema") != authorities["bundle_schema"] or
            bundle.get("source_sha") != source_sha or
            bundle.get("source_tree") != source_tree):
        raise AggregateError("aggregate bundle/source authority differs")
    bundle_shards = bundle.get("shards")
    if not isinstance(bundle_shards, list) or not bundle_shards:
        raise AggregateError("aggregate bundle shard authority is empty")
    bundle_shard_map: dict[str, dict[str, Any]] = {}
    for bundle_shard in bundle_shards:
        if not isinstance(bundle_shard, dict):
            raise AggregateError("aggregate bundle shard authority is malformed")
        shard_key = text(bundle_shard.get("shard_key"), "bundle shard key")
        if shard_key in bundle_shard_map:
            raise AggregateError("aggregate bundle shard authority is duplicated")
        bundle_shard_map[shard_key] = bundle_shard
    records = authorities["worker_evidence"]
    workers_expected = integer(
        summary.get("denominator", {}).get("workers"),
        "summary worker denominator", minimum=1)
    if not isinstance(records, list) or len(records) != workers_expected:
        raise AggregateError("aggregate worker authority denominator differs")
    result: dict[int, dict[str, Any]] = {}
    authority_bindings = {
        "bundle_sha256": authorities["bundle"]["sha256"],
        "workload_plan_sha256": authorities["workload_plan_sha256"],
        "workload_index_sha256": authorities["workload_index_sha256"],
        "master_sha256": authorities["master_sha256"],
        "assignment_sha256": authorities["assignment_sha256"],
        "device_homogeneity_sha256":
            authorities["device_homogeneity_sha256"],
    }
    observed_devices: set[str] = set()
    for raw in records:
        keys = {
            "worker_id", "device_identity_sha256", "evidence",
            "execution_authority", "completion_result", "executor_sha256",
            "runtime_linkage_sha256", "shards", "result_authorities",
        }
        if not isinstance(raw, dict) or set(raw) != keys:
            raise AggregateError("aggregate worker authority record differs")
        worker = integer(raw["worker_id"], "summary worker_id", minimum=0)
        if worker in result:
            raise AggregateError("aggregate worker authority is duplicated")
        evidence_record = raw.get("evidence")
        evidence_checksum = (evidence_record.get("sha256")
                             if isinstance(evidence_record, dict) else "")
        expected_prefix = (
            f"authorities/worker-{worker:04d}-{evidence_checksum[:16]}")
        expected_paths = {
            "evidence": f"{expected_prefix}/worker-evidence.json",
            "execution_authority":
                f"{expected_prefix}/execution-authority.json",
            "completion_result": f"{expected_prefix}/completion-result.json",
        }
        if any(not isinstance(raw.get(field), dict) or
               raw[field].get("path") != expected_path
               for field, expected_path in expected_paths.items()):
            raise AggregateError(
                f"worker {worker} content-addressed authority path differs")
        device_sha = sha(raw["device_identity_sha256"],
                         "summary worker device identity")
        executor_sha = sha(raw["executor_sha256"], "summary executor SHA-256")
        linkage_sha = sha(raw["runtime_linkage_sha256"],
                          "summary runtime linkage SHA-256")
        evidence_path = resolve_file(root, raw["evidence"],
                                     f"worker {worker} evidence snapshot")
        execution_path = resolve_file(
            root, raw["execution_authority"],
            f"worker {worker} execution snapshot")
        completion_path = resolve_file(
            root, raw["completion_result"],
            f"worker {worker} completion snapshot")
        evidence = load_json(evidence_path, f"worker {worker} evidence snapshot")
        execution = load_json(execution_path, f"worker {worker} execution snapshot")
        completion = load_json(completion_path, f"worker {worker} completion snapshot")
        evidence_fields = {
            "schema", "worker_id", "worker_count", *authority_bindings,
            "device_identity_sha256", "execution_authority",
            "completion_result", "result_authorities", "run_contract",
            "work_items",
        }
        if (set(evidence) != evidence_fields or
                evidence.get("schema") != EVIDENCE_SCHEMA or
                evidence.get("worker_id") != worker or
                evidence.get("worker_count") != workers_expected or
                evidence.get("device_identity_sha256") != device_sha or
                evidence.get("run_contract") != contract or
                evidence.get("execution_authority", {}).get("sha256") !=
                    raw["execution_authority"]["sha256"] or
                evidence.get("completion_result", {}).get("sha256") !=
                    raw["completion_result"]["sha256"]):
            raise AggregateError(f"worker {worker} evidence snapshot differs")
        for field, expected in authority_bindings.items():
            if evidence.get(field) != expected:
                raise AggregateError(
                    f"worker {worker} evidence {field} differs")
        if device_sha in observed_devices:
            raise AggregateError("aggregate worker device evidence is reused")
        observed_devices.add(device_sha)

        execution_fields = {
            "schema", "worker_id", "worker_count", "bundle_sha256",
            "workload_plan_sha256", "master_sha256", "assignment_sha256",
            "selection_sha256", "device_identity_sha256",
            "device_homogeneity_sha256", "visible_device_ordinal",
            "workload_index_sha256", "executor_sha256", "work_items",
            "validated_shards", "validated_shard_keys_sha256",
            "validated_partition_artifacts", "candidate_policy",
            "measurement", "runtime_linkage",
        }
        expected_measurement = {
            "screen_iterations":
                contract["screen"]["timing_samples_per_runtime"],
            "confirm_iterations":
                contract["confirm"]["timing_samples_per_runtime"],
            "confirm_rounds": len(contract["confirm"]["rounds"]),
            "correctness_repeats":
                contract["screen"]["correctness_repeats"],
            "grouped_warmups": contract["grouped_warmups"],
            "schedule_seed": contract["schedule_seed"],
            "outer_round_order": "ASSIGNMENT_REVERSE_THEN_HASHED_V1",
            "inner_candidate_order":
                "ROUND_SEED_VARIED_ALL_ROUTES_OPERATORS",
        }
        if (set(execution) != execution_fields or
                execution.get("schema") != worker_runner.EXECUTION_SCHEMA or
                execution.get("worker_id") != worker or
                execution.get("worker_count") != workers_expected or
                execution.get("device_identity_sha256") != device_sha or
                execution.get("executor_sha256") != executor_sha or
                execution.get("candidate_policy") !=
                    "SCREEN_ALL_COMPILED_CONFIRM_ALL_ADMISSIBLE_NO_TIMING_PRUNE" or
                execution.get("measurement") != expected_measurement or
                digest(_validate_runtime_linkage(
                    execution.get("runtime_linkage"),
                    f"worker {worker} execution")) != linkage_sha):
            raise AggregateError(f"worker {worker} execution snapshot differs")
        for field, expected in authority_bindings.items():
            if execution.get(field) != expected:
                raise AggregateError(
                    f"worker {worker} execution {field} differs")
        sha(execution.get("selection_sha256"),
            f"worker {worker} selection SHA-256")
        ordinal = text(execution.get("visible_device_ordinal"),
                       f"worker {worker} visible device ordinal")
        if not ordinal.isdecimal():
            raise AggregateError(
                f"worker {worker} visible device ordinal differs")

        completion_fields = {
            "schema", "worker_id", "bundle_sha256",
            "workload_plan_sha256", "master_sha256", "assignment_sha256",
            "device_homogeneity_sha256", "device_identity_sha256",
            "completed_work_item_ids",
        }
        if (set(completion) != completion_fields or
                completion.get("schema") != worker_plan.RESULT_SCHEMA or
                completion.get("worker_id") != worker or
                completion.get("device_identity_sha256") != device_sha):
            raise AggregateError(f"worker {worker} completion snapshot differs")
        for field, expected in authority_bindings.items():
            if field == "workload_index_sha256":
                continue
            if completion.get(field) != expected:
                raise AggregateError(
                    f"worker {worker} completion {field} differs")

        evidence_items = evidence.get("work_items")
        if not isinstance(evidence_items, list) or not evidence_items:
            raise AggregateError(f"worker {worker} evidence items are empty")
        evidence_item_ids = [
            row.get("work_item_id") if isinstance(row, dict) else None
            for row in evidence_items]
        if (any(not isinstance(item_id, str) for item_id in evidence_item_ids) or
                len(evidence_item_ids) != len(set(evidence_item_ids)) or
                completion.get("completed_work_item_ids") != evidence_item_ids or
                execution.get("work_items") != len(evidence_item_ids)):
            raise AggregateError(f"worker {worker} work-item authority differs")
        shards = raw["shards"]
        shard_map: dict[str, dict[str, Any]] = {}
        shard_keys = {
            "shard_key", "artifact_id", "payload_kind", "binary_sha256",
            "manifest_sha256", "binary_receipt_sha256",
            "structural_proof_sha256"}
        if not isinstance(shards, list) or not shards:
            raise AggregateError(f"worker {worker} shard authority is empty")
        for shard_row in shards:
            if not isinstance(shard_row, dict) or set(shard_row) != shard_keys:
                raise AggregateError(f"worker {worker} shard authority differs")
            key = text(shard_row["shard_key"], "summary shard key")
            if key in shard_map:
                raise AggregateError(f"worker {worker} shard authority duplicated")
            if shard_row["artifact_id"] is not None:
                text(shard_row["artifact_id"], "summary artifact ID")
            for field in ("binary_sha256", "manifest_sha256",
                          "binary_receipt_sha256"):
                sha(shard_row[field], f"summary shard.{field}")
            kind = shard_row["payload_kind"]
            proof_sha = shard_row["structural_proof_sha256"]
            if kind == worker_runner.NO_DEVICE_KERNEL_STRUCTURAL:
                sha(proof_sha, "summary shard structural proof SHA-256")
            elif kind != worker_runner.DEVICE_KERNEL or proof_sha is not None:
                raise AggregateError(
                    f"worker {worker} shard payload kind differs")
            bundle_shard = bundle_shard_map.get(key)
            bundle_files = (bundle_shard.get("files")
                            if isinstance(bundle_shard, dict) else None)
            expected_bundle_files = {
                "binary", "manifest", "binary_receipt"}
            if kind == worker_runner.NO_DEVICE_KERNEL_STRUCTURAL:
                expected_bundle_files.add("structural_proof")
            if (not isinstance(bundle_files, dict) or
                    set(bundle_files) != expected_bundle_files or any(
                    not isinstance(bundle_files[field], dict) or
                    not isinstance(bundle_files[field].get("sha256"), str)
                    for field in expected_bundle_files)):
                raise AggregateError(
                    f"worker {worker} shard/bundle authority differs")
            if (kind != bundle_shard.get(
                    "payload_kind", worker_runner.DEVICE_KERNEL) or
                    proof_sha != (None if kind == worker_runner.DEVICE_KERNEL
                                  else bundle_files[
                                      "structural_proof"]["sha256"]) or
                    shard_row["artifact_id"] != bundle_shard.get("artifact_id") or
                    any(shard_row[f"{field}_sha256"] !=
                        bundle_files[field]["sha256"]
                        for field in ("binary", "manifest",
                                      "binary_receipt"))):
                raise AggregateError(
                    f"worker {worker} shard/bundle authority differs")
            shard_map[key] = shard_row
        if [row["shard_key"] for row in shards] != sorted(shard_map):
            raise AggregateError(f"worker {worker} shard authority order differs")

        item_shards: dict[str, dict[str, Any]] = {}
        item_result_hashes: dict[str, str] = {}
        for evidence_item in evidence_items:
            item_id = evidence_item["work_item_id"]
            if set(evidence_item) != {
                    "work_item_id", "result_authority_sha256",
                    "execution_inputs", "screen", "retention", "confirm"}:
                raise AggregateError(
                    f"worker {worker} evidence item schema differs")
            item_result_hashes[item_id] = sha(
                evidence_item["result_authority_sha256"],
                f"worker {worker} item result-authority SHA-256")
            inputs = evidence_item.get("execution_inputs")
            base_input_keys = {
                "artifact_id", "shard_key", "binary", "manifest",
                "binary_receipt", "rows_file",
                "retention_symbols_executed_path"}
            if not isinstance(inputs, dict) or frozenset(inputs) not in {
                    frozenset(base_input_keys),
                    frozenset(base_input_keys | {
                        "payload_kind", "structural_proof"})}:
                raise AggregateError(
                    f"worker {worker} execution input schema differs")
            key = text(inputs.get("shard_key"), "evidence shard key")
            binary = inputs.get("binary")
            manifest = inputs.get("manifest")
            receipt = inputs.get("binary_receipt")
            proof = inputs.get("structural_proof")
            if (not isinstance(binary, dict) or set(binary) != {
                    "executed_path", "size", "sha256"} or
                    not isinstance(manifest, dict) or set(manifest) != {
                        "size", "sha256", "snapshot"} or
                    not isinstance(receipt, dict) or set(receipt) != {
                        "size", "sha256"}):
                raise AggregateError(
                    f"worker {worker} execution input file schema differs")
            kind = inputs.get("payload_kind", worker_runner.DEVICE_KERNEL)
            if kind == worker_runner.NO_DEVICE_KERNEL_STRUCTURAL:
                if not isinstance(proof, dict) or set(proof) != {
                        "size", "sha256", "snapshot"}:
                    raise AggregateError(
                        f"worker {worker} structural proof input differs")
                proof_sha = sha(
                    proof.get("sha256"), "evidence structural proof SHA-256")
            elif kind == worker_runner.DEVICE_KERNEL and proof is None:
                proof_sha = None
            else:
                raise AggregateError(
                    f"worker {worker} execution payload kind differs")
            candidate = {
                "shard_key": key, "artifact_id": inputs.get("artifact_id"),
                "payload_kind": kind,
                "binary_sha256": sha(
                    binary.get("sha256"), "evidence binary SHA-256"),
                "manifest_sha256": sha(
                    manifest.get("sha256"), "evidence manifest SHA-256"),
                "binary_receipt_sha256": sha(
                    receipt.get("sha256"), "evidence receipt SHA-256"),
                "structural_proof_sha256": proof_sha,
            }
            previous = item_shards.setdefault(key, candidate)
            if previous != candidate:
                raise AggregateError(
                    f"worker {worker} per-item shard authority differs")
        if item_shards != shard_map:
            raise AggregateError(
                f"worker {worker} evidence/shard authority differs")

        shard_keys = sorted(shard_map)
        if (execution.get("validated_shards") != len(shard_keys) or
                execution.get("validated_shard_keys_sha256") !=
                    digest(shard_keys)):
            raise AggregateError(
                f"worker {worker} validated-shard authority differs")
        expected_artifacts: list[dict[str, Any]] = []
        if bundle.get("schema") == worker_plan.CATALOG_SCHEMA:
            partition_map = {row["artifact_id"]: row
                             for row in bundle["partitions"]}
            for artifact_id in sorted({row["artifact_id"]
                                       for row in shard_map.values()}):
                partition = partition_map.get(artifact_id)
                if partition is None:
                    raise AggregateError(
                        f"worker {worker} catalog artifact differs")
                expected_artifacts.append({
                    "artifact_id": artifact_id,
                    "partition_manifest_sha256":
                        partition["partition_manifest"]["sha256"],
                })
        if execution.get("validated_partition_artifacts") != expected_artifacts:
            raise AggregateError(
                f"worker {worker} validated artifact authority differs")
        route_rows = raw["result_authorities"]
        if not isinstance(route_rows, list) or not route_rows:
            raise AggregateError(f"worker {worker} result authorities are empty")
        route_hashes: dict[str, str] = {}
        for route_row in route_rows:
            if not isinstance(route_row, dict) or set(route_row) != {"route", "file"}:
                raise AggregateError(f"worker {worker} result authority differs")
            route = text(route_row["route"], "summary result route")
            if route_row["file"].get("path") != \
                    f"{expected_prefix}/result-authority-{route}.json":
                raise AggregateError(
                    f"worker {worker} result authority path differs")
            route_path = resolve_file(
                root, route_row["file"],
                f"worker {worker} {route} result-authority snapshot")
            route_doc = load_json(route_path, "result-authority snapshot")
            expected_route_fields = {
                "schema", "route", "worker_id",
                "execution_authority_sha256", "candidate_policy",
                "work_item_ids", "screen_iterations", "confirm_iterations",
                "confirm_rounds", "correctness_repeats", "grouped_warmups",
                "schedule_seed",
            }
            if (set(route_doc) != expected_route_fields or
                    route_doc.get("schema") !=
                        "quactlize.kpack-discovery-worker-route-result-authority.v2" or
                    route in route_hashes or route_doc.get("route") != route or
                    route_doc.get("worker_id") != worker or
                    route_doc.get("execution_authority_sha256") !=
                        raw["execution_authority"]["sha256"] or
                    route_doc.get("candidate_policy") !=
                        "ALL_ADMISSIBLE_NO_TIMING_PRUNE" or
                    route_doc.get("screen_iterations") !=
                        contract["screen"]["timing_samples_per_runtime"] or
                    route_doc.get("confirm_iterations") !=
                        contract["confirm"]["timing_samples_per_runtime"] or
                    route_doc.get("confirm_rounds") !=
                        len(contract["confirm"]["rounds"]) or
                    route_doc.get("correctness_repeats") !=
                        contract["screen"]["correctness_repeats"] or
                    route_doc.get("grouped_warmups") !=
                        contract["grouped_warmups"] or
                    route_doc.get("schedule_seed") != contract["schedule_seed"]):
                raise AggregateError(f"worker {worker} result authority differs")
            route_hashes[route] = route_row["file"]["sha256"]
        evidence_routes = {
            row.get("route"): row.get("sha256")
            for row in evidence.get("result_authorities", [])
            if isinstance(row, dict)}
        if route_hashes != evidence_routes:
            raise AggregateError(f"worker {worker} result-authority union differs")
        routed_items: list[str] = []
        for route_row in route_rows:
            route_path = resolve_file(
                root, route_row["file"],
                f"worker {worker} result-authority snapshot")
            route_doc = load_json(route_path, "result-authority snapshot")
            route_items = route_doc.get("work_item_ids")
            if (not isinstance(route_items, list) or
                    any(not isinstance(item_id, str) for item_id in route_items)):
                raise AggregateError(
                    f"worker {worker} result work-item authority differs")
            for item_id in route_items:
                if item_result_hashes.get(item_id) != route_row["file"]["sha256"]:
                    raise AggregateError(
                        f"worker {worker} item/result authority differs")
            routed_items.extend(route_items)
        if (len(routed_items) != len(set(routed_items)) or
                set(routed_items) != set(evidence_item_ids)):
            raise AggregateError(
                f"worker {worker} result work-item union differs")
        result[worker] = {
            "record": raw, "evidence_sha256": raw["evidence"]["sha256"],
            "execution_authority_sha256":
                raw["execution_authority"]["sha256"],
            "device_identity_sha256": device_sha,
            "executor_sha256": executor_sha,
            "runtime_linkage_sha256": linkage_sha,
            "shards": shard_map, "result_authorities": route_hashes,
        }
    if ([raw["worker_id"] for raw in records] != list(range(workers_expected)) or
            set(result) != set(range(workers_expected))):
        raise AggregateError("aggregate worker authority IDs differ")
    return result


def _validate_raw_samples(run: dict[str, Any], expected: int,
                          label: str) -> list[float]:
    sha(run.get("argv_sha256"), f"{label}.argv_sha256")
    sha(run.get("source_log_sha256"), f"{label}.source_log_sha256")
    if run.get("sample_count") != expected:
        raise AggregateError(f"{label} sample count differs")
    rows = run.get("samples")
    if not isinstance(rows, list) or len(rows) != expected:
        raise AggregateError(f"{label} raw sample denominator differs")
    values: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"sample_index", "us"} or \
                row.get("sample_index") != index:
            raise AggregateError(f"{label} raw sample index differs")
        values.append(finite_positive(row.get("us"), f"{label}.us"))
    return values


def validate_output(path: Path) -> dict[str, Any]:
    summary = load_json(path / "summary.json", "aggregate summary")
    if summary.get("schema") != OUTPUT_SCHEMA or \
            summary.get("verdict") != "COMPLETE_RAW_BIT_CLEAN_CENSUS" or \
            summary.get("selection_performed") is not False or \
            summary.get("top_n") is not None:
        raise AggregateError("aggregate output identity differs")
    census = summary.get("census")
    if not isinstance(census, dict) or census.get("path") != "census.jsonl":
        raise AggregateError("aggregate census record differs")
    contract = validate_run_contract(summary.get("run_contract"))
    worker_authorities = _validate_summary_authorities(summary, path, contract)
    census_path = path / "census.jsonl"
    if not census_path.is_file() or census_path.is_symlink() or \
            file_sha(census_path) != census.get("sha256"):
        raise AggregateError("aggregate census hash differs")
    count = 0
    measured = 0
    structural = 0
    timing_samples = 0
    identities: set[tuple[str, str, str]] = set()
    with census_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AggregateError(f"malformed census JSONL: {exc}") from exc
            required = {
                "schema", "work_item_id", "worker_id", "route", "operator",
                "qtype", "shard_key", "workload_key", "source_class",
                "public_problem", "parent_id", "static_candidate_id", "symbol",
                "static", "runtime", "classification",
                "retained_for_confirmation", "raw_bad", "timing", "authority",
            }
            if not isinstance(row, dict) or set(row) != required or \
                    row.get("schema") != FILE_SCHEMA:
                raise AggregateError("census row schema differs")
            item_id = sha(row.get("work_item_id"), "census work_item_id")
            symbol = text(row.get("symbol"), "census symbol")
            runtime = row.get("runtime")
            if not isinstance(runtime, dict) or not runtime:
                raise AggregateError("census runtime identity differs")
            identity = (item_id, symbol, digest(runtime))
            if identity in identities:
                raise AggregateError("census contains duplicate runtime identity")
            identities.add(identity)
            worker = integer(row.get("worker_id"), "census worker_id", minimum=0)
            worker_authority = worker_authorities.get(worker)
            if worker_authority is None:
                raise AggregateError("census worker has no summary authority")
            if row.get("raw_bad") != 0:
                raise AggregateError("census raw-bit authority differs")
            authority = row.get("authority")
            authority_keys = {
                "payload_kind", "structural_proof_sha256",
                "manifest_sha256", "binary_sha256", "binary_receipt_sha256",
                "artifact_id", "screen_log_sha256", "confirm_log_sha256",
                "result_authority_sha256", "execution_authority_sha256",
                "executor_sha256", "worker_evidence_sha256",
                "device_identity_sha256",
                "device_homogeneity_sha256", "runtime_linkage_sha256",
            }
            if not isinstance(authority, dict) or set(authority) != authority_keys:
                raise AggregateError("census measurement authority differs")
            for field in authority_keys - {
                    "artifact_id", "confirm_log_sha256", "payload_kind",
                    "structural_proof_sha256"}:
                sha(authority[field], f"census authority.{field}")
            payload_kind = authority["payload_kind"]
            if payload_kind not in {
                    worker_runner.DEVICE_KERNEL,
                    worker_runner.NO_DEVICE_KERNEL_STRUCTURAL}:
                raise AggregateError("census payload kind differs")
            proof_sha = authority["structural_proof_sha256"]
            if payload_kind == worker_runner.NO_DEVICE_KERNEL_STRUCTURAL:
                sha(proof_sha, "census structural proof SHA-256")
            elif proof_sha is not None:
                raise AggregateError("device census carries structural proof")
            if authority["artifact_id"] is not None:
                text(authority["artifact_id"], "census authority.artifact_id")
            confirm_logs = authority["confirm_log_sha256"]
            if not isinstance(confirm_logs, list) or len(confirm_logs) != 3:
                raise AggregateError("census confirmation log denominator differs")
            for checksum in confirm_logs:
                sha(checksum, "census confirm log SHA-256")
            shard_authority = worker_authority["shards"].get(row["shard_key"])
            if (shard_authority is None or
                    authority["artifact_id"] != shard_authority["artifact_id"] or
                    authority["binary_sha256"] !=
                        shard_authority["binary_sha256"] or
                    authority["manifest_sha256"] !=
                        shard_authority["manifest_sha256"] or
                    authority["binary_receipt_sha256"] !=
                        shard_authority["binary_receipt_sha256"] or
                    payload_kind != shard_authority["payload_kind"] or
                    proof_sha != shard_authority[
                        "structural_proof_sha256"] or
                    authority["worker_evidence_sha256"] !=
                        worker_authority["evidence_sha256"] or
                    authority["execution_authority_sha256"] !=
                        worker_authority["execution_authority_sha256"] or
                    authority["executor_sha256"] !=
                        worker_authority["executor_sha256"] or
                    authority["device_identity_sha256"] !=
                        worker_authority["device_identity_sha256"] or
                    authority["device_homogeneity_sha256"] !=
                        summary["authorities"]["device_homogeneity_sha256"] or
                    authority["runtime_linkage_sha256"] !=
                        worker_authority["runtime_linkage_sha256"] or
                    authority["result_authority_sha256"] !=
                        worker_authority["result_authorities"].get(row["route"])):
                raise AggregateError("census/summary measurement authority differs")

            classification = row.get("classification")
            timing = row.get("timing")
            if (payload_kind == worker_runner.NO_DEVICE_KERNEL_STRUCTURAL and
                    classification != "STRUCTURAL_UNAVAILABLE"):
                raise AggregateError(
                    "structural no-kernel census contains a non-structural row")
            if classification == "MEASURED":
                measured += 1
                timing_keys = {
                    "provenance", "screen", "confirm_runs", "round_median_us",
                    "median_us", "min_us", "max_us", "sample_count",
                    "samples_sha256",
                }
                if not isinstance(timing, dict) or set(timing) != timing_keys:
                    raise AggregateError("measured census timing schema differs")
                provenance = timing["provenance"]
                provenance_keys = {
                    "worker_id", "device_identity_sha256",
                    "device_homogeneity_sha256", "artifact_id", "shard_key",
                    "binary_sha256", "binary_receipt_sha256", "manifest_sha256",
                    "execution_authority_sha256", "worker_evidence_sha256",
                    "executor_sha256", "runtime_linkage",
                    "runtime_linkage_sha256",
                }
                if not isinstance(provenance, dict) or set(provenance) != provenance_keys:
                    raise AggregateError("measured census provenance schema differs")
                linkage = _validate_runtime_linkage(
                    provenance["runtime_linkage"], "census provenance")
                if (provenance["worker_id"] != worker or
                        provenance["shard_key"] != row["shard_key"] or
                        provenance["artifact_id"] != authority["artifact_id"] or
                        provenance["runtime_linkage_sha256"] != digest(linkage)):
                    raise AggregateError("measured census provenance identity differs")
                for field in (
                        "device_identity_sha256", "device_homogeneity_sha256",
                        "binary_sha256",
                        "binary_receipt_sha256", "manifest_sha256",
                        "execution_authority_sha256", "executor_sha256",
                        "worker_evidence_sha256",
                        "runtime_linkage_sha256"):
                    if sha(provenance[field], f"census provenance.{field}") != \
                            authority[field]:
                        raise AggregateError(
                            "measured census provenance/authority differs")
                screen = timing["screen"]
                run_keys = {"round", "order", "schedule_seed", "argv_sha256",
                            "source_log_sha256", "provenance_sha256",
                            "sample_count", "samples"}
                if not isinstance(screen, dict) or set(screen) != run_keys or \
                        screen["round"] != 0 or screen["order"] != "SCREEN" or \
                        parse_seed(screen["schedule_seed"], "screen seed") != \
                        worker_runner.schedule_seed(item_id, "screen", 0) or \
                        screen["source_log_sha256"] != authority["screen_log_sha256"] or \
                        screen["provenance_sha256"] != digest(provenance):
                    raise AggregateError("census screen timing authority differs")
                expected_screen = contract["screen"]["timing_samples_per_runtime"]
                _validate_raw_samples(screen, expected_screen, "census screen")

                runs = timing["confirm_runs"]
                rounds = contract["confirm"]["rounds"]
                if not isinstance(runs, list) or len(runs) != len(rounds):
                    raise AggregateError("census confirm timing denominator differs")
                pooled: list[float] = []
                medians: list[float] = []
                for index, (run, expected_round) in enumerate(zip(runs, rounds)):
                    if not isinstance(run, dict) or set(run) != run_keys or \
                            (run["round"], run["order"]) != (
                                expected_round["round"], expected_round["order"]) or \
                            parse_seed(run["schedule_seed"], "confirm seed") != \
                            worker_runner.schedule_seed(
                                item_id, "confirm", expected_round["round"]) or \
                            run["source_log_sha256"] != confirm_logs[index] or \
                            run["provenance_sha256"] != digest(provenance):
                        raise AggregateError("census confirm timing authority differs")
                    values = _validate_raw_samples(
                        run, contract["confirm"]["timing_samples_per_runtime"],
                        f"census confirm round {index + 1}")
                    pooled.extend(values)
                    medians.append(statistics.median(values))
                if (len(pooled) != 33 or timing["sample_count"] != len(pooled) or
                        timing["samples_sha256"] != digest(pooled) or
                        timing["round_median_us"] != medians or
                        timing["median_us"] != statistics.median(pooled) or
                        timing["min_us"] != min(pooled) or
                        timing["max_us"] != max(pooled)):
                    raise AggregateError("census confirm raw timing reconstruction differs")
                timing_samples += len(pooled)
            elif classification == "STRUCTURAL_UNAVAILABLE":
                structural += 1
                if timing is not None:
                    raise AggregateError("structural census row carries timing")
            else:
                raise AggregateError("census classification differs")
            count += 1
    if count != census.get("records") or count != \
            summary.get("denominator", {}).get("runtime_instances"):
        raise AggregateError("census row denominator differs")
    denominator = summary.get("denominator", {})
    if (denominator.get("admissible_runtime_instances") != measured or
            denominator.get("structural_runtime_instances") != structural or
            denominator.get("timing_samples") != timing_samples):
        raise AggregateError("aggregate measured timing denominator differs")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    sub.add_parser("contract")
    check = sub.add_parser("validate-output")
    check.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("aggregate")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--workload-index", type=Path, required=True)
    run.add_argument("--master", type=Path, required=True)
    run.add_argument("--assignment", type=Path, required=True)
    run.add_argument("--device-homogeneity", type=Path, required=True)
    run.add_argument("--evidence", type=Path, action="append", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "contract":
            print(json.dumps(expected_contract(), indent=2, sort_keys=True))
        elif args.command == "validate-output":
            result = validate_output(args.output_dir)
            print("KPACK_DISCOVERY_CENSUS PASS "
                  f"records={result['census']['records']} "
                  f"sha256={result['census']['sha256']}")
        else:
            result = aggregate(
                bundle_path=args.bundle, plan_path=args.plan,
                workload_index_path=args.workload_index,
                master_path=args.master, assignment_path=args.assignment,
                device_path=args.device_homogeneity,
                evidence_paths=args.evidence, output_dir=args.output_dir)
            print("KPACK_DISCOVERY_AGGREGATE PASS "
                  f"items={result['denominator']['work_items']} "
                  f"runtime={result['denominator']['runtime_instances']} "
                  f"admissible={result['denominator']['admissible_runtime_instances']} "
                  f"census_sha256={result['census']['sha256']}")
        return 0
    except (AggregateError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[kpack-discovery-result-aggregate] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
