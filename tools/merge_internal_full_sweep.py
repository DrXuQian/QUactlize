#!/usr/bin/env python3
"""Fail-closed merger for the four internal full-sweep leaderboards.

This file is deliberately independent of every shipping kernel and collective.
It consumes one ScaleFirst component summary and one FullyQuantized component
summary, validates their finite denominators, then publishes four *separate*
leaderboards:

  1. ScaleFirst full output (non-persistent and persistent grid policies),
  2. ScaleFirst fixed Split-K producer only (S=2/4/8; reducer excluded),
  3. FullyQuantized full output (placed BC GEMV and tensor-core S=1), and
  4. FullyQuantized fixed Split-K producer only (S=2/4/8; reducer excluded).

The merger never repairs a partial run.  Runtime/correctness failures and lost
cells belong in a component's top-level ``failures``/``missing`` lists and make
that component INCOMPLETE.  A denominator cell may end only in one of the four
canonical terminal states below.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import pathlib
import re
import string
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


SCHEMA = "quactlize.internal_full_sweep.v1"
VALID_STATES = {"MEASURED", "INADMISSIBLE", "BUILD_REJECT", "UNSUPPORTED"}
RANKING_GROUPS = (
    "SCALEFIRST_FULL_OUTPUT",
    "SCALEFIRST_SPLITK_PRODUCER_ONLY",
    "FULLY_QUANTIZED_FULL_OUTPUT",
    "FULLY_QUANTIZED_SPLITK_PRODUCER_ONLY",
)
FULL_OUTPUT = "FULL_OUTPUT"
PRODUCER_ONLY = "PRODUCER_ONLY_REDUCER_EXCLUDED"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
STABLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHAPE_DIRECTORY_FIELDS = {
    "dense": {"m", "n", "k", "group_size"},
    "grouped": {
        "m", "n", "k", "group_size", "experts", "active", "ragged_profile"},
}


class ContractError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    raise ContractError(f"missing required field (one of {', '.join(names)})")


def integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be an integer, got {value!r}") from exc
    if result != value and not (isinstance(value, str) and str(result) == value):
        raise ContractError(f"{field} is not exactly integral: {value!r}")
    if result < minimum:
        raise ContractError(f"{field} must be >= {minimum}, got {result}")
    return result


def finite_float(value: Any, field: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ContractError(f"{field} must be {'positive ' if positive else ''}finite, got {result}")
    return result


def normalize_status(cell: dict[str, Any]) -> tuple[str, str]:
    raw_status = str(required(cell, "canonical_status", "status", "state")).upper()
    status, separator, suffix_reason = raw_status.partition(":")
    # Compiler-only aliases are unambiguous.  Missing/runtime/correctness
    # failures are intentionally *not* mapped: publishing them as a terminal
    # denominator cell would hide a broken experiment.
    if status in {"BUILD_FAILED", "COMPILE_REJECT", "COMPILE_FAILED"}:
        status = "BUILD_REJECT"
    if status not in VALID_STATES:
        raise ContractError(
            f"cell status {status!r} is not terminal; component must be INCOMPLETE")
    reason = str(cell.get("reason", suffix_reason))
    if separator and cell.get("reason") not in (None, "", suffix_reason):
        raise ContractError(
            f"status suffix reason {suffix_reason!r} contradicts reason={cell.get('reason')!r}")
    if status != "MEASURED" and not reason:
        raise ContractError(f"{status} cell lacks a named reason")
    return status, reason


def normalize_shape(cell: dict[str, Any]) -> dict[str, int]:
    shape = cell.get("shape", {})
    if not isinstance(shape, dict):
        raise ContractError("shape must be an object")
    result: dict[str, int] = {}
    for lower, upper in (("m", "M"), ("n", "N"), ("k", "K")):
        value = shape.get(lower, shape.get(upper, cell.get(lower, cell.get(upper))))
        result[lower] = integer(value, f"shape.{lower}", 1)
    l_value = shape.get("l", shape.get("L", cell.get("l", cell.get("L", 1))))
    result["l"] = integer(l_value, "shape.l", 1)
    return result


def normalize_fold(cell: dict[str, Any]) -> dict[str, int]:
    fold = cell.get("fold_n", cell.get("FoldN"))
    if isinstance(fold, dict):
        low = required(fold, "low", "low_bits")
        high = fold.get("high", fold.get("high_bits", low))
    elif isinstance(fold, (list, tuple)) and len(fold) == 2:
        low, high = fold
    elif fold is not None:
        low = high = fold
    else:
        low = cell.get("FoldN_low", cell.get("fold_n_low"))
        high = cell.get("FoldN_high", cell.get("fold_n_high", low))
    if low is None:
        raise ContractError("cell lacks FoldN/fold_n")
    return {
        "low": integer(low, "FoldN.low", 1),
        "high": integer(high, "FoldN.high", 1),
    }


def normalize_workload_identity(cell: dict[str, Any], *,
                                allow_not_applicable_group: bool = False
                                ) -> dict[str, Any]:
    model_id = str(required(cell, "model_id"))
    if not STABLE_ID_RE.fullmatch(model_id):
        raise ContractError(
            f"model_id must be a stable folder-safe identifier, got {model_id!r}")
    tp = cell.get("tp", {})
    if not isinstance(tp, dict):
        raise ContractError("tp must be an object when present")
    tp_world = integer(tp.get("world", cell.get("tp_world")), "tp_world", 1)
    tp_rank = integer(tp.get("rank", cell.get("tp_rank")), "tp_rank", 0)
    if tp_rank >= tp_world:
        raise ContractError(f"tp_rank={tp_rank} must be below tp_world={tp_world}")
    tp_partition = str(tp.get("partition", cell.get("tp_partition", "")))
    if not tp_partition:
        raise ContractError("tp_partition must be a nonempty measured/manifest identity")
    problem_route = str(required(cell, "problem_route", "workload_route", "problem_kind")).lower()
    if problem_route not in {"dense", "grouped"}:
        raise ContractError(f"problem_route must be dense or grouped, got {problem_route!r}")
    shape_id = str(required(cell, "shape_id"))
    if not STABLE_ID_RE.fullmatch(shape_id):
        raise ContractError(f"shape_id must be stable and folder-safe, got {shape_id!r}")
    raw_sources = required(cell, "source_tensors")
    if not isinstance(raw_sources, list) or not raw_sources or not all(
            isinstance(item, str) and item for item in raw_sources):
        raise ContractError("source_tensors must be a nonempty string array")
    source_tensors = sorted(set(raw_sources))
    raw_group_size = required(cell, "group_size", "gs")
    group_size: int | str
    if isinstance(raw_group_size, str) and raw_group_size.upper() == "UNKNOWN":
        group_size = "UNKNOWN"
    else:
        group_size = integer(raw_group_size, "group_size", 0)
        if group_size == 0 and not allow_not_applicable_group:
            raise ContractError(
                "group_size=0 is only valid for a terminal UNSUPPORTED "
                "non-group-quantized qtype")
    grouped: dict[str, Any] | None = None
    if problem_route == "grouped":
        raw_grouped = cell.get("grouped", {})
        if not isinstance(raw_grouped, dict):
            raise ContractError("grouped identity must be an object")

        def known_or_integer(name: str, *aliases: str) -> int | str:
            value = raw_grouped.get(name)
            if value is None:
                for alias in aliases:
                    value = cell.get(alias)
                    if value is not None:
                        break
            if value is None or str(value).upper() == "UNKNOWN":
                return "UNKNOWN"
            return integer(value, f"grouped.{name}", 0)

        expert_count = known_or_integer("experts", "experts", "E")
        active = known_or_integer("active", "active_experts", "active")
        if isinstance(expert_count, int) and isinstance(active, int) and active > expert_count:
            raise ContractError(
                f"grouped.active={active} exceeds grouped.experts={expert_count}")
        ragged_value = raw_grouped.get("ragged", cell.get("ragged", "UNKNOWN"))
        ragged = str(ragged_value).lower() if ragged_value is not None else "unknown"
        if not ragged:
            ragged = "unknown"
        grouped = {"experts": expert_count, "active": active, "ragged": ragged}
    return {
        "model_id": model_id,
        "shape_id": shape_id,
        "source_tensors": source_tensors,
        "tp_world": tp_world,
        "tp_rank": tp_rank,
        "tp_partition": tp_partition,
        "problem_route": problem_route,
        "group_size": group_size,
        "grouped": grouped,
    }


def normalize_shape_directory(value: Any, source: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(SHAPE_DIRECTORY_FIELDS):
        raise ContractError(f"{source}: shape_directory must define dense and grouped")
    result: dict[str, str] = {}
    formatter = string.Formatter()
    for route, required_fields in SHAPE_DIRECTORY_FIELDS.items():
        template = value[route]
        if not isinstance(template, str) or not template:
            raise ContractError(f"{source}: shape_directory.{route} must be nonempty text")
        try:
            parsed = list(formatter.parse(template))
        except ValueError as exc:
            raise ContractError(
                f"{source}: invalid shape_directory.{route}: {exc}") from exc
        fields = {field for _, field, _, _ in parsed if field is not None}
        if fields != required_fields:
            raise ContractError(
                f"{source}: shape_directory.{route} fields={sorted(fields)}, "
                f"expected={sorted(required_fields)}")
        if any(conversion is not None or format_spec for _, _, format_spec, conversion in parsed):
            raise ContractError(
                f"{source}: shape_directory.{route} may not use conversions/format specs")
        result[route] = template
    return result


def normalize_algorithm(component: str, cell: dict[str, Any]) -> tuple[str, str, str, int]:
    raw = str(required(cell, "algorithm")).strip()
    token = raw.lower().replace("_", "-")
    split = integer(cell.get("split_k_slices", cell.get("S", cell.get("s", 1))), "S", 1)

    if component == "scale_first":
        if token in {"non-persistent", "nonpersistent", "dp", "scale-first-non-persistent"}:
            algorithm = "SCALEFIRST_NONPERSISTENT"
        elif token in {"persistent", "scale-first-persistent"}:
            algorithm = "SCALEFIRST_PERSISTENT"
        elif token in {
                "splitk", "split-k", "fixed-splitk", "scale-first-splitk",
                "scale-first-fixed-splitk"}:
            if split not in {2, 4, 8}:
                raise ContractError(f"ScaleFirst fixed Split-K producer S must be 2/4/8, got {split}")
            return (f"SCALEFIRST_SPLITK_S{split}",
                    "SCALEFIRST_SPLITK_PRODUCER_ONLY", PRODUCER_ONLY, split)
        else:
            match = re.fullmatch(r"(?:scale-first-)?(?:fixed-)?splitk-s(2|4|8)", token)
            if match:
                encoded = int(match.group(1))
                if split != encoded:
                    raise ContractError(f"algorithm {raw} contradicts S={split}")
                return (f"SCALEFIRST_SPLITK_S{split}",
                        "SCALEFIRST_SPLITK_PRODUCER_ONLY", PRODUCER_ONLY, split)
            raise ContractError(f"unknown ScaleFirst algorithm {raw!r}")
        if split != 1:
            raise ContractError("ScaleFirst full-output board cannot contain Split-K S>1")
        return algorithm, "SCALEFIRST_FULL_OUTPUT", FULL_OUTPUT, split

    if component != "fully_quantized":
        raise ContractError(f"unknown component {component!r}")
    if token in {"bc-gemv", "placed-bc-gemv", "fq-bc-gemv", "gemv"}:
        if split != 1:
            raise ContractError("BC GEMV full-output cell must use S=1")
        return "FQ_BC_GEMV", "FULLY_QUANTIZED_FULL_OUTPUT", FULL_OUTPUT, split
    if token in {"tc-s1", "tensor-core-s1", "tensorcore-s1", "fq-tc-s1"}:
        if split != 1:
            raise ContractError("tensor-core S1 cell must use S=1")
        return "FQ_TC_S1", "FULLY_QUANTIZED_FULL_OUTPUT", FULL_OUTPUT, split
    if token in {"tc-splitk", "tensor-core-splitk", "tensorcore-splitk", "fq-tc-splitk"}:
        if split not in {2, 4, 8}:
            raise ContractError(f"fixed Split-K producer S must be 2/4/8, got {split}")
        return f"FQ_TC_SPLITK_S{split}", "FULLY_QUANTIZED_SPLITK_PRODUCER_ONLY", PRODUCER_ONLY, split
    match = re.fullmatch(r"(?:fq-)?tc-s(2|4|8)", token)
    if match:
        encoded = int(match.group(1))
        if split != encoded:
            raise ContractError(f"algorithm {raw} contradicts S={split}")
        return f"FQ_TC_SPLITK_S{split}", "FULLY_QUANTIZED_SPLITK_PRODUCER_ONLY", PRODUCER_ONLY, split
    raise ContractError(f"unknown FullyQuantized algorithm {raw!r}")


def normalize_provenance(doc: dict[str, Any], path: pathlib.Path) -> dict[str, Any]:
    provenance = doc.get("provenance")
    if not isinstance(provenance, dict):
        raise ContractError(f"{path}: missing provenance object")
    root_sha = str(required(provenance, "root_sha", "git_sha"))
    actlize_sha = str(required(provenance, "actlize_sha"))
    if not GIT_SHA_RE.fullmatch(root_sha) or not GIT_SHA_RE.fullmatch(actlize_sha):
        raise ContractError(f"{path}: root_sha/actlize_sha must be full lowercase git SHAs")
    device = required(provenance, "device")
    if not isinstance(device, dict) or not device:
        raise ContractError(f"{path}: device must be a nonempty measured identity object")
    if any(value in (None, "") for value in device.values()):
        raise ContractError(f"{path}: device identity contains an empty value")
    source_hashes = required(provenance, "source_hashes")
    binary_hashes = required(provenance, "binary_hashes")
    shape_manifest_sha256 = str(required(provenance, "shape_manifest_sha256"))
    gguf_hashes = required(provenance, "gguf_hashes")
    gguf_set_sha256 = str(required(provenance, "gguf_set_sha256"))
    shape_directory = normalize_shape_directory(
        required(provenance, "shape_directory"), str(path))
    if not SHA256_RE.fullmatch(shape_manifest_sha256) or not SHA256_RE.fullmatch(gguf_set_sha256):
        raise ContractError(
            f"{path}: shape_manifest_sha256/gguf_set_sha256 must be lowercase sha256")
    if not isinstance(gguf_hashes, dict) or not gguf_hashes:
        raise ContractError(f"{path}: gguf_hashes must be a nonempty model-id -> sha256 map")
    bad_gguf = {
        key: value for key, value in gguf_hashes.items()
        if not isinstance(key, str) or not STABLE_ID_RE.fullmatch(key) or
        not isinstance(value, str) or not SHA256_RE.fullmatch(value)
    }
    if bad_gguf:
        raise ContractError(f"{path}: gguf_hashes contains invalid stable ids/hashes: {bad_gguf}")
    computed_set_sha256 = hashlib.sha256(
        canonical_json(gguf_hashes).encode("utf-8")).hexdigest()
    if gguf_set_sha256 != computed_set_sha256:
        raise ContractError(
            f"{path}: gguf_set_sha256 does not hash the canonical gguf_hashes map")
    for name, hashes in (("source_hashes", source_hashes), ("binary_hashes", binary_hashes)):
        if not isinstance(hashes, dict) or not hashes:
            raise ContractError(f"{path}: {name} must be a nonempty object")
        bad = {key: value for key, value in hashes.items()
               if not isinstance(value, str) or not SHA256_RE.fullmatch(value)}
        if bad:
            raise ContractError(f"{path}: {name} contains non-sha256 values: {bad}")
    return {
        "root_sha": root_sha,
        "actlize_sha": actlize_sha,
        "device": device,
        "shape_manifest_sha256": shape_manifest_sha256,
        "gguf_hashes": gguf_hashes,
        "gguf_set_sha256": gguf_set_sha256,
        "shape_directory": shape_directory,
        "source_hashes": source_hashes,
        "binary_hashes": binary_hashes,
        "component_summary": str(path.resolve()),
        "component_summary_sha256": file_sha256(path),
    }


def extract_denominator(doc: dict[str, Any]) -> int:
    for name in ("expected_cells", "candidate_denominator", "denominator"):
        if name not in doc:
            continue
        value = doc[name]
        if isinstance(value, dict):
            for child in ("total_cells", "total", "expected_cells", "total_support_cells"):
                if child in value:
                    return integer(value[child], f"denominator.{child}", 1)
        else:
            return integer(value, name, 1)
    raise ContractError("component lacks exact expected-cell denominator")


def normalize_cell(component: str, source_schema: str, cell: dict[str, Any], ordinal: int) -> dict[str, Any]:
    if not isinstance(cell, dict):
        raise ContractError(f"cell {ordinal} is not an object")
    status, reason = normalize_status(cell)
    shape = normalize_shape(cell)
    workload = normalize_workload_identity(
        cell, allow_not_applicable_group=(status == "UNSUPPORTED"))
    algorithm, ranking_group, expected_scope, split = normalize_algorithm(component, cell)
    stated_scope = cell.get("metric_scope", cell.get("timing_scope"))
    if stated_scope is not None:
        canonical_scope = str(stated_scope).upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "FULL_END_TO_END_SHIPPING_RESULT": FULL_OUTPUT,
            "FULL_END_TO_END": FULL_OUTPUT,
            "FULL_OUTPUT": FULL_OUTPUT,
            "PRODUCER_ONLY_DIAGNOSTIC_REDUCER_EXCLUDED": PRODUCER_ONLY,
            "PRODUCER_ONLY_REDUCER_EXCLUDED": PRODUCER_ONLY,
        }
        canonical_scope = aliases.get(canonical_scope, canonical_scope)
        if canonical_scope != expected_scope:
            raise ContractError(
                f"cell {ordinal} metric_scope {stated_scope!r} contradicts {algorithm}")

    if status == "UNSUPPORTED" and not any(name in cell for name in ("layout", "arrangement")):
        layout = None
        fold = None
        artifact_tk = None
    else:
        layout = required(cell, "layout", "arrangement")
        fold = normalize_fold(cell)
        artifact_tk = integer(required(cell, "artifact_tile_k", "ArtifactTileK"), "ArtifactTileK", 1)
    qtype = str(required(cell, "qtype", "format"))
    config = str(cell.get("config", "<unsupported>" if status == "UNSUPPORTED" else ""))
    if not config:
        raise ContractError(f"{status} cell lacks config identity")
    policy = str(cell.get("policy", "n/a"))
    grid = cell.get("grid", "n/a")
    identity = {
        "component": component,
        **workload,
        "qtype": qtype,
        "shape": shape,
        "layout": layout,
        "fold_n": fold,
        "artifact_tile_k": artifact_tk,
        "algorithm": algorithm,
        "config": config,
        "split_k_slices": split,
        "policy": policy,
        "grid": grid,
    }
    cell_id = str(cell.get("cell_id", hashlib.sha256(canonical_json(identity).encode()).hexdigest()))

    result: dict[str, Any] = {
        "cell_id": cell_id,
        "source_schema": source_schema,
        "component": component,
        **workload,
        "ranking_group": ranking_group,
        "metric_scope": expected_scope,
        "status": status,
        "reason": reason,
        "qtype": qtype,
        "tensor": str(cell.get("tensor", cell.get("tensor_name", ""))),
        "route": str(cell.get("route", "")),
        "shape": shape,
        "layout": layout,
        "fold_n": fold,
        "artifact_tile_k": artifact_tk,
        "algorithm": algorithm,
        "config": config,
        "split_k_slices": split,
        "policy": policy,
        "grid": grid,
        "workspace_bytes": integer(cell.get("workspace_bytes", 0), "workspace_bytes", 0),
    }
    if status == "MEASURED":
        samples = required(cell, "raw_samples_us", "samples_us")
        if not isinstance(samples, list) or not samples:
            raise ContractError(f"MEASURED cell {cell_id} has no raw samples")
        raw_samples = [finite_float(value, "raw_samples_us", positive=True) for value in samples]
        median_us = finite_float(required(cell, "median_us"), "median_us", positive=True)
        lo, hi = min(raw_samples), max(raw_samples)
        if not (lo <= median_us <= hi):
            raise ContractError(f"MEASURED cell {cell_id} median {median_us} outside sample range [{lo},{hi}]")
        correctness = str(required(cell, "correctness")).upper()
        if "PASS" not in correctness:
            raise ContractError(f"MEASURED cell {cell_id} correctness is not PASS: {correctness}")
        result.update({
            "median_us": median_us,
            "mfu_pct": finite_float(required(cell, "mfu_pct", "MFU_pct"), "MFU_pct"),
            "mbu_pct": finite_float(required(
                cell, "mbu_pct", "MBU_pct", "distinct_MBU_model_pct"), "MBU_pct"),
            "mbu_kind": str(cell.get("mbu_kind", "DISTINCT_BYTE_MODEL")),
            "raw_samples_us": raw_samples,
            "correctness": correctness,
        })
    else:
        forbidden = [name for name in ("median_us", "raw_samples_us", "samples_us") if name in cell]
        if forbidden:
            raise ContractError(f"{status} cell {cell_id} carries timing fields {forbidden}")
        result.update({
            "median_us": None,
            "mfu_pct": None,
            "mbu_pct": None,
            "mbu_kind": None,
            "raw_samples_us": [],
            "correctness": None,
        })
    return result


@dataclass
class Component:
    name: str
    schema: str
    denominator: int
    status_counts: dict[str, int]
    cells: list[dict[str, Any]]
    provenance: dict[str, Any]


def load_component(path: pathlib.Path, expected_component: str) -> Component:
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ContractError(f"{path}: summary root must be an object")
    schema = str(required(doc, "schema"))
    component = str(doc.get("component", expected_component)).lower().replace("-", "_")
    if component != expected_component:
        raise ContractError(f"{path}: component={component}, expected {expected_component}")
    overall = str(doc.get("status", doc.get("overall", ""))).upper()
    if overall != "COMPLETE":
        raise ContractError(f"{path}: component is {overall or 'UNSPECIFIED'}, not COMPLETE")
    missing = doc.get("missing", [])
    failures = doc.get("failures", [])
    if missing or failures:
        raise ContractError(f"{path}: COMPLETE component has missing={len(missing)} failures={len(failures)}")
    raw_cells = required(doc, "cells")
    if not isinstance(raw_cells, list):
        raise ContractError(f"{path}: cells must be a list")
    denominator = extract_denominator(doc)
    if len(raw_cells) != denominator:
        raise ContractError(f"{path}: denominator={denominator}, cells={len(raw_cells)}")
    cells = [normalize_cell(component, schema, cell, ordinal)
             for ordinal, cell in enumerate(raw_cells)]
    ids = [cell["cell_id"] for cell in cells]
    duplicate_ids = [cell_id for cell_id, count in Counter(ids).items() if count != 1]
    if duplicate_ids:
        raise ContractError(f"{path}: duplicate cell ids: {duplicate_ids[:4]}")
    counts = dict(Counter(cell["status"] for cell in cells))
    stated_counts = doc.get("status_counts")
    if stated_counts is not None:
        canonical_stated = {str(key).upper(): integer(value, f"status_counts.{key}", 0)
                            for key, value in stated_counts.items()}
        if {key: counts.get(key, 0) for key in VALID_STATES} != {
                key: canonical_stated.get(key, 0) for key in VALID_STATES}:
            raise ContractError(f"{path}: status_counts disagree with cells")
    provenance = normalize_provenance(doc, path)
    cell_models = {cell["model_id"] for cell in cells}
    provenance_models = set(provenance["gguf_hashes"])
    if cell_models != provenance_models:
        raise ContractError(
            f"{path}: component model denominator differs from gguf_hashes: "
            f"cells={sorted(cell_models)} provenance={sorted(provenance_models)}")
    return Component(component, schema, denominator, counts, cells, provenance)


def ranking_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    shape = cell["shape"]
    return (
        cell["ranking_group"], cell["model_id"], cell["tp_world"],
        cell["tp_rank"], cell["tp_partition"], cell["problem_route"],
        canonical_json(cell["grouped"]), cell["group_size"],
        cell["shape_id"], cell["qtype"],
        shape["m"], shape["n"], shape["k"], shape["l"],
    )


def comparison_identity(cell: dict[str, Any]) -> tuple[Any, ...]:
    shape = cell["shape"]
    return (
        cell["component"], cell["model_id"], cell["tp_world"],
        cell["tp_rank"], cell["tp_partition"], cell["problem_route"],
        canonical_json(cell["grouped"]), cell["group_size"],
        cell["shape_id"], cell["qtype"],
        shape["m"], shape["n"], shape["k"], shape["l"],
    )


def validate_algorithm_denominators(cells: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[comparison_identity(cell)].append(cell)
    required = {
        "scale_first": {
            "SCALEFIRST_NONPERSISTENT", "SCALEFIRST_PERSISTENT",
            "SCALEFIRST_SPLITK_S2", "SCALEFIRST_SPLITK_S4", "SCALEFIRST_SPLITK_S8",
        },
        "fully_quantized": {
            "FQ_BC_GEMV", "FQ_TC_S1", "FQ_TC_SPLITK_S2",
            "FQ_TC_SPLITK_S4", "FQ_TC_SPLITK_S8",
        },
    }
    for identity, identity_cells in grouped.items():
        component = identity[0]
        algorithms = {cell["algorithm"] for cell in identity_cells}
        missing = required[component] - algorithms
        if missing:
            raise ContractError(
                f"algorithm denominator incomplete for {identity}: missing={sorted(missing)}")
        # Q8 has no shipping FullyQuantized reader.  Its finite denominator is
        # therefore exactly five visible UNSUPPORTED cells (BC, TC-S1, and
        # TC-S2/S4/S8), not an arbitrary number of unsupported variants.  A
        # set-only check would accept an extra fabricated cell and make a
        # closed-looking denominator larger after the fact.
        qtype_token = str(identity_cells[0]["qtype"]).upper().replace("-", "_")
        if component == "fully_quantized" and qtype_token in {"8", "Q8", "Q8_0"}:
            counts = Counter(cell["algorithm"] for cell in identity_cells)
            expected_counts = {
                "FQ_BC_GEMV": 1,
                "FQ_TC_S1": 1,
                "FQ_TC_SPLITK_S2": 1,
                "FQ_TC_SPLITK_S4": 1,
                "FQ_TC_SPLITK_S8": 1,
            }
            if counts != expected_counts or any(
                    cell["status"] != "UNSUPPORTED" for cell in identity_cells):
                raise ContractError(
                    f"Q8 FullyQuantized denominator must be exactly five "
                    f"UNSUPPORTED cells for {identity}: got={dict(counts)}")
        if component == "scale_first":
            np_policies = {
                cell["policy"] for cell in identity_cells
                if cell["algorithm"] == "SCALEFIRST_NONPERSISTENT"}
            persistent_policies = {
                cell["policy"] for cell in identity_cells
                if cell["algorithm"] == "SCALEFIRST_PERSISTENT"}
            witnessed_persistent_policies: set[str] = set()
            for policy in persistent_policies:
                token = policy.lower().replace("_", "-")
                if token in {"capacity", "capacity+balanced", "balanced+capacity"}:
                    witnessed_persistent_policies.add("capacity")
                if token in {"balanced", "capacity+balanced", "balanced+capacity"}:
                    witnessed_persistent_policies.add("balanced")
            if "non-persistent" not in np_policies:
                raise ContractError(f"ScaleFirst non-persistent policy missing for {identity}")
            if not {"capacity", "balanced"}.issubset(witnessed_persistent_policies):
                raise ContractError(
                    f"ScaleFirst persistent capacity/balanced policy missing for {identity}: "
                    f"got={sorted(persistent_policies)}")


def make_decisions(cells: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        groups[ranking_key(cell)].append(cell)
    decisions: list[dict[str, Any]] = []
    for key, candidates in sorted(groups.items(), key=lambda item: canonical_json(item[0])):
        measured = [cell for cell in candidates if cell["status"] == "MEASURED"]
        measured.sort(key=lambda cell: (cell["median_us"], cell["cell_id"]))
        winner = measured[0] if measured else None
        runner_up = measured[1] if len(measured) > 1 else None
        counts = Counter(cell["status"] for cell in candidates)
        decisions.append({
            "ranking_group": key[0],
            "model_id": key[1],
            "tp_world": key[2],
            "tp_rank": key[3],
            "tp_partition": key[4],
            "problem_route": key[5],
            "grouped": json.loads(key[6]),
            "group_size": key[7],
            "shape_id": key[8],
            "qtype": key[9],
            "source_tensors": sorted({
                tensor for cell in candidates for tensor in cell["source_tensors"]}),
            "shape": {"m": key[10], "n": key[11], "k": key[12], "l": key[13]},
            "disposition": "RESOLVED" if winner else "NO_MEASURED_CANDIDATE",
            "candidate_denominator": len(candidates),
            "candidate_status_counts": {state: counts.get(state, 0) for state in sorted(VALID_STATES)},
            "measured_candidates": len(measured),
            "winner_cell_id": winner["cell_id"] if winner else None,
            "winner_algorithm": winner["algorithm"] if winner else None,
            "winner_config": winner["config"] if winner else None,
            "winner_layout": winner["layout"] if winner else None,
            "winner_fold_n": winner["fold_n"] if winner else None,
            "winner_S": winner["split_k_slices"] if winner else None,
            "winner_grid": winner["grid"] if winner else None,
            "winner_median_us": winner["median_us"] if winner else None,
            "winner_mfu_pct": winner["mfu_pct"] if winner else None,
            "winner_mbu_pct": winner["mbu_pct"] if winner else None,
            "runner_up_cell_id": runner_up["cell_id"] if runner_up else None,
            "runner_up_algorithm": runner_up["algorithm"] if runner_up else None,
            "runner_up_config": runner_up["config"] if runner_up else None,
            "runner_up_median_us": runner_up["median_us"] if runner_up else None,
            "runner_up_gap_us": (runner_up["median_us"] - winner["median_us"])
            if runner_up else None,
            "runner_up_gap_pct": (
                100.0 * (runner_up["median_us"] / winner["median_us"] - 1.0)
                if runner_up else None),
        })
    return decisions


def merge_components(scale: Component, fq: Component) -> dict[str, Any]:
    if scale.provenance["root_sha"] != fq.provenance["root_sha"]:
        raise ContractError("component root SHAs differ")
    if scale.provenance["actlize_sha"] != fq.provenance["actlize_sha"]:
        raise ContractError("component actlize SHAs differ")
    if canonical_json(scale.provenance["device"]) != canonical_json(fq.provenance["device"]):
        raise ContractError("component measured device identities differ")
    if scale.provenance["shape_manifest_sha256"] != fq.provenance["shape_manifest_sha256"]:
        raise ContractError("component shape-manifest hashes differ")
    if canonical_json(scale.provenance["shape_directory"]) != canonical_json(
            fq.provenance["shape_directory"]):
        raise ContractError("component shape-directory authorities differ")
    if scale.provenance["gguf_set_sha256"] != fq.provenance["gguf_set_sha256"]:
        raise ContractError("component GGUF-set digests differ")
    if canonical_json(scale.provenance["gguf_hashes"]) != canonical_json(
            fq.provenance["gguf_hashes"]):
        raise ContractError(
            "component GGUF member maps differ even though their set digest may match")
    cells = scale.cells + fq.cells
    ids = [cell["cell_id"] for cell in cells]
    if len(ids) != len(set(ids)):
        raise ContractError("cell IDs are not globally unique across components")
    validate_algorithm_denominators(cells)
    present_groups = set(cell["ranking_group"] for cell in cells)
    missing_groups = set(RANKING_GROUPS) - present_groups
    if missing_groups:
        raise ContractError(f"missing ranking-group denominator(s): {sorted(missing_groups)}")
    # A group may legitimately contain only explicit UNSUPPORTED cells for a
    # format, but across the complete run each requested board needs at least
    # one actual measurement or it is not an overnight performance sweep.
    unmeasured_groups = {
        group for group in RANKING_GROUPS
        if not any(cell["ranking_group"] == group and cell["status"] == "MEASURED" for cell in cells)
    }
    if unmeasured_groups:
        raise ContractError(f"ranking group(s) have zero measured cells: {sorted(unmeasured_groups)}")
    counts = Counter(cell["status"] for cell in cells)
    by_group = {}
    for group in RANKING_GROUPS:
        group_cells = [cell for cell in cells if cell["ranking_group"] == group]
        by_group[group] = {
            "denominator": len(group_cells),
            "status_counts": {state: sum(cell["status"] == state for cell in group_cells)
                              for state in sorted(VALID_STATES)},
        }
    decisions = make_decisions(cells)
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "COMPLETE",
        "ranking_rule": (
            "rank by median_us only within (ranking_group,model,TP,problem-route,"
            "grouped-identity,group-size,shape-id,qtype,M,N,K,L); "
            "the four ranking groups never compete"),
        "shape_directory": copy.deepcopy(scale.provenance["shape_directory"]),
        "denominator": {
            "total_cells": len(cells),
            "component_cells": {
                "scale_first": scale.denominator,
                "fully_quantized": fq.denominator,
            },
            "ranking_groups": by_group,
            "status_counts": {state: counts.get(state, 0) for state in sorted(VALID_STATES)},
        },
        "components": {
            "scale_first": {"schema": scale.schema, "provenance": scale.provenance},
            "fully_quantized": {"schema": fq.schema, "provenance": fq.provenance},
        },
        "cells": cells,
        "leaderboard_decisions": decisions,
        "winners": [decision for decision in decisions if decision["winner_cell_id"] is not None],
    }


CELL_COLUMNS = (
    "cell_id", "ranking_group", "metric_scope", "status", "reason",
    "model_id", "shape_id", "source_tensors", "tp_world", "tp_rank",
    "tp_partition", "problem_route", "group_size", "grouped", "qtype",
    "tensor", "route", "M", "N", "K", "L",
    "layout", "FoldN_low",
    "FoldN_high", "ArtifactTileK", "algorithm", "config", "S", "policy",
    "grid", "median_us", "MFU_pct", "MBU_pct", "MBU_kind", "raw_samples_us",
    "correctness", "workspace_bytes",
)


def tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return canonical_json(value)
    return str(value)


def cells_tsv(cells: list[dict[str, Any]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CELL_COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for cell in cells:
        shape = cell["shape"]
        row = {
            "cell_id": cell["cell_id"], "ranking_group": cell["ranking_group"],
            "metric_scope": cell["metric_scope"], "status": cell["status"], "reason": cell["reason"],
            "model_id": cell["model_id"], "shape_id": cell["shape_id"],
            "source_tensors": tsv_value(cell["source_tensors"]),
            "tp_world": cell["tp_world"],
            "tp_rank": cell["tp_rank"], "tp_partition": cell["tp_partition"],
            "problem_route": cell["problem_route"], "group_size": cell["group_size"],
            "grouped": tsv_value(cell["grouped"]),
            "qtype": cell["qtype"], "tensor": cell["tensor"], "route": cell["route"],
            "M": shape["m"], "N": shape["n"], "K": shape["k"], "L": shape["l"],
            "layout": tsv_value(cell["layout"]),
            "FoldN_low": tsv_value(cell["fold_n"]["low"] if cell["fold_n"] else None),
            "FoldN_high": tsv_value(cell["fold_n"]["high"] if cell["fold_n"] else None),
            "ArtifactTileK": tsv_value(cell["artifact_tile_k"]),
            "algorithm": cell["algorithm"], "config": cell["config"], "S": cell["split_k_slices"],
            "policy": cell["policy"], "grid": tsv_value(cell["grid"]),
            "median_us": tsv_value(cell["median_us"]), "MFU_pct": tsv_value(cell["mfu_pct"]),
            "MBU_pct": tsv_value(cell["mbu_pct"]), "MBU_kind": tsv_value(cell["mbu_kind"]),
            "raw_samples_us": tsv_value(cell["raw_samples_us"]), "correctness": tsv_value(cell["correctness"]),
            "workspace_bytes": cell["workspace_bytes"],
        }
        writer.writerow(row)
    return stream.getvalue()


WINNER_COLUMNS = (
    "ranking_group", "model_id", "tp_world", "tp_rank", "tp_partition",
    "problem_route", "group_size", "grouped", "shape_id", "source_tensors",
    "qtype", "M", "N", "K", "L",
    "disposition", "candidate_denominator", "candidate_status_counts", "measured_candidates",
    "winner_cell_id", "winner_algorithm", "winner_config",
    "winner_layout", "winner_FoldN", "winner_S", "winner_grid", "winner_median_us",
    "winner_MFU_pct", "winner_MBU_pct", "runner_up_cell_id", "runner_up_algorithm",
    "runner_up_config", "runner_up_median_us", "runner_up_gap_us", "runner_up_gap_pct",
)


def winners_tsv(winners: list[dict[str, Any]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=WINNER_COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for winner in winners:
        shape = winner["shape"]
        writer.writerow({
            "ranking_group": winner["ranking_group"], "qtype": winner["qtype"],
            "model_id": winner["model_id"], "tp_world": winner["tp_world"],
            "tp_rank": winner["tp_rank"], "tp_partition": winner["tp_partition"],
            "problem_route": winner["problem_route"], "group_size": winner["group_size"],
            "grouped": tsv_value(winner["grouped"]),
            "shape_id": winner["shape_id"],
            "source_tensors": tsv_value(winner["source_tensors"]),
            "M": shape["m"], "N": shape["n"],
            "K": shape["k"], "L": shape["l"], "disposition": winner["disposition"],
            "candidate_denominator": winner["candidate_denominator"],
            "candidate_status_counts": tsv_value(winner["candidate_status_counts"]),
            "measured_candidates": winner["measured_candidates"],
            "winner_cell_id": winner["winner_cell_id"], "winner_algorithm": winner["winner_algorithm"],
            "winner_config": winner["winner_config"], "winner_layout": tsv_value(winner["winner_layout"]),
            "winner_FoldN": tsv_value(winner["winner_fold_n"]), "winner_S": winner["winner_S"],
            "winner_grid": tsv_value(winner["winner_grid"]), "winner_median_us": winner["winner_median_us"],
            "winner_MFU_pct": winner["winner_mfu_pct"], "winner_MBU_pct": winner["winner_mbu_pct"],
            "runner_up_cell_id": tsv_value(winner["runner_up_cell_id"]),
            "runner_up_algorithm": tsv_value(winner["runner_up_algorithm"]),
            "runner_up_config": tsv_value(winner["runner_up_config"]),
            "runner_up_median_us": tsv_value(winner["runner_up_median_us"]),
            "runner_up_gap_us": tsv_value(winner["runner_up_gap_us"]),
            "runner_up_gap_pct": tsv_value(winner["runner_up_gap_pct"]),
        })
    return stream.getvalue()


def safe_segment(value: Any, field: str) -> str:
    text_value = str(value)
    if not STABLE_ID_RE.fullmatch(text_value):
        raise ContractError(f"{field} is not folder-safe: {text_value!r}")
    return text_value


def shape_folder_name(cell: dict[str, Any], templates: dict[str, str]) -> str:
    shape = cell["shape"]
    route = cell["problem_route"]
    grouped = cell.get("grouped") or {}
    values = {
        "m": shape["m"], "n": shape["n"], "k": shape["k"],
        "group_size": cell["group_size"],
        "experts": grouped.get("experts", "UNKNOWN"),
        "active": grouped.get("active", "UNKNOWN"),
        "ragged_profile": safe_segment(
            grouped.get("ragged", "unknown"), "grouped.ragged"),
    }
    try:
        rendered = templates[route].format_map(values)
    except (KeyError, ValueError) as exc:
        raise ContractError(f"cannot render shape_directory.{route}: {exc}") from exc
    return safe_segment(rendered, f"shape_directory.{route} output")


def materialize_model_folders(out: pathlib.Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    templates = normalize_shape_directory(
        result.get("shape_directory"), "merged result")
    by_shape: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in result["cells"]:
        by_shape[(cell["model_id"], shape_folder_name(cell, templates))].append(cell)
    decisions_by_shape: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for decision in result["leaderboard_decisions"]:
        probe = {
            "shape": decision["shape"], "group_size": decision["group_size"],
            "problem_route": decision["problem_route"], "grouped": decision["grouped"],
        }
        decisions_by_shape[(
            decision["model_id"], shape_folder_name(probe, templates))].append(decision)

    manifest: list[dict[str, Any]] = []
    models_root = out / "models"
    models_root.mkdir()
    for (model_id, folder), scoped_cells in sorted(by_shape.items()):
        model_dir = models_root / safe_segment(model_id, "model_id")
        model_dir.mkdir(exist_ok=True)
        shape_dir = model_dir / folder
        shape_dir.mkdir()
        scoped_decisions = decisions_by_shape[(model_id, folder)]
        (shape_dir / "cells.tsv").write_text(cells_tsv(scoped_cells))
        (shape_dir / "winners.tsv").write_text(winners_tsv(scoped_decisions))
        scope = {
            "model_id": model_id,
            "shape_folder": folder,
            "problem_routes": sorted({cell["problem_route"] for cell in scoped_cells}),
            "tp_partitions": sorted({
                (cell["tp_world"], cell["tp_rank"], cell["tp_partition"])
                for cell in scoped_cells}),
            "qtypes": sorted({str(cell["qtype"]) for cell in scoped_cells}),
            "cell_count": len(scoped_cells),
            "decision_count": len(scoped_decisions),
        }
        (shape_dir / "scope.json").write_text(
            json.dumps(scope, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        manifest.append({
            "model_id": model_id, "shape_folder": folder,
            "path": str(shape_dir.relative_to(out)),
            "cell_count": len(scoped_cells), "decision_count": len(scoped_decisions),
        })
    return manifest


def ensure_workspace_child(path: pathlib.Path) -> pathlib.Path:
    workspace = pathlib.Path("/workspace").resolve(strict=True)
    candidate = path.resolve(strict=False)
    if candidate == workspace or workspace not in candidate.parents:
        raise ContractError(f"output must be a strict /workspace child: {candidate}")
    return candidate


def synthetic_doc(component: str) -> dict[str, Any]:
    zero = "0" * 64
    gguf_hashes = {"model-a": "6" * 64}
    provenance = {
        "root_sha": "1" * 40,
        "actlize_sha": "2" * 40,
        "device": {"uuid": "PPU-test-0", "cu": 72, "driver": "test"},
        "source_hashes": {"authority": zero},
        "binary_hashes": {component: "3" * 64},
        "shape_manifest_sha256": "5" * 64,
        "gguf_hashes": gguf_hashes,
        "gguf_set_sha256": hashlib.sha256(
            canonical_json(gguf_hashes).encode("utf-8")).hexdigest(),
        "shape_directory": {
            "dense": "m{m}_n{n}_k{k}_g{group_size}",
            "grouped": (
                "m{m}_n{n}_k{k}_g{group_size}_e{experts}_a{active}_{ragged_profile}"),
        },
    }
    common = {
        "model_id": "model-a", "tp_world": 1, "tp_rank": 0,
        "tp_partition": "replicated", "problem_route": "dense", "group_size": 32,
        "shape_id": "dense-m1-n4096-k4096-g32",
        "source_tensors": ["blk.0.attn_q.weight"],
        "qtype": "Q4_K", "tensor": "blk.0.attn_q.weight",
        "shape": {"m": 1, "n": 4096, "k": 4096, "l": 1},
        "layout": {"name": "xplane-q4-a64-f1"}, "FoldN": [1, 1],
        "ArtifactTileK": 64, "config": "8x128x128_w8x16_s2_bc0",
        "raw_samples_us": [10.0, 10.2, 10.1], "median_us": 10.1,
        "MFU_pct": 10.0, "distinct_MBU_model_pct": 20.0, "correctness": "RAW-BIT/PASS",
        "status": "MEASURED", "grid": 72,
    }
    if component == "scale_first":
        cells = [
            dict(common, algorithm="non-persistent", policy="non-persistent"),
            dict(common, algorithm="persistent", policy="capacity", grid=72,
                 median_us=9.8, raw_samples_us=[9.7, 9.8, 9.9]),
            dict(common, algorithm="persistent", policy="balanced", grid=64,
                 median_us=9.9, raw_samples_us=[9.8, 9.9, 10.0]),
            dict(common, algorithm="scale-first-splitk", S=2, metric_scope=PRODUCER_ONLY,
                 config="sf-split2", median_us=8.7, raw_samples_us=[8.6, 8.7, 8.8]),
            dict(common, algorithm="scale-first-splitk", S=4, metric_scope=PRODUCER_ONLY,
                 config="sf-split4", median_us=8.4, raw_samples_us=[8.3, 8.4, 8.5]),
            dict(common, algorithm="scale-first-splitk", S=8, metric_scope=PRODUCER_ONLY,
                 config="sf-split8", median_us=8.6, raw_samples_us=[8.5, 8.6, 8.7]),
        ]
    else:
        cells = [
            dict(common, algorithm="bc-gemv", config="bc-cfg", median_us=9.0,
                 raw_samples_us=[8.9, 9.0, 9.1]),
            dict(common, algorithm="tc-s1", config="tc-cfg"),
            dict(common, algorithm="tc-splitk", S=2, metric_scope=PRODUCER_ONLY,
                 config="split-cfg", median_us=8.0, raw_samples_us=[7.9, 8.0, 8.1]),
            dict(common, algorithm="tc-splitk", S=4, metric_scope=PRODUCER_ONLY,
                 config="split-reject", status="INADMISSIBLE", reason="PIPELINE_DEPTH",
                 raw_samples_us=None, median_us=None, MFU_pct=None,
                 distinct_MBU_model_pct=None, correctness=None),
            dict(common, algorithm="tc-splitk", S=8, metric_scope=PRODUCER_ONLY,
                 config="split-build-reject", status="BUILD_REJECT", reason="COMPILER_EVIDENCE_ID=17",
                 raw_samples_us=None, median_us=None, MFU_pct=None,
                 distinct_MBU_model_pct=None, correctness=None),
            dict(common, qtype="Q8_0", algorithm="tc-s1", config="unsupported-q8",
                 status="UNSUPPORTED", reason="NO_SHIPPING_FQ_Q8_READER",
                 raw_samples_us=None, median_us=None, MFU_pct=None,
                 distinct_MBU_model_pct=None, correctness=None),
        ]
        for algorithm, split in (("bc-gemv", 1), ("tc-splitk", 2),
                                 ("tc-splitk", 4), ("tc-splitk", 8)):
            cells.append(dict(
                common, qtype="Q8_0", algorithm=algorithm, S=split,
                metric_scope=PRODUCER_ONLY if split > 1 else FULL_OUTPUT,
                config=f"unsupported-q8-s{split}-{algorithm}", status="UNSUPPORTED",
                reason="NO_SHIPPING_FQ_Q8_READER", raw_samples_us=None,
                median_us=None, MFU_pct=None, distinct_MBU_model_pct=None,
                correctness=None))
        for rejected in cells[3:]:
            for key in ("raw_samples_us", "median_us", "MFU_pct", "distinct_MBU_model_pct", "correctness"):
                rejected.pop(key)
        for rejected in cells[5:]:
            for key in ("layout", "FoldN", "ArtifactTileK", "config"):
                rejected.pop(key)
    return {
        "schema": f"synthetic.{component}.v1", "component": component,
        "status": "COMPLETE", "expected_cells": len(cells),
        "status_counts": dict(Counter(cell["status"] for cell in cells)),
        "missing": [], "failures": [], "provenance": provenance, "cells": cells,
    }


def component_from_doc(doc: dict[str, Any], component: str) -> Component:
    """In-memory equivalent of load_component, used only by negative controls."""
    denominator = extract_denominator(doc)
    if doc.get("status") != "COMPLETE" or doc.get("missing") or doc.get("failures"):
        raise ContractError("synthetic component incomplete")
    if len(doc["cells"]) != denominator:
        raise ContractError("synthetic denominator mismatch")
    cells = [normalize_cell(component, doc["schema"], cell, i) for i, cell in enumerate(doc["cells"])]
    counts = dict(Counter(cell["status"] for cell in cells))
    provenance = copy.deepcopy(doc["provenance"])
    provenance.update({"component_summary": "synthetic", "component_summary_sha256": "4" * 64})
    return Component(component, doc["schema"], denominator, counts, cells, provenance)


def expect_red(label: str, action: Any) -> None:
    try:
        action()
    except ContractError:
        return
    raise AssertionError(f"negative control stayed green: {label}")


def self_test() -> None:
    scale_doc = synthetic_doc("scale_first")
    fq_doc = synthetic_doc("fully_quantized")
    scale = component_from_doc(scale_doc, "scale_first")
    fq = component_from_doc(fq_doc, "fully_quantized")
    merged = merge_components(scale, fq)
    assert merged["denominator"]["total_cells"] == 16
    assert merged["denominator"]["status_counts"] == {
        "BUILD_REJECT": 1, "INADMISSIBLE": 1, "MEASURED": 9, "UNSUPPORTED": 5}
    assert {winner["ranking_group"] for winner in merged["winners"]} == set(RANKING_GROUPS)
    assert len(merged["leaderboard_decisions"]) == 6
    q8 = [decision for decision in merged["leaderboard_decisions"] if decision["qtype"] == "Q8_0"]
    assert len(q8) == 2 and all(item["disposition"] == "NO_MEASURED_CANDIDATE" for item in q8)
    assert sum(item["candidate_denominator"] for item in q8) == 5
    fq_full = next(winner for winner in merged["winners"]
                   if winner["ranking_group"] == "FULLY_QUANTIZED_FULL_OUTPUT")
    assert fq_full["winner_algorithm"] == "FQ_BC_GEMV"
    assert fq_full["runner_up_algorithm"] == "FQ_TC_S1"
    assert "\trunner_up_gap_us\t" in winners_tsv(merged["leaderboard_decisions"]).splitlines()[0]

    unsupported_g0 = copy.deepcopy(scale_doc["cells"][0])
    unsupported_g0.update(
        group_size=0, status="UNSUPPORTED", reason="QTYPE_NOT_REGISTERED")
    for key in ("raw_samples_us", "median_us", "MFU_pct",
                "distinct_MBU_model_pct", "correctness"):
        unsupported_g0.pop(key, None)
    assert normalize_cell(
        "scale_first", scale_doc["schema"], unsupported_g0, 0)["group_size"] == 0
    measured_g0 = copy.deepcopy(scale_doc["cells"][0])
    measured_g0["group_size"] = 0
    expect_red("measured g0", lambda: normalize_cell(
        "scale_first", scale_doc["schema"], measured_g0, 0))

    layer_a = copy.deepcopy(scale.cells[0])
    layer_b = copy.deepcopy(layer_a)
    layer_b["cell_id"] = "same-shape-other-layer"
    layer_b["tensor"] = "blk.17.attn_q.weight"
    layer_b["source_tensors"] = ["blk.17.attn_q.weight"]
    assert ranking_key(layer_a) == ranking_key(layer_b)
    dedup = make_decisions([layer_a, layer_b])
    assert len(dedup) == 1
    assert dedup[0]["source_tensors"] == [
        "blk.0.attn_q.weight", "blk.17.attn_q.weight"]
    for label, mutation in (
            ("qtype", {"qtype": "Q5_K"}),
            ("TP", {"tp_world": 2, "tp_partition": "n-shard"}),
            ("route", {
                "problem_route": "grouped",
                "grouped": {"experts": 256, "active": 8, "ragged": "one-heavy"}})):
        isolated = copy.deepcopy(layer_a)
        isolated.update(mutation)
        assert ranking_key(layer_a) != ranking_key(isolated), label

    changed_templates = copy.deepcopy(merged["shape_directory"])
    changed_templates["dense"] = "shape-m{m}_n{n}_k{k}_g{group_size}"
    original_folder = shape_folder_name(layer_a, merged["shape_directory"])
    changed_folder = shape_folder_name(layer_a, changed_templates)
    assert original_folder != changed_folder and changed_folder.startswith("shape-")
    mismatched_template = component_from_doc(fq_doc, "fully_quantized")
    mismatched_template.provenance["shape_directory"]["dense"] = (
        "shape-m{m}_n{n}_k{k}_g{group_size}")
    expect_red("shape-directory authority mismatch", lambda: merge_components(
        scale, mismatched_template))

    dropped = copy.deepcopy(fq_doc)
    dropped["cells"].pop()
    expect_red("missing denominator cell", lambda: component_from_doc(dropped, "fully_quantized"))
    omitted = copy.deepcopy(fq_doc)
    omitted["cells"] = [cell for cell in omitted["cells"]
                        if not (cell["qtype"] == "Q4_K" and cell["algorithm"] == "tc-splitk" and cell.get("S") == 8)]
    omitted["expected_cells"] = len(omitted["cells"])
    omitted["status_counts"] = dict(Counter(cell["status"] for cell in omitted["cells"]))
    expect_red("whole algorithm omitted", lambda: merge_components(
        scale, component_from_doc(omitted, "fully_quantized")))
    omitted_sf = copy.deepcopy(scale_doc)
    omitted_sf["cells"] = [cell for cell in omitted_sf["cells"]
                           if not (cell["algorithm"] == "scale-first-splitk" and cell.get("S") == 4)]
    omitted_sf["expected_cells"] = len(omitted_sf["cells"])
    omitted_sf["status_counts"] = dict(Counter(cell["status"] for cell in omitted_sf["cells"]))
    expect_red("whole ScaleFirst S4 algorithm omitted", lambda: merge_components(
        component_from_doc(omitted_sf, "scale_first"), fq))
    extra_q8 = copy.deepcopy(fq_doc)
    planted = copy.deepcopy(next(
        cell for cell in extra_q8["cells"]
        if cell["qtype"] == "Q8_0" and cell["algorithm"] == "tc-s1"))
    planted["config"] = "unsupported-q8-extra-tc-s1"
    extra_q8["cells"].append(planted)
    extra_q8["expected_cells"] = len(extra_q8["cells"])
    extra_q8["status_counts"] = dict(Counter(
        cell["status"] for cell in extra_q8["cells"]))
    expect_red("extra Q8 unsupported cell", lambda: merge_components(
        scale, component_from_doc(extra_q8, "fully_quantized")))
    unknown = copy.deepcopy(fq_doc)
    unknown["cells"][0]["status"] = "MISSING"
    expect_red("unknown terminal status", lambda: component_from_doc(unknown, "fully_quantized"))
    mixed = copy.deepcopy(fq_doc)
    mixed["cells"][2]["metric_scope"] = FULL_OUTPUT
    expect_red("producer/full-output scope mixing", lambda: component_from_doc(mixed, "fully_quantized"))
    mismatched = component_from_doc(fq_doc, "fully_quantized")
    mismatched.provenance["device"] = {"uuid": "PPU-other", "cu": 72, "driver": "test"}
    expect_red("mixed device provenance", lambda: merge_components(scale, mismatched))
    same_digest_other_member = component_from_doc(fq_doc, "fully_quantized")
    same_digest_other_member.provenance["gguf_hashes"] = {"model-b": "6" * 64}
    # Deliberately preserve the old digest: exact member-map comparison, not a
    # caller-supplied aggregate token, must catch the substitution.
    expect_red("same GGUF-set digest but different member map", lambda: merge_components(
        scale, same_digest_other_member))
    missing_gguf_member = component_from_doc(fq_doc, "fully_quantized")
    scale_with_two_models = copy.deepcopy(scale)
    scale_with_two_models.provenance["gguf_hashes"] = {
        "model-a": "6" * 64, "model-b": "7" * 64}
    # Again forge equality of the aggregate digest so only the exact map check
    # can reject the missing model.
    scale_with_two_models.provenance["gguf_set_sha256"] = (
        missing_gguf_member.provenance["gguf_set_sha256"])
    expect_red("same GGUF-set digest but one model missing", lambda: merge_components(
        scale_with_two_models, missing_gguf_member))
    no_runner = copy.deepcopy(fq_doc)
    tc_s1 = next(cell for cell in no_runner["cells"]
                 if cell["qtype"] == "Q4_K" and cell["algorithm"] == "tc-s1")
    tc_s1.update(status="UNSUPPORTED", reason="PLANTED_NO_TC_S1")
    for key in ("raw_samples_us", "median_us", "MFU_pct", "distinct_MBU_model_pct", "correctness"):
        tc_s1.pop(key)
    no_runner["status_counts"] = dict(Counter(cell["status"] for cell in no_runner["cells"]))
    merged_one = merge_components(scale, component_from_doc(no_runner, "fully_quantized"))
    full = next(winner for winner in merged_one["winners"]
                if winner["ranking_group"] == "FULLY_QUANTIZED_FULL_OUTPUT")
    assert full["runner_up_cell_id"] is None
    print("[internal-full-sweep:self-test] PASS positive=4-separated-leaderboards+shape-layer-dedup+unsupported-g0 "
          "negative=missing-cell+missing-fq-algorithm+missing-scalefirst-s4+extra-q8-cell+unknown-state+scope-mix+device-mix+gguf-map-substitution+gguf-member-loss "
          "shape-template-mismatch+identity-isolation runner-up=BOUND")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    merge = sub.add_parser("merge")
    merge.add_argument("--scale-first", type=pathlib.Path, required=True)
    merge.add_argument("--fully-quantized", type=pathlib.Path, required=True)
    merge.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        out = ensure_workspace_child(args.out)
        if out.exists():
            raise ContractError(f"refusing to overwrite output directory: {out}")
        scale = load_component(args.scale_first.resolve(strict=True), "scale_first")
        fq = load_component(args.fully_quantized.resolve(strict=True), "fully_quantized")
        result = merge_components(scale, fq)
        out.mkdir(parents=True, exist_ok=False)
        result["model_outputs"] = materialize_model_folders(out, result)
        (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        (out / "cells.tsv").write_text(cells_tsv(result["cells"]))
        (out / "winners.tsv").write_text(winners_tsv(result["leaderboard_decisions"]))
        print(f"[internal-full-sweep] PASS denominator={result['denominator']['total_cells']} "
              f"winners={len(result['winners'])} out={out}")
        return 0
    except (ContractError, FileNotFoundError, OSError, AssertionError) as exc:
        print(f"[internal-full-sweep] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
