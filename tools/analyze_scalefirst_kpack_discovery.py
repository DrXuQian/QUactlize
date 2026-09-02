#!/usr/bin/env python3
"""Retain and summarize canonical K-pack ScaleFirst dense/grouped discovery.

Screening may remove only rows with no admissible full-output measurement; it
never ranks timings or truncates to a top-N.  Confirmation reuses the same
precompiled shard through a symbol file and measures every raw-bit-clean,
structurally available symbol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import statistics
import sys
from collections import defaultdict


QTYPES = (10, 11, 12, 13, 14)
LAYOUT = {10: 2, 11: 2, 12: 1, 13: 2, 14: 2}
MAPPING = {1: "0x51344b5034540001", 2: "0x514b504b54000001"}
Q4_HISTORICAL_SOURCE_SYMBOL = \
    "sf_q12_a64_tm64_tn64_tk64_wm64_wn32_s3_bc0"
Q4_HISTORICAL_GEOMETRY_ANCHOR = \
    "sf_q12_a0_tm64_tn64_tk64_wm64_wn32_s3_bc0_ap0_dn64"
def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurement_contract(dense_shape: str, grouped_tokens: int,
                         grouped_topk: int, grouped_experts: int,
                         grouped_n: int, grouped_k: int,
                         screen_iterations: int, confirm_iterations: int,
                         correctness_repeats: int) -> dict:
    try:
        dense = [int(value) for value in dense_shape.split("x")]
    except ValueError as error:
        raise ValueError("dense shape must be positive MxNxK") from error
    values = (grouped_tokens, grouped_topk, grouped_experts, grouped_n,
              grouped_k, screen_iterations, confirm_iterations,
              correctness_repeats)
    if len(dense) != 3 or any(value <= 0 for value in dense) or \
            any(not isinstance(value, int) or isinstance(value, bool) or
                value <= 0 for value in values) or \
            grouped_topk > grouped_experts:
        raise ValueError("ScaleFirst measurement controls are inadmissible")
    return {
        "dense_shape": dense_shape,
        "grouped": {"tokens": grouped_tokens, "topk": grouped_topk,
                    "experts": grouped_experts, "n": grouped_n,
                    "k": grouped_k},
        "screen_iterations": screen_iterations,
        "confirm_iterations": confirm_iterations,
        "correctness_repeats": correctness_repeats,
        "dense_algorithm": "full-output",
        "algorithm_denominator": ["NONPERSISTENT", "PERSISTENT"],
    }


def write_stable_json(path: pathlib.Path, document: dict) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError("result authority changed on resume")
        return
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def kv_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in shlex.split(line):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def dense_candidates(path: pathlib.Path) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    fixed: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("SF_CELL "):
            continue
        row = json.loads(line[len("SF_CELL "):])
        algorithm = str(row.get("algorithm", ""))
        full = row.get("metric_scope") == "FULL_OUTPUT" and algorithm in {
            "NONPERSISTENT", "PERSISTENT"}
        if (row.get("status") != "MEASURED" or not full or
                int(row.get("raw_bad", 1)) != 0 or
                float(row.get("sample_us", 0)) <= 0):
            continue
        key = (row.get("shape"), row["symbol"], row["algorithm"], row["policy"],
               int(row["grid"]))
        grouped[key].append(float(row["sample_us"]))
        fixed[key] = row
    out = []
    for key, samples in grouped.items():
        row = dict(fixed[key])
        producer_us = statistics.median(samples)
        row["median_us"] = producer_us
        row["samples"] = len(samples)
        out.append(row)
    return out


def grouped_candidates(path: pathlib.Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.startswith("SF_GROUPED_CELL "):
            continue
        row = kv_line(line)
        if (row.get("state") != "MEASURED" or
                int(row.get("raw_bad", "1")) != 0 or
                float(row.get("median_us", "0")) <= 0):
            continue
        out.append({
            "qtype": int(row["q"]), "symbol": row["symbol"],
            "config": row["config"], "algorithm": row["algorithm"],
            "policy": row["policy"], "grid": int(row["grid"]),
            "median_us": float(row["median_us"]),
            "samples": 1,
        })
    return out


def load_candidates(operator: str, path: pathlib.Path) -> list[dict]:
    return dense_candidates(path) if operator == "dense" else grouped_candidates(path)


def product_census(operator: str, path: pathlib.Path) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if operator == "dense" and line.startswith("SF_CELL "):
            row = json.loads(line[len("SF_CELL "):])
            if row.get("metric_scope") == "FULL_OUTPUT" and \
                    row.get("algorithm") in {"NONPERSISTENT", "PERSISTENT"}:
                records[str(row["symbol"])].append(row)
        elif operator == "grouped" and line.startswith("SF_GROUPED_CELL "):
            row = kv_line(line)
            if row.get("scope") == "FULL_OUTPUT" and row.get("algorithm") in {
                    "GROUPED_NONPERSISTENT", "GROUPED_PERSISTENT"}:
                records[str(row["symbol"])].append(row)
    return records


def validate_manifest(operator: str, qtype: int, path: pathlib.Path) -> dict:
    doc = json.loads(path.read_text())
    identity = doc.get("identity", {})
    layout = LAYOUT[qtype]
    if operator == "dense":
        exact = (doc.get("schema") == "quactlize.scalefirst.generated_shard.v3" and
                 identity.get("qtype") == qtype and
                 identity.get("artifact_tile_k") == 0 and
                 identity.get("bchunk") == 0 and
                 identity.get("weight_layout") == layout)
        typed = doc.get("denominator", {}).get("typed_rows", 0)
        authority_typed = doc.get("denominator", {}).get(
            "authority_typed_rows", 0)
        exact = exact and doc.get("runtime", {}).get("cuda") == {
            "status": "STRUCTURAL_UNAVAILABLE",
            "reason": "NO_CANONICAL_KPACK_CUDA_READER",
        }
        exact = exact and doc.get("runtime", {}).get("full_output") == [
            "NONPERSISTENT", "PERSISTENT_CAPACITY_BALANCED"]
        exact = exact and doc.get("runtime", {}).get("producer_only") == [
            "SPLITK_S2_PRODUCER", "SPLITK_S4_PRODUCER",
            "SPLITK_S8_PRODUCER"]
    else:
        exact = (doc.get("schema") ==
                 "quactlize.scalefirst_grouped_kpack_shard.v2" and
                 identity.get("qtype") == qtype and
                 identity.get("artifact_tile_k") == 0 and
                 identity.get("weight_layout") == layout and
                 identity.get("quant_mode") == "ScaleZero" and
                 identity.get("metadata_planes") == 2)
        denominator = doc.get("denominator", {})
        typed = denominator.get("compiled_rows", 0)
        authority_typed = denominator.get("authority_typed_rows", 0)
        exact = exact and doc.get("algorithms", {}).get("full_output") == {
            "nonpersistent": "RAW_BIT_THEN_TIMING",
            "persistent": "RAW_BIT_THEN_TIMING",
        }
        exact = exact and doc.get("algorithms", {}).get("persistent") == \
            "AVAILABLE_RUNTIME_EXACT_OCCUPANCY_CAPACITY_BALANCED"
        exact = exact and doc.get("algorithms", {}).get("cuda") == {
            "status": "STRUCTURAL_UNAVAILABLE",
            "reason": "NO_CANONICAL_KPACK_CUDA_READER",
        }
        exact = exact and doc.get("algorithms", {}).get("split_k") == {
            "S2": "STRUCTURAL_UNAVAILABLE",
            "S4": "STRUCTURAL_UNAVAILABLE",
            "S8": "STRUCTURAL_UNAVAILABLE",
        }
    parent_range = doc.get("parent_range") or {}
    selection = doc.get("selection") or {}
    begin, end = parent_range.get("begin"), parent_range.get("end")
    parents = doc.get("compiled_parents")
    typed_rows = doc.get("typed_rows")
    exact = exact and isinstance(begin, int) and isinstance(end, int) and \
        isinstance(authority_typed, int) and 0 <= begin < end <= authority_typed
    exact = exact and parent_range == {
        "begin": begin, "end": end, "count": end - begin,
        "authority_count": authority_typed}
    mode = selection.get("mode")
    exact = exact and mode in {"authority-full", "parent-range",
                               "exact-symbol"}
    exact = exact and all(selection.get(key) == value for key, value in {
        "begin": begin, "end": end,
        "authority_typed_rows": authority_typed,
        "compiled_rows": end - begin,
    }.items())
    if mode == "authority-full":
        exact = exact and begin == 0 and end == authority_typed
    elif mode == "parent-range":
        exact = exact and not (begin == 0 and end == authority_typed)
    else:
        exact = exact and end == begin + 1 and isinstance(
            selection.get("symbol"), str)
    exact = exact and isinstance(parents, list) and isinstance(typed_rows, list)
    if exact:
        expected_parents = [
            {"parent_id": parent_id, "symbol": row.get("symbol"),
             "static_candidate_id": row.get("static_candidate_id"),
             "a_provider": row.get("a_provider"),
             "resolved_delivery_n": row.get("resolved_delivery_n")}
            for parent_id, row in zip(range(begin, end), typed_rows)]
        exact = (typed == end - begin == len(typed_rows) and
                 parents == expected_parents and
                 [row.get("parent_id") for row in typed_rows] ==
                 list(range(begin, end)) and
                 all(row.get("static_candidate_id") == row.get("symbol") and
                     row.get("a_provider") in (0, 1) and
                     row.get("resolved_delivery_n") in (16, 32, 64)
                     for row in typed_rows))
    if operator == "dense":
        rejected = doc.get("non_typed_authority") or {}
        rejected_count = doc.get("denominator", {}).get("non_typed_rows")
        exact = exact and rejected == {
            "count": rejected_count,
            "sha256": rejected.get("sha256"),
            "encoding": "JSON_SORT_KEYS_COMPACT_V1",
        }
        exact = exact and isinstance(rejected_count, int) and \
            rejected_count >= 0 and isinstance(rejected.get("sha256"), str) and \
            len(rejected["sha256"]) == 64
        expanded = doc.get("non_typed_rows")
        if mode == "authority-full":
            exact = exact and isinstance(expanded, list) and \
                len(expanded) == rejected_count
            if exact:
                encoded = json.dumps(
                    expanded, sort_keys=True,
                    separators=(",", ":")).encode()
                exact = hashlib.sha256(encoded).hexdigest() == rejected["sha256"]
        else:
            exact = exact and "non_typed_rows" not in doc
    if not exact or not isinstance(typed, int) or typed <= 0:
        raise ValueError(f"q{qtype}/{operator}: manifest is not one canonical parent shard")
    return doc


def retain(operator: str, qtype: int, screen: pathlib.Path,
           manifest: pathlib.Path, output: pathlib.Path,
           sidecar: pathlib.Path) -> None:
    doc = validate_manifest(operator, qtype, manifest)
    candidates = load_candidates(operator, screen)
    symbols = sorted({row["symbol"] for row in candidates})
    if not symbols:
        raise ValueError(f"q{qtype}/{operator}: screen has no raw-bit-clean measurement")
    authority_symbols = {row["symbol"] for row in doc["typed_rows"]}
    records = product_census(operator, screen)
    observed_symbols = set(records)
    if observed_symbols != authority_symbols:
        missing = sorted(authority_symbols - observed_symbols)[:4]
        extra = sorted(observed_symbols - authority_symbols)[:4]
        raise ValueError(
            f"q{qtype}/{operator}: product census differs; missing={missing} extra={extra}")
    structural = []
    for symbol in sorted(authority_symbols - set(symbols)):
        cells = records[symbol]
        explicit = (all(row.get("status") == "INADMISSIBLE" for row in cells)
                    if operator == "dense" else
                    all(row.get("state") in {
                        "INADMISSIBLE_SHARED_STORAGE",
                        "INADMISSIBLE_OCCUPANCY"} for row in cells))
        if not explicit:
            raise ValueError(
                f"q{qtype}/{operator}: {symbol} lacks a clean measurement "
                "and is not explicitly structural-unavailable")
        structural.append(symbol)
    anchor_status = "NOT_APPLICABLE"
    if operator == "dense" and qtype == 12:
        authority = {row["symbol"] for row in doc["typed_rows"]}
        if Q4_HISTORICAL_GEOMETRY_ANCHOR in authority:
            anchor_status = ("RETAINED_CANDIDATE" if
                             Q4_HISTORICAL_GEOMETRY_ANCHOR in symbols
                             else "STRUCTURAL_UNAVAILABLE")
        else:
            anchor_status = "OUTSIDE_THIS_PARENT_RANGE"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{symbol}\n" for symbol in symbols))
    typed = (doc["denominator"]["typed_rows"] if operator == "dense" else
             doc["denominator"]["compiled_rows"])
    payload = {
        "schema": "quactlize.scalefirst_kpack_retention.v1",
        "operator": operator, "qtype": qtype, "layout": LAYOUT[qtype],
        "mapping_id": MAPPING[LAYOUT[qtype]],
        "parent_range": doc["parent_range"],
        "parent_ids": [row["parent_id"] for row in doc["compiled_parents"]],
        "manifest_sha256": sha256(manifest), "screen_sha256": sha256(screen),
        "authority_typed_symbols": typed,
        "retained_symbols": symbols,
        "retained_count": len(symbols),
        "structural_unavailable_symbols": structural,
        "structural_unavailable_count": len(structural),
        "elimination_rule": "NO_RAW_BIT_CLEAN_PRODUCT_FULL_OUTPUT_MEASUREMENT",
        "timing_rank_used_for_elimination": False,
        "heuristic_algorithm_denominator": ["NONPERSISTENT", "PERSISTENT"],
        "split_k_policy": "EXCLUDED_DIAGNOSTIC_ONLY",
        "q4_historical_geometry_anchor_status": anchor_status,
        "q4_historical_geometry_translation": {
            "source_xplane_symbol": Q4_HISTORICAL_SOURCE_SYMBOL,
            "canonical_kpack_candidate": Q4_HISTORICAL_GEOMETRY_ANCHOR,
            "authority": "CANDIDATE_ONLY_NOT_WINNER",
        },
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def summarize(root: pathlib.Path, output_json: pathlib.Path,
              output_tsv: pathlib.Path,
              bundle_index: pathlib.Path | None = None) -> None:
    if bundle_index is None:
        scope = "legacy-unsharded"
        entries = [
            {"shard_id": f"q{qtype}-{operator}", "qtype": qtype,
             "operator": operator}
            for qtype in QTYPES for operator in ("dense", "grouped")]
    else:
        index = json.loads(bundle_index.read_text())
        scope = index.get("scope")
        if index.get("schema") != \
                "quactlize.scalefirst_kpack_prebuilt_bundle.v2" or \
                index.get("route") != "scalefirst" or \
                scope not in {"full", "pilot"}:
            raise ValueError("summary bundle index identity differs")
        entries = index.get("shards")
        if not isinstance(entries, list) or not entries:
            raise ValueError("summary bundle index has no parent shards")

    grouped_entries: dict[tuple[int, str], list[dict]] = defaultdict(list)
    seen_shards: set[str] = set()
    for entry in entries:
        shard_id = entry.get("shard_id")
        qtype, operator = entry.get("qtype"), entry.get("operator")
        if not isinstance(shard_id, str) or shard_id in seen_shards or \
                qtype not in QTYPES or operator not in {"dense", "grouped"}:
            raise ValueError("summary bundle shard identity differs")
        seen_shards.add(shard_id)
        log = root / f"{shard_id}.confirm.log"
        retention = root / f"{shard_id}.retention.json"
        if not log.is_file() or not retention.is_file():
            raise ValueError(f"missing confirm authority for {shard_id}")
        candidates = load_candidates(operator, log)
        if not candidates:
            raise ValueError(f"{shard_id}: confirm has no clean candidate")
        retained = json.loads(retention.read_text())
        retained_symbols = set(retained.get("retained_symbols", []))
        observed_symbols = set(product_census(operator, log))
        clean_symbols = {row["symbol"] for row in candidates}
        if (retained.get("qtype"), retained.get("operator")) != \
                (qtype, operator) or observed_symbols != retained_symbols or \
                clean_symbols != retained_symbols:
            raise ValueError(
                f"{shard_id}: confirm census/clean set differs from retention")
        for candidate in candidates:
            candidate = dict(candidate)
            candidate["shard_id"] = shard_id
            candidate["confirm_sha256"] = sha256(log)
            candidate["retention_sha256"] = sha256(retention)
            grouped_entries[(qtype, operator)].append(candidate)

    rows = []
    for (qtype, operator), candidates in grouped_entries.items():
        winner = min(candidates, key=lambda row: (
            row["median_us"], row["symbol"], row["shard_id"]))
        rows.append({
            "qtype": qtype, "operator": operator,
            "layout": LAYOUT[qtype], "mapping_id": MAPPING[LAYOUT[qtype]],
            "shard_id": winner["shard_id"],
            "symbol": winner["symbol"], "config": winner.get("config", ""),
            "algorithm": winner["algorithm"], "policy": winner["policy"],
            "grid": winner["grid"], "median_us": winner["median_us"],
            "confirm_sha256": winner["confirm_sha256"],
            "retention_sha256": winner["retention_sha256"],
        })
    rows.sort(key=lambda row: (row["qtype"], row["operator"]))
    expected_pairs = ({(qtype, operator) for qtype in QTYPES
                       for operator in ("dense", "grouped")} if scope in {
                           "full", "legacy-unsharded"} else
                      {(10, "dense"), (10, "grouped")})
    if set(grouped_entries) != expected_pairs:
        raise ValueError("summary qtype/operator denominator differs from scope")
    payload = {
        "schema": "quactlize.scalefirst_kpack_discovery.v1",
        "scope": scope,
        "formats": sorted({row["qtype"] for row in rows}),
        "operators": sorted({row["operator"] for row in rows}),
        "parent_shards": len(entries),
        "q4_historical_geometry_translation": {
            "source_xplane_symbol": Q4_HISTORICAL_SOURCE_SYMBOL,
            "canonical_kpack_candidate": Q4_HISTORICAL_GEOMETRY_ANCHOR,
            "authority": "CANDIDATE_ONLY_NOT_WINNER",
        },
        "heuristic_algorithm_denominator": ["NONPERSISTENT", "PERSISTENT"],
        "split_k_policy": "EXCLUDED_DIAGNOSTIC_ONLY",
        "winner_contract": "MEASURED_FULL_OUTPUT_ONLY",
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    columns = ("qtype", "operator", "layout", "mapping_id", "shard_id", "symbol",
               "config", "algorithm", "policy", "grid", "median_us")
    output_tsv.write_text("\t".join(columns) + "\n" + "".join(
        "\t".join(str(row[column]) for column in columns) + "\n" for row in rows))


def self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="qz-sf-kpack-analysis-") as temp:
        root = pathlib.Path(temp)
        dense = root / "dense.log"
        dense.write_text("\n".join([
            "SF_CELL " + json.dumps({
                "symbol": "fast", "algorithm": "NONPERSISTENT",
                "metric_scope": "FULL_OUTPUT", "policy": "ordinary", "grid": 0,
                "status": "MEASURED", "raw_bad": 0, "sample_us": value,
            }) for value in (2.0, 1.0, 3.0)
        ] + ["SF_CELL " + json.dumps({
            "symbol": "slow", "algorithm": "NONPERSISTENT",
            "metric_scope": "FULL_OUTPUT", "policy": "ordinary", "grid": 0,
            "status": "MEASURED", "raw_bad": 0, "sample_us": 1000.0,
        })] + ["SF_CELL " + json.dumps({
            "symbol": "red", "algorithm": "NONPERSISTENT",
            "metric_scope": "FULL_OUTPUT", "policy": "ordinary", "grid": 0,
            "status": "MEASURED", "raw_bad": 1, "sample_us": .1,
        })] + ["SF_CELL " + json.dumps({
            "shape": "1x1024x5120", "symbol": "split",
            "algorithm": "SPLITK_S4_PRODUCER",
            "metric_scope": "PRODUCER_ONLY_NOT_PRODUCT_E2E",
            "policy": "fixed-split-k", "grid": 0, "status": "MEASURED",
            "raw_bad": 0, "sample_us": 5.0,
            "reducer_correctness_untimed": 1,
        })] + ["SF_CELL " + json.dumps({
            "shape": "1x1024x5120", "symbol": "split-red",
            "algorithm": "SPLITK_S4_PRODUCER",
            "metric_scope": "PRODUCER_ONLY_NOT_PRODUCT_E2E",
            "policy": "fixed-split-k", "grid": 0, "status": "MEASURED",
            "raw_bad": 0, "sample_us": .01,
            "reducer_correctness_untimed": 0,
        })]) + "\n")
        if len(dense_candidates(dense)) != 2 or \
                dense_candidates(dense)[0]["median_us"] != 2.0:
            raise AssertionError("dense raw-bit/sample denominator negative failed")
        retention_dense = root / "dense-retention.log"
        retention_dense.write_text("\n".join([
            "SF_CELL " + json.dumps({
                "symbol": "fast", "algorithm": "NONPERSISTENT",
                "metric_scope": "FULL_OUTPUT", "policy": "ordinary",
                "grid": 0, "status": "MEASURED", "raw_bad": 0,
                "sample_us": 2.0}),
            "SF_CELL " + json.dumps({
                "symbol": "slow", "algorithm": "NONPERSISTENT",
                "metric_scope": "FULL_OUTPUT", "policy": "ordinary",
                "grid": 0, "status": "MEASURED", "raw_bad": 0,
                "sample_us": 1000.0}),
            "SF_CELL " + json.dumps({
                "symbol": "structural", "algorithm": "NONPERSISTENT",
                "metric_scope": "FULL_OUTPUT", "policy": "ordinary",
                "grid": 0, "status": "INADMISSIBLE",
                "reason": "INADMISSIBLE_SHIPPING_SMEM", "raw_bad": 0,
                "sample_us": 0.0}),
        ]) + "\n")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": "quactlize.scalefirst.generated_shard.v3",
            "identity": {"qtype": 10, "artifact_tile_k": 0,
                         "bchunk": 0, "weight_layout": 2},
            "runtime": {
                "full_output": ["NONPERSISTENT",
                                "PERSISTENT_CAPACITY_BALANCED"],
                "producer_only": ["SPLITK_S2_PRODUCER",
                                  "SPLITK_S4_PRODUCER",
                                  "SPLITK_S8_PRODUCER"],
                "cuda": {
                    "status": "STRUCTURAL_UNAVAILABLE",
                    "reason": "NO_CANONICAL_KPACK_CUDA_READER"}},
            "denominator": {"typed_rows": 3, "authority_typed_rows": 3,
                            "non_typed_rows": 0},
            "parent_range": {"begin": 0, "end": 3, "count": 3,
                             "authority_count": 3},
            "selection": {"mode": "authority-full", "begin": 0,
                          "end": 3, "authority_typed_rows": 3,
                          "compiled_rows": 3},
            "non_typed_authority": {
                "count": 0,
                "sha256": hashlib.sha256(b"[]").hexdigest(),
                "encoding": "JSON_SORT_KEYS_COMPACT_V1"},
            "non_typed_rows": [],
            "compiled_parents": [
                {"parent_id": parent_id, "symbol": symbol,
                 "static_candidate_id": symbol, "a_provider": 0,
                 "resolved_delivery_n": 16}
                for parent_id, symbol in enumerate(
                    ("fast", "slow", "structural"))],
            "typed_rows": [
                {"parent_id": parent_id, "symbol": symbol,
                 "static_candidate_id": symbol, "a_provider": 0,
                 "resolved_delivery_n": 16}
                for parent_id, symbol in enumerate(
                    ("fast", "slow", "structural"))],
        }))
        symbols, sidecar = root / "symbols", root / "retention.json"
        missing_census = root / "missing-census.log"
        missing_census.write_text("\n".join(
            retention_dense.read_text().splitlines()[:2]) + "\n")
        try:
            retain("dense", 10, missing_census, manifest, symbols, sidecar)
        except ValueError as error:
            if "product census differs" not in str(error):
                raise
        else:
            raise AssertionError("missing product census stayed green")
        nonstructural = root / "nonstructural-census.log"
        rows = retention_dense.read_text().splitlines()[:2]
        rows.append("SF_CELL " + json.dumps({
            "symbol": "structural", "algorithm": "NONPERSISTENT",
            "metric_scope": "FULL_OUTPUT", "policy": "ordinary",
            "grid": 0, "status": "MEASURED", "raw_bad": 1,
            "sample_us": 0.1}))
        nonstructural.write_text("\n".join(rows) + "\n")
        try:
            retain("dense", 10, nonstructural, manifest, symbols, sidecar)
        except ValueError as error:
            if "not explicitly structural-unavailable" not in str(error):
                raise
        else:
            raise AssertionError("nonstructural elimination stayed green")
        retain("dense", 10, retention_dense, manifest, symbols, sidecar)
        if symbols.read_text().splitlines() != ["fast", "slow"]:
            raise AssertionError("timing rank incorrectly eliminated a clean candidate")
        retained = json.loads(sidecar.read_text())
        if retained["timing_rank_used_for_elimination"] or \
                retained["structural_unavailable_symbols"] != ["structural"]:
            raise AssertionError("retention sidecar does not prove safe elimination")
        grouped = root / "grouped.log"
        grouped.write_text(
            "SF_GROUPED_CELL q=10 layout=2 symbol=g config=c scope=FULL_OUTPUT "
            "algorithm=GROUPED_PERSISTENT policy=balanced grid=72 "
            "state=MEASURED raw_bad=0 median_us=4.0\n"
            "SF_GROUPED_CELL q=10 layout=2 symbol=red config=c scope=FULL_OUTPUT "
            "algorithm=GROUPED_NONPERSISTENT policy=ordinary grid=0 "
            "state=RAW_FP16_MISMATCH raw_bad=1 median_us=0.1\n")
        if [row["symbol"] for row in grouped_candidates(grouped)] != ["g"]:
            raise AssertionError("grouped raw-bit negative failed")
        # A pilot bundle is a named two-shard result, never a five-format
        # result in disguise. Dynamic summary consumes shard IDs rather than
        # assuming one binary per qtype/operator.
        dense_shard = "q10-dense-p000000-000003"
        grouped_shard = "q10-grouped-p000000-000001"
        (root / f"{dense_shard}.confirm.log").write_text(
            "\n".join(retention_dense.read_text().splitlines()[:2]) + "\n")
        (root / f"{dense_shard}.retention.json").write_text(json.dumps({
            "qtype": 10, "operator": "dense",
            "retained_symbols": ["fast", "slow"]}))
        grouped_clean = root / f"{grouped_shard}.confirm.log"
        grouped_clean.write_text(grouped.read_text().splitlines()[0] + "\n")
        (root / f"{grouped_shard}.retention.json").write_text(json.dumps({
            "qtype": 10, "operator": "grouped",
            "retained_symbols": ["g"]}))
        index = root / "pilot-index.json"
        index.write_text(json.dumps({
            "schema": "quactlize.scalefirst_kpack_prebuilt_bundle.v2",
            "scope": "pilot", "route": "scalefirst",
            "shards": [
                {"shard_id": dense_shard, "qtype": 10,
                 "operator": "dense"},
                {"shard_id": grouped_shard, "qtype": 10,
                 "operator": "grouped"}],
        }))
        summary_json, summary_tsv = root / "summary.json", root / "summary.tsv"
        summarize(root, summary_json, summary_tsv, index)
        summary_doc = json.loads(summary_json.read_text())
        if summary_doc["scope"] != "pilot" or \
                {(row["qtype"], row["operator"])
                 for row in summary_doc["rows"]} != {
                    (10, "dense"), (10, "grouped")}:
            raise AssertionError("pilot dynamic shard summary differs")
        planted = json.loads(index.read_text()); planted["scope"] = "full"
        index.write_text(json.dumps(planted))
        try:
            summarize(root, summary_json, summary_tsv, index)
        except ValueError as error:
            if "denominator differs" not in str(error):
                raise
        else:
            raise AssertionError("pilot summary masqueraded as full")
        measurement = measurement_contract(
            "2048x1024x5120", 4, 2, 16, 256, 512, 1, 7, 2)
        result_authority = root / "result-authority.json"
        write_stable_json(result_authority, {"measurement": measurement})
        write_stable_json(result_authority, {"measurement": measurement})
        planted_measurement = dict(measurement)
        planted_measurement["confirm_iterations"] = 9
        try:
            write_stable_json(result_authority,
                              {"measurement": planted_measurement})
        except ValueError as error:
            if "changed on resume" not in str(error):
                raise
        else:
            raise AssertionError("stale measurement parameters stayed green")
    print("[sf-kpack-discovery-analysis:self-test] PASS raw-bit-before-time="
          "BOUND timing-rank-elimination=FORBIDDEN q4-anchor=CANDIDATE-ONLY "
          "splitk=EXCLUDED-DIAGNOSTIC full-output=NP+P census=FAIL-CLOSED "
          "dynamic-shards=PILOT-BOUND resume-parameters=BOUND")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    choose = sub.add_parser("retain")
    choose.add_argument("--operator", choices=("dense", "grouped"), required=True)
    choose.add_argument("--qtype", type=int, choices=QTYPES, required=True)
    choose.add_argument("--screen", type=pathlib.Path, required=True)
    choose.add_argument("--manifest", type=pathlib.Path, required=True)
    choose.add_argument("--output", type=pathlib.Path, required=True)
    choose.add_argument("--sidecar", type=pathlib.Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--results", type=pathlib.Path, required=True)
    summary.add_argument("--bundle-index", type=pathlib.Path)
    summary.add_argument("--output-json", type=pathlib.Path, required=True)
    summary.add_argument("--output-tsv", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "retain":
            retain(args.operator, args.qtype, args.screen, args.manifest,
                   args.output, args.sidecar)
        else:
            summarize(args.results, args.output_json, args.output_tsv,
                      args.bundle_index)
        return 0
    except (KeyError, OSError, ValueError, AssertionError,
            json.JSONDecodeError) as error:
        print(f"[sf-kpack-discovery-analysis] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
