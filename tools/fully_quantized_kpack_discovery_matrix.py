#!/usr/bin/env python3
"""Exact discovery denominator for canonical K-pack FullyQuantized kernels.

This module is intentionally independent of the shipping selector.  The raw
topology product comes from the shared ScaleFirst/CuTe policy emitter because
canonical K-pack has no ArtifactTileK axis: the same offline bytes are consumed
by every legal tactic.  FullyQuantized adds only packed-metadata ownership,
the legal A-provider expansion, and explicit runtime algorithm boards.

No candidate is selected here.  Every statically admitted TC parent must be
compiled and every runtime-admitted variant must pass raw-fp16 correctness
before timing.  Canonical K-pack has no BC reader today, so BC remains a named
independent algorithm with STRUCTURAL_UNAVAILABLE status rather than being
silently dropped or represented by a tensor-core row.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from dataclasses import asdict

import scalefirst_internal_matrix as topology


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "quactlize.fully_quantized_kpack_discovery.v1"
QTYPES = (10, 11, 12, 13, 14)
SPLITS = (1, 2, 4, 8)
GROUPED_ALGORITHMS = ("GROUPED_NONPERSISTENT", "GROUPED_PERSISTENT")
STRUCTURAL_GROUPED_ALGORITHMS = (
    "GROUPED_SPLITK_S2", "GROUPED_SPLITK_S4", "GROUPED_SPLITK_S8",
    "GROUPED_BC_FULL_OUTPUT",
)
BC_REASON = "NO_CANONICAL_KPACK_BC_READER"
GROUPED_SPLIT_REASON = "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER"

# Geometry tokens selected by the previous measured K-pack policy.  They are
# anchors, not a reduced search space: exhaustive discovery must contain each
# one in addition to every newly generated tactic.
MEASURED_DENSE_GEOMETRY_ANCHORS = (
    ("Default", 64, 64, 32, 32, 3),
    ("SmallSquare", 32, 32, 16, 16, 3),
    ("ShortWide", 16, 128, 16, 32, 3),
    ("ShortWideM8S2", 8, 128, 8, 32, 2),
    ("ShortWideM8S3", 8, 128, 8, 32, 3),
    ("MidWide", 32, 128, 32, 32, 3),
    ("Tall", 128, 64, 64, 32, 2),
)
FQ_TILE_K = {10: 256, 11: 256, 12: 256, 13: 256, 14: 128}
Q4_POLICY_V2_ANCHORS = (
    (8, 32, 256, 8, 16, 3, 4),
    (8, 64, 256, 8, 16, 2, 4),
    (8, 128, 256, 8, 16, 2, 4),
    (8, 64, 256, 8, 16, 2, 1),
    (64, 128, 256, 64, 16, 2, 1),
)


def layout_for(qtype: int) -> int:
    if qtype not in QTYPES:
        raise ValueError(f"qtype {qtype} has no canonical FQ K-pack route")
    return topology.KPACK_LAYOUT_BY_QTYPE[qtype]


def format_for(qtype: int) -> topology.Format:
    fmt = topology.format_for(qtype)
    if fmt.quant_mode != "ScaleZero" or fmt.metadata_planes != 2:
        raise ValueError(f"qtype {qtype} does not retain packed scale+zero metadata")
    return fmt


def raw_rows(qtype: int) -> tuple[topology.Tactic, ...]:
    rows = topology.emitted_tactics(qtype, 0, weight_layout=layout_for(qtype))
    if len(rows) != topology.RAW_ROWS_PER_PAIR:
        raise ValueError(f"qtype {qtype} raw topology denominator differs")
    return rows


def admitted_rows(qtype: int) -> tuple[topology.Tactic, ...]:
    fmt = format_for(qtype)
    layout = layout_for(qtype)
    rows = tuple(row for row in raw_rows(qtype)
                 if topology.classify(fmt, 0, row, layout)[0] ==
                 "TYPE_ADMISSION_REQUIRED")
    if not rows or any(row.bchunk != 0 for row in rows):
        raise ValueError(f"qtype {qtype} canonical FQ admission is empty/non-bc0")
    return rows


def ap1_legal(qtype: int, row: topology.Tactic) -> bool:
    # KPackMainloopPolicy admits AP1 only for its single-plane Q2 arm; Q4 has
    # the separately proved DenseQ4KPack4KernelTypes AP1 arm.  Both require
    # the exact m8/warp-m8 decode geometry.
    return qtype in (10, 12) and row.tile_m == 8 and row.warp_m == 8


def parent_id(qtype: int, row: topology.Tactic, provider: int,
              delivery_n: int, operator: str, algorithm: str = "") -> str:
    suffix = f"_{algorithm.lower()}" if algorithm else ""
    return (f"fqk_{operator}_q{qtype}_l{layout_for(qtype)}_"
            f"tm{row.tile_m}_tn{row.tile_n}_tk{row.tactic_tile_k}_"
            f"wm{row.warp_m}_wn{row.warp_n}_s{row.stages}_"
            f"bc0_ap{provider}_dn{delivery_n}{suffix}")


def provider_rows(qtype: int) -> tuple[tuple[topology.Tactic, int, int], ...]:
    result: list[tuple[topology.Tactic, int, int]] = []
    for row in admitted_rows(qtype):
        for provider in ((0, 1) if ap1_legal(qtype, row) else (0,)):
            for delivery_n in topology.resolved_delivery_ns(row.tile_n):
                result.append((row, provider, delivery_n))
    return tuple(result)


def grouped_rows(qtype: int) -> tuple[tuple[topology.Tactic, int, str], ...]:
    return tuple(
        (row, delivery_n, algorithm)
        for row in admitted_rows(qtype)
        for delivery_n in topology.resolved_delivery_ns(row.tile_n)
        for algorithm in GROUPED_ALGORITHMS
    )


def _row_record(qtype: int, row: topology.Tactic, status: str,
                reason: str) -> dict:
    return asdict(row) | {
        "qtype": qtype,
        "weight_layout": layout_for(qtype),
        "artifact_tile_k": 0,
        "bchunk": 0,
        "status": status,
        "reason": reason,
    }


def format_manifest(qtype: int, expand: bool) -> dict:
    fmt = format_for(qtype)
    layout = layout_for(qtype)
    source = raw_rows(qtype)
    admitted = admitted_rows(qtype)
    providers = provider_rows(qtype)
    dense_parents = []
    for row, provider, delivery_n in providers:
        item = _row_record(
            qtype, row, "TYPE_ADMISSION_REQUIRED",
            "DENSE_CANONICAL_KPACK_TYPE_COMPILE_AND_CAN_IMPLEMENT_REQUIRED")
        item.update({
            "static_candidate_id": parent_id(
                qtype, row, provider, delivery_n, "dense"),
            "a_provider": provider,
            "a_provider_name": "packed-row" if provider else "standard-aiu",
            "resolved_delivery_n": delivery_n,
            "runtime_algorithms": [f"TC_S{split}" for split in SPLITS],
        })
        dense_parents.append(item)
    grouped_parents = []
    for row, delivery_n, algorithm in grouped_rows(qtype):
        item = _row_record(
            qtype, row, "TYPE_ADMISSION_REQUIRED",
            "GROUPED_CANONICAL_KPACK_TYPE_COMPILE_AND_CAN_IMPLEMENT_REQUIRED")
        item.update({
            "static_candidate_id": parent_id(
                qtype, row, 0, delivery_n, "grouped", algorithm),
            "a_provider": 0,
            "resolved_delivery_n": delivery_n,
            "algorithm": algorithm,
            "runtime_variant_authority": (
                "EXACT_RUNTIME_GRID_SPACE_FROM_COMPILED_KERNEL"
                if algorithm == "GROUPED_PERSISTENT" else
                "ONE_NONPERSISTENT_VARIANT_PER_CELL"),
        })
        grouped_parents.append(item)
    source_records = []
    if expand:
        for row in source:
            status, reason = topology.classify(fmt, 0, row, layout)
            source_records.append(_row_record(qtype, row, status, reason))
    result = {
        "qtype": qtype,
        "format": fmt.name,
        "low_bits": fmt.low_bits,
        "high_bits": fmt.high_bits,
        "group_size": fmt.group_size,
        "quant_mode": "ScaleZero",
        "metadata": "PACKED_UNITS_SCALE_AND_ZERO",
        "weight_layout": layout,
        "weight_layout_name": topology.layout_name(layout),
        "artifact_tile_k": 0,
        "bchunk": 0,
        "source_raw_rows": len(source),
        "admitted_topologies": len(admitted),
        "dense_provider_parents": len(providers),
        "dense_runtime_tc_cells": len(providers) * len(SPLITS),
        "grouped_type_parents": len(grouped_parents),
        "algorithms": {
            "DENSE_TC": {
                "status": "TYPE_ADMISSION_REQUIRED",
                "splits": list(SPLITS),
                "correctness": "FULL_RAW_FP16_BEFORE_TIMING",
                "split_scope": {
                    "S1": "FULL_OUTPUT",
                    "S2_S4_S8": "MEASURED_PRODUCER_PLUS_VERSIONED_REDUCER_MODEL",
                },
            },
            "DENSE_BC_FULL_OUTPUT": {
                "status": "STRUCTURAL_UNAVAILABLE", "reason": BC_REASON,
            },
            "GROUPED_NONPERSISTENT": {
                "status": "TYPE_ADMISSION_REQUIRED",
                "correctness": "REAL_RAGGED_OFFSETS_EMPTY_EXPERT_RAW_FP16_BEFORE_TIMING",
            },
            "GROUPED_PERSISTENT": {
                "status": "TYPE_ADMISSION_REQUIRED",
                "correctness": "REAL_RAGGED_OFFSETS_EMPTY_EXPERT_RAW_FP16_BEFORE_TIMING",
            },
            **{
                name: {
                    "status": "STRUCTURAL_UNAVAILABLE",
                    "reason": BC_REASON if name.endswith("BC_FULL_OUTPUT")
                    else GROUPED_SPLIT_REASON,
                }
                for name in STRUCTURAL_GROUPED_ALGORITHMS
            },
        },
        "dense_parents": dense_parents if expand else None,
        "grouped_parents": grouped_parents if expand else None,
        "source_rows": source_records if expand else None,
    }
    return result


def make_manifest(expand: bool = False) -> dict:
    formats = [format_manifest(qtype, expand) for qtype in QTYPES]
    denominator = {
        "formats": 5,
        "source_raw_rows": sum(row["source_raw_rows"] for row in formats),
        "admitted_topologies": sum(row["admitted_topologies"] for row in formats),
        "dense_provider_parents": sum(row["dense_provider_parents"] for row in formats),
        "dense_runtime_tc_cells": sum(row["dense_runtime_tc_cells"] for row in formats),
        "dense_bc_structural_cells": 5,
        "grouped_type_parents": sum(row["grouped_type_parents"] for row in formats),
        "grouped_splitk_structural_cells": 15,
        "grouped_bc_structural_cells": 5,
    }
    return {
        "schema": SCHEMA,
        "scope": "canonical-kpack-fully-quantized-exhaustive-discovery-only",
        "fixed_identity": {
            "artifact_tile_k": 0,
            "bchunk": 0,
            "packed_scale": 1,
            "shipping_selector_mutated": False,
            "xplane_discovery_added": False,
        },
        "confirmation": {
            "elimination": "NONE_AFTER_RAW_BIT_CLEAN_AND_CAN_IMPLEMENT",
            "top_n": None,
            "all_clean_candidates_confirmed": True,
        },
        "formats": formats,
        "denominator": denominator,
    }


def validate_manifest(document: dict, *, expanded: bool) -> None:
    if document.get("schema") != SCHEMA:
        raise ValueError("FQ K-pack discovery schema differs")
    fixed = document.get("fixed_identity") or {}
    if fixed != {
            "artifact_tile_k": 0, "bchunk": 0, "packed_scale": 1,
            "shipping_selector_mutated": False,
            "xplane_discovery_added": False}:
        raise ValueError("FQ K-pack fixed identity differs")
    if document.get("confirmation") != {
            "elimination": "NONE_AFTER_RAW_BIT_CLEAN_AND_CAN_IMPLEMENT",
            "top_n": None, "all_clean_candidates_confirmed": True}:
        raise ValueError("FQ K-pack discovery introduced a ranking screen")
    formats = document.get("formats")
    if not isinstance(formats, list) or [row.get("qtype") for row in formats] != list(QTYPES):
        raise ValueError("FQ K-pack format denominator differs")
    totals = {name: 0 for name in (
        "source_raw_rows", "admitted_topologies", "dense_provider_parents",
        "dense_runtime_tc_cells", "grouped_type_parents")}
    for record in formats:
        qtype = int(record["qtype"])
        if (record.get("weight_layout") != layout_for(qtype) or
                record.get("artifact_tile_k") != 0 or
                record.get("bchunk") != 0 or
                record.get("metadata") != "PACKED_UNITS_SCALE_AND_ZERO" or
                record.get("source_raw_rows") != topology.RAW_ROWS_PER_PAIR):
            raise ValueError(f"qtype {qtype} canonical identity differs")
        expected_topologies = len(admitted_rows(qtype))
        expected_providers = len(provider_rows(qtype))
        expected_grouped = len(grouped_rows(qtype))
        if (record.get("admitted_topologies") != expected_topologies or
                record.get("dense_provider_parents") != expected_providers or
                record.get("dense_runtime_tc_cells") != expected_providers * 4 or
                record.get("grouped_type_parents") != expected_grouped):
            raise ValueError(f"qtype {qtype} candidate denominator differs")
        algorithms = record.get("algorithms") or {}
        if algorithms.get("DENSE_BC_FULL_OUTPUT") != {
                "status": "STRUCTURAL_UNAVAILABLE", "reason": BC_REASON}:
            raise ValueError(f"qtype {qtype} fabricated a canonical BC reader")
        for name in STRUCTURAL_GROUPED_ALGORITHMS:
            expected_reason = BC_REASON if name.endswith("BC_FULL_OUTPUT") else GROUPED_SPLIT_REASON
            if algorithms.get(name) != {
                    "status": "STRUCTURAL_UNAVAILABLE", "reason": expected_reason}:
                raise ValueError(f"qtype {qtype} fabricated {name}")
        if expanded:
            source = record.get("source_rows")
            dense = record.get("dense_parents")
            grouped = record.get("grouped_parents")
            if not isinstance(source, list) or len(source) != topology.RAW_ROWS_PER_PAIR:
                raise ValueError(f"qtype {qtype} source row missing")
            if not isinstance(dense, list) or len(dense) != expected_providers or \
                    len({row["static_candidate_id"] for row in dense}) != len(dense):
                raise ValueError(f"qtype {qtype} dense parent missing/duplicate")
            if not isinstance(grouped, list) or len(grouped) != expected_grouped or \
                    len({row["static_candidate_id"] for row in grouped}) != len(grouped):
                raise ValueError(f"qtype {qtype} grouped parent missing/duplicate")
            for row in dense:
                if row["artifact_tile_k"] != 0 or row["bchunk"] != 0 or \
                        row["weight_layout"] != layout_for(qtype):
                    raise ValueError(f"qtype {qtype} dense parent changed layout")
                if row["a_provider"] == 1 and not ap1_legal(
                        qtype, topology.Tactic(**{
                            key: row[key] for key in (
                                "tile_m", "tile_n", "tactic_tile_k", "warp_m",
                                "warp_n", "stages", "bchunk", "source_status",
                                "source_reason", "fold_low", "fold_high")
                        })):
                    raise ValueError(f"qtype {qtype} illegal AP1 parent")
                if (row["resolved_delivery_n"] not in
                        topology.resolved_delivery_ns(row["tile_n"])):
                    raise ValueError(f"qtype {qtype} illegal delivery parent")
            for row in grouped:
                if (row["a_provider"] != 0 or
                        row["resolved_delivery_n"] not in
                        topology.resolved_delivery_ns(row["tile_n"])):
                    raise ValueError(f"qtype {qtype} illegal grouped provider/delivery")
        for name in totals:
            totals[name] += int(record[name])
    expected_denominator = {
        "formats": 5,
        **totals,
        "dense_bc_structural_cells": 5,
        "grouped_splitk_structural_cells": 15,
        "grouped_bc_structural_cells": 5,
    }
    if document.get("denominator") != expected_denominator:
        raise ValueError("FQ K-pack aggregate denominator differs")


def self_test() -> None:
    manifest = make_manifest(True)
    validate_manifest(manifest, expanded=True)
    den = manifest["denominator"]
    if den != {
            "formats": 5, "source_raw_rows": 115200,
            "admitted_topologies": 5182, "dense_provider_parents": 14750,
            "dense_runtime_tc_cells": 59000, "dense_bc_structural_cells": 5,
            "grouped_type_parents": 27412,
            "grouped_splitk_structural_cells": 15,
            "grouped_bc_structural_cells": 5}:
        raise ValueError(f"canonical FQ denominator drifted: {den}")
    for qtype in (10, 11, 13, 14):
        parents = provider_rows(qtype)
        for name, tm, tn, wm, wn, stages in MEASURED_DENSE_GEOMETRY_ANCHORS:
            present = any(
                (row.tile_m, row.tile_n, row.tactic_tile_k,
                 row.warp_m, row.warp_n, row.stages, provider, delivery_n) ==
                (tm, tn, FQ_TILE_K[qtype], wm, wn, stages, 0, min(64, tn))
                for row, provider, delivery_n in parents)
            if not present:
                raise ValueError(
                    f"qtype {qtype} lost measured dense anchor {name}")
    q4_parents = provider_rows(12)
    for tm, tn, tk, wm, wn, stages, split in Q4_POLICY_V2_ANCHORS:
        present = split in SPLITS and any(
            (row.tile_m, row.tile_n, row.tactic_tile_k,
             row.warp_m, row.warp_n, row.stages, provider, delivery_n) ==
            (tm, tn, tk, wm, wn, stages, 0, min(64, tn))
            for row, provider, delivery_n in q4_parents)
        if not present:
            raise ValueError(
                "Q4 policy-v2 candidate left the exhaustive denominator")
    for qtype in QTYPES:
        if not any(
                (row.tile_m, row.tile_n, row.tactic_tile_k,
                 row.warp_m, row.warp_n, row.stages, delivery_n, algorithm) ==
                (16, 128, FQ_TILE_K[qtype], 16, 16, 2, 64,
                 "GROUPED_NONPERSISTENT")
                for row, delivery_n, algorithm in grouped_rows(qtype)):
            raise ValueError(
                f"qtype {qtype} lost the measured grouped default anchor")
    plants = []
    missing = copy.deepcopy(manifest)
    missing["formats"][2]["dense_parents"].pop(len(
        missing["formats"][2]["dense_parents"]) // 2)
    plants.append(missing)
    missing_delivery = copy.deepcopy(manifest)
    q4_dense = missing_delivery["formats"][2]["dense_parents"]
    target = next(row for row in q4_dense
                  if row["tile_n"] >= 64 and row["resolved_delivery_n"] == 32)
    q4_dense.remove(target)
    plants.append(missing_delivery)
    wrong_layout = copy.deepcopy(manifest)
    wrong_layout["formats"][0]["weight_layout"] = 1
    plants.append(wrong_layout)
    fake_bc = copy.deepcopy(manifest)
    fake_bc["formats"][1]["algorithms"]["DENSE_BC_FULL_OUTPUT"] = {
        "status": "AVAILABLE", "reason": "TC_READER"}
    plants.append(fake_bc)
    fake_split = copy.deepcopy(manifest)
    fake_split["formats"][3]["algorithms"]["GROUPED_SPLITK_S4"] = {
        "status": "AVAILABLE", "reason": "MODELED"}
    plants.append(fake_split)
    illegal_ap = copy.deepcopy(manifest)
    q3 = illegal_ap["formats"][1]
    q3["dense_parents"][0]["a_provider"] = 1
    plants.append(illegal_ap)
    filtered = copy.deepcopy(manifest)
    filtered["confirmation"]["top_n"] = 16
    plants.append(filtered)
    for planted in plants:
        try:
            validate_manifest(planted, expanded=True)
        except (TypeError, ValueError):
            continue
        raise ValueError("FQ K-pack negative plant stayed green")
    print("[fq-kpack-discovery-matrix:self-test] PASS "
          "formats=5 raw=115200 topology=5182 dense-parents=14750 "
          "dense-cells=59000 grouped-parents=27412 "
          "BC=5+5xSTRUCTURAL_UNAVAILABLE grouped-split=15xSTRUCTURAL_UNAVAILABLE "
          "historical-dense/grouped/Q4-policy-anchors=INCLUDED "
          "negatives=missing+layout+fake-bc+fake-split+illegal-ap+top-n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("emit", "self-test"))
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--out", default="-")
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        document = make_manifest(args.expand)
        validate_manifest(document, expanded=args.expand)
        payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
        if args.out == "-":
            sys.stdout.write(payload)
        else:
            path = pathlib.Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[fq-kpack-discovery-matrix] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
