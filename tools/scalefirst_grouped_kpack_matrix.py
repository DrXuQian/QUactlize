#!/usr/bin/env python3
"""Exact static denominator for canonical K-pack grouped ScaleFirst discovery.

This is deliberately a discovery graph, not a product routing table.  It
reuses the complete ScaleFirst topology emitter and changes only the operator
driver: one grouped full-output kernel is available, while grouped Split-K is
named STRUCTURAL_UNAVAILABLE rather than assigned a modeled reducer.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from dataclasses import asdict

import scalefirst_internal_matrix as dense


SCHEMA = "quactlize.scalefirst_grouped_kpack_matrix.v1"
SPLITS = (2, 4, 8)
QTYPES = (10, 11, 12, 13, 14)


def layout_for(qtype: int) -> int:
    try:
        return dense.KPACK_LAYOUT_BY_QTYPE[qtype]
    except KeyError as error:
        raise ValueError(f"qtype {qtype} has no canonical K-pack layout") from error


def admitted_rows(qtype: int) -> tuple[dense.Tactic, ...]:
    fmt = dense.format_for(qtype)
    layout = layout_for(qtype)
    rows = dense.emitted_tactics(qtype, 0, weight_layout=layout)
    admitted = tuple(row for row in rows
                     if dense.classify(fmt, 0, row, layout)[0] ==
                     "TYPE_ADMISSION_REQUIRED")
    if not admitted or any(row.bchunk != 0 for row in admitted):
        raise ValueError(f"qtype {qtype} canonical K-pack admission is empty/non-bc0")
    return admitted


def candidate_rows(qtype: int) -> tuple[tuple[dense.Tactic, int], ...]:
    return tuple((row, delivery_n) for row in admitted_rows(qtype)
                 for delivery_n in dense.resolved_delivery_ns(row.tile_n))


def row_key(row: dense.Tactic) -> tuple[int, ...]:
    return (row.tile_m, row.tile_n, row.tactic_tile_k,
            row.warp_m, row.warp_n, row.stages, row.bchunk)


def format_manifest(qtype: int, expand: bool) -> dict:
    fmt = dense.format_for(qtype)
    layout = layout_for(qtype)
    rows = admitted_rows(qtype)
    candidates = candidate_rows(qtype)
    result = {
        "qtype": qtype,
        "format": fmt.name,
        "quant_mode": "ScaleZero",
        "metadata_planes": 2,
        "group_size": fmt.group_size,
        "weight_layout": layout,
        "weight_layout_name": dense.layout_name(layout),
        "artifact_tile_k": 0,
        "source_raw_rows": dense.RAW_ROWS_PER_PAIR,
        "admitted_rows": len(rows),
        "delivery_expanded_rows": len(candidates),
        "algorithms": {
            "GROUPED_NONPERSISTENT": {
                "status": "AVAILABLE",
                "scope": "FULL_OUTPUT_RAW_BIT_CORRECTNESS_THEN_TIMING",
            },
            "GROUPED_PERSISTENT": {
                "status": "AVAILABLE",
                "scope": "FULL_OUTPUT_RAW_BIT_CORRECTNESS_THEN_TIMING",
                "grid_space": "RUNTIME_EXACT_OCCUPANCY_CAPACITY_BALANCED",
            },
            "SCALEFIRST_CUDA": {
                "status": "STRUCTURAL_UNAVAILABLE",
                "scope": "NO_CANONICAL_KPACK_CUDA_READER",
            },
            **{
                f"GROUPED_SPLITK_S{split}": {
                    "status": "STRUCTURAL_UNAVAILABLE",
                    "scope": "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER",
                }
                for split in SPLITS
            },
        },
    }
    if expand:
        result["rows"] = [{**asdict(row), "resolved_delivery_n": delivery_n,
                           "a_provider": 0}
                          for row, delivery_n in candidates]
    return result


def make_manifest(expand: bool = False) -> dict:
    formats = [format_manifest(qtype, expand) for qtype in QTYPES]
    return {
        "schema": SCHEMA,
        "scope": "canonical-kpack-grouped-scalefirst-discovery-only",
        "formats": formats,
        "denominator": {
            "formats": len(formats),
            "source_raw_rows": sum(row["source_raw_rows"] for row in formats),
            "admitted_rows": sum(row["admitted_rows"] for row in formats),
            "delivery_expanded_rows": sum(row["delivery_expanded_rows"] for row in formats),
            "full_output_boards": 2 * len(formats),
            "structural_cuda_cells": len(formats),
            "structural_splitk_cells": len(formats) * len(SPLITS),
        },
    }


def validate_manifest(document: dict) -> None:
    if document.get("schema") != SCHEMA:
        raise ValueError("grouped K-pack matrix schema differs")
    formats = document.get("formats")
    if not isinstance(formats, list) or len(formats) != len(QTYPES):
        raise ValueError("grouped K-pack format denominator differs")
    if [row.get("qtype") for row in formats] != list(QTYPES):
        raise ValueError("grouped K-pack qtype order/set differs")
    total = 0
    for row in formats:
        qtype = int(row["qtype"])
        if row.get("weight_layout") != layout_for(qtype) or \
                row.get("artifact_tile_k") != 0 or \
                row.get("quant_mode") != "ScaleZero" or \
                row.get("metadata_planes") != 2 or \
                row.get("source_raw_rows") != dense.RAW_ROWS_PER_PAIR or \
                int(row.get("admitted_rows", 0)) <= 0 or \
                int(row.get("delivery_expanded_rows", 0)) <= 0:
            raise ValueError(f"qtype {qtype} grouped K-pack identity differs")
        algorithms = row.get("algorithms") or {}
        full = algorithms.get("GROUPED_NONPERSISTENT") or {}
        if full.get("status") != "AVAILABLE" or \
                full.get("scope") != "FULL_OUTPUT_RAW_BIT_CORRECTNESS_THEN_TIMING":
            raise ValueError(f"qtype {qtype} full-output board differs")
        if algorithms.get("GROUPED_PERSISTENT") != {
                "status": "AVAILABLE",
                "scope": "FULL_OUTPUT_RAW_BIT_CORRECTNESS_THEN_TIMING",
                "grid_space": "RUNTIME_EXACT_OCCUPANCY_CAPACITY_BALANCED"}:
            raise ValueError(f"qtype {qtype} grouped persistent status differs")
        if algorithms.get("SCALEFIRST_CUDA") != {
                "status": "STRUCTURAL_UNAVAILABLE",
                "scope": "NO_CANONICAL_KPACK_CUDA_READER"}:
            raise ValueError(f"qtype {qtype} CUDA status differs")
        for split in SPLITS:
            cell = algorithms.get(f"GROUPED_SPLITK_S{split}") or {}
            if cell != {"status": "STRUCTURAL_UNAVAILABLE",
                        "scope": "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER"}:
                raise ValueError(f"qtype {qtype} S{split} is not structural-unavailable")
        rows = row.get("rows")
        if rows is not None:
            if any(int(item.get("a_provider", -1)) != 0 or
                   int(item.get("resolved_delivery_n", 0)) not in
                   dense.resolved_delivery_ns(int(item["tile_n"]))
                   for item in rows):
                raise ValueError(f"qtype {qtype} delivery/provider identity differs")
            keys = [tuple(item[name] for name in (
                "tile_m", "tile_n", "tactic_tile_k", "warp_m", "warp_n",
                "stages", "bchunk")) for item in rows]
            keys = [key + (int(item["resolved_delivery_n"]), int(item["a_provider"]))
                    for key, item in zip(keys, rows)]
            if len(keys) != row["delivery_expanded_rows"] or len(set(keys)) != len(keys):
                raise ValueError(f"qtype {qtype} expanded type denominator differs")
        total += int(row["admitted_rows"])
    denominator = document.get("denominator") or {}
    if denominator != {
            "formats": 5,
            "source_raw_rows": 5 * dense.RAW_ROWS_PER_PAIR,
            "admitted_rows": total,
            "delivery_expanded_rows": sum(r["delivery_expanded_rows"] for r in formats),
            "full_output_boards": 10,
            "structural_cuda_cells": 5,
            "structural_splitk_cells": 15}:
        raise ValueError("grouped K-pack aggregate denominator differs")


def self_test() -> None:
    manifest = make_manifest(True)
    validate_manifest(manifest)
    if manifest["denominator"]["delivery_expanded_rows"] != 13706:
        raise ValueError("canonical grouped delivery denominator drifted from 13706")
    negatives = []
    missing = copy.deepcopy(manifest)
    missing["formats"][0]["rows"].pop()
    negatives.append(missing)
    fake_split = copy.deepcopy(manifest)
    fake_split["formats"][0]["algorithms"]["GROUPED_SPLITK_S2"] = {
        "status": "AVAILABLE", "scope": "MODELED_REDUCER"}
    negatives.append(fake_split)
    fake_persistent = copy.deepcopy(manifest)
    fake_persistent["formats"][0]["algorithms"]["GROUPED_PERSISTENT"] = {
        "status": "AVAILABLE", "scope": "ASSUMED", "grid_space": "ONE_GUESSED_GRID"}
    negatives.append(fake_persistent)
    fake_cuda = copy.deepcopy(manifest)
    fake_cuda["formats"][0]["algorithms"]["SCALEFIRST_CUDA"] = {
        "status": "AVAILABLE", "scope": "ASSUMED_READER"}
    negatives.append(fake_cuda)
    wrong_layout = copy.deepcopy(manifest)
    wrong_layout["formats"][1]["weight_layout"] = 1
    negatives.append(wrong_layout)
    old_q6 = copy.deepcopy(manifest)
    old_q6["formats"][-1]["quant_mode"] = "ScaleOnly"
    old_q6["formats"][-1]["metadata_planes"] = 1
    negatives.append(old_q6)
    for planted in negatives:
        try:
            validate_manifest(planted)
        except ValueError:
            continue
        raise ValueError("grouped K-pack negative plant stayed green")
    print("[sf-grouped-kpack-matrix:self-test] PASS formats=5 "
          "admitted=5182 delivery-expanded=13706 full-output=NP+P persistent-grid=EXACT-OCCUPANCY "
          "cuda=STRUCTURAL_UNAVAILABLE splitk=15xSTRUCTURAL_UNAVAILABLE "
          "negatives=missing-row+fake-cuda+fake-persistent+fake-reducer+"
          "layout+Q6-scaleonly")


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
        validate_manifest(document)
        payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
        if args.out == "-":
            sys.stdout.write(payload)
        else:
            path = pathlib.Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[sf-grouped-kpack-matrix] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
