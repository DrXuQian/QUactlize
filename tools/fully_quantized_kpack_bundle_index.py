#!/usr/bin/env python3
"""Canonical parent-range index for FullyQuantized K-pack discovery.

The generator authority is much larger than one safe hgcc image.  This module
is the single source of truth for partitioning that authority into binaries.
Every range is half-open, contiguous, non-overlapping, and contains at most 32
compiled parents.  PILOT mode deliberately emits only the first q10 dense and
grouped range; it is never accepted as a full discovery bundle.
"""

from __future__ import annotations

import copy
import pathlib
import sys


TOOLS = pathlib.Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import fully_quantized_kpack_discovery_matrix as matrix  # noqa: E402


BUNDLE_SCHEMA = "quactlize.fully_quantized_kpack_prebuilt_bundle.v2"
RECEIPT_SCHEMA = "quactlize.fully_quantized_kpack_binary_receipt.v2"
MAX_PARENTS_PER_BINARY = 32


def authority_count(qtype: int, operator: str) -> int:
    if qtype not in matrix.QTYPES:
        raise ValueError(f"unknown qtype {qtype}")
    if operator == "dense":
        return len(matrix.provider_rows(qtype))
    if operator == "grouped":
        return len(matrix.grouped_rows(qtype))
    raise ValueError(f"unknown operator {operator}")


def authority_parent_ids(qtype: int, operator: str) -> list[str]:
    if operator == "dense":
        return [matrix.parent_id(qtype, row, provider, delivery_n, "dense")
                for row, provider, delivery_n in matrix.provider_rows(qtype)]
    if operator == "grouped":
        return [matrix.parent_id(qtype, row, 0, delivery_n, "grouped", algorithm)
                for row, delivery_n, algorithm in matrix.grouped_rows(qtype)]
    raise ValueError(f"unknown operator {operator}")


def shard_key(qtype: int, operator: str, begin: int, end: int) -> str:
    return f"q{qtype}-{operator}-p{begin:05d}-{end:05d}"


def plan(pilot: bool = False,
         max_parents: int = MAX_PARENTS_PER_BINARY) -> list[dict]:
    if isinstance(max_parents, bool) or not 1 <= max_parents <= MAX_PARENTS_PER_BINARY:
        raise ValueError("max parents per binary must be in [1,32]")
    rows: list[dict] = []
    for qtype in ((10,) if pilot else matrix.QTYPES):
        for operator in ("dense", "grouped"):
            total = authority_count(qtype, operator)
            stops = (min(total, max_parents),) if pilot else tuple(
                range(max_parents, total, max_parents)) + (total,)
            begin = 0
            for end in stops:
                parent_ids = authority_parent_ids(qtype, operator)[begin:end]
                rows.append({
                    "shard_key": shard_key(qtype, operator, begin, end),
                    "qtype": qtype,
                    "operator": operator,
                    "route": "fully-quantized",
                    "layout": matrix.layout_for(qtype),
                    "parent_begin": begin,
                    "parent_end": end,
                    "parent_count": end - begin,
                    "authority_count": total,
                    "parent_ids": parent_ids,
                })
                begin = end
    return rows


def validate_index(document: dict) -> list[dict]:
    if not isinstance(document, dict) or document.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("FQ K-pack bundle schema differs")
    mode = document.get("mode")
    if mode not in ("FULL", "PILOT"):
        raise ValueError("FQ K-pack bundle mode differs")
    maximum = document.get("max_parents_per_binary")
    expected = plan(mode == "PILOT", maximum)
    shards = document.get("shards")
    if not isinstance(shards, dict) or set(shards) != {
            row["shard_key"] for row in expected}:
        raise ValueError("FQ K-pack shard key union has a gap or overlap")
    by_key = {row["shard_key"]: row for row in expected}
    for key, shard in shards.items():
        if not isinstance(shard, dict):
            raise ValueError(f"{key}: shard record is malformed")
        wanted = by_key[key]
        for field, value in wanted.items():
            if shard.get(field) != value:
                raise ValueError(f"{key}: {field} differs from parent authority")
        if shard.get("typed_rows") != wanted["parent_count"]:
            raise ValueError(f"{key}: linked binary parent count differs")
    return expected


def validate_receipt(receipt: dict, shard: dict,
                     manifest_sha256: str, binary_sha256: str) -> None:
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("FQ K-pack binary receipt schema differs")
    for field in ("shard_key", "qtype", "operator", "route",
                  "parent_begin", "parent_end", "parent_count",
                  "authority_count", "parent_ids"):
        if receipt.get(field) != shard.get(field):
            raise ValueError(f"binary receipt stale range: {field}")
    if (receipt.get("manifest_sha256") != manifest_sha256 or
            receipt.get("binary_sha256") != binary_sha256):
        raise ValueError("binary receipt stale payload hash")


def self_test() -> None:
    full = plan(False, 32)
    pilot = plan(True, 32)
    if len(full) != 1323 or len(pilot) != 2:
        raise ValueError(f"range count differs full={len(full)} pilot={len(pilot)}")
    if any(not 1 <= row["parent_count"] <= 32 for row in full + pilot):
        raise ValueError("one binary exceeds the 32-parent cap")
    base = {
        "schema": BUNDLE_SCHEMA, "mode": "FULL",
        "max_parents_per_binary": 32,
        "shards": {row["shard_key"]: {**row, "typed_rows": row["parent_count"]}
                   for row in full},
    }
    validate_index(base)
    plants = []
    gap = copy.deepcopy(base); gap["shards"].pop(full[len(full)//2]["shard_key"])
    plants.append(gap)
    overlap = copy.deepcopy(base)
    target = full[len(full)//2]
    overlap["shards"][target["shard_key"]]["parent_begin"] -= 1
    plants.append(overlap)
    out_of_range = copy.deepcopy(base)
    last = full[-1]
    out_of_range["shards"][last["shard_key"]]["parent_end"] += 1
    plants.append(out_of_range)
    for planted in plants:
        try:
            validate_index(planted)
        except ValueError:
            continue
        raise ValueError("range gap/overlap/out-of-range plant stayed green")
    shard = {**full[0], "typed_rows": full[0]["parent_count"]}
    receipt = {"schema": RECEIPT_SCHEMA, **full[0],
               "manifest_sha256": "a" * 64, "binary_sha256": "b" * 64}
    validate_receipt(receipt, shard, "a" * 64, "b" * 64)
    stale = copy.deepcopy(receipt); stale["parent_end"] += 1
    try:
        validate_receipt(stale, shard, "a" * 64, "b" * 64)
    except ValueError:
        pass
    else:
        raise ValueError("stale range receipt plant stayed green")
    print("[fq-kpack-bundle-index:self-test] PASS full=1323 pilot=2 "
          "parents-per-binary<=32 gap+overlap+out-of-range+stale=RED")


if __name__ == "__main__":
    self_test()
