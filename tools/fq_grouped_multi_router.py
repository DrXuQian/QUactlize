#!/usr/bin/env python3
"""Deterministic grouped-routing histograms for K-pack selector training."""

from __future__ import annotations

import hashlib
import json

EXPERTS = 256
SCHEMA = "quactlize.fq-grouped-multi-router-fixture.v1"


def rows_fnv64(rows: list[int]) -> str:
    value = 14695981039346656037
    for row in rows:
        for byte in row.to_bytes(4, "little", signed=False):
            value ^= byte
            value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"0x{value:016x}"


def _place(values: list[int], positions: list[int]) -> list[int]:
    rows = [0] * EXPERTS
    for value, position in zip(values, positions):
        rows[position] = value
    return rows


def profiles() -> dict[str, list[int]]:
    multiset = [1, 2, 3, 15, 16, 17, 31, 32, 33, 127, 128, 129]
    return {
        "balanced": [4] * EXPERTS,
        "hot-skewed": _place([48] * 16 + [8] * 16, list(range(32))),
        "sparse-empty": _place([1, 2, 3, 5], [0, 17, 129, 255]),
        "tilem-boundary": _place(
            [15, 16, 17, 31, 32, 33, 127, 128, 129], list(range(9))
        ),
        "permutation-a": _place(multiset, list(range(len(multiset)))),
        "permutation-b": _place(
            multiset, [(37 * i + 53) % EXPERTS for i in range(len(multiset))]
        ),
    }


def summarize(name: str, rows: list[int]) -> dict:
    if len(rows) != EXPERTS or any(type(x) is not int or x < 0 for x in rows):
        raise ValueError("router rows must be 256 nonnegative integers")
    active = sum(x > 0 for x in rows)
    result = {
        "schema": SCHEMA,
        "profile": name,
        "experts": EXPERTS,
        "rows": rows,
        "total_rows": sum(rows),
        "max_rows": max(rows),
        "active": active,
        "zero": EXPERTS - active,
        "work_tm16": sum((x + 15) // 16 for x in rows if x),
        "work_tm32": sum((x + 31) // 32 for x in rows if x),
        "work_tm128": sum((x + 127) // 128 for x in rows if x),
    }
    result["rows_sha256"] = hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()
    result["rows_hash"] = rows_fnv64(rows)
    return result


def materialize() -> dict[str, dict]:
    return {name: summarize(name, rows) for name, rows in profiles().items()}


def self_test() -> None:
    value = materialize()
    if set(value) != {
        "balanced",
        "hot-skewed",
        "sparse-empty",
        "tilem-boundary",
        "permutation-a",
        "permutation-b",
    }:
        raise AssertionError("router profile denominator differs")
    a, b = value["permutation-a"], value["permutation-b"]
    if sorted(a["rows"]) != sorted(b["rows"]) or a["rows"] == b["rows"]:
        raise AssertionError("same-multiset permutation control differs")
    if a["rows_hash"] == b["rows_hash"]:
        raise AssertionError("same-multiset permutation hashes collided")
    if not all(
        x in value["tilem-boundary"]["rows"]
        for x in (15, 16, 17, 31, 32, 33, 127, 128, 129)
    ):
        raise AssertionError("TileM boundary ladder differs")
    broken = value["sparse-empty"]["rows"][:-1]
    try:
        summarize("red", broken)
    except ValueError:
        pass
    else:
        raise AssertionError("short histogram negative stayed green")
    print(
        "[fq-grouped-router:self-test] PASS profiles=6 balanced/hot/sparse/TileM/permutation; short RED"
    )


if __name__ == "__main__":
    self_test()
