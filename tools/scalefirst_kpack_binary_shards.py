#!/usr/bin/env python3
"""Plan and validate binary-level ScaleFirst canonical K-pack parent shards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys

import scalefirst_grouped_kpack_matrix as grouped
import scalefirst_internal_matrix as dense


SCHEMA = "quactlize.scalefirst_kpack_binary_shards.v1"
QTYPES = (10, 11, 12, 13, 14)
OPERATORS = ("dense", "grouped")


def authority_symbols(operator: str, qtype: int) -> list[str]:
    if operator == "grouped":
        return [
            f"sfg_q{qtype}_tm{row.tile_m}_tn{row.tile_n}_"
            f"tk{row.tactic_tile_k}_wm{row.warp_m}_wn{row.warp_n}_"
            f"s{row.stages}_ap0_dn{delivery_n}"
            for row, delivery_n in grouped.candidate_rows(qtype)]
    fmt = dense.format_for(qtype)
    layout = dense.KPACK_LAYOUT_BY_QTYPE[qtype]
    rows = dense.kpack_dense_candidates(qtype)
    return [
        f"sf_q{qtype}_a0_tm{row.tile_m}_tn{row.tile_n}_"
        f"tk{row.tactic_tile_k}_wm{row.warp_m}_wn{row.warp_n}_"
        f"s{row.stages}_bc{row.bchunk}_ap{provider}_dn{delivery_n}"
        for row, provider, delivery_n in rows]


def authority_count(operator: str, qtype: int) -> int:
    return len(authority_symbols(operator, qtype))


def symbols_sha256(symbols: list[str]) -> str:
    return hashlib.sha256(("\n".join(symbols) + "\n").encode()).hexdigest()


def shard_id(qtype: int, operator: str, begin: int, end: int) -> str:
    return f"q{qtype}-{operator}-p{begin:06d}-{end:06d}"


def make_plan(scope: str, parents_per_binary: int = 32) -> dict:
    if scope not in {"full", "pilot"}:
        raise ValueError("scope must be full or pilot")
    if parents_per_binary <= 0 or parents_per_binary > 32:
        raise ValueError("parents-per-binary must be in [1,32]")
    pairs = [(qtype, operator) for qtype in QTYPES for operator in OPERATORS]
    if scope == "pilot":
        pairs = [(10, "dense"), (10, "grouped")]
    pair_rows = []
    shards = []
    for qtype, operator in pairs:
        symbols = authority_symbols(operator, qtype)
        total = len(symbols)
        pair_rows.append({
            "qtype": qtype, "operator": operator,
            "layout": grouped.layout_for(qtype),
            "authority_parents": total,
            "authority_parent_symbols_sha256": symbols_sha256(symbols),
        })
        limit = total if scope == "full" else min(total, parents_per_binary)
        for begin in range(0, limit, parents_per_binary):
            end = min(begin + parents_per_binary, limit)
            shards.append({
                "shard_id": shard_id(qtype, operator, begin, end),
                "qtype": qtype, "operator": operator,
                "layout": grouped.layout_for(qtype),
                "parent_begin": begin, "parent_end": end,
                "compiled_parents": end - begin,
                "authority_parents": total,
            })
    document = {
        "schema": SCHEMA,
        "scope": scope,
        "parents_per_binary": parents_per_binary,
        "pairs": pair_rows,
        "shards": shards,
    }
    validate_plan(document)
    return document


def validate_plan(document: dict) -> None:
    if document.get("schema") != SCHEMA:
        raise ValueError("binary shard schema differs")
    scope = document.get("scope")
    width = document.get("parents_per_binary")
    if scope not in {"full", "pilot"} or not isinstance(width, int) or \
            width <= 0 or width > 32:
        raise ValueError("binary shard scope/width differs")
    expected_pairs = ([(qtype, operator) for qtype in QTYPES
                       for operator in OPERATORS] if scope == "full" else
                      [(10, "dense"), (10, "grouped")])
    pairs = document.get("pairs")
    if not isinstance(pairs, list) or [
            (row.get("qtype"), row.get("operator")) for row in pairs
            ] != expected_pairs:
        raise ValueError("binary shard pair denominator differs")
    live = {}
    for row in pairs:
        qtype, operator = int(row["qtype"]), str(row["operator"])
        symbols = authority_symbols(operator, qtype)
        total = len(symbols)
        if row != {"qtype": qtype, "operator": operator,
                   "layout": grouped.layout_for(qtype),
                   "authority_parents": total,
                   "authority_parent_symbols_sha256": symbols_sha256(symbols)}:
            raise ValueError(f"q{qtype}/{operator}: authority identity differs")
        live[(qtype, operator)] = total
    shards = document.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("binary shard list is empty")
    seen_ids = set()
    by_pair = {pair: [] for pair in expected_pairs}
    for row in shards:
        qtype, operator = row.get("qtype"), row.get("operator")
        pair = (qtype, operator)
        begin, end = row.get("parent_begin"), row.get("parent_end")
        if pair not in by_pair or not isinstance(begin, int) or \
                not isinstance(end, int) or begin < 0 or end <= begin or \
                end > live[pair] or end - begin > width:
            raise ValueError("binary shard range is out of authority")
        expected = {
            "shard_id": shard_id(qtype, operator, begin, end),
            "qtype": qtype, "operator": operator,
            "layout": grouped.layout_for(qtype),
            "parent_begin": begin, "parent_end": end,
            "compiled_parents": end - begin,
            "authority_parents": live[pair],
        }
        if row != expected or row["shard_id"] in seen_ids:
            raise ValueError("binary shard identity is duplicate or noncanonical")
        seen_ids.add(row["shard_id"])
        by_pair[pair].append((begin, end))
    for pair, ranges in by_pair.items():
        ranges.sort()
        expected_end = live[pair] if scope == "full" else min(live[pair], width)
        cursor = 0
        for begin, end in ranges:
            if begin < cursor:
                raise ValueError(f"q{pair[0]}/{pair[1]}: binary shard overlap")
            if begin > cursor:
                raise ValueError(f"q{pair[0]}/{pair[1]}: binary shard gap")
            cursor = end
        if cursor != expected_end:
            raise ValueError(f"q{pair[0]}/{pair[1]}: binary shard tail gap")
        if scope == "pilot" and ranges != [(0, expected_end)]:
            raise ValueError("pilot must contain exactly the first parent shard")


def self_test() -> None:
    full = make_plan("full", 32)
    pilot = make_plan("pilot", 32)
    q4_dense_anchor = (
        "sf_q12_a0_tm64_tn64_tk64_wm64_wn32_s3_bc0_ap0_dn64")
    q4_grouped_anchor = (
        "sfg_q12_tm64_tn128_tk64_wm64_wn64_s3_ap0_dn64")
    if authority_symbols("dense", 12).count(q4_dense_anchor) != 1:
        raise ValueError("historical Q4 ScaleFirst geometry left the denominator")
    if authority_symbols("grouped", 12).count(q4_grouped_anchor) != 1:
        raise ValueError(
            "historical grouped Q4 ScaleFirst geometry left the denominator")
    if max(row["compiled_parents"] for row in full["shards"]) > 32 or \
            len(pilot["shards"]) != 2 or \
            {(row["qtype"], row["operator"]) for row in pilot["shards"]} != {
                (10, "dense"), (10, "grouped")}:
        raise ValueError("binary shard positive denominator differs")
    plants = []
    overlap = copy.deepcopy(full)
    overlap["shards"][1]["parent_begin"] -= 1
    overlap["shards"][1]["compiled_parents"] += 1
    overlap["shards"][1]["shard_id"] = shard_id(
        overlap["shards"][1]["qtype"], overlap["shards"][1]["operator"],
        overlap["shards"][1]["parent_begin"],
        overlap["shards"][1]["parent_end"])
    plants.append(overlap)
    gap = copy.deepcopy(full)
    gap["shards"][1]["parent_begin"] += 1
    gap["shards"][1]["compiled_parents"] -= 1
    gap["shards"][1]["shard_id"] = shard_id(
        gap["shards"][1]["qtype"], gap["shards"][1]["operator"],
        gap["shards"][1]["parent_begin"], gap["shards"][1]["parent_end"])
    plants.append(gap)
    out_of_range = copy.deepcopy(full)
    out_of_range["shards"][0]["parent_begin"] = -1
    plants.append(out_of_range)
    masquerade = copy.deepcopy(pilot)
    masquerade["scope"] = "full"
    plants.append(masquerade)
    for planted in plants:
        try:
            validate_plan(planted)
        except ValueError:
            continue
        raise ValueError("binary shard negative stayed green")
    print("[sf-kpack-binary-shards:self-test] PASS max-parents=32 "
          "historical-q4-dense/grouped-anchors=INCLUDED "
          "full=EXACT-PARTITION pilot=Q10-FIRST-SHARDS "
          "negatives=overlap+gap+out-of-range+pilot-masquerade")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("emit", "self-test"))
    parser.add_argument("--scope", choices=("full", "pilot"), default="full")
    parser.add_argument("--parents-per-binary", type=int, default=32)
    parser.add_argument("--out", default="-")
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        document = make_plan(args.scope, args.parents_per_binary)
        payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
        if args.out == "-":
            sys.stdout.write(payload)
        else:
            path = pathlib.Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[sf-kpack-binary-shards] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
