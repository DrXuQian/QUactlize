#!/usr/bin/env python3
"""Adjudicate matched plain/fused-store K-pack4 over all real decode shapes."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import tempfile
from typing import Any

import analyze_fq_q4k_decode_real_shapes as decode
import analyze_fq_q4k_kpack4_pilot as pilot
import gen_fully_quantized_splitk_producer_units as generator
import plan_fq_q4k_decode_real_shapes as planner


SCHEMA = "quactlize.fq_q4k_kpack4_fused_store_real_shapes.v1"
DELIVERY_N = 32
ARMS = {"plain": False, "store": True}


class AnalysisError(ValueError):
    pass


def union_symbols(manifest_path: pathlib.Path,
                  inputs: list[pathlib.Path], output: pathlib.Path) -> int:
    rows, _ = pilot.load_manifest(manifest_path)
    if len(inputs) != 2:
        raise AnalysisError("symbol union requires exact plain/store inputs")
    selected: set[str] = set()
    for path in inputs:
        values = decode.read_symbols(path)
        if not values or any(symbol not in rows for symbol in values):
            raise AnalysisError(f"{path}: symbol set is empty or foreign")
        selected.update(values)
    ordered = [symbol for symbol in rows if symbol in selected]
    if len(ordered) != len(selected):
        raise AnalysisError("symbol union lost a manifest row")
    decode.atomic_text(output, "".join(f"{symbol}\n" for symbol in ordered))
    return len(ordered)


def load_aggregate(path: pathlib.Path, expected_fused: bool,
                   plan_path: pathlib.Path, policy_path: pathlib.Path
                   ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = json.loads(path.read_text())
    if value.get("schema") != pilot.REAL_SHAPES_SCHEMA or \
            value.get("shape_count") != 20 or value.get("family_count") != 5 or \
            value.get("plan_sha256") != decode.sha256(plan_path) or \
            value.get("policy_sha256") != decode.sha256(policy_path) or \
            value.get("layout") != pilot.KPACK4_CLASS or \
            value.get("scalezero_fused") is not expected_fused or \
            value.get("weight_delivery_n") != DELIVERY_N:
        raise AnalysisError(f"{path}: aggregate identity differs")
    rows = value.get("shape_winners")
    if not isinstance(rows, list) or len(rows) != 20:
        raise AnalysisError(f"{path}: shape winner denominator differs")
    by_key = {str(row.get("shape_key")): row for row in rows}
    if len(by_key) != 20:
        raise AnalysisError(f"{path}: duplicate shape winner")
    return value, by_key


def winner_contract(row: dict[str, Any], shape: list[int]) -> dict[str, Any]:
    if row.get("shape") != shape:
        raise AnalysisError("shape winner coordinates differ")
    winner = row.get("winner")
    if not isinstance(winner, dict):
        raise AnalysisError("shape winner is missing")
    required = ("symbol", "config", "algorithm", "split", "median_us",
                "min_us", "max_us", "producer_median_us",
                "modeled_reducer_us")
    if any(key not in winner for key in required) or \
            float(winner["median_us"]) <= 0 or \
            float(winner["min_us"]) <= 0 or \
            float(winner["max_us"]) < float(winner["min_us"]):
        raise AnalysisError("shape winner timing identity differs")
    return winner


def compare(plan_path: pathlib.Path, policy_path: pathlib.Path,
            manifest_path: pathlib.Path, raw_root: pathlib.Path,
            plain_path: pathlib.Path, store_path: pathlib.Path,
            output_json: pathlib.Path, output_tsv: pathlib.Path,
            threshold: float) -> dict[str, Any]:
    if not 0 < threshold < 1:
        raise AnalysisError("material threshold must be in (0,1)")
    plan = json.loads(plan_path.read_text())
    planner.validate_plan(plan)
    rows, _ = pilot.load_manifest(manifest_path)
    _, plain_rows = load_aggregate(
        plain_path, False, plan_path, policy_path)
    _, store_rows = load_aggregate(
        store_path, True, plan_path, policy_path)
    expected_keys = {str(row["shape_key"]) for row in plan["shapes"]}
    if set(plain_rows) != expected_keys or set(store_rows) != expected_keys:
        raise AnalysisError("plain/store shape denominator differs from plan")

    output_rows: list[dict[str, Any]] = []
    census: collections.Counter[str] = collections.Counter()
    for shape_obj in sorted(plan["shapes"], key=lambda row: row["shape_key"]):
        key = str(shape_obj["shape_key"])
        shape = [int(shape_obj[name]) for name in ("m", "n", "k")]
        plain_detail = json.loads(
            (raw_root / "plain" / key / "summary.json").read_text())
        store_detail = json.loads(
            (raw_root / "store" / key / "summary.json").read_text())
        if plain_detail.get("scalezero_fused") is not False or \
                store_detail.get("scalezero_fused") is not True or \
                plain_detail.get("weight_delivery_n") != DELIVERY_N or \
                store_detail.get("weight_delivery_n") != DELIVERY_N or \
                plain_detail.get("symbols_sha256") != store_detail.get("symbols_sha256") or \
                plain_detail.get("confirmed_symbols") != store_detail.get("confirmed_symbols"):
            raise AnalysisError(f"{key}: matched confirmation identity differs")
        plain = winner_contract(plain_rows[key], shape)
        store = winner_contract(store_rows[key], shape)
        if plain["symbol"] not in rows or store["symbol"] not in rows:
            raise AnalysisError(f"{key}: winning symbol is foreign")
        delta = float(store["median_us"]) / float(plain["median_us"]) - 1.0
        if float(store["max_us"]) < float(plain["min_us"]):
            envelope = "RESOLVED_STORE_FASTER"
        elif float(plain["max_us"]) < float(store["min_us"]):
            envelope = "RESOLVED_STORE_SLOWER"
        else:
            envelope = "OVERLAPPING_ENVELOPES"
        if delta <= -threshold:
            material = "STORE_FASTER"
        elif delta >= threshold:
            material = "STORE_SLOWER"
        else:
            material = "WITHIN_THRESHOLD"
        census[material] += 1
        row = {
            "shape_key": key, "shape": shape, "delta": delta,
            "material_verdict": material, "envelope_verdict": envelope,
            "confirmed_symbols": int(plain_detail["confirmed_symbols"]),
            "plain": plain, "store": store,
            "plain_provider": rows[plain["symbol"]]["a_provider"],
            "store_provider": rows[store["symbol"]]["a_provider"],
        }
        output_rows.append(row)
        print("FQ_KPACK4_FUSED_STORE_SHAPE "
              f"shape={key} verdict={material} envelope={envelope} "
              f"plain_us={float(plain['median_us']):.9f} "
              f"store_us={float(store['median_us']):.9f} "
              f"delta_pct={100 * delta:.6f} "
              f"plain={plain['config']}/{plain['algorithm']} "
              f"store={store['config']}/{store['algorithm']}")

    families = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = \
        collections.defaultdict(list)
    for row in output_rows:
        grouped[(row["shape"][1], row["shape"][2])].append(row)
    for (n, k), family_rows in sorted(grouped.items()):
        if {row["shape"][0] for row in family_rows} != set(planner.DECODE_M):
            raise AnalysisError(f"family {n}x{k} lost one decode M")
        families.append({
            "N": n, "K": k,
            "max_store_regression": max(row["delta"] for row in family_rows),
            "mean_delta": sum(row["delta"] for row in family_rows) /
                          len(family_rows),
        })
    result = {
        "schema": SCHEMA,
        "plan_sha256": decode.sha256(plan_path),
        "policy_sha256": decode.sha256(policy_path),
        "manifest_sha256": decode.sha256(manifest_path),
        "shape_count": len(output_rows), "family_count": len(families),
        "delivery_n": DELIVERY_N, "material_threshold": threshold,
        "census": dict(sorted(census.items())),
        "worst_store_regression": max(row["delta"] for row in output_rows),
        "best_store_gain": min(row["delta"] for row in output_rows),
        "shapes": output_rows, "families": families,
    }
    decode.atomic_json(output_json, result)
    lines = [
        "shape\tM\tN\tK\tverdict\tenvelope\tplain_us\tstore_us\t"
        "delta_pct\tplain_provider\tstore_provider\tplain_config\t"
        "store_config\tplain_algorithm\tstore_algorithm\tconfirmed"
    ]
    for row in output_rows:
        lines.append("\t".join(map(str, (
            row["shape_key"], *row["shape"], row["material_verdict"],
            row["envelope_verdict"], row["plain"]["median_us"],
            row["store"]["median_us"], 100 * row["delta"],
            row["plain_provider"], row["store_provider"],
            row["plain"]["config"], row["store"]["config"],
            row["plain"]["algorithm"], row["store"]["algorithm"],
            row["confirmed_symbols"],
        ))))
    decode.atomic_text(output_tsv, "\n".join(lines) + "\n")
    print("FQ_KPACK4_FUSED_STORE_CENSUS "
          f"shapes={len(output_rows)} families={len(families)} "
          f"verdicts={json.dumps(dict(sorted(census.items())), sort_keys=True)} "
          f"worst_regression_pct={100 * result['worst_store_regression']:.6f} "
          f"best_gain_pct={100 * result['best_store_gain']:.6f}")
    return result


def synthetic_plan(root: pathlib.Path, policy: pathlib.Path) -> pathlib.Path:
    families = ((1024, 5120), (5120, 8192), (5120, 25600),
                (8192, 5120), (25600, 5120))
    materialized = {"cells": [{
        "qtype": 12, "inventory_status": "SUPPORTED",
        "problem_route": "dense", "grouped": None, "group_size": 32,
        "model_id": "model", "m": 1, "n": n, "k": k,
        "tp_world": 1, "tp_rank": 0, "tp_partition": "replicated",
        "sources": [[f"tensor_{n}_{k}", "weight", f"{index + 1:064x}"]],
    } for index, (n, k) in enumerate(families)]}
    value = planner.build_plan(
        materialized, decode.load_policy(policy), "a" * 64)
    value["policy_sha256"] = decode.sha256(policy)
    path = root / "plan.json"
    decode.atomic_json(path, value)
    return path


def self_test(policy: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-fused-store-") as temp:
        root = pathlib.Path(temp)
        manifest_dir = root / "generated"
        generator.generate(12, 0, 0, manifest_dir, 4, False, 8, "q4-kpack4")
        manifest = manifest_dir / "manifest.json"
        manifest_rows, _ = pilot.load_manifest(manifest)
        symbols = list(manifest_rows)[:2]
        left, right = root / "left.txt", root / "right.txt"
        decode.atomic_text(left, symbols[0] + "\n")
        decode.atomic_text(right, symbols[1] + "\n")
        union = root / "union.txt"
        if union_symbols(manifest, [left, right], union) != 2:
            raise AnalysisError("synthetic symbol union differs")
        plan = synthetic_plan(root, policy)
        plan_value = json.loads(plan.read_text())
        raw = root / "raw"
        aggregates = {}
        for arm, fused in ARMS.items():
            shape_rows = []
            for shape_obj in plan_value["shapes"]:
                key = shape_obj["shape_key"]
                shape = [shape_obj[name] for name in ("m", "n", "k")]
                directory = raw / arm / key
                directory.mkdir(parents=True)
                median = 9.5 if fused else 10.0
                winner = {
                    "symbol": symbols[0], "config": "fixture",
                    "algorithm": "TC_SPLITK_S4_MODELED_E2E", "split": 4,
                    "producer_median_us": median - .1,
                    "modeled_reducer_us": .1, "median_us": median,
                    "min_us": median - .05, "max_us": median + .05,
                }
                detail = {
                    "schema": pilot.SCHEMA, "shape": shape,
                    "scalezero_fused": fused, "weight_delivery_n": DELIVERY_N,
                    "confirmed_symbols": 2, "symbols_sha256": "b" * 64,
                    "global": {"winner": winner},
                }
                decode.atomic_json(directory / "summary.json", detail)
                shape_rows.append({"shape_key": key, "shape": shape,
                                   "winner": winner})
            aggregate = {
                "schema": pilot.REAL_SHAPES_SCHEMA,
                "shape_count": 20, "family_count": 5,
                "plan_sha256": decode.sha256(plan),
                "policy_sha256": decode.sha256(policy),
                "layout": pilot.KPACK4_CLASS,
                "scalezero_fused": fused, "weight_delivery_n": DELIVERY_N,
                "shape_winners": shape_rows,
            }
            path = root / f"{arm}.json"
            decode.atomic_json(path, aggregate)
            aggregates[arm] = path
        result = compare(
            plan, policy, manifest, raw, aggregates["plain"],
            aggregates["store"], root / "summary.json", root / "summary.tsv",
            .02)
        if result["census"] != {"STORE_FASTER": 20}:
            raise AnalysisError("synthetic fused-store census differs")
        victim = raw / "store" / plan_value["shapes"][0]["shape_key"] / \
            "summary.json"
        broken = json.loads(victim.read_text())
        broken["symbols_sha256"] = "c" * 64
        decode.atomic_json(victim, broken)
        try:
            compare(plan, policy, manifest, raw, aggregates["plain"],
                    aggregates["store"], root / "red.json", root / "red.tsv",
                    .02)
        except AnalysisError:
            pass
        else:
            raise AnalysisError("mismatched confirm union stayed green")
    print("[fq-kpack4-fused-store-analysis:self-test] PASS exact 20-shape "
          "plain/store D32 denominator, matched confirm unions and one RED plant")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    union_parser = sub.add_parser("union-symbols")
    union_parser.add_argument("--manifest", type=pathlib.Path, required=True)
    union_parser.add_argument("--input", type=pathlib.Path, action="append",
                              required=True)
    union_parser.add_argument("--output", type=pathlib.Path, required=True)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--plan", type=pathlib.Path, required=True)
    compare_parser.add_argument("--policy", type=pathlib.Path, required=True)
    compare_parser.add_argument("--manifest", type=pathlib.Path, required=True)
    compare_parser.add_argument("--raw-root", type=pathlib.Path, required=True)
    compare_parser.add_argument("--plain", type=pathlib.Path, required=True)
    compare_parser.add_argument("--store", type=pathlib.Path, required=True)
    compare_parser.add_argument("--output-json", type=pathlib.Path, required=True)
    compare_parser.add_argument("--output-tsv", type=pathlib.Path, required=True)
    compare_parser.add_argument("--threshold", type=float, default=.02)
    self_parser = sub.add_parser("self-test")
    self_parser.add_argument("--policy", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "union-symbols":
            count = union_symbols(args.manifest, args.input, args.output)
            print(f"[fq-kpack4-fused-store-union] PASS symbols={count} output={args.output}")
        elif args.command == "compare":
            compare(args.plan, args.policy, args.manifest, args.raw_root,
                    args.plain, args.store, args.output_json, args.output_tsv,
                    args.threshold)
        else:
            self_test(args.policy)
        return 0
    except (AnalysisError, KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError, pilot.PilotError) as error:
        print(f"[fq-kpack4-fused-store-analysis] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
