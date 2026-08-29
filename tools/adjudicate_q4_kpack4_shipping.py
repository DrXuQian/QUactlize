#!/usr/bin/env python3
"""Combine decode, ScaleFirst prefill and production K-pack4 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any


DECODE_SCHEMA = "quactlize.fq_q4k_kpack4_vs_xplane_decode.v1"
PREFILL_SCHEMA = "quactlize.scalefirst-q4k-kpack4-prefill-real-result.v2"
OUTPUT_SCHEMA = "quactlize.q4_kpack4_shipping_verdict.v1"
FAMILIES = {
    (1024, 5120), (5120, 8192), (5120, 25600),
    (8192, 5120), (25600, 5120),
}


class VerdictError(ValueError):
    pass


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_production(results: pathlib.Path) -> dict[str, Any]:
    required = {
        "source": results / "source-head.txt",
        "inventory": results / "inventory-audit.log",
        "pytest": results / "pytest-production.log",
        "authority": results / "authority.sha256",
    }
    if any(not path.is_file() or not path.stat().st_size
           for path in required.values()):
        raise VerdictError("production closure evidence is incomplete")
    source = required["source"].read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise VerdictError("production source SHA is malformed")
    inventory = required["inventory"].read_text()
    pytest = required["pytest"].read_text()
    if "Q4_KPACK4_PRODUCTION_INVENTORY" not in inventory or \
            "decode_split4=PASS" not in inventory or \
            "prefill_persistent=PASS" not in inventory or \
            "bad_mapping=EXPECTED_RED" not in inventory:
        raise VerdictError("production inventory closure differs")
    if "Q4_KPACK4_PRODUCTION dense=M1/M4/M64/M2048" not in pytest or \
            not re.search(r"(^|\s)1 passed(\s|,|$)", pytest) or \
            re.search(r"(^|\s)[1-9][0-9]* skipped(\s|,|$)", pytest):
        raise VerdictError("production numeric closure is not one unskipped pass")
    authority_rows = []
    for line in required["authority"].read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise VerdictError("production authority row is malformed")
        bound = pathlib.Path(fields[1].lstrip("*"))
        if not bound.is_file() or sha256(bound) != fields[0]:
            raise VerdictError(f"production authority target differs: {bound}")
        authority_rows.append(bound.resolve())
    if len(authority_rows) != 4 or \
            required["inventory"].resolve() not in authority_rows or \
            required["pytest"].resolve() not in authority_rows:
        raise VerdictError("production authority denominator differs")
    return {
        "results": str(results.resolve()),
        "source_sha": source,
        "inventory_sha256": sha256(required["inventory"]),
        "pytest_sha256": sha256(required["pytest"]),
        "authority_sha256": sha256(required["authority"]),
        "verdict": "PRODUCTION_NUMERIC_AND_ABI_PASS",
    }


def decode_score(value: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    if value.get("schema") != DECODE_SCHEMA or \
            value.get("shape_count") != 20 or value.get("family_count") != 5:
        raise VerdictError("decode comparison denominator/schema differs")
    rows = value.get("family_minimax")
    if not isinstance(rows, list) or \
            {(int(row["N"]), int(row["K"])) for row in rows} != FAMILIES:
        raise VerdictError("decode family denominator differs")
    scores = []
    for row in rows:
        candidates = row.get("all_scores")
        if not isinstance(candidates, list):
            raise VerdictError("decode family lacks all_scores")
        matches = [item for item in candidates if item.get("family") == "kpack4"]
        if len(matches) != 1 or matches[0].get("physical_layout_class") != \
                "q4-kpack4-transpose-v1":
            raise VerdictError("decode family lost canonical K-pack4 score")
        score = float(matches[0]["max_regret"]) * 100.0
        scores.append({"N": int(row["N"]), "K": int(row["K"]),
                       "max_regret_pct": score})
    return max(row["max_regret_pct"] for row in scores), scores


def prefill_score(value: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    if value.get("schema") != PREFILL_SCHEMA or \
            value.get("shape_count") != 15 or value.get("family_count") != 5 or \
            value.get("scope") != "PERSISTENT_SCALEFIRST_FULL_OUTPUT_REAL_SHAPES":
        raise VerdictError("prefill comparison denominator/schema differs")
    rows = value.get("families")
    if not isinstance(rows, list) or \
            {(int(row["N"]), int(row["K"])) for row in rows} != FAMILIES:
        raise VerdictError("prefill family denominator differs")
    if any(row.get("M_values") != [64, 2048, 4096] for row in rows):
        raise VerdictError("prefill family lost one M")
    score = float(value["worst_kpack4_regression_pct"])
    recomputed = max(float(row["max_kpack4_regression_pct"]) for row in rows)
    if abs(score - recomputed) > 1e-9:
        raise VerdictError("prefill worst regression differs from family rows")
    return score, rows


def adjudicate(decode_path: pathlib.Path, prefill_path: pathlib.Path,
               production_results: pathlib.Path, decode_threshold: float,
               prefill_threshold: float) -> dict[str, Any]:
    if not 0 < decode_threshold < 100 or not 0 < prefill_threshold < 100:
        raise VerdictError("performance thresholds are invalid")
    decode = json.loads(decode_path.read_text())
    prefill = json.loads(prefill_path.read_text())
    decode_worst, decode_families = decode_score(decode)
    prefill_worst, prefill_families = prefill_score(prefill)
    production = require_production(production_results)
    decode_pass = decode_worst <= decode_threshold
    prefill_pass = prefill_worst <= prefill_threshold
    ready = decode_pass and prefill_pass
    return {
        "schema": OUTPUT_SCHEMA,
        "canonical_offline_format": {
            "name": "q4-kpack4-transpose-v1",
            "mapping_id": "0x51344b5034540001",
            "descriptor": "v2/layout1/artifact_tile_k0/transport_tile_k64",
        },
        "coverage": {"decode_shapes": 20, "prefill_shapes": 15,
                     "total_shapes": 35, "families": 5,
                     "decode_m": [1, 2, 4, 8],
                     "prefill_m": [64, 2048, 4096]},
        "decode": {
            "comparison": str(decode_path.resolve()),
            "sha256": sha256(decode_path),
            "threshold_pct": decode_threshold,
            "worst_family_regret_pct": decode_worst,
            "within_threshold": decode_pass,
            "families": decode_families,
        },
        "prefill": {
            "comparison": str(prefill_path.resolve()),
            "sha256": sha256(prefill_path),
            "threshold_pct": prefill_threshold,
            "worst_family_regression_pct": prefill_worst,
            "within_threshold": prefill_pass,
            "families": prefill_families,
        },
        "production_closure": production,
        "verdict": ("KPACK4_SHIP_READY" if ready else
                    "KPACK4_HOLD_PERFORMANCE_THRESHOLD"),
        "archive": ({
            "action": "REMOVE_FROM_Q4_PRODUCTION_SELECTION_AND_SWEEP_ONLY",
            "keep_shared_reader_source_until_cross_format_audit": True,
            "candidates": [
                "xplane-q4k-fold2-a32",
                "xplane-q4k-tile-free-f1-le256",
                "xplane-q4k-a128-scalefirst-fp16",
                "xplane-q4k-a256-scalefirst-fp16",
            ],
        } if ready else {
            "action": "NONE",
            "reason": "performance threshold is not closed",
            "candidates": [],
        }),
    }


def self_test() -> None:
    families = sorted(FAMILIES)
    decode = {
        "schema": DECODE_SCHEMA, "shape_count": 20, "family_count": 5,
        "family_minimax": [{
            "N": n, "K": k,
            "all_scores": [{"family": "kpack4",
                            "physical_layout_class": "q4-kpack4-transpose-v1",
                            "max_regret": .04}],
        } for n, k in families],
    }
    prefill = {
        "schema": PREFILL_SCHEMA, "shape_count": 15, "family_count": 5,
        "scope": "PERSISTENT_SCALEFIRST_FULL_OUTPUT_REAL_SHAPES",
        "worst_kpack4_regression_pct": 2.0,
        "families": [{"N": n, "K": k, "M_values": [64, 2048, 4096],
                      "max_kpack4_regression_pct": 2.0}
                     for n, k in families],
    }
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-ship-") as temp:
        root = pathlib.Path(temp)
        decode_path, prefill_path = root / "decode.json", root / "prefill.json"
        results = root / "production"; results.mkdir()
        decode_path.write_text(json.dumps(decode)); prefill_path.write_text(json.dumps(prefill))
        (results / "source-head.txt").write_text("a" * 40 + "\n")
        (results / "inventory-audit.log").write_text(
            "Q4_KPACK4_PRODUCTION_INVENTORY decode_split4=PASS "
            "prefill_persistent=PASS bad_mapping=EXPECTED_RED\n")
        (results / "pytest-production.log").write_text(
            "Q4_KPACK4_PRODUCTION dense=M1/M4/M64/M2048 grouped_rows=[2,0,3,1]\n"
            "1 passed, 82 deselected in 2.9s\n")
        packed = root / "packed.so"; scalefirst = root / "scalefirst.so"
        packed.write_text("packed"); scalefirst.write_text("scalefirst")
        bound = (packed, scalefirst, results / "inventory-audit.log",
                 results / "pytest-production.log")
        (results / "authority.sha256").write_text("".join(
            f"{sha256(path)}  {path}\n" for path in bound))
        positive = adjudicate(decode_path, prefill_path, results, 5.0, 3.0)
        if positive["verdict"] != "KPACK4_SHIP_READY" or \
                positive["coverage"]["total_shapes"] != 35 or \
                len(positive["archive"]["candidates"]) != 4:
            raise AssertionError("shipping positive differs")
        broken = json.loads(json.dumps(decode))
        broken["family_minimax"][0]["all_scores"][0]["max_regret"] = .06
        decode_path.write_text(json.dumps(broken))
        held = adjudicate(decode_path, prefill_path, results, 5.0, 3.0)
        if held["verdict"] != "KPACK4_HOLD_PERFORMANCE_THRESHOLD" or \
                held["archive"]["candidates"]:
            raise AssertionError("shipping hold differs")
    print("[q4-kpack4-shipping:self-test] PASS production closure plus exact "
          "20-decode/15-prefill denominator; threshold and archive RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("run")
    run.add_argument("--decode-comparison", type=pathlib.Path, required=True)
    run.add_argument("--prefill-summary", type=pathlib.Path, required=True)
    run.add_argument("--production-results", type=pathlib.Path, required=True)
    run.add_argument("--decode-threshold-pct", type=float, default=5.0)
    run.add_argument("--prefill-threshold-pct", type=float, default=3.0)
    run.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            result = adjudicate(
                args.decode_comparison, args.prefill_summary,
                args.production_results, args.decode_threshold_pct,
                args.prefill_threshold_pct)
            atomic_json(args.output, result)
            print("Q4_KPACK4_SHIPPING_VERDICT "
                  f"verdict={result['verdict']} shapes=35 families=5 "
                  f"decode_worst_pct={result['decode']['worst_family_regret_pct']:.6f} "
                  f"decode_threshold_pct={result['decode']['threshold_pct']:.6f} "
                  f"prefill_worst_pct={result['prefill']['worst_family_regression_pct']:.6f} "
                  f"prefill_threshold_pct={result['prefill']['threshold_pct']:.6f} "
                  f"archive={result['archive']['action']}")
        return 0
    except (AssertionError, KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        print(f"[q4-kpack4-shipping] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
