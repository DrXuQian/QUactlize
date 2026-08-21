#!/usr/bin/env python3
"""Materialize and summarize the shape-specific real-Q4_K ScaleFirst sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys
from typing import Any

import analyze_fully_quantized_internal_sweep as inventory


MASTER_SCHEMA = "quactlize.scalefirst_q4k_real_shapes_policy.v1"
CELL_POLICY_SCHEMA = "quactlize.scalefirst_q4k_shape_policy.v1"
PLAN_SCHEMA = "quactlize.scalefirst_q4k_real_shapes_plan.v1"
SUMMARY_SCHEMA = "quactlize.scalefirst_q4k_real_shapes_summary.v1"
RESULT_SCHEMA = "quactlize.scalefirst_q4k_pruned_result.v1"
ARTIFACTS = (32, 64, 128, 256)
BOARDS = ("FULL_OUTPUT", "SPLITK_S2_PRODUCER",
          "SPLITK_S4_PRODUCER", "SPLITK_S8_PRODUCER")


class PlanError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_master(path: pathlib.Path) -> dict[str, Any]:
    policy = json.loads(path.read_text())
    expected = {
        "schema": MASTER_SCHEMA, "qtype": 12, "format": "Q4_K",
        "quant_mode": "FinegrainedScaleZero", "group_size": 32,
        "artifact_tile_k": list(ARTIFACTS), "bchunk": 0,
        "problem_route": "dense", "minimum_m": 8,
        "prefill_m": [64, 2048, 4096],
        "boards": list(BOARDS),
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise PlanError(f"master policy {key} differs: {policy.get(key)!r}")
    for phase in ("screen", "scheduler", "confirm"):
        if not isinstance(policy.get(phase), dict):
            raise PlanError(f"master policy lacks {phase}")
    # These values were registered before the first pilot result.  A real-shape
    # runner may materialize them, never silently tune them per shape.
    frozen = {
        "screen": {"iterations": 2, "correctness_repeats": 1,
                   "top_n": 32, "relative_to_leader": 1.20,
                   "top_per_axis_value": 2, "top_per_q": 2,
                   "retain_if_relative_spread_exceeds": 0.05,
                   "algorithm": "NONPERSISTENT"},
        "scheduler": {"iterations": 1, "correctness_repeats": 1,
                      "top_n_per_board": 8,
                      "relative_to_leader": 1.05},
        "confirm": {"iterations": 7, "correctness_repeats": 2,
                    "unresolved_if_sample_envelopes_overlap": True},
    }
    for phase, value in frozen.items():
        if policy[phase] != value:
            raise PlanError(f"master policy changed preregistered {phase}")
    return policy


def dim(cell: dict[str, Any], name: str) -> int:
    value = cell.get(name, cell.get(name.upper()))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlanError(f"inventory cell has invalid {name}={value!r}")
    return value


def shape_key(shape: tuple[int, int, int]) -> str:
    return f"m{shape[0]}_n{shape[1]}_k{shape[2]}_g32"


def layout_identity(artifact: int) -> dict[str, Any]:
    if artifact not in ARTIFACTS:
        raise PlanError(f"unregistered ArtifactTileK {artifact}")
    fold_low = 2 if artifact == 32 else 1
    return {
        "name": f"xplane-q4k-a{artifact}-f{fold_low}x1-scalefirst-fp16",
        "artifact_tile_k": artifact,
        "fold_n": {"low": fold_low, "high": 1},
        "metadata": "FP16_SCALE_ZERO_PLANES",
        # FoldN alone is not a complete consumer contract for the folded A32
        # bytes.  The canonical producer map is shared by exactly the two
        # currently exposed reader classes TN64/WN32 and TN128/WN64.  Keep
        # this next to the offline identity so a report cannot imply that an
        # arbitrary FoldN=2 tactic read the measured artifact.
        "reader_contract": (
            "Q4_A32_CANONICAL_TN_EQ_2WN_WN_GE_32"
            if artifact == 32 else "UNFOLDED_TACTIC_INVARIANT"
        ),
    }


def reference(cell: dict[str, Any]) -> dict[str, Any]:
    sources = cell.get("source_tensors")
    if not isinstance(sources, list) or not sources or \
            any(not isinstance(value, str) or not value for value in sources):
        raise PlanError("inventory cell lacks source_tensors")
    return {
        "model_id": str(cell["model_id"]),
        "shape_id": str(cell["shape_id"]),
        "source_tensors": sorted(set(sources)),
        "tp_world": int(cell.get("tp_world", 1)),
        "tp_rank": int(cell.get("tp_rank", 0)),
        "tp_partition": str(cell.get("tp_partition", "replicated")),
    }


def build_plan(materialized: dict[str, Any], master: dict[str, Any],
               inventory_sha256: str) -> dict[str, Any]:
    cells = materialized.get("cells")
    if not isinstance(cells, list) or not cells:
        raise PlanError("materialized inventory has no cells")
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    census: dict[str, int] = {}
    for cell in cells:
        reason = "SELECTED"
        if int(cell.get("qtype", -1)) != 12:
            reason = "NOT_Q4_K"
        elif cell.get("inventory_status") != "SUPPORTED":
            reason = "INVENTORY_UNSUPPORTED"
        elif cell.get("problem_route") != "dense" or cell.get("grouped"):
            reason = "NOT_DENSE"
        elif int(cell.get("group_size", -1)) != 32:
            raise PlanError("Q4_K inventory cell lost gs=32")
        else:
            shape = tuple(dim(cell, name) for name in ("m", "n", "k"))
            if shape[0] < int(master["minimum_m"]):
                reason = "DECODE_NOT_SCALEFIRST_PREFILL"
            elif shape[0] not in set(map(int, master["prefill_m"])):
                reason = "OUTSIDE_REGISTERED_PREFILL_M"
            else:
                grouped.setdefault(shape, []).append(reference(cell))
        census[reason] = census.get(reason, 0) + 1
    if not grouped:
        raise PlanError("inventory contains no dense Q4_K prefill shapes")
    shapes = []
    for shape in sorted(grouped):
        refs = sorted(grouped[shape], key=canonical)
        if len({canonical(ref) for ref in refs}) != len(refs):
            raise PlanError(f"duplicate source reference for {shape}")
        shapes.append({"shape_key": shape_key(shape), "m": shape[0],
                       "n": shape[1], "k": shape[2], "group_size": 32,
                       "references": refs})
    result_cells = [
        {"cell_key": f"a{artifact}/{entry['shape_key']}",
         "artifact_tile_k": artifact,
         "layout": layout_identity(artifact),
         "shape_key": entry["shape_key"],
         "shape": [entry["m"], entry["n"], entry["k"]],
         "policy": f"policies/a{artifact}/{entry['shape_key']}.json"}
        for artifact in ARTIFACTS for entry in shapes
    ]
    return {
        "schema": PLAN_SCHEMA,
        "inventory_sha256": inventory_sha256,
        "master_policy_sha256": "BOUND_BY_MATERIALIZER",
        "format": {"qtype": 12, "name": "Q4_K",
                   "quant_mode": "FinegrainedScaleZero", "group_size": 32,
                   "bchunk": 0},
        "artifact_tile_k": list(ARTIFACTS),
        "shape_count": len(shapes), "cell_count": len(result_cells),
        "shapes": shapes, "cells": result_cells,
        "inventory_census": dict(sorted(census.items())),
    }


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise PlanError("plan schema differs")
    shapes = plan.get("shapes")
    cells = plan.get("cells")
    if not isinstance(shapes, list) or not shapes or \
            not isinstance(cells, list):
        raise PlanError("plan shape/cell arrays are malformed")
    if plan.get("shape_count") != len(shapes) or \
            plan.get("cell_count") != len(cells):
        raise PlanError("plan declared denominator differs")
    keys = [entry["shape_key"] for entry in shapes]
    if len(keys) != len(set(keys)):
        raise PlanError("plan shape key duplicate")
    expected = {(artifact, key) for artifact in ARTIFACTS for key in keys}
    observed = {(int(cell["artifact_tile_k"]), str(cell["shape_key"]))
                for cell in cells}
    if observed != expected or len(cells) != len(expected):
        raise PlanError(
            f"shape/layout cross product differs missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}")
    for cell in cells:
        if cell.get("layout") != layout_identity(int(cell["artifact_tile_k"])):
            raise PlanError(f"layout identity differs for {cell['cell_key']}")


def cell_policy(master: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": CELL_POLICY_SCHEMA,
        "name": "real-gguf-q4k-prefill-shape-specific",
        "qtype": 12, "format": "Q4_K",
        "quant_mode": "FinegrainedScaleZero", "group_size": 32,
        "artifact_tile_k": int(cell["artifact_tile_k"]), "bchunk": 0,
        "layout": layout_identity(int(cell["artifact_tile_k"])),
        "shape": list(map(int, cell["shape"])), "anchor_symbol": None,
        "source_cell_key": cell["cell_key"],
        "screen": master["screen"], "scheduler": master["scheduler"],
        "confirm": master["confirm"], "boards": list(BOARDS),
    }


def materialize(inventory_path: pathlib.Path, master_path: pathlib.Path,
                output: pathlib.Path, policies_dir: pathlib.Path) -> dict[str, Any]:
    master = load_master(master_path)
    source = json.loads(inventory_path.read_text())
    materialized = inventory._materialize_spec(
        source, inventory_path, sha256(inventory_path))
    plan = build_plan(materialized, master, sha256(inventory_path))
    plan["master_policy_sha256"] = sha256(master_path)
    validate_plan(plan)
    for cell in plan["cells"]:
        path = policies_dir / pathlib.Path(cell["policy"]).relative_to("policies")
        atomic_json(path, cell_policy(master, cell))
        cell["policy_sha256"] = sha256(path)
    atomic_json(output, plan)
    return plan


def winner_candidates(summary: dict[str, Any], artifact: int,
                      board: str) -> list[dict[str, Any]]:
    result = summary["boards"][board]
    candidates = []
    for rank, key in ((0, "winner"), (1, "runner_up")):
        item = result.get(key)
        if item is not None:
            candidates.append({"artifact_tile_k": artifact, "rank": rank,
                               "layout": layout_identity(artifact),
                               "within_layout_verdict": result["verdict"],
                               **item})
    return candidates


def unavailable_board(shape: dict[str, Any], key: str, board: str,
                      layout_terminals: dict[str, Any]
                      ) -> tuple[dict[str, Any], dict[str, Any]]:
    result = {
        "verdict": "UNAVAILABLE", "winner": None, "runner_up": None,
        "confirmed_candidate_count": 0,
        "layout_terminals": layout_terminals,
    }
    row = {
        "shape_key": key, "M": shape["m"], "N": shape["n"],
        "K": shape["k"], "board": board, "verdict": "UNAVAILABLE",
        "ArtifactTileK": "", "FoldN_low": "", "FoldN_high": "",
        "layout": "", "reader_contract": "", "config": "",
        "algorithm": board, "grid": "", "policy": "",
        "median_us": "", "MFU_pct": "", "distinct_MBU_pct": "",
        "runner_gap_us": "",
    }
    return result, row


def summarize(plan_path: pathlib.Path, results_root: pathlib.Path,
              output: pathlib.Path, tsv: pathlib.Path,
              models_root: pathlib.Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    by_shape = {entry["shape_key"]: entry for entry in plan["shapes"]}
    cell_results: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in plan["cells"]:
        artifact = int(cell["artifact_tile_k"])
        key = str(cell["shape_key"])
        path = results_root / f"a{artifact}" / key / "summary.json"
        if not path.is_file():
            raise PlanError(f"missing result {path}")
        result = json.loads(path.read_text())
        if result.get("schema") != RESULT_SCHEMA or \
                result.get("phase") != "CONFIRM" or \
                set(result.get("boards", {})) != set(BOARDS):
            raise PlanError(f"malformed result {path}")
        if result.get("policy_sha256") != cell.get("policy_sha256"):
            raise PlanError(f"result policy binding differs for {cell['cell_key']}")
        cell_results[(key, artifact)] = result
    if len(cell_results) != len(plan["cells"]):
        raise PlanError("result cell denominator differs")

    rows = []
    shape_summaries = []
    for key, shape in by_shape.items():
        boards = {}
        for board in BOARDS:
            candidates = []
            layout_terminals = {}
            for artifact in ARTIFACTS:
                board_result = cell_results[(key, artifact)]["boards"][board]
                if board_result.get("winner") is None:
                    layout_terminals[str(artifact)] = {
                        "terminal_cells": board_result.get("terminal_cells", 0),
                        "terminal_reasons": board_result.get(
                            "terminal_reasons", {}),
                    }
                candidates += winner_candidates(
                    cell_results[(key, artifact)], artifact, board)
            if not candidates:
                boards[board], row = unavailable_board(
                    shape, key, board, layout_terminals)
                rows.append(row)
                continue
            candidates.sort(key=lambda item: (float(item["median_us"]),
                                               item["artifact_tile_k"],
                                               item["cell"]))
            winner = candidates[0]
            runner = candidates[1] if len(candidates) > 1 else None
            overlap = bool(runner and
                           max(map(float, (winner["range_us"][0],
                                           runner["range_us"][0]))) <=
                           min(map(float, (winner["range_us"][1],
                                           runner["range_us"][1]))))
            verdict = "UNRESOLVED" if overlap or \
                winner["within_layout_verdict"] == "UNRESOLVED" else "RESOLVED"
            boards[board] = {"verdict": verdict, "winner": winner,
                             "runner_up": runner,
                             "confirmed_candidate_count": len(candidates)}
            rows.append({"shape_key": key, "M": shape["m"], "N": shape["n"],
                         "K": shape["k"], "board": board, "verdict": verdict,
                         "ArtifactTileK": winner["artifact_tile_k"],
                         "FoldN_low": winner["layout"]["fold_n"]["low"],
                         "FoldN_high": winner["layout"]["fold_n"]["high"],
                         "layout": winner["layout"]["name"],
                         "reader_contract": winner["layout"]["reader_contract"],
                         "config": winner["config"],
                         "algorithm": winner["algorithm"],
                         "grid": winner["grid"], "policy": winner["policy"],
                         "median_us": winner["median_us"],
                         "MFU_pct": winner["MFU_pct_500TF"],
                         "distinct_MBU_pct": winner["distinct_MBU_pct_2766GBs"],
                         "runner_gap_us": "" if runner is None else
                             float(runner["median_us"]) - float(winner["median_us"])})
        shape_summaries.append({**shape, "boards": boards})
    document = {"schema": SUMMARY_SCHEMA,
                "plan_sha256": sha256(plan_path),
                "shape_count": len(shape_summaries),
                "cell_count": len(cell_results),
                "shapes": shape_summaries}
    atomic_json(output, document)
    tsv.parent.mkdir(parents=True, exist_ok=True)
    temporary = tsv.with_name(f".{tsv.name}.current.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, tsv)
    model_shapes: dict[str, dict[str, dict[str, Any]]] = {}
    for shape in shape_summaries:
        for ref in shape["references"]:
            model_shapes.setdefault(ref["model_id"], {})[shape["shape_key"]] = {
                "schema": "quactlize.scalefirst_q4k_model_shape_result.v1",
                "model_id": ref["model_id"],
                "shape_key": shape["shape_key"],
                "shape": [shape["m"], shape["n"], shape["k"]],
                "group_size": shape["group_size"],
                "source_references": [item for item in shape["references"]
                                      if item["model_id"] == ref["model_id"]],
                "deduplicated_result": os.path.relpath(
                    output, models_root / ref["model_id"] / shape["shape_key"]),
                "boards": shape["boards"],
            }
    for model_id, shapes in sorted(model_shapes.items()):
        for key, payload in sorted(shapes.items()):
            atomic_json(models_root / model_id / key / "summary.json", payload)
    for row in rows:
        print("Q4K_REAL_SHAPE_WINNER " + " ".join(
            f"{key}={value}" for key, value in row.items()))
    return document


def self_test() -> None:
    master = {
        "minimum_m": 8,
        "prefill_m": [64, 2048, 4096],
    }
    base = {"inventory_status": "SUPPORTED", "problem_route": "dense",
            "grouped": None, "group_size": 32, "qtype": 12,
            "source_tensors": ["blk.0.ffn_up.weight"], "tp_world": 1,
            "tp_rank": 0, "tp_partition": "replicated"}
    cells = [
        {**base, "m": 64, "n": 4096, "k": 4096,
         "model_id": "a", "shape_id": "1" * 64},
        {**base, "m": 64, "n": 4096, "k": 4096,
         "model_id": "b", "shape_id": "2" * 64},
        {**base, "m": 2048, "n": 4096, "k": 4096,
         "model_id": "a", "shape_id": "3" * 64},
        {**base, "m": 1, "n": 4096, "k": 4096,
         "model_id": "a", "shape_id": "4" * 64},
        {**base, "qtype": 8, "m": 64, "n": 4096, "k": 4096,
         "model_id": "a", "shape_id": "5" * 64},
        {**base, "problem_route": "grouped", "grouped": {"experts": 8},
         "m": 64, "n": 4096, "k": 4096,
         "model_id": "a", "shape_id": "6" * 64},
    ]
    plan = build_plan({"cells": cells}, master, "a" * 64)
    validate_plan(plan)
    if plan["shape_count"] != 2 or plan["cell_count"] != 8 or \
            len(plan["shapes"][0]["references"]) != 2:
        raise AssertionError("shape aggregation/cross product differs")
    planted = json.loads(json.dumps(plan)); planted["cells"].pop()
    try:
        validate_plan(planted)
    except PlanError:
        pass
    else:
        raise AssertionError("drop-one shape/layout negative stayed green")
    unavailable, row = unavailable_board(
        plan["shapes"][0], plan["shapes"][0]["shape_key"],
        "SPLITK_S8_PRODUCER",
        {"32": {"terminal_cells": 3,
                "terminal_reasons": {"K_TILE_DOES_NOT_DIVIDE": 3}}})
    if unavailable["winner"] is not None or \
            unavailable["verdict"] != "UNAVAILABLE" or \
            row["ArtifactTileK"] != "" or row["median_us"] != "":
        raise AssertionError("cross-layout all-terminal board was not explicit")
    print("[q4k-real-shapes:self-test] PASS shape-specific aggregation, "
          "decode/grouped/qtype exclusions, drop-one cross-product RED, "
          "and cross-layout all-terminal board=UNAVAILABLE")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    materializer = sub.add_parser("materialize")
    materializer.add_argument("--inventory", type=pathlib.Path, required=True)
    materializer.add_argument("--master-policy", type=pathlib.Path, required=True)
    materializer.add_argument("--output", type=pathlib.Path, required=True)
    materializer.add_argument("--policies-dir", type=pathlib.Path, required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--plan", type=pathlib.Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--plan", type=pathlib.Path, required=True)
    summary.add_argument("--results-root", type=pathlib.Path, required=True)
    summary.add_argument("--output", type=pathlib.Path, required=True)
    summary.add_argument("--tsv", type=pathlib.Path, required=True)
    summary.add_argument("--models-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test(); return 0
    if args.command == "materialize":
        plan = materialize(args.inventory, args.master_policy, args.output,
                           args.policies_dir)
        print(f"[q4k-real-shapes] PLAN shapes={plan['shape_count']} "
              f"layouts={len(ARTIFACTS)} cells={plan['cell_count']}")
        return 0
    if args.command == "list":
        plan = json.loads(args.plan.read_text()); validate_plan(plan)
        for cell in plan["cells"]:
            print("\t".join((str(cell["artifact_tile_k"]),
                              str(cell["shape_key"]),
                              "x".join(map(str, cell["shape"])),
                              str(cell["policy"]))))
        return 0
    summarize(args.plan, args.results_root, args.output, args.tsv,
              args.models_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[q4k-real-shapes] FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
