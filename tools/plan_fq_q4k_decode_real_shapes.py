#!/usr/bin/env python3
"""Materialize real-GGUF Q4_K decode shapes for M={1,2,4,8,16}.

The inventory owns model/tensor/qtype/N/K/TP identity.  This planner only
expands the registered decode-M axis.  It deliberately does not consume the
ScaleFirst prefill measurements or let an inventory's historical M list
silently shrink the decode denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any

import analyze_fully_quantized_internal_sweep as inventory

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from quactlize import formats as qformats


POLICY_SCHEMA = "quactlize.fq_q4k_decode_real_shapes_policy.v1"
PLAN_SCHEMA = "quactlize.fq_q4k_decode_real_shapes_plan.v1"
ARTIFACTS = (32, 64, 128, 256)
DECODE_M = (1, 2, 4, 8, 16)


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


def layout_class(artifact: int) -> dict[str, Any]:
    if artifact not in ARTIFACTS:
        raise PlanError(f"unregistered ArtifactTileK {artifact}")
    arrangement = qformats.PlacedArrangement(4, artifact, 0)
    if arrangement.layout_is_tile_free() and artifact <= 256:
        identity = {
            "schema": "quactlize.xplane_canonical_mapping.v1",
            "producer": "xplane::place_derived",
            "logical_code_planes": [4],
            "fold_n": [1],
            "interleave_codes": 256,
            "equivalence_domain": {"artifact_tile_k": [64, 128, 256]},
        }
        name = "xplane-q4k-tile-free-f1-le256"
        readers = [64, 128, 256]
    else:
        identity = {
            "schema": "quactlize.xplane_canonical_mapping.v1",
            "producer": "xplane::place_derived",
            "logical_code_planes": [4],
            "artifact_tile_k": artifact,
            "fold_n": [arrangement.fold],
            "interleave_codes": 256,
        }
        name = f"xplane-q4k-fold{arrangement.fold}-a{artifact}"
        readers = [artifact]
    return {
        "name": name,
        "mapping_sha256": hashlib.sha256(
            canonical(identity).encode("utf-8")).hexdigest(),
        "canonical_mapping_identity": identity,
        "reader_artifact_tile_k": readers,
        "cute_debug_string_role": "DIAGNOSTIC_ONLY_NOT_CANONICAL",
    }


def load_policy(path: pathlib.Path) -> dict[str, Any]:
    policy = json.loads(path.read_text())
    expected = {
        "schema": POLICY_SCHEMA,
        "qtype": 12,
        "format": "Q4_K",
        "quant_mode": "FinegrainedScaleZero",
        "group_size": 32,
        "decode_m": list(DECODE_M),
        "artifact_tile_k": list(ARTIFACTS),
        "bchunk": 0,
        "problem_route": "dense",
        "bc_batch_policy": "native-grid-y-m-lt8",
        "split_k": [1, 2, 4, 8],
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise PlanError(f"policy {key} differs: {policy.get(key)!r}")
    expected_classes = {str(a): layout_class(a)["name"] for a in ARTIFACTS}
    if policy.get("physical_layout_classes") != expected_classes:
        raise PlanError("policy physical-layout classes differ from canonical map")
    for phase in ("screen", "scheduler", "confirm"):
        obj = policy.get(phase)
        if not isinstance(obj, dict):
            raise PlanError(f"policy lacks {phase}")
        for key in ("iterations", "correctness_repeats"):
            value = obj.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PlanError(f"{phase}.{key} must be positive")
    reducer = policy.get("reducer_model")
    if not isinstance(reducer, dict) or reducer.get("bandwidth_fraction") != .8 or \
            reducer.get("hbm_gbs") != 2766.0 or reducer.get("launch_us") != 0.0 or \
            reducer.get("partial_element_bytes") != 4 or \
            reducer.get("output_element_bytes") != 2:
        raise PlanError("policy reducer model differs from the registered contract")
    return policy


def source_references(cell: dict[str, Any]) -> list[dict[str, Any]]:
    sources = cell.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PlanError("inventory cell lacks source triples")
    result = []
    for source in sources:
        if not isinstance(source, (list, tuple)) or len(source) != 3:
            raise PlanError("inventory source is not [tensor,role,cell_id]")
        tensor, role, cell_id = map(str, source)
        if not tensor or not role or not cell_id:
            raise PlanError("inventory source contains an empty identity")
        result.append({
            "model_id": str(cell["model_id"]),
            "tensor": tensor,
            "role": role,
            "inventory_cell_id": cell_id,
            "tp_world": int(cell["tp_world"]),
            "tp_rank": int(cell["tp_rank"]),
            "tp_partition": str(cell["tp_partition"]),
        })
    return result


def shape_key(m: int, n: int, k: int) -> str:
    return f"m{m}_n{n}_k{k}_g32"


def build_plan(materialized: dict[str, Any], policy: dict[str, Any],
               inventory_sha: str) -> dict[str, Any]:
    cells = materialized.get("cells")
    if not isinstance(cells, list) or not cells:
        raise PlanError("materialized inventory has no cells")
    families: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
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
            raise PlanError("Q4_K inventory cell lost group_size=32")
        else:
            n, k = int(cell["n"]), int(cell["k"])
            refs = families.setdefault((n, k), {})
            for ref in source_references(cell):
                identity = canonical({key: ref[key] for key in (
                    "model_id", "tensor", "role", "tp_world", "tp_rank",
                    "tp_partition")})
                previous = refs.get(identity)
                if previous is None:
                    refs[identity] = ref
                elif previous["inventory_cell_id"] != ref["inventory_cell_id"]:
                    # The inventory may give the same tensor a distinct cell ID
                    # at each historical M.  Preserve all IDs while keeping one
                    # layer identity for the new M expansion.
                    if "inventory_cell_ids" in previous:
                        ids = set(previous["inventory_cell_ids"])
                    else:
                        ids = {previous["inventory_cell_id"]}
                    ids.add(ref["inventory_cell_id"])
                    previous["inventory_cell_ids"] = sorted(ids)
        census[reason] = census.get(reason, 0) + 1
    if not families:
        raise PlanError("inventory contains no supported dense Q4_K families")
    shapes = []
    for (n, k), refs_by_id in sorted(families.items()):
        refs = sorted(refs_by_id.values(), key=canonical)
        for m in DECODE_M:
            shapes.append({
                "shape_key": shape_key(m, n, k),
                "m": m, "n": n, "k": k, "group_size": 32,
                "references": refs,
            })
    plan_cells = []
    for artifact in ARTIFACTS:
        for shape in shapes:
            plan_cells.append({
                "cell_key": f"a{artifact}/{shape['shape_key']}",
                "artifact_tile_k": artifact,
                "physical_layout_class": layout_class(artifact),
                "shape_key": shape["shape_key"],
                "shape": [shape["m"], shape["n"], shape["k"]],
            })
    return {
        "schema": PLAN_SCHEMA,
        "inventory_sha256": inventory_sha,
        "policy_sha256": "BOUND_BY_MATERIALIZER",
        "format": {"qtype": 12, "name": "Q4_K",
                   "quant_mode": "FinegrainedScaleZero", "group_size": 32,
                   "bchunk": 0},
        "decode_m": list(DECODE_M),
        "artifact_tile_k": list(ARTIFACTS),
        "family_count": len(families),
        "shape_count": len(shapes),
        "cell_count": len(plan_cells),
        "shapes": shapes,
        "cells": plan_cells,
        "inventory_census": dict(sorted(census.items())),
        "inventory_provenance": materialized.get("provenance"),
    }


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("decode_m") != list(DECODE_M):
        raise PlanError("decode plan schema/M axis differs")
    shapes, cells = plan.get("shapes"), plan.get("cells")
    if not isinstance(shapes, list) or not shapes or not isinstance(cells, list):
        raise PlanError("decode plan shape/cell denominator is malformed")
    if plan.get("shape_count") != len(shapes) or plan.get("cell_count") != len(cells):
        raise PlanError("decode plan declared denominator differs")
    keys = [str(shape["shape_key"]) for shape in shapes]
    if len(keys) != len(set(keys)):
        raise PlanError("decode plan has duplicate shape keys")
    by_family: dict[tuple[int, int], set[int]] = {}
    for shape in shapes:
        by_family.setdefault((int(shape["n"]), int(shape["k"])), set()).add(
            int(shape["m"]))
    if any(values != set(DECODE_M) for values in by_family.values()):
        raise PlanError("a real Q4_K family lost one decode M")
    expected = {(a, key) for a in ARTIFACTS for key in keys}
    observed = {(int(cell["artifact_tile_k"]), str(cell["shape_key"]))
                for cell in cells}
    if observed != expected or len(cells) != len(expected):
        raise PlanError("ArtifactTileK x shape denominator differs")
    for cell in cells:
        if cell.get("physical_layout_class") != layout_class(
                int(cell["artifact_tile_k"])):
            raise PlanError(f"physical layout differs for {cell['cell_key']}")


def materialize(inventory_path: pathlib.Path, policy_path: pathlib.Path,
                output: pathlib.Path) -> dict[str, Any]:
    policy = load_policy(policy_path)
    source = json.loads(inventory_path.read_text())
    materialized = inventory._materialize_spec(
        source, inventory_path, sha256(inventory_path))
    plan = build_plan(materialized, policy, sha256(inventory_path))
    plan["policy_sha256"] = sha256(policy_path)
    validate_plan(plan)
    atomic_json(output, plan)
    return plan


def list_plan(plan_path: pathlib.Path) -> None:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    by_shape = {shape["shape_key"]: shape for shape in plan["shapes"]}
    for cell in plan["cells"]:
        shape = by_shape[cell["shape_key"]]
        print("\t".join(map(str, (
            cell["artifact_tile_k"], shape["shape_key"], shape["m"],
            shape["n"], shape["k"], cell["physical_layout_class"]["name"],
            len(shape["references"])))))


def self_test() -> None:
    policy = load_policy(
        ROOT / "benchmarks/fq_q4k_decode_real_shapes_policy.json")
    base = {
        "qtype": 12, "inventory_status": "SUPPORTED", "problem_route": "dense",
        "grouped": None, "group_size": 32, "model_id": "model", "n": 1024,
        "k": 5120, "tp_world": 1, "tp_rank": 0,
        "tp_partition": "replicated", "sources": [
            ["blk.0.attn_k.weight", "attn_k", "1" * 64]],
    }
    materialized = {"cells": [dict(base, m=1), dict(base, m=64,
        sources=[["blk.0.attn_k.weight", "attn_k", "2" * 64]])],
        "provenance": {"fixture": True}}
    plan = build_plan(materialized, policy, "a" * 64)
    plan["policy_sha256"] = "b" * 64
    validate_plan(plan)
    if plan["family_count"] != 1 or plan["shape_count"] != 5 or \
            plan["cell_count"] != 20:
        raise AssertionError("decode M/layout cross product differs")
    refs = plan["shapes"][0]["references"]
    if refs[0].get("inventory_cell_ids") != ["1" * 64, "2" * 64]:
        raise AssertionError("cross-M source identity did not coalesce")
    if layout_class(64) != layout_class(128) or \
            layout_class(32) == layout_class(64):
        raise AssertionError("Q4_K byte-class equivalence differs")
    planted = json.loads(json.dumps(plan))
    planted["shapes"].pop()
    planted["shape_count"] -= 1
    try:
        validate_plan(planted)
    except PlanError:
        pass
    else:
        raise AssertionError("missing decode M stayed green")
    print("[fq-q4k-decode-plan:self-test] PASS: inventory-owned real families, "
          "exact M=1/2/4/8/16 expansion, per-layer source coalescing, and "
          "canonical fold2/fold1 byte classes")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--inventory", type=pathlib.Path, required=True)
    materialize_parser.add_argument("--policy", type=pathlib.Path, required=True)
    materialize_parser.add_argument("--output", type=pathlib.Path, required=True)
    list_parser = sub.add_parser("list")
    list_parser.add_argument("--plan", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "materialize":
            plan = materialize(args.inventory, args.policy, args.output)
            print(f"[fq-q4k-decode-plan] PASS families={plan['family_count']} "
                  f"shapes={plan['shape_count']} cells={plan['cell_count']} "
                  f"output={args.output}")
        else:
            list_plan(args.plan)
        return 0
    except (AssertionError, KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        print(f"[fq-q4k-decode-plan] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
