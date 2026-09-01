#!/usr/bin/env python3
"""Materialize the production Xplane/K-pack real-shape A/B denominator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.workloads import MODELS, N_TOKENS, projections  # noqa: E402
from tools.gguf_internal_shape_inventory import _routing_fixture  # noqa: E402


SCHEMA = "quactlize.fq-kquant-kpack-perf-plan.v2"
HEURISTIC_SCHEMA = "quactlize.fq-kquant-config-training-plan.v1"
FORMATS = {
    10: {"name": "Q2_K", "packed_format": 2, "low_bits": 2,
         "high_bits": 0, "group_size": 16, "tactic_tile_k": 256,
         "operators": ["dense", "grouped"], "layout_key": "kpack"},
    11: {"name": "Q3_K", "packed_format": 3, "low_bits": 2,
         "high_bits": 1, "group_size": 16, "tactic_tile_k": 256,
         "operators": ["dense", "grouped"], "layout_key": "kpack"},
    12: {"name": "Q4_K", "packed_format": 0, "low_bits": 4,
         "high_bits": 0, "group_size": 32, "tactic_tile_k": 256,
         "operators": ["grouped"], "layout_key": "q4-kpack4"},
    13: {"name": "Q5_K", "packed_format": 1, "low_bits": 4,
         "high_bits": 1, "group_size": 32, "tactic_tile_k": 256,
         "operators": ["dense", "grouped"], "layout_key": "kpack"},
    14: {"name": "Q6_K", "packed_format": 4, "low_bits": 4,
         "high_bits": 2, "group_size": 16, "tactic_tile_k": 128,
         "operators": ["dense", "grouped"], "layout_key": "kpack"},
}
DENSE_M = (1, 2, 4, 8, 64, 2048, 4096)
GROUPED_TOKENS = tuple(N_TOKENS)
HEURISTIC_DYNAMIC = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                     1024, 2048, 4096)
EXPERTS = 256
TOPK = 8
ROUTER = "token-topk-hot16x4-wor-sm64-s44-v1"
MAPPING_ID = "0x514b504b54000001"
Q4_MAPPING_ID = "0x51344b5034540001"


def operators(qtype: int) -> tuple[str, ...]:
    return tuple(FORMATS[qtype]["operators"])


def mapping_id(qtype: int) -> str:
    return Q4_MAPPING_ID if qtype == 12 else MAPPING_ID


class PlanError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def source_families() -> tuple[dict[tuple[int, int], set[str]],
                               dict[tuple[int, int], set[str]]]:
    dense: dict[tuple[int, int], set[str]] = {}
    grouped: dict[tuple[int, int], set[str]] = {}
    for model, config in MODELS.items():
        for role, n, k, _ in projections(config):
            target = (grouped if config["kind"] == "moe" and
                      role.startswith("expert_") else dense)
            target.setdefault((n, k), set()).add(f"{model}:{role}")
    return dense, grouped


def profile_axes(profile: str) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    if profile == "layout-ab":
        return SCHEMA, DENSE_M, GROUPED_TOKENS
    if profile == "heuristic":
        return HEURISTIC_SCHEMA, HEURISTIC_DYNAMIC, HEURISTIC_DYNAMIC
    raise PlanError(f"unknown profile {profile!r}")


def materialize(profile: str = "layout-ab") -> dict[str, Any]:
    schema, dense_m, grouped_tokens = profile_axes(profile)
    dense_families, grouped_families = source_families()
    dense = [
        {"key": f"dense_m{m}_n{n}_k{k}", "m": m, "n": n, "k": k,
         "sources": sorted(sources)}
        for (n, k), sources in sorted(dense_families.items())
        for m in dense_m
    ]
    grouped = []
    for (n, k), sources in sorted(grouped_families.items()):
        for tokens in grouped_tokens:
            route = _routing_fixture(EXPERTS, TOPK, tokens)
            if route["fixture"] != ROUTER:
                raise PlanError("routing fixture identity differs")
            grouped.append({
                "key": f"grouped_t{tokens}_n{n}_k{k}_e{EXPERTS}_top{TOPK}",
                "tokens": tokens, "n": n, "k": k, "experts": EXPERTS,
                "topk": TOPK, "router": ROUTER,
                "total_rows": route["total_rows"], "active": route["active"],
                "zero": route["zero"], "max_rows": route["max_rows"],
                "sources": sorted(sources),
            })
    result = {
        "schema": schema,
        "formats": {str(q): row for q, row in FORMATS.items()},
        "layouts": {
            "xplane": {"layout": 0, "mapping_id": "0x0000000000000000",
                       "artifact_tile_k": "format-fully-quantized-default"},
            "kpack": {"layout": 2, "mapping_id": MAPPING_ID,
                      "artifact_tile_k": 0},
            "q4-kpack4": {"layout": 1, "mapping_id": Q4_MAPPING_ID,
                           "artifact_tile_k": 0},
        },
        "policy": {
            "dense_config": "production-shape-default",
            "grouped_config": "production-default",
            "split_k": 1,
            "timing": "same-binary-distinct-event-pairs",
            "correctness": "full-output-raw-bit-device-compare",
            "archive_threshold_pct": 3.0,
        },
        "dense_m": list(dense_m),
        "grouped_tokens": list(grouped_tokens),
        "dense_families": len(dense_families),
        "grouped_families": len(grouped_families),
        "dense": dense,
        "grouped": grouped,
        "source_sha256": {
            "workloads.py": hashlib.sha256(
                (ROOT / "benchmarks/workloads.py").read_bytes()).hexdigest(),
            "moe_router_fixture.hpp": hashlib.sha256(
                (ROOT / "benchmarks/moe_router_fixture.hpp").read_bytes()).hexdigest(),
        },
    }
    # Preserve the byte-for-byte v2 layout-A/B plan used by resumable evidence.
    # The expanded training profile has its own schema and names itself.
    if profile != "layout-ab":
        result["profile"] = profile
    return result


def validate(value: dict[str, Any]) -> None:
    profile = value.get("profile", "layout-ab")
    expected = materialize(profile)
    if value != expected:
        raise PlanError("plan differs from the workloads/router authority")
    dense = value["dense"]
    grouped = value["grouped"]
    dense_count = 77 if profile == "layout-ab" else 143
    grouped_count = 24 if profile == "layout-ab" else 52
    if len(dense) != dense_count or len(grouped) != grouped_count:
        raise PlanError(
            f"shape denominator differs: dense={len(dense)} grouped={len(grouped)}")
    if len({row["key"] for row in dense + grouped}) != dense_count + grouped_count:
        raise PlanError("shape keys are not unique identities")


def self_test() -> None:
    value = materialize("layout-ab")
    validate(value)
    heuristic = materialize("heuristic")
    validate(heuristic)
    if len(heuristic["dense"]) != 143 or len(heuristic["grouped"]) != 52:
        raise AssertionError("heuristic profile denominator differs")
    plants = []
    broken = copy.deepcopy(value); broken["dense"].pop(); plants.append(broken)
    broken = copy.deepcopy(value); broken["grouped"][0]["experts"] = 255; plants.append(broken)
    broken = copy.deepcopy(value); broken["formats"]["11"]["packed_format"] = 2; plants.append(broken)
    broken = copy.deepcopy(value); broken["layouts"]["kpack"]["mapping_id"] = "0x0"; plants.append(broken)
    broken = copy.deepcopy(value); broken["layouts"]["q4-kpack4"]["mapping_id"] = "0x0"; plants.append(broken)
    for broken in plants:
        try:
            validate(broken)
        except PlanError:
            pass
        else:
            raise AssertionError("plan negative stayed green")
    print("[fq-kquant-perf-plan:self-test] PASS formats=5 layout=77+24 "
          "heuristic=143+52, 11 dense/4 grouped families, Q4=grouped-only; "
          "five plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    emit = sub.add_parser("materialize")
    emit.add_argument("--output", type=pathlib.Path, required=True)
    emit.add_argument("--profile", choices=("layout-ab", "heuristic"),
                      default="layout-ab")
    check = sub.add_parser("validate")
    check.add_argument("--plan", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "materialize":
            value = materialize(args.profile); validate(value); atomic_json(args.output, value)
            print(f"[fq-kquant-perf-plan] PASS profile={args.profile} "
                  f"dense={len(value['dense'])} grouped={len(value['grouped'])} "
                  f"output={args.output}")
        else:
            validate(json.loads(args.plan.read_text()))
            print(f"[fq-kquant-perf-plan] PASS validated={args.plan}")
        return 0
    except (AssertionError, OSError, PlanError, ValueError) as error:
        print(f"[fq-kquant-perf-plan] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
