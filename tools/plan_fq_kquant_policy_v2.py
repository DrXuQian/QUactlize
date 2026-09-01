#!/usr/bin/env python3
"""Materialize the isolated Q4 K-pack-only policy-v2 pilot."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
from typing import Any


SCHEMA = "quactlize.fq-kquant-kpack-policy-plan.v2"
MAPPING_ID = "0x51344b5034540001"
M_VALUES = tuple(range(1, 65))
N, K = 1024, 5120
CANDIDATES = (
    "kpack4:8x32x256:8x16:s3:S4",
    "kpack4:8x64x256:8x16:s2:S4",
    "kpack4:8x128x256:8x16:s2:S4",
    "kpack4:8x64x256:8x16:s2:S1",
    "kpack4:64x128x256:64x16:s2:S1",
)


class PlanError(ValueError):
    pass


def materialize() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "profile": "kpack-policy-v2",
        "format": {
            "qtype": 12, "name": "Q4_K", "packed_format": 0,
            "low_bits": 4, "high_bits": 0, "group_size": 32,
        },
        "layout": {
            "name": "q4-kpack4", "layout": 1,
            "mapping_id": MAPPING_ID, "artifact_tile_k": 0,
        },
        "operator": "dense",
        "family": {"n": N, "k": K, "identity": "q4-dense-n1024-k5120"},
        "m_values": list(M_VALUES),
        "boundary_controls": {
            "decode_max_m": 8,
            "required": [1, 8, 9, 64],
            "m64_role": "prefill-boundary-control",
        },
        "policy": {
            "candidate_source": "runtime-valid-config-inventory-v4",
            "include_split_k": True,
            "include_cuda_kernel": False,
            "timing": "same-binary-distinct-event-pairs",
            "correctness": "full-output-raw-bit-device-compare",
            "production_policy_mutation": False,
        },
        "candidate_names": list(CANDIDATES),
        "dense": [
            {"key": f"q4_policy_m{m}_n{N}_k{K}", "m": m, "n": N, "k": K,
             "boundary_control": m in (1, 8, 9, 64)}
            for m in M_VALUES
        ],
        "grouped": [],
    }


def validate(value: dict[str, Any]) -> None:
    if value != materialize():
        raise PlanError("plan differs from the policy-v2 authority")
    if value["schema"] != SCHEMA or value["profile"] != "kpack-policy-v2":
        raise PlanError("schema/profile identity differs")
    if value["layout"] != {"name": "q4-kpack4", "layout": 1,
                            "mapping_id": MAPPING_ID, "artifact_tile_k": 0}:
        raise PlanError("the pilot is not canonical K-pack-only")
    if value["m_values"] != list(M_VALUES) or len(value["dense"]) != 64:
        raise PlanError("M=1..64 denominator differs")
    if value["grouped"] or value["operator"] != "dense":
        raise PlanError("the pilot must be dense-only")
    if not value["policy"]["include_split_k"]:
        raise PlanError("split-K candidates were disabled")


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def self_test() -> None:
    value = materialize(); validate(value)
    plants = []
    broken = copy.deepcopy(value); broken["layout"]["layout"] = 0; plants.append(broken)
    broken = copy.deepcopy(value); broken["format"]["qtype"] = 11; plants.append(broken)
    broken = copy.deepcopy(value); broken["dense"].pop(); plants.append(broken)
    broken = copy.deepcopy(value); broken["boundary_controls"]["required"] = [1, 64]; plants.append(broken)
    broken = copy.deepcopy(value); broken["policy"]["include_split_k"] = False; plants.append(broken)
    broken = copy.deepcopy(value); broken["grouped"] = [{}]; plants.append(broken)
    broken = copy.deepcopy(value); broken["candidate_names"].pop(); plants.append(broken)
    for broken in plants:
        try: validate(broken)
        except PlanError: pass
        else: raise AssertionError("policy-v2 negative stayed green")
    print("[fq-kquant-policy-v2-plan:self-test] PASS Q4 dense K-pack-only "
          "M=1..64 boundaries=8/9/64 candidates=5 split-K=enabled; seven plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    emit = sub.add_parser("materialize"); emit.add_argument("--output", type=pathlib.Path, required=True)
    check = sub.add_parser("validate"); check.add_argument("--plan", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test": self_test()
        elif args.command == "materialize":
            value = materialize(); validate(value); atomic_json(args.output, value)
            print(f"[fq-kquant-policy-v2-plan] PASS dense=64 grouped=0 output={args.output}")
        else:
            validate(json.loads(args.plan.read_text()))
            print(f"[fq-kquant-policy-v2-plan] PASS validated={args.plan}")
        return 0
    except (AssertionError, OSError, PlanError, ValueError) as error:
        print(f"[fq-kquant-policy-v2-plan] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
