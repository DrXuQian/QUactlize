#!/usr/bin/env python3
"""Materialize the five-family Q4 K-pack dense policy measurement."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any

import plan_fq_kquant_policy_v2 as pilot


SCHEMA = "quactlize.fq-kquant-kpack-policy-real-families-plan.v1"
FAMILIES = (
    (1024, 5120),
    (5120, 8192),
    (5120, 25600),
    (8192, 5120),
    (25600, 5120),
)
M_VALUES = tuple(range(1, 65))
CANDIDATES = pilot.CANDIDATES
ROUNDS = 3
ITERATIONS = 11
WARMUPS = 3
REGRET_THRESHOLD_PCT = 3.0


class PlanError(ValueError):
    pass


def family_identity(n: int, k: int) -> str:
    return f"q4-dense-n{n}-k{k}"


def materialize() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "profile": "kpack-policy-v2",
        "format": {
            "qtype": 12,
            "name": "Q4_K",
            "packed_format": 0,
            "low_bits": 4,
            "high_bits": 0,
            "group_size": 32,
        },
        "layout": {
            "name": "q4-kpack4",
            "layout": 1,
            "mapping_id": pilot.MAPPING_ID,
            "artifact_tile_k": 0,
        },
        "operator": "dense",
        "m_values": list(M_VALUES),
        "candidate_names": list(CANDIDATES),
        "families": [
            {
                "identity": family_identity(n, k),
                "n": n,
                "k": k,
                "dense": [
                    {
                        "key": f"q4_policy_m{m}_n{n}_k{k}",
                        "m": m,
                        "n": n,
                        "k": k,
                    }
                    for m in M_VALUES
                ],
            }
            for n, k in FAMILIES
        ],
        "measurement": {
            "execution_unit": "one-family-one-round",
            "rounds": ROUNDS,
            "samples_per_round": ITERATIONS,
            "warmups_per_round": WARMUPS,
            "dense_cases_per_execution": len(M_VALUES),
            "categorical_candidates_per_case": len(CANDIDATES),
            "correctness": "full-output-raw-bit-device-compare",
        },
        "policy": {
            "regret_threshold_pct": REGRET_THRESHOLD_PCT,
            "config_names_are_categorical": True,
            "compiled_default_is_measured_policy": False,
            "cross_family_extrapolation": False,
            "production_policy_mutation": False,
        },
        "outside_scope": {
            "unknown_n_k": "NO_MEASURED_POLICY",
            "m_greater_than_64": "A07_SCALEFIRST",
            "grouped": "SEPARATE_GROUPED_POLICY_GATE",
        },
    }


def validate(value: dict[str, Any]) -> None:
    expected = materialize()
    if value != expected:
        raise PlanError("plan differs from the five-real-family authority")
    if value["schema"] != SCHEMA or value["profile"] != "kpack-policy-v2":
        raise PlanError("schema/profile identity differs")
    if value["layout"] != {
        "name": "q4-kpack4",
        "layout": 1,
        "mapping_id": pilot.MAPPING_ID,
        "artifact_tile_k": 0,
    }:
        raise PlanError("plan is not canonical Q4 K-pack-only")
    observed = tuple((row["n"], row["k"]) for row in value["families"])
    if observed != FAMILIES or len(set(observed)) != len(FAMILIES):
        raise PlanError("real-family denominator differs")
    if any(
        len(row["dense"]) != len(M_VALUES)
        or [cell["m"] for cell in row["dense"]] != list(M_VALUES)
        for row in value["families"]
    ):
        raise PlanError("per-family M=1..64 denominator differs")
    if tuple(value["candidate_names"]) != CANDIDATES:
        raise PlanError("categorical candidate denominator differs")
    if value["outside_scope"]["m_greater_than_64"] != "A07_SCALEFIRST":
        raise PlanError("M>64 must be handed to A07 ScaleFirst")
    if value["policy"]["compiled_default_is_measured_policy"] or value["policy"]["cross_family_extrapolation"]:
        raise PlanError("compiled default or cross-family extrapolation was enabled")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def self_test() -> None:
    value = materialize()
    validate(value)
    plants = []
    broken = copy.deepcopy(value); broken["families"].pop(); plants.append(broken)
    broken = copy.deepcopy(value); broken["families"][0]["dense"].pop(); plants.append(broken)
    broken = copy.deepcopy(value); broken["candidate_names"].pop(); plants.append(broken)
    broken = copy.deepcopy(value); broken["layout"]["mapping_id"] = "0x0"; plants.append(broken)
    broken = copy.deepcopy(value); broken["measurement"]["rounds"] = 2; plants.append(broken)
    broken = copy.deepcopy(value); broken["policy"]["compiled_default_is_measured_policy"] = True; plants.append(broken)
    broken = copy.deepcopy(value); broken["policy"]["cross_family_extrapolation"] = True; plants.append(broken)
    broken = copy.deepcopy(value)
    broken["outside_scope"]["m_greater_than_64"] = "COMPILED_DEFAULT"
    plants.append(broken)
    for index, broken in enumerate(plants):
        try:
            validate(broken)
        except PlanError:
            pass
        else:
            raise AssertionError(f"plan negative {index} stayed green")
    print(
        "[fq-kquant-policy-real-plan:self-test] PASS five exact N/K families, "
        "per-family M=1..64, five categorical configs, 3x11+3 timing, "
        "no cross-family/default policy and M>64=A07_SCALEFIRST; eight plants RED"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    emit = commands.add_parser("materialize")
    emit.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "materialize":
            value = materialize()
            validate(value)
            atomic_json(args.output, value)
            print(
                f"[fq-kquant-policy-real-plan] PASS families={len(FAMILIES)} "
                f"dense={len(FAMILIES) * len(M_VALUES)} output={args.output}"
            )
        else:
            validate(json.loads(args.plan.read_text()))
            print(f"[fq-kquant-policy-real-plan] PASS validated={args.plan}")
        return 0
    except (AssertionError, OSError, PlanError, TypeError, ValueError) as error:
        print(f"[fq-kquant-policy-real-plan] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
