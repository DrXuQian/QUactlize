#!/usr/bin/env python3
"""Materialize the exact shared FQ Split-K reducer measurement denominator.

The reducer is shared by every dense FullyQuantized candidate with the same
``(M,N,S,partial dtype,output dtype)``.  This plan intentionally contains no
qtype, K, or tactic axis: repeating those axes would measure the same shipping
reducer implementation many times and would make a per-candidate model look
like independent evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import fully_quantized_kpack_discovery_matrix as fq_matrix
import plan_fq_kpack_route_optimal as route_plan


SCHEMA = "quactlize.fq-splitk-reducer-lookup-plan.v1"
SPLITS = (2, 4, 8)
PARTIAL_DTYPE = "fp32"
OUTPUT_DTYPE = "fp16"
REDUCER_TYPE = (
    "cutlass::gemm::device::splitk_parallel::"
    "PpuMixedInputSplitKParallelM1FastReduction<2>"
)
FAST_IMPLEMENTATION = "M1_FAST_E2"
GENERIC_IMPLEMENTATION = "GENERIC_E8"
DEFAULT_WARMUPS = 3
DEFAULT_SAMPLES = 11
SCHEDULE_SEED_SCHEMA = "quactlize.fq-splitk-reducer-schedule.v1"
ROUND_SEEDS = (
    0x6A09E667F3BCC909,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
)
MAX_I64 = (1 << 63) - 1


class PlanError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_product(*values: int) -> int:
    result = 1
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PlanError(f"invalid positive extent {value!r}")
        result *= value
        if result > MAX_I64:
            raise PlanError("reducer byte extent overflows signed 64-bit authority")
    return result


def expected_implementation(m: int, n: int) -> str:
    # The benchmark uses allocator-aligned workspace/output and compact D
    # stride.  With the canonical dense N denominator (all N are multiples of
    # 256), these are the remaining shipping dispatch predicates.
    return FAST_IMPLEMENTATION if m == 1 and n % 64 == 0 else GENERIC_IMPLEMENTATION


def _canonical_route_plan() -> dict[str, Any]:
    value = route_plan.materialize()
    route_plan.validate_plan(value)
    return value


def materialize() -> dict[str, Any]:
    source = _canonical_route_plan()
    grouped_split = fq_matrix.make_manifest(False)["denominator"][
        "grouped_splitk_structural_cells"]
    if grouped_split != 15:
        raise PlanError("grouped Split-K structural denominator drifted")

    dense_cells = [cell for cell in source["cells"]
                   if cell["operator"] == "dense"]
    if len(dense_cells) != 1001:
        raise PlanError(f"dense workload denominator drifted: {len(dense_cells)}")
    mn = sorted({
        (int(cell["public_problem"]["m"]),
         int(cell["public_problem"]["n"]))
        for cell in dense_cells
    })
    if len(mn) != 345:
        raise PlanError(f"unique dense M/N denominator drifted: {len(mn)}")
    if any(n % 256 for _m, n in mn):
        raise PlanError("canonical dense reducer denominator contains N%256")

    cases = []
    for ordinal, (m, n, split) in enumerate(
            (m, n, split) for m, n in mn for split in SPLITS):
        elements = checked_product(m, n)
        if elements >= (1 << 32):
            raise PlanError("raw-bit validator index exceeds uint32 authority")
        workspace_bytes = checked_product(m, n, split, 4)
        output_bytes = checked_product(m, n, 2)
        cases.append({
            "ordinal": ordinal,
            "case_id": f"m{m}-n{n}-s{split}-fp32-fp16",
            "m": m,
            "n": n,
            "split": split,
            "partial_dtype": PARTIAL_DTYPE,
            "output_dtype": OUTPUT_DTYPE,
            "workspace_bytes": workspace_bytes,
            "output_bytes": output_bytes,
            "expected_implementation": expected_implementation(m, n),
        })

    return {
        "schema": SCHEMA,
        "scope": "dense-fully-quantized-shared-splitk-reducer",
        "route_plan_schema": source["schema"],
        "route_plan_sha256": digest(source),
        "reducer": {
            "type": REDUCER_TYPE,
            "partial_dtype": PARTIAL_DTYPE,
            "output_dtype": OUTPUT_DTYPE,
            "fixed_partition_order": "INCREASING_S",
            "m1_fast_implementation": FAST_IMPLEMENTATION,
            "generic_implementation": GENERIC_IMPLEMENTATION,
            "workspace_layout": "ROW_MAJOR_[S][M][N]",
        },
        "measurement": {
            "rounds": [
                {"round": index + 1, "schedule_seed": f"0x{seed:016x}"}
                for index, seed in enumerate(ROUND_SEEDS)
            ],
            "warmups": DEFAULT_WARMUPS,
            "samples": DEFAULT_SAMPLES,
            "schedule_seed_schema": SCHEDULE_SEED_SCHEMA,
            "case_order": "SPLITMIX64_FISHER_YATES_V1",
            "raw_bit_correctness": "EVERY_OUTPUT_ELEMENT",
            "workspace_initialization": "OUTSIDE_TIMED_SPAN",
            "output_poison": "0x7b7b",
            "top_n": None,
            "point_estimate_pruning": False,
        },
        "denominator": {
            "source_dense_cells": len(dense_cells),
            "unique_m_n": len(mn),
            "splits": list(SPLITS),
            "cases": len(cases),
            "grouped_splitk": "STRUCTURAL_UNAVAILABLE",
            "grouped_splitk_structural_cells": grouped_split,
        },
        "authorities": {
            "reducer_plan_generator": {
                "path": "tools/plan_fq_splitk_reducer_lookup.py",
                "sha256": file_sha(
                    ROOT / "tools/plan_fq_splitk_reducer_lookup.py"),
            },
            "benchmark": {
                "path": "benchmarks/test_fq_splitk_reducer_lookup.cu",
                "sha256": file_sha(
                    ROOT / "benchmarks/test_fq_splitk_reducer_lookup.cu"),
            },
            "route_plan_source": {
                "path": "tools/plan_fq_kpack_route_optimal.py",
                "sha256": file_sha(ROOT / "tools/plan_fq_kpack_route_optimal.py"),
            },
            "fq_discovery_matrix": {
                "path": "tools/fully_quantized_kpack_discovery_matrix.py",
                "sha256": file_sha(
                    ROOT / "tools/fully_quantized_kpack_discovery_matrix.py"),
            },
            "shipping_reducer": {
                "path": (
                    "quactlize/include/actlize_extensions/cutlass/gemm/device/"
                    "ppu_mixed_input_splitk_parallel.hpp"),
                "sha256": file_sha(
                    ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/"
                    "device/ppu_mixed_input_splitk_parallel.hpp"),
            },
            "shipping_type_alias": {
                "path": "quactlize/include/dense_splitk_parallel_ppu.cuh",
                "sha256": file_sha(
                    ROOT / "quactlize/include/dense_splitk_parallel_ppu.cuh"),
            },
        },
        "cases": cases,
    }


def validate_plan(value: Any, expected: dict[str, Any] | None = None) -> None:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise PlanError("reducer lookup plan schema differs")
    if expected is None:
        expected = materialize()
    if value != expected:
        raise PlanError("reducer lookup plan differs from canonical denominator")


def write_frozen(path: Path, payload: bytes) -> None:
    if not payload:
        raise PlanError(f"refusing empty output {path}")
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise PlanError(f"refusing to replace stale output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cpp_include(value: dict[str, Any]) -> bytes:
    validate_plan(value, value)
    lines = [
        "// Generated by tools/plan_fq_splitk_reducer_lookup.py; do not edit.",
        f'#define FQ_SPLITK_REDUCER_PLAN_SCHEMA "{SCHEMA}"',
        f'#define FQ_SPLITK_REDUCER_PLAN_SHA256 "{digest(value)}"',
        f"#define FQ_SPLITK_REDUCER_CASE_COUNT {len(value['cases'])}",
        "#define FQ_SPLITK_REDUCER_CASES(X) \\",
    ]
    for index, row in enumerate(value["cases"]):
        continuation = " \\" if index + 1 < len(value["cases"]) else ""
        lines.append(
            "  X({ordinal},{m},{n},{split},{workspace_bytes},{output_bytes},\"{case_id}\",\"{impl}\"){cont}".format(
                ordinal=row["ordinal"], m=row["m"], n=row["n"],
                split=row["split"], workspace_bytes=row["workspace_bytes"],
                output_bytes=row["output_bytes"], case_id=row["case_id"],
                impl=row["expected_implementation"], cont=continuation))
    return ("\n".join(lines) + "\n").encode("ascii")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read {path}: {exc}") from exc


def self_test() -> None:
    value = materialize()
    validate_plan(value, value)
    if value["denominator"] != {
            "source_dense_cells": 1001,
            "unique_m_n": 345,
            "splits": [2, 4, 8],
            "cases": 1035,
            "grouped_splitk": "STRUCTURAL_UNAVAILABLE",
            "grouped_splitk_structural_cells": 15}:
        raise AssertionError("canonical reducer denominator drifted")
    if [row["ordinal"] for row in value["cases"]] != list(range(1035)):
        raise AssertionError("case ordinals are not exact and contiguous")
    identities = {(row["m"], row["n"], row["split"])
                  for row in value["cases"]}
    if len(identities) != 1035:
        raise AssertionError("reducer cases are not unique")
    if sum(row["expected_implementation"] == FAST_IMPLEMENTATION
           for row in value["cases"]) != 27:
        raise AssertionError("M=1 fast reducer denominator drifted")
    if value["measurement"]["rounds"] != [
            {"round": index + 1, "schedule_seed": f"0x{seed:016x}"}
            for index, seed in enumerate(ROUND_SEEDS)]:
        raise AssertionError("three-round seed contract drifted")
    include = cpp_include(value).decode("ascii")
    if include.count("  X(") != 1035 or \
            FQ_INCLUDE_SENTINEL not in include:
        raise AssertionError("generated include denominator differs")

    plants = []
    broken = copy.deepcopy(value); broken["cases"].pop(); plants.append(broken)
    broken = copy.deepcopy(value); broken["cases"][1] = broken["cases"][0]; plants.append(broken)
    broken = copy.deepcopy(value); broken["cases"][0]["split"] = 3; plants.append(broken)
    broken = copy.deepcopy(value); broken["cases"][0]["workspace_bytes"] += 4; plants.append(broken)
    broken = copy.deepcopy(value); broken["cases"][0]["expected_implementation"] = GENERIC_IMPLEMENTATION; plants.append(broken)
    broken = copy.deepcopy(value); broken["measurement"]["top_n"] = 8; plants.append(broken)
    broken = copy.deepcopy(value); broken["route_plan_sha256"] = "0" * 64; plants.append(broken)
    for index, broken in enumerate(plants):
        try:
            validate_plan(broken, value)
        except PlanError:
            pass
        else:
            raise AssertionError(f"reducer plan negative {index} stayed green")
    print(
        "[fq-splitk-reducer-plan:self-test] PASS dense=1001 "
        "unique_mn=345 S=2/4/8 cases=1035 fast=27 generic=1008 "
        "warmups=3 samples=11 top_n=NONE seven_plants=RED")


FQ_INCLUDE_SENTINEL = "X(1034,4096,25600,8,3355443200,209715200"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    emit = commands.add_parser("materialize")
    emit.add_argument("--output", type=Path, required=True)
    include = commands.add_parser("emit-cpp")
    include.add_argument("--plan-output", type=Path, required=True)
    include.add_argument("--include-output", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--plan", type=Path, required=True)
    seed = commands.add_parser("round-seed")
    seed.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "round-seed":
            if args.round < 1 or args.round > len(ROUND_SEEDS):
                raise PlanError("round must be in [1,3]")
            print(f"0x{ROUND_SEEDS[args.round - 1]:016x}")
        elif args.command == "validate":
            validate_plan(read_json(args.plan))
            print(f"[fq-splitk-reducer-plan] PASS plan={args.plan}")
        else:
            value = materialize()
            plan_payload = json.dumps(
                value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
            if args.command == "materialize":
                write_frozen(args.output, plan_payload)
                print(
                    f"[fq-splitk-reducer-plan] PASS cases={len(value['cases'])} "
                    f"output={args.output}")
            else:
                write_frozen(args.plan_output, plan_payload)
                write_frozen(args.include_output, cpp_include(value))
                print(
                    f"[fq-splitk-reducer-plan] PASS cases={len(value['cases'])} "
                    f"plan={args.plan_output} include={args.include_output}")
        return 0
    except (PlanError, OSError) as exc:
        print(f"[fq-splitk-reducer-plan] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
