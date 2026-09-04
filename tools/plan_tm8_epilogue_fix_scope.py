#!/usr/bin/env python3
"""Plan the exact K-pack rebuild/resweep scope of the PPU TM8 epilogue fix.

The plan never enumerates tactics or invents shard boundaries itself.  It
joins the ordered FullyQuantized and ScaleFirst parent authorities to their
canonical 32-parent shard planners, then evaluates the same capacity formula
as CUTLASS's PPU epilogue builder.  A parent is invalidated only when the old
requested output vector width exceeds the fragment actually owned by one
thread in the shared epilogue tile.

Compile invalidation and runtime admission are deliberately separate.  Dense
TM8 parents are attempted only below the shipping M=8 boundary (and packed-A
parents only at M=1), while grouped parents have no local-expert-M ceiling.
The current discovery protocol publishes one atomic shard/workload log, so the
plan also reports the larger whole-shard evidence-rebind set independently of
the semantically affected runtime set.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import fully_quantized_kpack_bundle_index as fq_shards  # noqa: E402
import fully_quantized_kpack_discovery_matrix as fq_matrix  # noqa: E402
import plan_fq_kpack_route_optimal as workload_matrix  # noqa: E402
import scalefirst_grouped_kpack_matrix as sf_grouped  # noqa: E402
import scalefirst_internal_matrix as sf_dense  # noqa: E402
import scalefirst_kpack_binary_shards as sf_shards  # noqa: E402


SCHEMA = "quactlize.tm8_epilogue_fix_selective_scope.v1"
ROUTES = ("scalefirst", "fully-quantized")
OPERATORS = ("dense", "grouped")
QTYPES = (10, 11, 12, 13, 14)
PARENTS_PER_BINARY = 32
OUTPUT_ELEMENT_BITS = 16
REQUESTED_ALIGNMENT = 128 // OUTPUT_ELEMENT_BITS
OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")

AUTHORITY_PATHS = (
    "tools/emit_scalefirst_internal_superset.cpp",
    "tools/fully_quantized_kpack_bundle_index.py",
    "tools/fully_quantized_kpack_discovery_matrix.py",
    "tools/plan_fq_kpack_route_optimal.py",
    "tools/scalefirst_grouped_kpack_matrix.py",
    "tools/scalefirst_internal_matrix.py",
    "tools/scalefirst_kpack_binary_shards.py",
    "quactlize/include/ppu_format_config.inc",
    "quactlize/include/ppu_tactic_space.hpp",
    "quactlize/include/ppu_dense_shipping_policy.hpp",
    "quactlize/include/dense_splitk_parallel_ppu.cuh",
    "quactlize/include/fpA_intB_ppu.cuh",
    "quactlize/include/moe_grouped_ppu.cuh",
    "benchmarks/fully_quantized_splitk_producer_bench.hpp",
    "benchmarks/scalefirst_internal_sweep_bench.hpp",
    "benchmarks/fully_quantized_grouped_kpack_discovery.hpp",
    "benchmarks/scalefirst_grouped_kpack_discovery.hpp",
    "third_party/actlize/include/cutlass/epilogue/collective/builders/ppu_builder.inl",
)


class ScopeError(ValueError):
    """The selective scope does not match its live generator authorities."""


@dataclass(frozen=True)
class Parent:
    parent_id: str
    tactic: Any
    provider: int
    delivery_n: int
    algorithm: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_oid(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    value = result.stdout.strip()
    if result.returncode or not OID.fullmatch(value):
        raise ScopeError(f"cannot resolve Git identity for {path}")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScopeError(f"{label} must be a positive integer")
    return value


def epilogue_geometry(tactic: Any) -> dict[str, Any]:
    """Mirror the PPU builder's integer capacity derivation for one row."""
    block_m = _positive_int(tactic.tile_m, "tile_m")
    block_n = _positive_int(tactic.tile_n, "tile_n")
    warp_m = _positive_int(tactic.warp_m, "warp_m")
    warp_n = _positive_int(tactic.warp_n, "warp_n")
    if block_m % warp_m or block_n % warp_n:
        raise ScopeError("warp shape does not exactly tile the CTA shape")
    warp_on_m, warp_on_n = block_m // warp_m, block_n // warp_n
    threads = warp_on_m * warp_on_n * 32
    tile_values = block_m * block_n
    use_m8 = block_m == 8 and warp_m == 8
    instruction_m = 8 if use_m8 else 16
    shared_m = warp_on_m * instruction_m
    shared_values = shared_m * block_n
    if tile_values % threads or shared_values % threads:
        raise ScopeError("epilogue fragment capacity is not integral")
    fragment = tile_values // threads
    shared_fragment = shared_values // threads
    effective = min(REQUESTED_ALIGNMENT, fragment, shared_fragment)
    if (effective <= 0 or fragment % effective or
            shared_fragment % effective or block_n % effective):
        raise ScopeError("effective alignment does not tile the epilogue")
    return {
        "use_m8_instruction": use_m8,
        "instruction_m": instruction_m,
        "warp_on_m": warp_on_m,
        "warp_on_n": warp_on_n,
        "cta_threads": threads,
        "fragment_size": fragment,
        "shared_m": shared_m,
        "shared_fragment_size": shared_fragment,
        "requested_alignment": REQUESTED_ALIGNMENT,
        "effective_alignment": effective,
        "affected": effective < REQUESTED_ALIGNMENT,
    }


def _parents(route: str, operator: str, qtype: int) -> list[Parent]:
    if route == "fully-quantized":
        identities = fq_shards.authority_parent_ids(qtype, operator)
        if operator == "dense":
            rows = [Parent(identity, row, provider, delivery_n, "")
                    for identity, (row, provider, delivery_n) in zip(
                        identities, fq_matrix.provider_rows(qtype))]
        else:
            rows = [Parent(identity, row, 0, delivery_n, algorithm)
                    for identity, (row, delivery_n, algorithm) in zip(
                        identities, fq_matrix.grouped_rows(qtype))]
    elif route == "scalefirst":
        identities = sf_shards.authority_symbols(operator, qtype)
        if operator == "dense":
            rows = [Parent(identity, row, provider, delivery_n, "")
                    for identity, (row, provider, delivery_n) in zip(
                        identities, sf_dense.kpack_dense_candidates(qtype))]
        else:
            rows = [Parent(identity, row, 0, delivery_n, "FULL_OUTPUT_BOTH")
                    for identity, (row, delivery_n) in zip(
                        identities, sf_grouped.candidate_rows(qtype))]
    else:
        raise ScopeError(f"unknown route {route}")
    if len(rows) != len(identities) or [row.parent_id for row in rows] != identities:
        raise ScopeError(f"{route}/q{qtype}/{operator} parent order differs")
    return rows


def _canonical_shards() -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    result: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    fq = fq_shards.plan(False, PARENTS_PER_BINARY)
    sf = sf_shards.make_plan("full", PARENTS_PER_BINARY)
    fq_rows = {(qtype, operator): [] for qtype in QTYPES for operator in OPERATORS}
    sf_rows = {(qtype, operator): [] for qtype in QTYPES for operator in OPERATORS}
    for row in fq:
        fq_rows[(row["qtype"], row["operator"])].append({
            "shard_key": row["shard_key"],
            "parent_begin": row["parent_begin"],
            "parent_end": row["parent_end"],
            "compiled_parents": row["parent_count"],
        })
    for row in sf["shards"]:
        sf_rows[(row["qtype"], row["operator"])].append({
            "shard_key": row["shard_id"],
            "parent_begin": row["parent_begin"],
            "parent_end": row["parent_end"],
            "compiled_parents": row["compiled_parents"],
        })
    for qtype in QTYPES:
        for operator in OPERATORS:
            result[("fully-quantized", qtype, operator)] = fq_rows[(qtype, operator)]
            result[("scalefirst", qtype, operator)] = sf_rows[(qtype, operator)]
    return result


def _workloads() -> tuple[dict[tuple[int, str], list[dict[str, Any]]], str]:
    document = workload_matrix.materialize()
    try:
        workload_matrix.validate_plan(document)
    except (AssertionError, KeyError, workload_matrix.PlanError) as error:
        raise ScopeError(f"canonical workload plan differs: {error}") from error
    result = {(qtype, operator): [] for qtype in QTYPES for operator in OPERATORS}
    for row in document["cells"]:
        qtype, operator = int(row["qtype"]), str(row["operator"])
        result[(qtype, operator)].append({
            "workload_key": row["workload_key"],
            "source_class": row["source_class"],
            "public_problem": row["public_problem"],
        })
    for key, rows in result.items():
        rows.sort(key=lambda row: row["workload_key"])
        keys = [row["workload_key"] for row in rows]
        if not rows or len(keys) != len(set(keys)):
            raise ScopeError(f"q{key[0]}/{key[1]} workload denominator differs")
    return result, workload_matrix.digest(document)


def _workload_sets(workloads: dict[tuple[int, str], list[dict[str, Any]]],
                   dense_boundary: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for qtype in QTYPES:
        for operator in OPERATORS:
            rows = workloads[(qtype, operator)]
            variants = {"all": rows}
            if operator == "dense":
                variants["tm8-ap0"] = [
                    row for row in rows
                    if int(row["public_problem"]["m"]) < dense_boundary]
                variants["tm8-ap1"] = [
                    row for row in rows
                    if int(row["public_problem"]["m"]) == 1]
            for name, selected in variants.items():
                identity = f"q{qtype}-{operator}-{name}"
                result[identity] = {
                    "qtype": qtype,
                    "operator": operator,
                    "admission": (
                        "ALL_GROUPED_WORKLOADS_NO_LOCAL_EXPERT_M_CEILING"
                        if operator == "grouped" else
                        "DENSE_TM8_M_LT_SHIPPING_BOUNDARY"
                        if name == "tm8-ap0" else
                        "DENSE_PACKED_A_EXACT_M1" if name == "tm8-ap1" else
                        "ALL_DENSE_WORKLOADS_FOR_ATOMIC_EVIDENCE_REBIND"),
                    "count": len(selected),
                    "workload_keys": [row["workload_key"] for row in selected],
                }
    return result


def _dense_boundary() -> int:
    path = ROOT / "quactlize/include/ppu_dense_shipping_policy.hpp"
    match = re.search(
        r"kDecodeDefaultExclusiveM\s*=\s*(\d+)\s*;", path.read_text())
    if match is None or int(match.group(1)) <= 1:
        raise ScopeError("cannot resolve dense TM8 shipping M boundary")
    boundary = int(match.group(1))
    fq_source = (ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp").read_text()
    sf_source = (ROOT / "benchmarks/scalefirst_internal_sweep_bench.hpp").read_text()
    if ("ppu_dense_shipping::kDecodeDefaultExclusiveM - 1" not in fq_source or
            "default_config_for_m(in.m) !=\n        ppu_dense_shipping::kDecodeDefault" not in sf_source):
        raise ScopeError("dense sweep TM8 admission is no longer bound to shipping policy")
    for relative in (
            "benchmarks/fully_quantized_grouped_kpack_discovery.hpp",
            "benchmarks/scalefirst_grouped_kpack_discovery.hpp"):
        source = (ROOT / relative).read_text()
        if "if constexpr (TM == 8)" in source:
            raise ScopeError(f"{relative} gained a grouped TM8 M ceiling")
    return boundary


def _parent_record(ordinal: int, parent: Parent,
                   geometry: dict[str, Any]) -> dict[str, Any]:
    tactic = parent.tactic
    return {
        "ordinal": ordinal,
        "parent_id": parent.parent_id,
        "tile_m": tactic.tile_m,
        "tile_n": tactic.tile_n,
        "tactic_tile_k": tactic.tactic_tile_k,
        "warp_m": tactic.warp_m,
        "warp_n": tactic.warp_n,
        "stages": tactic.stages,
        "provider": parent.provider,
        "delivery_n": parent.delivery_n,
        "algorithm": parent.algorithm,
        "fragment_size": geometry["fragment_size"],
        "shared_fragment_size": geometry["shared_fragment_size"],
        "requested_alignment": geometry["requested_alignment"],
        "effective_alignment": geometry["effective_alignment"],
    }


def make_plan() -> dict[str, Any]:
    shards_by_pair = _canonical_shards()
    workloads, workload_sha = _workloads()
    boundary = _dense_boundary()
    workload_sets = _workload_sets(workloads, boundary)
    pair_records: list[dict[str, Any]] = []
    rebuild_shards: list[dict[str, Any]] = []
    totals = {
        "authority_parents": 0,
        "binary_shards": 0,
        "affected_parents": 0,
        "affected_shards": 0,
        "recompiled_parents": 0,
        "runtime_candidate_work_items": 0,
        "whole_shard_evidence_rebind_work_items": 0,
    }
    for route in ROUTES:
        for qtype in QTYPES:
            for operator in OPERATORS:
                parents = _parents(route, operator, qtype)
                shards = shards_by_pair[(route, qtype, operator)]
                cursor = 0
                for shard in shards:
                    if (shard["parent_begin"] != cursor or
                            shard["parent_end"] - shard["parent_begin"] !=
                            shard["compiled_parents"]):
                        raise ScopeError(f"{route}/q{qtype}/{operator} shard gap")
                    cursor = shard["parent_end"]
                if cursor != len(parents):
                    raise ScopeError(f"{route}/q{qtype}/{operator} shard tail differs")

                affected: dict[int, tuple[Parent, dict[str, Any]]] = {}
                for ordinal, parent in enumerate(parents):
                    geometry = epilogue_geometry(parent.tactic)
                    if geometry["affected"]:
                        affected[ordinal] = (parent, geometry)
                if not affected:
                    raise ScopeError(f"{route}/q{qtype}/{operator} lost affected rows")
                # This assertion is a consequence check, not the selector.
                # It makes a future builder/tactic change fail loudly instead
                # of silently broadening this historical invalidation class.
                if any(not (parent.tactic.tile_m == 8 and
                            parent.tactic.warp_m == 8 and
                            parent.tactic.warp_n == 16)
                       for parent, _geometry in affected.values()):
                    raise ScopeError("capacity formula affected an unexpected topology")

                pair_shards = []
                semantic_items = 0
                all_workloads = len(workloads[(qtype, operator)])
                for shard in shards:
                    ordinals = [ordinal for ordinal in affected
                                if shard["parent_begin"] <= ordinal <
                                shard["parent_end"]]
                    if not ordinals:
                        continue
                    selected = [affected[ordinal][0] for ordinal in ordinals]
                    if operator == "grouped":
                        workload_set = f"q{qtype}-grouped-all"
                    elif any(parent.provider == 0 for parent in selected):
                        workload_set = f"q{qtype}-dense-tm8-ap0"
                    else:
                        workload_set = f"q{qtype}-dense-tm8-ap1"
                    runtime_count = workload_sets[workload_set]["count"]
                    semantic_items += runtime_count
                    record = {
                        "route": route,
                        "qtype": qtype,
                        "operator": operator,
                        **shard,
                        "affected_parent_count": len(ordinals),
                        "affected_parent_ordinals": ordinals,
                        "affected_parent_ids": [parents[index].parent_id
                                                for index in ordinals],
                        "runtime_workload_set": workload_set,
                        "runtime_candidate_work_items": runtime_count,
                        "whole_shard_evidence_rebind_work_items": all_workloads,
                        "native_build_selector": {
                            "parent_begin": shard["parent_begin"],
                            "parent_count": shard["compiled_parents"],
                        },
                    }
                    pair_shards.append(record)
                    rebuild_shards.append(record)

                pair_record = {
                    "route": route,
                    "qtype": qtype,
                    "format": sf_dense.format_for(qtype).name,
                    "operator": operator,
                    "authority_parents": len(parents),
                    "binary_shards": len(shards),
                    "affected_parents": len(affected),
                    "affected_shards": len(pair_shards),
                    "recompiled_parents": sum(row["compiled_parents"]
                                              for row in pair_shards),
                    "runtime_candidate_work_items": semantic_items,
                    "whole_shard_evidence_rebind_work_items":
                        len(pair_shards) * all_workloads,
                    "affected_parent_records": [
                        _parent_record(ordinal, *affected[ordinal])
                        for ordinal in sorted(affected)],
                    "affected_shard_keys": [row["shard_key"]
                                            for row in pair_shards],
                }
                pair_records.append(pair_record)
                for field in totals:
                    totals[field] += pair_record[field]

    authority_files = {}
    for relative in AUTHORITY_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise ScopeError(f"authority file is missing: {relative}")
        authority_files[relative] = _sha256(path)
    return {
        "schema": SCHEMA,
        "scope": "CANONICAL_KPACK_TM8_EPILOGUE_FIX_SELECTIVE_INVALIDATION",
        "source": {
            "repository_git_commit": _git_oid(ROOT),
            "actlize_git_commit": _git_oid(ROOT / "third_party/actlize"),
            "authority_files_sha256": authority_files,
            "workload_plan_canonical_sha256": workload_sha,
        },
        "builder_contract": {
            "architecture": "PPU0010",
            "output_element_bits": OUTPUT_ELEMENT_BITS,
            "requested_alignment_elements": REQUESTED_ALIGNMENT,
            "thread_count_rule": "(TM/WM)*(TN/WN)*32",
            "instruction_m_rule": "8_if_TM8_and_WM8_else_16",
            "shared_m_rule": "(TM/WM)*instruction_m",
            "fragment_size_rule": "TM*TN/thread_count",
            "shared_fragment_size_rule": "shared_m*TN/thread_count",
            "effective_alignment_rule":
                "min(requested_alignment,fragment_size,shared_fragment_size)",
            "affected_rule": "effective_alignment<requested_alignment",
            "derived_historical_class": "TM8/WM8/WN16; TN_AND_QTYPE_UNCONSTRAINED",
        },
        "runtime_contract": {
            "dense_tm8_exclusive_m": boundary,
            "dense_ap0": "M<dense_tm8_exclusive_m",
            "dense_ap1": "M==1",
            "grouped": "NO_LOCAL_EXPERT_M_CEILING",
            "device_can_implement":
                "NOT_STATICALLY_PRUNED;TERMINAL_SHARED_STORAGE_REMAINS_MEASURED",
            "evidence_unit": "ONE_ATOMIC_BINARY_SHARD_X_WORKLOAD_LOG",
        },
        "parents_per_binary": PARENTS_PER_BINARY,
        "denominator": {
            "routes": len(ROUTES),
            "formats": len(QTYPES),
            "qtype_operator_route_pairs": len(pair_records),
            **totals,
        },
        "workload_sets": workload_sets,
        "pairs": pair_records,
        "rebuild_shards": rebuild_shards,
    }


def _validate_against(document: dict[str, Any], expected: dict[str, Any]) -> None:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ScopeError("selective scope schema differs")
    if document != expected:
        raise ScopeError("selective scope differs from live parent/shard/workload authority")


def validate_plan(document: dict[str, Any]) -> None:
    _validate_against(document, make_plan())


def self_test() -> None:
    affected = epilogue_geometry(SimpleNamespace(
        tile_m=8, tile_n=64, warp_m=8, warp_n=16))
    affected_tn32 = epilogue_geometry(SimpleNamespace(
        tile_m=8, tile_n=32, warp_m=8, warp_n=16))
    wn32 = epilogue_geometry(SimpleNamespace(
        tile_m=8, tile_n=64, warp_m=8, warp_n=32))
    tm16 = epilogue_geometry(SimpleNamespace(
        tile_m=16, tile_n=64, warp_m=16, warp_n=16))
    if not affected["affected"] or affected["effective_alignment"] != 4 or \
            not affected_tn32["affected"] or wn32["affected"] or tm16["affected"]:
        raise ScopeError("epilogue capacity formula controls differ")

    plan = make_plan()
    denominator = plan["denominator"]
    wanted = {
        "routes": 2, "formats": 5, "qtype_operator_route_pairs": 20,
        "authority_parents": 70618, "binary_shards": 2216,
        "affected_parents": 3537, "affected_shards": 226,
        "recompiled_parents": 7232,
    }
    if any(denominator.get(key) != value for key, value in wanted.items()):
        raise ScopeError(f"selective denominator drifted: {denominator}")
    pair_keys = {(row["route"], row["qtype"], row["operator"])
                 for row in plan["pairs"]}
    if pair_keys != {(route, qtype, operator) for route in ROUTES
                     for qtype in QTYPES for operator in OPERATORS}:
        raise ScopeError("one route/qtype/operator affected pair is missing")
    if any(not row["affected_parent_records"] or
           any(parent["effective_alignment"] != 4 or
               parent["requested_alignment"] != 8
               for parent in row["affected_parent_records"])
           for row in plan["pairs"]):
        raise ScopeError("affected row formula evidence differs")

    plants = []
    missing = copy.deepcopy(plan)
    missing["rebuild_shards"].pop()
    plants.append(missing)
    extra = copy.deepcopy(plan)
    planted = copy.deepcopy(extra["rebuild_shards"][-1])
    planted["shard_key"] += "-extra"
    extra["rebuild_shards"].append(planted)
    plants.append(extra)
    predicate = copy.deepcopy(plan)
    predicate["builder_contract"]["affected_rule"] = \
        "tile_m==8_and_qtype==12"
    plants.append(predicate)
    missing_parent = copy.deepcopy(plan)
    missing_parent["pairs"][0]["affected_parent_records"].pop()
    plants.append(missing_parent)
    runtime = copy.deepcopy(plan)
    key = next(key for key in runtime["workload_sets"]
               if key.endswith("dense-tm8-ap0"))
    runtime["workload_sets"][key]["workload_keys"].pop()
    runtime["workload_sets"][key]["count"] -= 1
    plants.append(runtime)
    for planted in plants:
        try:
            _validate_against(planted, plan)
        except ScopeError:
            continue
        raise ScopeError("selective scope negative plant stayed green")
    print(
        "[tm8-epilogue-selective-scope:self-test] PASS "
        f"parents={denominator['affected_parents']}/"
        f"{denominator['authority_parents']} "
        f"shards={denominator['affected_shards']}/"
        f"{denominator['binary_shards']} "
        f"recompiled={denominator['recompiled_parents']} "
        f"runtime-items={denominator['runtime_candidate_work_items']} "
        f"evidence-rebind-items="
        f"{denominator['whole_shard_evidence_rebind_work_items']} "
        "formula=TM8/WM8/WN16-Q2..Q6 negatives=missing+extra+predicate+runtime")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    emit = commands.add_parser("emit")
    emit.add_argument("--out", type=Path, default=Path("-"))
    validate = commands.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        if args.command == "emit":
            document = make_plan()
            payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
            if str(args.out) == "-":
                sys.stdout.write(payload)
            else:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(payload)
            return 0
        document = json.loads(args.input.read_text())
        validate_plan(document)
        print("[tm8-epilogue-selective-scope] PASS "
              f"shards={document['denominator']['affected_shards']} "
              f"output={args.input}")
        return 0
    except (OSError, json.JSONDecodeError, ScopeError) as error:
        print(f"[tm8-epilogue-selective-scope] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
