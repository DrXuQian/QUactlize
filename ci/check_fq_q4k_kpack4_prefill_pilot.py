#!/usr/bin/env python3
"""Fail-closed source contract for the first native K-pack4 prefill pilot."""

from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_fq_q4k_kpack4_prefill_pilot_box.sh"
ANALYZER = ROOT / "tools/analyze_fq_q4k_kpack4_prefill_pilot.py"
POLICY = ROOT / "benchmarks/fq_q4k_kpack4_prefill_pilot_policy.json"
GENERATOR = ROOT / "tools/gen_fully_quantized_splitk_producer_units.py"
DRIVER = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
PLANNER = ROOT / "tools/plan_scalefirst_q4k_real_shapes.py"
MASTER = ROOT / "benchmarks/scalefirst_q4k_real_shapes_pruned_policy.json"
COLLECTIVE = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"


class CheckError(ValueError):
    pass


def require(text: str, needles: tuple[str, ...], label: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise CheckError(f"{label} lost source seams: {missing}")


def check(texts: list[str]) -> None:
    (runner, analyzer, policy_text, generator, driver, planner, master_text,
     collective) = texts
    require(runner, (
        "INTERNAL_SWEEP_SPEC must name COMPLETE inventory-v2 JSON",
        'value["shape_count"] == 15 and value["cell_count"] == 60',
        '[row["m"], row["n"], row["k"]] == [2048, 1024, 5120]',
        "--qtype 12 --artifact-tk 0 --bchunk 0 --weight-layout q4-kpack4",
        "--kpack4-subsuperblocks --tile-m-filter 64",
        "--tile-m-filter 64 --per-unit \"$per_unit\"",
        '"source_typed_rows": 2754, "typed_rows": 630',
        'assert {row["tile_m"] for row in value["typed_rows"]} == {64}',
        'assert {row["a_provider"] for row in value["typed_rows"]} == {"standard-aiu"}',
        'assert {row["tactic_tile_k"] for row in value["typed_rows"]} == {64,128,256}',
        "FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1",
        "pilot binary carries a delivery/fused-metadata experiment",
        "phase=screen typed=630 S=1",
        "--iterations=2 --correctness-repeats=1 --only-split=1",
        "--iterations=1 --correctness-repeats=1 --only-split=0",
        "--iterations=7 --correctness-repeats=2 --only-split=0",
        "analyze_fq_q4k_kpack4_prefill_pilot.py\" screen",
        "analyze_fq_q4k_kpack4_prefill_pilot.py\" scheduler",
        "analyze_fq_q4k_kpack4_prefill_pilot.py\" finalize",
        "FQ_KPACK4_PREFILL_TK",
        "source-state.sha256", "authority.sha256",
    ), "runner")
    if runner.count("TARGET=test_fully_quantized_internal_sweep") != 1:
        raise CheckError("prefill pilot must build the TM64 graph exactly once")
    require(analyzer, (
        "SHAPE = (2048, 1024, 5120)",
        "TYPED_ROWS = 630", "SOURCE_TYPED_ROWS = 2754",
        '"tile_m_filter": 64',
        '"selection_reject_rows": 2124',
        '"runtime_tc_cells": 48000',
        'row.get("tactic_tile_k") not in (64, 128, 256)',
        '"weight_delivery_n": "0"',
        '"scalezero_fused": "0"',
        '"top_per_tactic_tile_k"',
        '"by_tactic_tile_k"',
        "fixture_contract(log)", "core.load_log(",
        'row["partial_bytes_int"] != expected',
        '"PRODUCER_PLUS_MODELED_80PCT_HBM_REDUCER_ZERO_LAUNCH"',
        'winner["max_us"] >= runner["min_us"]',
        "prefill mapping negative stayed green",
        "prefill missing-cell negative stayed green",
        "prefill raw-bit negative stayed green",
    ), "analyzer")
    policy = json.loads(policy_text)
    if policy != {
        "schema": "quactlize.fq_q4k_kpack4_prefill_pilot_policy.v1",
        "name": "q4-kpack4-prefill-m2048-tm64-first",
        "qtype": 12, "format": "Q4_K",
        "quant_mode": "FinegrainedScaleZero", "group_size": 32,
        "shape": [2048, 1024, 5120],
        "weight_layout": "q4-kpack4-transpose-v1",
        "mapping_id": "0x51344b5034540001", "tile_m": [64],
        "a_provider": ["standard-aiu"], "scheduled_delivery_n": 0,
        "screen": {"only_split": 1, "iterations": 2,
                   "correctness_repeats": 1, "top_n": 32,
                   "relative_to_leader": 1.2, "top_per_axis_value": 2},
        "scheduler": {"iterations": 1, "correctness_repeats": 1,
                      "top_n_per_board": 8, "relative_to_leader": 1.05,
                      "top_per_tactic_tile_k": 2},
        "confirm": {"iterations": 7, "correctness_repeats": 2,
                    "unresolved_if_sample_envelopes_overlap": True},
        "split_k": [1, 2, 4, 8],
        "reducer_model": {"bandwidth_fraction": 0.8, "hbm_gbs": 2766.0,
                          "launch_us": 0.0, "partial_element_bytes": 4,
                          "output_element_bytes": 2,
                          "formula": "(M*N*S*4 + M*N*2) / (0.8*2766 GB/s); producer workspace write is already timed"},
    }:
        raise CheckError("prefill pilot policy differs from the registered factorial")
    require(generator, (
        '"q4-kpack4 requires qtype=12, artifact-tk=0 and bchunk=0"',
        'identity["weight_layout"] = weight_layout',
        '"mapping_id": "0x51344b5034540001"',
    ), "generator")
    require(driver, (
        '"weight_delivery_n=%d "',
        '"provider_capacity_rows=%d scalezero_fused=%d "',
        'if (wanted.erase(row.symbol)) selected.push_back(row);',
    ), "driver")
    require(planner, (
        '"prefill_m": [64, 2048, 4096]',
        'reason = "DECODE_NOT_SCALEFIRST_PREFILL"',
        'reason = "OUTSIDE_REGISTERED_PREFILL_M"',
    ), "planner")
    require(master_text, (
        '"prefill_m": [64, 2048, 4096]',
        '"iterations": 7', '"correctness_repeats": 2',
    ), "master policy")
    require(collective, (
        "kPackedKpack4Subtile",
        "kPackedTilesPerSb",
        "kPackedTilesPerUnit",
        "packed_tile_in_unit[write_stage] = scale_load_k % kPackedTilesPerUnit",
        "scale_load_k / kPackedTilesPerUnit",
        "packed_tile_in_unit[stage]",
        "constexpr int GroupBase =",
        "decode_group(cute::Int<GroupBase + G>{}, cute::Int<G>{})",
        "if constexpr (kPackedTilesPerUnit == 1)",
    ), "single-plane packed sub-superblock collective")


def main() -> int:
    paths = (RUNNER, ANALYZER, POLICY, GENERATOR, DRIVER, PLANNER, MASTER,
             COLLECTIVE)
    texts = [path.read_text() for path in paths]
    check(texts)
    plants = (
        (0, "--tile-m-filter 64", "--tile-m-filter 8"),
        (0, '"source_typed_rows": 2754, "typed_rows": 630',
         '"source_typed_rows": 2754, "typed_rows": 629'),
        (0, "--iterations=7 --correctness-repeats=2 --only-split=0",
         "--iterations=1 --correctness-repeats=1 --only-split=0"),
        (1, "TYPED_ROWS = 630", "TYPED_ROWS = 629"),
        (1, '"weight_delivery_n": "0"', '"weight_delivery_n": "32"'),
        (2, '"bandwidth_fraction": 0.8', '"bandwidth_fraction": 1.0'),
        (2, '"top_per_tactic_tile_k": 2', '"top_per_tactic_tile_k": 0'),
        (3, 'identity["weight_layout"] = weight_layout', "pass"),
        (4, "if (wanted.erase(row.symbol)) selected.push_back(row);", ""),
        (5, 'reason = "DECODE_NOT_SCALEFIRST_PREFILL"', 'reason = "SELECTED"'),
        (7, "scale_load_k / kPackedTilesPerUnit", "scale_load_k"),
    )
    for index, old, new in plants:
        broken = list(texts)
        if old not in broken[index]:
            raise CheckError(f"negative seam is absent: {old}")
        broken[index] = broken[index].replace(old, new, 1)
        try:
            check(broken)
        except (CheckError, json.JSONDecodeError):
            pass
        else:
            raise CheckError(f"negative stayed green: {old}")
    print("[fq-kpack4-prefill:self-test] PASS inventory-owned M2048, exact "
          "TM64=630/2754 AP0 TK64/TK128/TK256 graph, raw-bit "
          "screen/scheduler/confirm and eleven plants RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, CheckError, AssertionError, json.JSONDecodeError) as error:
        print(f"[fq-kpack4-prefill:self-test] FAIL: {error}")
        raise SystemExit(2)
