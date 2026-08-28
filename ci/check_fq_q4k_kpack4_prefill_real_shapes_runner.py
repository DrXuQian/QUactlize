#!/usr/bin/env python3
"""Fail-closed source contract for the full K-pack4 real-prefill sweep."""

from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_fq_q4k_kpack4_prefill_real_shapes_box.sh"
ANALYZER = ROOT / "tools/analyze_fq_q4k_kpack4_prefill_real_shapes.py"
POLICY = ROOT / "benchmarks/fq_q4k_kpack4_prefill_real_shapes_policy.json"
GENERATOR = ROOT / "tools/gen_fully_quantized_splitk_producer_units.py"
DRIVER = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
PLANNER = ROOT / "tools/plan_scalefirst_q4k_real_shapes.py"
MASTER = ROOT / "benchmarks/scalefirst_q4k_real_shapes_pruned_policy.json"


class CheckError(ValueError):
    pass


def require(text: str, needles: tuple[str, ...], label: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise CheckError(f"{label} lost source seams: {missing}")


def check(texts: list[str]) -> None:
    runner, analyzer, policy_text, generator, driver, planner, master = texts
    require(runner, (
        "INTERNAL_SWEEP_SPEC must name COMPLETE inventory-v2 JSON",
        'value["shape_count"] == 15 and value["cell_count"] == 60',
        'sorted({row["m"] for row in value["shapes"]}) == [64, 2048, 4096]',
        'len({(row["n"], row["k"]) for row in value["shapes"]}) == 5',
        "--qtype 12 --artifact-tk 0 --bchunk 0 --weight-layout q4-kpack4",
        '"source_typed_rows": 918, "typed_rows": 918',
        'len(symbols) == len(set(symbols)) == 774',
        'assert {row["tile_m"] for row in selected} == {16, 32, 64, 128, 256}',
        'assert {row["a_provider"] for row in selected} == {"standard-aiu"}',
        "FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1",
        "full sweep binary carries a delivery/fused-metadata experiment",
        "phase=screen selected=774 S1",
        "--iterations=2 --correctness-repeats=1 --only-split=1",
        "--iterations=1 --correctness-repeats=1 --only-split=0",
        "--iterations=7 --correctness-repeats=2 --only-split=0",
        "prefill-symbols.txt", "screen-symbols.txt", "confirm-symbols.txt",
        'aggregate --plan "$plan" --raw-root "$out/raw" --policy "$policy"',
        "source-state.sha256", "raw-authority.sha256", "authority.sha256",
        "shapes=15 families=5 M=64,2048,4096 AP0=774 auto64/plain",
        'key=lambda item: (item["n"], item["k"], item["m"])',
    ), "runner")
    if runner.count("TARGET=test_fully_quantized_internal_sweep") != 1:
        raise CheckError("real prefill sweep must build its 918-row graph once")
    if runner.count("run_phase \"$screen_log\"") != 1 or \
            runner.count("run_phase \"$scheduler_log\"") != 1 or \
            runner.count("run_phase \"$confirm_log\"") != 1:
        raise CheckError("real prefill runner lost one looped three-phase seam")
    require(analyzer, (
        "MANIFEST_TYPED = 918", "PREFILL_TYPED = 774",
        "PREFILL_TM = (16, 32, 64, 128, 256)",
        "PREFILL_M = (64, 2048, 4096)",
        '(8, "packed-row"): 72', '(256, "standard-aiu"): 156',
        '"typed_runtime_tc_cells": 3672',
        'row["tile_m"] in PREFILL_TM and row["a_provider"] == "standard-aiu"',
        '"weight_delivery_n": "0"', '"scalezero_fused": "0"',
        "fixture_contract(log, shape)", "core.load_log(",
        'row["partial_bytes_int"] != expected',
        '"PRODUCER_PLUS_MODELED_80PCT_HBM_REDUCER_ZERO_LAUNCH"',
        'winner["max_us"] >= runner["min_us"]',
        'plan.get("shape_count") != 15 or plan.get("cell_count") != 60',
        'len({(int(row["n"]), int(row["k"])) for row in shapes}) != 5',
        '(MAPPING_ID, "0x0", "mapping")',
        '("raw_bad=0", "raw_bad=1", "raw-bit")',
        'raise PrefillRealError(f"real prefill {label} negative stayed green")',
        "real prefill missing-shape negative stayed green",
        "FQ_KPACK4_PREFILL_CENSUS shapes=15 families=5",
    ), "analyzer")
    policy = json.loads(policy_text)
    exact = {
        "schema": "quactlize.fq_q4k_kpack4_prefill_real_shapes_policy.v1",
        "problem_route": "dense", "prefill_m": [64, 2048, 4096],
        "weight_layout": "q4-kpack4-transpose-v1",
        "mapping_id": "0x51344b5034540001",
        "tile_m": [16, 32, 64, 128, 256],
        "a_provider": ["standard-aiu"], "scheduled_delivery_n": 0,
        "scalezero_fused": False, "split_k": [1, 2, 4, 8],
    }
    if any(policy.get(key) != value for key, value in exact.items()) or \
            policy.get("screen") != {
                "only_split": 1, "iterations": 2,
                "correctness_repeats": 1, "top_n": 32,
                "relative_to_leader": 1.2, "top_per_axis_value": 2} or \
            policy.get("scheduler") != {
                "iterations": 1, "correctness_repeats": 1,
                "top_n_per_board": 8, "relative_to_leader": 1.05} or \
            policy.get("confirm") != {
                "iterations": 7, "correctness_repeats": 2,
                "unresolved_if_sample_envelopes_overlap": True}:
        raise CheckError("real prefill policy differs from its frozen factorial")
    model = policy.get("reducer_model", {})
    if any(model.get(key) != value for key, value in {
            "bandwidth_fraction": .8, "hbm_gbs": 2766.0,
            "launch_us": 0.0, "partial_element_bytes": 4,
            "output_element_bytes": 2}.items()):
        raise CheckError("real prefill reducer model differs")
    require(generator, (
        '"q4-kpack4 requires qtype=12, artifact-tk=0 and bchunk=0"',
        'identity["weight_layout"] = weight_layout',
        '"mapping_id": "0x51344b5034540001"',
    ), "generator")
    require(driver, (
        'if (wanted.erase(row.symbol)) selected.push_back(row);',
        '"weight_delivery_n=%d "',
        '"provider_capacity_rows=%d scalezero_fused=%d "',
    ), "driver")
    require(planner, (
        '"prefill_m": [64, 2048, 4096]',
        'reason = "DECODE_NOT_SCALEFIRST_PREFILL"',
        'reason = "OUTSIDE_REGISTERED_PREFILL_M"',
    ), "planner")
    require(master, ('"prefill_m": [64, 2048, 4096]',), "master")


def main() -> int:
    paths = (RUNNER, ANALYZER, POLICY, GENERATOR, DRIVER, PLANNER, MASTER)
    texts = [path.read_text() for path in paths]
    check(texts)
    plants = (
        (0, 'len(symbols) == len(set(symbols)) == 774',
         'len(symbols) == len(set(symbols)) == 773'),
        (0, 'sorted({row["m"] for row in value["shapes"]}) == [64, 2048, 4096]',
         'sorted({row["m"] for row in value["shapes"]}) == [2048]'),
        (0, "--iterations=7 --correctness-repeats=2 --only-split=0",
         "--iterations=1 --correctness-repeats=1 --only-split=0"),
        (1, "PREFILL_TYPED = 774", "PREFILL_TYPED = 773"),
        (1, '"weight_delivery_n": "0"', '"weight_delivery_n": "32"'),
        (1, 'plan.get("shape_count") != 15 or plan.get("cell_count") != 60',
         "False"),
        (2, '"bandwidth_fraction": 0.8', '"bandwidth_fraction": 1.0'),
        (3, 'identity["weight_layout"] = weight_layout', "pass"),
        (4, "if (wanted.erase(row.symbol)) selected.push_back(row);", ""),
        (5, 'reason = "DECODE_NOT_SCALEFIRST_PREFILL"', 'reason = "SELECTED"'),
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
    print("[fq-kpack4-prefill-real:self-test] PASS one 918-row binary, exact "
          "774 AP0 graph, M=64/2048/4096 x five families, three phases and "
          "ten source plants RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, CheckError, AssertionError, json.JSONDecodeError) as error:
        print(f"[fq-kpack4-prefill-real:self-test] FAIL: {error}")
        raise SystemExit(2)
