#!/usr/bin/env python3
"""Fail-closed source contract for the native K-pack4 performance pilot."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_fq_q4k_kpack4_pilot_box.sh"
ANALYZER = ROOT / "tools/analyze_fq_q4k_kpack4_pilot.py"
GENERATOR = ROOT / "tools/gen_fully_quantized_splitk_producer_units.py"
BENCH = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
POLICY = ROOT / "benchmarks/fq_q4k_decode_real_shapes_policy.json"


class CheckError(ValueError):
    pass


def check(runner: str, analyzer: str, generator: str,
          bench: str, policy: str) -> None:
    runner_needles = (
        "--qtype 12 --artifact-tk 0 --bchunk 0 --weight-layout q4-kpack4",
        "--tile-m-filter 8 --per-unit \"$per_unit\"",
        "FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0",
        "FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1",
        "typed=144 S=1",
        "--iterations=2 \\",
        "--correctness-repeats=1 --only-split=1",
        "--symbols-file=\"$out/results/screen-symbols.txt\"",
        "--iterations=1 \\",
        "--correctness-repeats=1 --only-split=0",
        "--symbols-file=\"$out/results/confirm-symbols.txt\"",
        "--iterations=7 \\",
        "--correctness-repeats=2 --only-split=0",
        "analyze_fq_q4k_kpack4_pilot.py\" screen",
        "analyze_fq_q4k_kpack4_pilot.py\" scheduler",
        "analyze_fq_q4k_kpack4_pilot.py\" finalize",
        "check_fq_q4k_kpack4_generator.py",
        "mapping=0x51344b5034540001",
        "source-authority.sha256",
        "authority.sha256",
    )
    if any(token not in runner for token in runner_needles):
        raise CheckError("K-pack4 pilot lost a generation/build/phase authority")
    if runner.count("TARGET=test_fully_quantized_internal_sweep") != 1:
        raise CheckError("K-pack4 pilot must build its 144-row binary exactly once")
    if "select_fq_q4k_kpack4_closure.py" in runner:
        raise CheckError("K-pack4 pilot regressed to the one-row closure selector")
    analyzer_needles = (
        "TYPED_ROWS = 144",
        '"source_typed_rows": 918',
        '"packed-row": 72',
        '"artifact_tile_k_is_not_an_axis": True',
        "decode.select_screen(",
        "decode.select_scheduler(",
        "decode.reducer_us(SHAPE[0], SHAPE[1], split, policy)",
        '"PRODUCER_PLUS_MODELED_80PCT_HBM_REDUCER_ZERO_LAUNCH"',
        's8["terminal_states"] != {"SPLIT_PARTITION": len(symbols)}',
        'winner["max_us"] >= runner["min_us"]',
        "confirmation sample/repeat denominator differs from policy",
        "mapping-id negative control stayed green",
        "missing confirmation cell stayed green",
    )
    if any(token not in analyzer for token in analyzer_needles):
        raise CheckError("K-pack4 pilot analyzer lost a denominator/scope gate")
    generator_needles = (
        'weight_layout: str = "xplane"',
        '"q4-kpack4 requires qtype=12, artifact-tk=0 and bchunk=0"',
        "provider_artifact = emitter_artifact if kpack4 else artifact",
        "matrix.packed_a_provider_candidate(fmt, row, provider_artifact)",
        'identity["weight_layout"] = weight_layout',
        '"mapping_id": "0x51344b5034540001"',
    )
    if any(token not in generator for token in generator_needles):
        raise CheckError("native K-pack4 generator contract is incomplete")
    if '"--symbols-file="' not in bench or \
            "if (wanted.erase(row.symbol)) selected.push_back(row);" not in bench:
        raise CheckError("pilot binary lost runtime symbol selection")
    policy_needles = (
        '"top_n": 24', '"relative_to_leader": 1.2',
        '"top_n_per_board": 8', '"relative_to_leader": 1.08',
        '"iterations": 7', '"correctness_repeats": 2',
        '"bandwidth_fraction": 0.8', '"launch_us": 0.0',
    )
    if any(token not in policy for token in policy_needles):
        raise CheckError("K-pack4 pilot policy no longer matches the bound phases")


def main() -> int:
    texts = [path.read_text() for path in
             (RUNNER, ANALYZER, GENERATOR, BENCH, POLICY)]
    check(*texts)
    plants = (
        (0, "--artifact-tk 0", "--artifact-tk 64"),
        (0, "FQ_SWEEP_WEIGHT_LAYOUT=1", "FQ_SWEEP_WEIGHT_LAYOUT=0"),
        (0, "--iterations=7", "--iterations=1"),
        (0, '--symbols-file="$out/results/screen-symbols.txt"',
         '--symbols-file="$out/results/all-symbols.txt"'),
        (1, "decode.reducer_us(SHAPE[0], SHAPE[1], split, policy)", "0.0"),
        (1, '"SPLIT_PARTITION": len(symbols)', '"SPLIT_PARTITION": 0'),
        (2, 'identity["weight_layout"] = weight_layout', "pass"),
        (3, "if (wanted.erase(row.symbol)) selected.push_back(row);", ""),
        (4, '"bandwidth_fraction": 0.8', '"bandwidth_fraction": 1.0'),
    )
    for index, old, new in plants:
        broken = list(texts)
        if old not in broken[index]:
            raise CheckError(f"pilot negative-control seam is missing: {old}")
        broken[index] = broken[index].replace(old, new, 1)
        try:
            check(*broken)
        except CheckError:
            pass
        else:
            raise CheckError(f"pilot negative control stayed green: {old}")
    print("[fq-q4k-kpack4-pilot:self-test] PASS one native 144-row AP0/AP1 binary, "
          "S1/scheduler/confirm runtime pruning, exact layout and 80%-HBM "
          "scope; nine source plants RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, CheckError, AssertionError) as error:
        print(f"[fq-q4k-kpack4-pilot:self-test] FAIL: {error}")
        raise SystemExit(2)
