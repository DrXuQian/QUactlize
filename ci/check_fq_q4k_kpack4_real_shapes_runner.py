#!/usr/bin/env python3
"""Fail-closed source contract for the K-pack4 20-shape decode runner."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_fq_q4k_kpack4_decode_real_shapes_box.sh"
ANALYZER = ROOT / "tools/analyze_fq_q4k_kpack4_pilot.py"
BUNDLE_CHECKER = ROOT / "tools/check_fq_q4k_kpack4_pilot_bundle.py"


class CheckError(ValueError):
    pass


def check(runner: str, analyzer: str, bundle_checker: str) -> None:
    runner_needles = (
        "INTERNAL_SWEEP_SPEC must name COMPLETE inventory-v2 JSON",
        "PILOT_BUNDLE",
        'check_fq_q4k_kpack4_pilot_bundle.py" validate',
        'check_fq_q4k_kpack4_pilot_bundle.py" self-test',
        "policy must be byte-identical to the pilot policy",
        'value["family_count"]==5 and value["shape_count"]==20',
        "One 144-row TM8 AP0/AP1 binary serves every shape",
        "--iterations=2 --correctness-repeats=1 --only-split=1",
        '--symbols-file="$directory/screen-symbols.txt"',
        "--iterations=1 --correctness-repeats=1",
        '--symbols-file="$directory/confirm-symbols.txt"',
        "--iterations=7 --correctness-repeats=2",
        'analyze_fq_q4k_kpack4_pilot.py" screen',
        'analyze_fq_q4k_kpack4_pilot.py" scheduler',
        'analyze_fq_q4k_kpack4_pilot.py" finalize',
        'analyze_fq_q4k_kpack4_pilot.py" aggregate',
        'sha256sum "$inventory" "$policy" "$plan" "$manifest" "$binary"',
        "source-state.sha256",
        "raw-authority.sha256",
        "families=5 shapes=20",
    )
    if any(token not in runner for token in runner_needles):
        raise CheckError("K-pack4 real-shape runner lost an authority or phase seam")
    if runner.count("run_fq_q4k_kpack4_pilot_box.sh") != 1:
        raise CheckError("runner must have exactly one no-bundle build fallback")
    analyzer_needles = (
        'REAL_SHAPES_SCHEMA = "quactlize.fq_q4k_kpack4_decode_real_shapes.v1"',
        '"mapping_id": MAPPING_ID',
        "def set_shape(text: str)",
        "shape[0] not in planner.DECODE_M",
        'SHAPE == (1, 1024, 5120)',
        '"artifact_tile_k": 0,\n        "physical_layout_class": KPACK4_CLASS,',
        "def aggregate(plan_path: pathlib.Path",
        'len(shape_rows) != 20',
        '"PRODUCER_PLUS_MODELED_80PCT_HBM_REDUCER_ZERO_LAUNCH"',
        '"max_internal_regret": 0.0',
        "missing real-shape summary stayed green",
        'sub.add_parser("aggregate")',
    )
    if any(token not in analyzer for token in analyzer_needles):
        raise CheckError("K-pack4 real-shape analyzer lost a denominator/scope seam")
    bundle_needles = (
        'result_authority = bundle / "results/authority.sha256"',
        'summary_value.get("manifest_sha256") != manifest_sha',
        'one_suffix(final_records, "/generated/manifest.json"',
        'one_suffix(final_records, "/results/summary.json"',
        '"/ppu_targets/test_fully_quantized_internal_sweep"',
        'source_rows.append(("/original/pilot/generated/manifest.json"',
        '"final-manifest"', '"summary-manifest"', '"binary"',
    )
    if any(token not in bundle_checker for token in bundle_needles):
        raise CheckError("K-pack4 pilot-bundle checker lost a final authority seam")


def main() -> int:
    texts = [RUNNER.read_text(), ANALYZER.read_text(), BUNDLE_CHECKER.read_text()]
    check(*texts)
    plants = (
        (0, 'value["family_count"]==5 and value["shape_count"]==20',
         'value["shape_count"]==20'),
        (0, "--iterations=7 --correctness-repeats=2",
         "--iterations=1 --correctness-repeats=1"),
        (0, 'sha256sum "$inventory" "$policy" "$plan" "$manifest" "$binary"',
         'sha256sum "$inventory" "$policy" "$plan" "$manifest"'),
        (1, "shape[0] not in planner.DECODE_M", "False"),
        (1, '"artifact_tile_k": 0,\n        "physical_layout_class": KPACK4_CLASS,',
         '"artifact_tile_k": 64,\n        "physical_layout_class": KPACK4_CLASS,'),
        (1, 'len(shape_rows) != 20', 'len(shape_rows) != 19'),
        (1, '"max_internal_regret": 0.0', '"max_internal_regret": 0.1'),
        (2, 'summary_value.get("manifest_sha256") != manifest_sha',
         'summary_value.get("manifest_sha256") == manifest_sha'),
        (2, 'one_suffix(final_records, "/generated/manifest.json"',
         'one_suffix(final_records, "/generated/wrong.json"'),
    )
    for index, old, new in plants:
        broken = list(texts)
        if old not in broken[index]:
            raise CheckError(f"missing negative-control seam: {old}")
        broken[index] = broken[index].replace(old, new, 1)
        try:
            check(*broken)
        except CheckError:
            pass
        else:
            raise CheckError(f"negative control stayed green: {old}")
    print("[fq-kpack4-real:self-test] PASS one reused 144-row AP0/AP1 binary, exact "
          "M=1/2/4/8 x five-family denominator, three phases, aggregate and "
          "seven authority plants RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, CheckError, AssertionError) as error:
        print(f"[fq-kpack4-real:self-test] FAIL: {error}")
        raise SystemExit(2)
