#!/usr/bin/env python3
"""Fail-closed source contract for producer-only Split-K timing."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SHARED = ROOT / "benchmarks/splitk_producer_timing.hpp"
FQ = ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp"
SF = ROOT / "benchmarks/scalefirst_internal_sweep_bench.hpp"
MAIN = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
RUNNER = ROOT / "tools/run_fq_q4k_split_timing_closure_box.sh"
SELECT = ROOT / "tools/select_fq_split_timing_closure.py"
ADJUDICATE = ROOT / "tools/check_fq_split_timing_closure.py"


class CheckError(ValueError):
    pass


def check(shared: str, fq: str, sf: str, main: str, runner: str,
          select: str, adjudicate: str) -> None:
    order = (
        "hggcEventRecord(events.start, nullptr)",
        "producer()",
        "hggcEventRecord(events.stop, nullptr)",
        "consumer()",
        "hggcEventSynchronize(events.stop)",
        "hggcEventElapsedTime(&ms, events.start, events.stop)",
        "hggcDeviceSynchronize()",
    )
    positions = []
    cursor = shared.find("Result measure(")
    if cursor < 0:
        raise CheckError("shared ordered-close helper is missing")
    for token in order:
        position = shared.find(token, cursor)
        if position < 0:
            raise CheckError(f"ordered-close seam missing: {token}")
        positions.append(position)
        cursor = position + len(token)
    if positions != sorted(positions):
        raise CheckError("producer/stop/reducer/wait order changed")
    if "The stop event is recorded after the producer and before the reducer" \
            not in shared or "partial workspace is therefore never reused" \
            not in shared:
        raise CheckError("timing/performance invariant is not documented")
    fq_needles = (
        "bool legacy_split_timing = false;",
        "splitk_producer_timing::measure(",
        "if (options.legacy_split_timing)",
        "POST_TIMING_RAW_FP16_MISMATCH",
        "ORDERED_CLOSE_RAW_FP16_MISMATCH",
    )
    sf_needles = (
        "splitk_producer_timing::measure(",
        "ORDERED_CLOSE_RAW_FP16_MISMATCH",
    )
    main_needles = (
        '"--legacy-split-timing"',
        '"legacy-host-gap" : "ordered-close"',
    )
    runner_needles = (
        "--legacy-split-timing",
        "legacy_rc",
        "candidate_rc",
        "check_fq_split_timing_closure.py",
    )
    select_needles = (
        "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0",
        "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1",
        '"selection_denominator": len(rows)',
    )
    adjudicate_needles = (
        'row.get("raw_bad") == "32"',
        'first % 64 == 32',
        'row.get("first_want") == "0x4e80"',
        'row.get("first_got") == "0x4fc0"',
        "AP0/AP1 x S1/S2/S4/S8 census differs",
    )
    for label, text, needles in (
            ("FQ", fq, fq_needles), ("ScaleFirst", sf, sf_needles),
            ("main", main, main_needles), ("runner", runner, runner_needles),
            ("selector", select, select_needles),
            ("adjudicator", adjudicate, adjudicate_needles)):
        missing = [token for token in needles if token not in text]
        if missing:
            raise CheckError(f"{label} contract missing: {missing}")


def main() -> int:
    paths = (SHARED, FQ, SF, MAIN, RUNNER, SELECT, ADJUDICATE)
    texts = [path.read_text() for path in paths]
    check(*texts)
    plants = (
        (0, "consumer()", "cutlass::Status::kSuccess"),
        (0, "hggcEventSynchronize(events.stop)",
         "hggcDeviceSynchronize()"),
        (1, "if (options.legacy_split_timing)", "if (false)"),
        (2, "splitk_producer_timing::measure(", "measure("),
        (3, '"--legacy-split-timing"', '"--lost-legacy-arm"'),
        (4, "--legacy-split-timing", "--lost-legacy-arm"),
        (5, "_ap1\"", "_ap2\""),
        (6, 'row.get("raw_bad") == "32"',
         'row.get("raw_bad") == "31"'),
    )
    for index, old, new in plants:
        planted = list(texts)
        if old not in planted[index]:
            raise CheckError(f"negative seam missing: {old}")
        planted[index] = planted[index].replace(old, new, 1)
        try:
            check(*planted)
        except CheckError:
            pass
        else:
            raise CheckError(f"negative stayed green: {old} -> {new}")
    print("[splitk-producer-timing:self-test] PASS: stop-before-reducer span, "
          "reducer-before-host-wait publication, shared FQ/ScaleFirst use, "
          "exact legacy arm, AP0/AP1 denominator, and eight negatives")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, OSError) as error:
        print(f"[splitk-producer-timing:self-test] FAIL: {error}")
        raise SystemExit(2)
