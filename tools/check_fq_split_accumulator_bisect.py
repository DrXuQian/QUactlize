#!/usr/bin/env python3
"""Adjudicate shared partial epilogue versus direct accumulator delivery."""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import pathlib
import sys

from check_fq_packed_owner_candidate import (
    PROVIDER_NAMES,
    direct_summary,
    probe_groups,
    probe_summary,
)
from check_fq_split_workspace_probe import AP0, AP1, exact_cells, records


PERF_SPLITS = (1, 2, 4)
PERF_REGRESSION_LIMIT = 0.03


def totals(cells: dict[tuple[str, str], dict[str, object]],
           provider: str | None = None) -> dict[str, int]:
    result = {
        "samples": 0,
        "bad_samples": 0,
        "partial_bad": 0,
        "canary": 0,
        "host_bad": 0,
        "sync_bad": 0,
        "observed_bad": 0,
    }
    for (symbol, _), cell in cells.items():
        if provider is not None and symbol != provider:
            continue
        for key in result:
            result[key] += int(cell[key])
    return result


def print_cells(arm: str,
                cells: dict[tuple[str, str], dict[str, object]],
                direct_failed: set[tuple[str, str]]) -> None:
    for (symbol, split), cell in sorted(
            cells.items(), key=lambda item: (PROVIDER_NAMES[item[0][0]],
                                             int(item[0][1]))):
        origins = sorted(cell["stripe_origins"])
        origin_text = ",".join(str(value) for value in origins) or "NONE"
        print(
            f"FQ_ACCUMULATOR_BISECT_CELL arm={arm} "
            f"provider={PROVIDER_NAMES[symbol]} S={split} "
            f"samples={cell['samples']} bad_samples={cell['bad_samples']} "
            f"partial_value_raw_bad={cell['partial_bad']} "
            f"bad_plane_mask=0x{int(cell['bad_plane_mask']):x} "
            f"local_n_half_mask=0x{int(cell['local_n_half_mask']):x} "
            f"stripe_origins={origin_text} canary_words={cell['canary']} "
            f"host_reduce_raw_bad={cell['host_bad']} "
            f"sync_only_raw_bad={cell['sync_bad']} "
            f"observed_reducer_raw_bad={cell['observed_bad']} "
            f"direct_failed={int((symbol, split) in direct_failed)}")


def check(epilogue_direct: str, epilogue_probe: str,
          accumulator_direct: str, accumulator_probe: str) -> str:
    epilogue_groups = probe_groups(epilogue_probe)
    accumulator_groups = probe_groups(accumulator_probe)
    _, epilogue_cells = probe_summary(epilogue_groups, False)
    _, accumulator_cells = probe_summary(accumulator_groups, False)
    epilogue_valid, _, epilogue_direct_failed = direct_summary(epilogue_direct)
    accumulator_valid, _, accumulator_direct_failed = direct_summary(
        accumulator_direct)
    if epilogue_valid != 6 or accumulator_valid != 6:
        raise ValueError("direct valid-cell denominator differs")

    epilogue = totals(epilogue_cells)
    accumulator = totals(accumulator_cells)
    epilogue_ap0 = totals(epilogue_cells, AP0)
    accumulator_ap0 = totals(accumulator_cells, AP0)
    accumulator_ap1 = totals(accumulator_cells, AP1)

    if epilogue["canary"] or accumulator["canary"]:
        verdict = "UNADJUDICATED_UNWRITTEN_PARTIALS"
    elif not epilogue_ap0["partial_bad"]:
        verdict = "UNADJUDICATED_EPILOGUE_ARM_DID_NOT_REPRODUCE"
    elif accumulator_ap1["partial_bad"] or any(
            key[0] == AP1 for key in accumulator_direct_failed):
        verdict = "DIRECT_STORE_NEGATIVE_CONTROL_FAILED"
    elif accumulator_ap0["partial_bad"]:
        verdict = "MAINLOOP_ACCUMULATOR_CORRUPTION_CONFIRMED"
    elif accumulator["partial_bad"]:
        verdict = "DIRECT_STORE_UNCLASSIFIED_PROVIDER_FAILURE"
    elif accumulator_ap0["host_bad"] or accumulator_ap0["observed_bad"]:
        verdict = "DIRECT_PARTIALS_EXACT_REDUCER_SEAM_REMAINS"
    else:
        verdict = "PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED"

    print_cells("shared-epilogue", epilogue_cells, epilogue_direct_failed)
    print_cells("direct-accumulator", accumulator_cells,
                accumulator_direct_failed)
    print(
        "FQ_ACCUMULATOR_BISECT "
        f"verdict={verdict} "
        f"epilogue_bad_samples={epilogue['bad_samples']} "
        f"epilogue_partial_value_raw_bad={epilogue['partial_bad']} "
        f"direct_bad_samples={accumulator['bad_samples']} "
        f"direct_partial_value_raw_bad={accumulator['partial_bad']} "
        f"direct_ap1_partial_value_raw_bad={accumulator_ap1['partial_bad']} "
        "mainloop_prefix=IDENTICAL_BY_COMPILE_TIME_POST_MAINLOOP_SEAM")
    return verdict


def parse_samples(row: dict[str, str]) -> list[float]:
    raw = row.get("samples", "")
    if len(raw) < 2 or raw[0] != "[" or raw[-1] != "]":
        raise ValueError(f"invalid timing samples: {row}")
    if raw == "[]":
        return []
    try:
        values = [float(value) for value in raw[1:-1].split(",")]
    except ValueError as error:
        raise ValueError(f"invalid timing sample value: {row}") from error
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"nonpositive/nonfinite timing sample: {row}")
    if values != sorted(values):
        raise ValueError(f"timing samples are not sorted: {row}")
    return values


def median(values: list[float]) -> float:
    middle = len(values) // 2
    return (values[middle] if len(values) & 1 else
            0.5 * (values[middle - 1] + values[middle]))


def performance_cells(text: str, *, production: bool) -> dict[
        tuple[str, int], tuple[dict[str, str], list[float]]]:
    cells = records(text, "FQ_TC_CELL ")
    exact_cells(cells)
    shards = records(text, "FQ_SHARD ")
    done = records(text, "FQ_SHAPE_DONE ")
    if len(shards) != 1 or len(done) != 1:
        raise ValueError("performance shard/done denominator differs")
    shard = shards[0]
    try:
        iterations = int(shard["iterations"])
        repeats = int(shard["correctness_repeats"])
    except (KeyError, ValueError) as error:
        raise ValueError("performance iteration authority missing") from error
    if iterations < 7 or repeats < 8:
        raise ValueError("performance timing/correctness denominator too small")
    if (shard.get("q"), shard.get("A"), shard.get("bchunk"),
            shard.get("shape"), shard.get("split_workspace_probe")) != (
                "12", "64", "0", "1x1024x5120", "0"):
        raise ValueError("performance shard identity differs")
    if production and done[0].get("status") != "PASS":
        raise ValueError("production performance run did not close correctness")

    result: dict[tuple[str, int], tuple[dict[str, str], list[float]]] = {}
    for row in cells:
        symbol = row.get("symbol", "")
        split = int(row.get("S", "0"))
        provider = PROVIDER_NAMES.get(symbol)
        if provider is None or row.get("provider") != provider:
            raise ValueError(f"performance provider identity differs: {row}")
        if (row.get("q"), row.get("A"), row.get("bchunk"),
                row.get("shape"), row.get("tm"), row.get("tn"),
                row.get("tk"), row.get("wm"), row.get("wn"),
                row.get("stages")) != (
                    "12", "64", "0", "1x1024x5120", "8", "64",
                    "256", "8", "16", "2"):
            raise ValueError(f"performance tactic identity differs: {row}")
        raw_bad = int(row.get("raw_bad", "-1"), 0)
        samples = parse_samples(row)
        if split == 8:
            if row.get("state") != "SPLIT_PARTITION" or raw_bad or samples:
                raise ValueError(f"performance S8 control differs: {row}")
        elif row.get("state") == "MEASURED":
            if raw_bad or len(samples) != iterations:
                raise ValueError(f"measured performance cell differs: {row}")
            reported = float(row.get("us", "nan"))
            if not math.isclose(reported, median(samples),
                                rel_tol=0.0, abs_tol=2e-6):
                raise ValueError(f"reported median differs from samples: {row}")
            expected_scope = ("FULL_OUTPUT" if split == 1 else
                              "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS")
            if row.get("scope") != expected_scope:
                raise ValueError(f"performance scope differs: {row}")
        elif production:
            raise ValueError(f"production performance cell is not measured: {row}")
        elif samples:
            raise ValueError(f"failed historical cell retained timings: {row}")
        result[(symbol, split)] = (row, samples)
    return result


def check_performance(epilogue: str, accumulator: str) -> str:
    historical = performance_cells(epilogue, production=False)
    production = performance_cells(accumulator, production=True)
    for symbol in sorted((AP0, AP1), key=lambda value: PROVIDER_NAMES[value]):
        for split in PERF_SPLITS:
            row, samples = production[(symbol, split)]
            print(
                "FQ_ACCUMULATOR_PRODUCTION_PERF "
                f"provider={PROVIDER_NAMES[symbol]} S={split} "
                f"scope={row['scope']} median_us={float(row['us']):.9f} "
                f"range=[{samples[0]:.9f},{samples[-1]:.9f}] "
                f"samples={len(samples)} raw_bit=PASS")

    old_row, old_samples = historical[(AP1, 4)]
    new_row, _ = production[(AP1, 4)]
    if old_row.get("state") != "MEASURED" or not old_samples:
        raise ValueError(
            "historical packed-row S4 timing arm did not remain correct")
    old_us = float(old_row["us"])
    new_us = float(new_row["us"])
    delta = new_us / old_us - 1.0
    verdict = ("PASS" if delta <= PERF_REGRESSION_LIMIT else
               "PERFORMANCE_REGRESSION")
    print(
        "FQ_ACCUMULATOR_PRODUCTION_CLOSURE "
        f"verdict={verdict} comparison=packed-row-S4-same-run "
        f"legacy_shared_us={old_us:.9f} direct_store_us={new_us:.9f} "
        f"delta_pct={100.0 * delta:+.3f} "
        f"regression_limit_pct={100.0 * PERF_REGRESSION_LIMIT:.1f} "
        "correctness=RAW-BIT/PASS metadata=SHIPPING")
    if verdict != "PASS":
        raise ValueError("direct FP32 partial store regressed packed-row S4")
    return verdict


def self_test() -> None:
    from check_fq_packed_owner_candidate import self_test as owner_self_test
    from check_fq_split_workspace_probe import EXPECTED

    # First keep the imported parser/denominator contract live.
    with contextlib.redirect_stdout(io.StringIO()):
        owner_self_test()

    def cell(symbol: str, split: int, probe: bool,
             direct_fail: bool = False) -> str:
        state = (
            "SPLIT_PARTITION" if split == 8 else
            "WORKSPACE_PROBE_COMPLETE" if probe and split in (2, 4) else
            "RAW_FP16_MISMATCH" if direct_fail else
            "MEASURED")
        bad = 32 if direct_fail else 0
        first = 32 if direct_fail else 18446744073709551615
        return (f"FQ_TC_CELL symbol={symbol} S={split} state={state} "
                f"raw_bad={bad} first_bad={first}")

    def logs(ap0_bad: bool, ap1_bad: bool = False) -> tuple[str, str]:
        direct: list[str] = []
        probe: list[str] = [
            "FQ_SHARD split_workspace_probe=1",
            "FQ_WORKSPACE_ORACLE exact=1 S1=0x1 S2=0x2 S4=0x4 S8=0x8",
        ]
        for symbol in sorted(EXPECTED):
            provider_bad = ap0_bad if symbol == AP0 else ap1_bad
            for split in (1, 2, 4, 8):
                direct.append(cell(symbol, split, False,
                                   provider_bad and split == 4))
                probe.append(cell(symbol, split, True))
            for split in (2, 4):
                for repeat in range(8):
                    bad = provider_bad and split == 4 and repeat == 0
                    probe.append(
                        f"FQ_WORKSPACE_PROBE symbol={symbol} S={split} "
                        f"repeat={repeat} canary_words=0 "
                        f"partial_value_raw_bad={32 if bad else 0} "
                        f"partial_bad_plane_mask=0x{1 if bad else 0:x} "
                        f"partial_first_bad_plane={0 if bad else 18446744073709551615} "
                        f"partial_first_bad_index={32 if bad else 18446744073709551615} "
                        "host_reduce_raw_bad=0 sync_only_raw_bad=0 "
                        "observed_reducer_raw_bad=0")
        return "\n".join(direct), "\n".join(probe)

    epilogue_direct, epilogue_probe = logs(True)
    clean_direct, clean_probe = logs(False)
    bad_direct, bad_probe = logs(True)
    control_direct, control_probe = logs(False, True)
    with contextlib.redirect_stdout(io.StringIO()):
        assert check(epilogue_direct, epilogue_probe,
                     clean_direct, clean_probe) == \
            "PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED"
        assert check(epilogue_direct, epilogue_probe,
                     bad_direct, bad_probe) == \
            "MAINLOOP_ACCUMULATOR_CORRUPTION_CONFIRMED"
        assert check(epilogue_direct, epilogue_probe,
                     control_direct, control_probe) == \
            "DIRECT_STORE_NEGATIVE_CONTROL_FAILED"
        assert check(clean_direct, clean_probe,
                     clean_direct, clean_probe) == \
            "UNADJUDICATED_EPILOGUE_ARM_DID_NOT_REPRODUCE"

    def performance_log(direct_us: float, *, historical: bool) -> str:
        iterations = 7
        lines = [
            "FQ_SHARD q=12 A=64 bchunk=0 shape=1x1024x5120 "
            "typed_rows=2 selected_rows=2 split_workspace_probe=0 "
            f"iterations={iterations} correctness_repeats=8"
        ]
        for symbol in sorted(EXPECTED):
            provider = PROVIDER_NAMES[symbol]
            for split in (1, 2, 4, 8):
                base = (28.0 if split == 1 else 17.0 if split == 2 else
                        direct_us if symbol == AP1 else 12.0)
                samples = [base + 0.01 * offset
                           for offset in (-3, -2, -1, 0, 1, 2, 3)]
                state = "SPLIT_PARTITION" if split == 8 else "MEASURED"
                sample_text = ("[]" if split == 8 else
                               "[" + ",".join(f"{x:.9f}" for x in samples) + "]")
                lines.append(
                    "FQ_TC_CELL q=12 A=64 bchunk=0 shape=1x1024x5120 "
                    f"symbol={symbol} tm=8 tn=64 tk=256 wm=8 wn=16 "
                    f"stages=2 provider={provider} S={split} "
                    f"scope={'FULL_OUTPUT' if split == 1 else 'PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS'} "
                    f"state={state} us={0.0 if split == 8 else base:.9f} "
                    f"raw_bad=0 samples={sample_text}")
        lines.append(
            "FQ_SHAPE_DONE q=12 A=64 bchunk=0 shape=1x1024x5120 "
            f"iterations={iterations} status={'FAIL' if historical else 'PASS'}")
        return "\n".join(lines)

    historical_perf = performance_log(11.40, historical=True)
    production_perf = performance_log(11.20, historical=False)
    with contextlib.redirect_stdout(io.StringIO()):
        assert check_performance(historical_perf, production_perf) == "PASS"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            check_performance(historical_perf,
                              performance_log(12.0, historical=False))
    except ValueError:
        pass
    else:
        raise AssertionError("performance regression stayed green")
    print("[fq-accumulator-bisect:self-test] PASS mainloop/epilogue/control/"
          "nonreproduction verdicts, exact performance A/B and regression RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epilogue-direct", type=pathlib.Path)
    parser.add_argument("--epilogue-probe", type=pathlib.Path)
    parser.add_argument("--accumulator-direct", type=pathlib.Path)
    parser.add_argument("--accumulator-probe", type=pathlib.Path)
    parser.add_argument("--epilogue-performance", type=pathlib.Path)
    parser.add_argument("--accumulator-performance", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            required = (args.epilogue_direct, args.epilogue_probe,
                        args.accumulator_direct, args.accumulator_probe)
            if any(path is None for path in required):
                parser.error("all four logs are required")
            verdict = check(*(path.read_text() for path in required))
            performance = (args.epilogue_performance,
                           args.accumulator_performance)
            if any(path is not None for path in performance):
                if any(path is None for path in performance):
                    parser.error("both performance logs are required")
                if verdict != "PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED":
                    raise ValueError(
                        f"production closure rejected semantic verdict {verdict}")
                check_performance(*(path.read_text() for path in performance))
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-accumulator-bisect] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
