#!/usr/bin/env python3
"""Adjudicate shared partial epilogue versus direct accumulator delivery."""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import sys

from check_fq_packed_owner_candidate import (
    PROVIDER_NAMES,
    direct_summary,
    probe_groups,
    probe_summary,
)
from check_fq_split_workspace_probe import AP0, AP1


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
    print("[fq-accumulator-bisect:self-test] PASS mainloop/epilogue/control/"
          "nonreproduction verdicts and imported exact denominator")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epilogue-direct", type=pathlib.Path)
    parser.add_argument("--epilogue-probe", type=pathlib.Path)
    parser.add_argument("--accumulator-direct", type=pathlib.Path)
    parser.add_argument("--accumulator-probe", type=pathlib.Path)
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
            check(*(path.read_text() for path in required))
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-accumulator-bisect] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
