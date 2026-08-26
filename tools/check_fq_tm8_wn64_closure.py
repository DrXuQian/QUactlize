#!/usr/bin/env python3
"""Validate aligned and N-tail Q4_K packed-metadata device closure."""

from __future__ import annotations

import argparse
import pathlib
import shlex
import sys

from select_fq_tm8_wn64_closure import EXPECTED


SAFE_STATES = {
    "MEASURED",
    "REAL_CAN_IMPLEMENT",
    "SPLIT_PARTITION",
    "INADMISSIBLE_PIPELINE_DEPTH",
}


def check_one(text: str, shape: str) -> None:
    cells = []
    for line in text.splitlines():
        if line.startswith("FQ_TC_CELL "):
            fields = {}
            for token in shlex.split(line.removeprefix("FQ_TC_CELL ")):
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key] = value
            cells.append(fields)
    symbols = {cell.get("symbol") for cell in cells}
    if symbols != EXPECTED:
        raise ValueError(
            f"device symbol denominator changed: rows={len(cells)} "
            f"missing={sorted(EXPECTED-symbols)} extra={sorted(symbols-EXPECTED)}")
    census_keys = [(cell.get("symbol"), cell.get("S")) for cell in cells]
    expected_census = {(symbol, split) for symbol in EXPECTED
                       for split in ("1", "2", "4", "8")}
    if len(census_keys) != len(expected_census) or set(census_keys) != expected_census:
        raise ValueError(
            f"device census denominator changed: rows={len(census_keys)} "
            f"missing={sorted(expected_census-set(census_keys))} "
            f"extra={sorted(set(census_keys)-expected_census)}")

    s1 = [cell for cell in cells
          if cell.get("S") == "1" and
          cell.get("scope") == "FULL_OUTPUT" and
          cell.get("state") == "MEASURED"]
    measured_symbols = {cell["symbol"] for cell in s1}
    if measured_symbols != EXPECTED or len(s1) != len(EXPECTED):
        raise ValueError(
            f"S1 measured denominator changed: rows={len(s1)} "
            f"missing={sorted(EXPECTED-measured_symbols)} "
            f"extra={sorted(measured_symbols-EXPECTED)}")

    for cell in cells:
        symbol = cell["symbol"]
        expected_stage = symbol.split("_s", 1)[1].split("_", 1)[0]
        expected_wn = symbol.split("_wn", 1)[1].split("_", 1)[0]
        expected_provider = "packed-row" if symbol.endswith("_ap1") else "standard-aiu"
        if not (cell.get("shape") == shape and
                cell.get("q") == "12" and cell.get("A") == "64" and
                cell.get("bchunk") == "0" and cell.get("tm") == "8" and
                cell.get("tn") == "64" and cell.get("tk") == "256" and
                cell.get("wm") == "8" and cell.get("wn") == expected_wn and
                cell.get("stages") == expected_stage and
                cell.get("provider") == expected_provider and
                cell.get("state") in SAFE_STATES and
                cell.get("raw_bad") == "0"):
            raise ValueError(f"row did not close raw-bit exact: {cell}")
    required_split = {
        (symbol, split)
        for symbol in EXPECTED if "_wn16_" in symbol
        for split in ("1", "2", "4")
    }
    measured_split = {
        (cell["symbol"], cell["S"])
        for cell in cells
        if cell.get("state") == "MEASURED"
    }
    if not required_split <= measured_split:
        raise ValueError(
            f"WN16 S1/S2/S4 closure missing: {sorted(required_split-measured_split)}")
    done = [line for line in text.splitlines()
            if line.startswith(
                f"FQ_SHAPE_DONE q=12 A=64 bchunk=0 shape={shape} ")]
    if len(done) != 1 or "typed_rows=6" not in done[0] or \
            "selected_rows=6" not in done[0] or \
            "only_split=0" not in done[0] or "bc_mode=skip" not in done[0] or \
            "status=PASS" not in done[0]:
        raise ValueError("shape-level PASS marker missing")


def check(aligned: str, tail: str) -> None:
    check_one(aligned, "1x1024x5120")
    check_one(tail, "1x992x5120")
    print("[fq-tm8-wn64-check] PASS shapes=2 tactics=6 census_rows=48 "
          "raw_bad=0 WN64-control=4 WN16-S1/S2/S4=6 tail=EXACT")


def fixture(shape: str) -> str:
    lines = []
    for symbol in sorted(EXPECTED):
        stage = symbol.split("_s", 1)[1].split("_", 1)[0]
        wn = symbol.split("_wn", 1)[1].split("_", 1)[0]
        provider = "packed-row" if symbol.endswith("_ap1") else "standard-aiu"
        for split in (1, 2, 4, 8):
            measured = split == 1 or (wn == "16" and split in (2, 4))
            scope = "FULL_OUTPUT" if split == 1 else \
                "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS"
            state = "MEASURED" if measured else (
                "SPLIT_PARTITION" if split == 8 else "REAL_CAN_IMPLEMENT")
            lines.append(
                f"FQ_TC_CELL q=12 A=64 bchunk=0 shape={shape} "
                f"symbol={symbol} tm=8 tn=64 tk=256 wm=8 wn={wn} stages={stage} "
                f"provider={provider} S={split} scope={scope} state={state} "
                f"us={'1.0' if measured else '0.0'} raw_bad=0 "
                f"samples={'[1.0]' if measured else '[]'}")
    lines.append(
        f"FQ_SHAPE_DONE q=12 A=64 bchunk=0 shape={shape} "
        "typed_rows=6 selected_rows=6 only_split=0 bc_mode=skip "
        "iterations=3 status=PASS")
    return "\n".join(lines)


def self_test() -> None:
    aligned = fixture("1x1024x5120")
    tail = fixture("1x992x5120")
    check(aligned, tail)
    first_s2 = next(line for line in aligned.splitlines()
                    if "_wn16_" in line and " S=2 " in line)
    for broken_aligned, broken_tail in (
            (aligned.replace(first_s2 + "\n", ""), tail),
            (aligned.replace("raw_bad=0", "raw_bad=32", 1), tail),
            (aligned.replace("_wn64_", "_wn32_", 1), tail),
            (aligned.replace("state=MEASURED", "state=RAW_FP16_MISMATCH", 1), tail),
            (aligned, tail.replace("status=PASS", "status=FAIL", 1))):
        try:
            check(broken_aligned, broken_tail)
        except ValueError:
            pass
        else:
            raise AssertionError("closure negative stayed green")
    print("[fq-tm8-wn64-check:self-test] PASS aligned/tail denominators, "
          "raw_bad, WN-substitution, failure-state and tail-status RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-log", type=pathlib.Path)
    parser.add_argument("--tail-log", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if args.aligned_log is None or args.tail_log is None:
                parser.error("--aligned-log and --tail-log are required")
            check(args.aligned_log.read_text(), args.tail_log.read_text())
        return 0
    except (OSError, ValueError, AssertionError) as exc:
        print(f"[fq-tm8-wn64-check] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
