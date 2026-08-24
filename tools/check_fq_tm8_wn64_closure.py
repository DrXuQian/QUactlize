#!/usr/bin/env python3
"""Validate the exact four-row Q4_K TM8/WM8/WN64 device closure."""

from __future__ import annotations

import argparse
import pathlib
import shlex
import sys

from select_fq_tm8_wn64_closure import EXPECTED


def check(text: str) -> None:
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

    measured = [cell for cell in cells
                if cell.get("S") == "1" and
                cell.get("scope") == "FULL_OUTPUT" and
                cell.get("state") == "MEASURED"]
    measured_symbols = {cell["symbol"] for cell in measured}
    if measured_symbols != EXPECTED or len(measured) != len(EXPECTED):
        raise ValueError(
            f"measured denominator changed: rows={len(measured)} "
            f"missing={sorted(EXPECTED-measured_symbols)} "
            f"extra={sorted(measured_symbols-EXPECTED)}")

    for cell in measured:
        symbol = cell["symbol"]
        expected_stage = "3" if "_s3_" in symbol else "4"
        expected_provider = "packed-row" if symbol.endswith("_ap1") else "standard-aiu"
        if not (cell.get("shape") == "1x1024x5120" and
                cell.get("q") == "12" and cell.get("A") == "64" and
                cell.get("bchunk") == "0" and cell.get("tm") == "8" and
                cell.get("tn") == "64" and cell.get("tk") == "256" and
                cell.get("wm") == "8" and cell.get("wn") == "64" and
                cell.get("stages") == expected_stage and
                cell.get("provider") == expected_provider and
                cell.get("S") == "1" and cell.get("scope") == "FULL_OUTPUT" and
                cell.get("state") == "MEASURED" and cell.get("raw_bad") == "0"):
            raise ValueError(f"row did not close raw-bit exact: {cell}")
    for cell in cells:
        if cell in measured:
            continue
        if not (cell.get("state") == "REAL_CAN_IMPLEMENT" and
                cell.get("raw_bad") == "0"):
            raise ValueError(f"unselected census row changed state: {cell}")
    done = [line for line in text.splitlines()
            if line.startswith(
                "FQ_SHAPE_DONE q=12 A=64 bchunk=0 shape=1x1024x5120 ")]
    if len(done) != 1 or "typed_rows=4" not in done[0] or \
            "selected_rows=4" not in done[0] or \
            "only_split=1" not in done[0] or "bc_mode=skip" not in done[0] or \
            "status=PASS" not in done[0]:
        raise ValueError("shape-level PASS marker missing")
    print("[fq-tm8-wn64-check] PASS measured_rows=4 census_rows=16 "
          "raw_bad=0 providers=2 stages=2 WN=64-retained")


def self_test() -> None:
    lines = []
    for symbol in sorted(EXPECTED):
        stage = 3 if "_s3_" in symbol else 4
        provider = "packed-row" if symbol.endswith("_ap1") else "standard-aiu"
        for split in (1, 2, 4, 8):
            measured = split == 1
            scope = "FULL_OUTPUT" if measured else \
                "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS"
            state = "MEASURED" if measured else "REAL_CAN_IMPLEMENT"
            lines.append(
                "FQ_TC_CELL q=12 A=64 bchunk=0 shape=1x1024x5120 "
                f"symbol={symbol} tm=8 tn=64 tk=256 wm=8 wn=64 stages={stage} "
                f"provider={provider} S={split} scope={scope} state={state} "
                f"us={'1.0' if measured else '0.0'} raw_bad=0 "
                f"samples={'[1.0]' if measured else '[]'}")
    lines.append(
        "FQ_SHAPE_DONE q=12 A=64 bchunk=0 shape=1x1024x5120 "
        "typed_rows=4 selected_rows=4 only_split=1 bc_mode=skip "
        "iterations=3 status=PASS")
    good = "\n".join(lines)
    check(good)
    first_s1 = next(i for i, line in enumerate(lines)
                    if " S=1 " in line and line.startswith("FQ_TC_CELL "))
    first_s2 = next(i for i, line in enumerate(lines)
                    if " S=2 " in line and line.startswith("FQ_TC_CELL "))
    without_one_measured = lines[:first_s1] + lines[first_s1 + 1:]
    measured_s2 = list(lines)
    measured_s2[first_s2] = measured_s2[first_s2].replace(
        "state=REAL_CAN_IMPLEMENT", "state=MEASURED")
    for broken in ("\n".join(without_one_measured),
                   good.replace("raw_bad=0", "raw_bad=512", 1),
                   good.replace("wn=64", "wn=32", 1),
                   "\n".join(measured_s2)):
        try:
            check(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("closure negative stayed green")
    print("[fq-tm8-wn64-check:self-test] PASS measured/census denominators, "
          "raw_bad, WN-substitution and extra-measurement RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if args.log is None:
                parser.error("--log is required")
            check(args.log.read_text())
        return 0
    except (OSError, ValueError, AssertionError) as exc:
        print(f"[fq-tm8-wn64-check] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
