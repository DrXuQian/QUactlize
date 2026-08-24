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
    relevant = [cell for cell in cells if cell.get("symbol") in EXPECTED]
    symbols = {cell["symbol"] for cell in relevant}
    if symbols != EXPECTED or len(relevant) != 4:
        raise ValueError(
            f"device denominator changed: rows={len(relevant)} "
            f"missing={sorted(EXPECTED-symbols)} extra={sorted(symbols-EXPECTED)}")
    for cell in relevant:
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
    done = [line for line in text.splitlines()
            if line.startswith(
                "FQ_SHAPE_DONE q=12 A=64 bchunk=0 shape=1x1024x5120 ")]
    if len(done) != 1 or "selected_rows=4" not in done[0] or \
            "only_split=1" not in done[0] or "bc_mode=skip" not in done[0] or \
            "status=PASS" not in done[0]:
        raise ValueError("shape-level PASS marker missing")
    print("[fq-tm8-wn64-check] PASS rows=4 raw_bad=0 providers=2 stages=2 WN=64-retained")


def self_test() -> None:
    lines = []
    for symbol in sorted(EXPECTED):
        stage = 3 if "_s3_" in symbol else 4
        provider = "packed-row" if symbol.endswith("_ap1") else "standard-aiu"
        lines.append(
            "FQ_TC_CELL q=12 A=64 bchunk=0 shape=1x1024x5120 "
            f"symbol={symbol} tm=8 tn=64 tk=256 wm=8 wn=64 stages={stage} "
            f"provider={provider} S=1 scope=FULL_OUTPUT state=MEASURED "
            "us=1.0 raw_bad=0 samples=[1.0]")
    lines.append(
        "FQ_SHAPE_DONE q=12 A=64 bchunk=0 shape=1x1024x5120 "
        "typed_rows=4 selected_rows=4 only_split=1 bc_mode=skip "
        "iterations=3 status=PASS")
    good = "\n".join(lines)
    check(good)
    for broken in ("\n".join(lines[:-2] + lines[-1:]),
                   good.replace("raw_bad=0", "raw_bad=512", 1),
                   good.replace("wn=64", "wn=32", 1)):
        try:
            check(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("closure negative stayed green")
    print("[fq-tm8-wn64-check:self-test] PASS missing-row, raw_bad and WN-substitution RED")


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
