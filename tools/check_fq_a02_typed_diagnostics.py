#!/usr/bin/env python3
"""Adjudicate exact A02 Q4 provider and Q3 effective-BChunk device rows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

from select_fq_a02_typed_diagnostics import Q3, Q4


class CheckError(ValueError):
    pass


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", line))


def check_text(q4_text: str, q3_text: str, repeats: int) -> list[str]:
    result = []
    for label, text, symbols in (("q4", q4_text, Q4), ("q3", q3_text, Q3)):
        shards = [fields(line) for line in text.splitlines() if line.startswith("FQ_SHARD ")]
        done = [fields(line) for line in text.splitlines() if line.startswith("FQ_SHAPE_DONE ")]
        cells = [fields(line) for line in text.splitlines() if line.startswith("FQ_TC_CELL ")]
        if len(shards) != 1 or len(done) != 1 or \
           shards[0].get("correctness_repeats") != str(repeats) or \
           shards[0].get("selected_rows") != "2":
            raise CheckError(f"{label}: shard/repeat denominator differs")
        if done[0].get("status") != "PASS" or done[0].get("selected_rows") != "2":
            raise CheckError(f"{label}: completion differs")
        s1 = {row.get("symbol"): row for row in cells if row.get("S") == "1"}
        if set(s1) != set(symbols):
            raise CheckError(f"{label}: exact S1 rows differ")
        for symbol in symbols:
            row = s1[symbol]
            if row.get("scope") != "FULL_OUTPUT" or row.get("state") != "MEASURED" or \
               row.get("raw_bad") != "0" or row.get("failure_step") != "NONE":
                raise CheckError(f"{symbol}: raw-bit/full-output closure differs")
    result.append(
        "A02_DEVICE_CELL classification=PRODUCT_SHIPPING route=dense "
        "q=12 metadata=InterleavedHalf2 provider=AP0 bchunk=request0-effective0 "
        f"symbol={Q4[0]} raw_bad=0 repeats={repeats} status=PASS")
    result.append(
        "A02_DEVICE_CELL classification=TYPED_DIAGNOSTIC_NONPRODUCT route=dense "
        "q=12 metadata=InterleavedHalf2 provider=AP1 bchunk=request0-effective0 "
        f"symbol={Q4[1]} raw_bad=0 repeats={repeats} status=PASS")
    for bc, symbol in enumerate(Q3):
        result.append(
            "A02_DEVICE_CELL classification=TYPED_DIAGNOSTIC_NONPRODUCT route=dense "
            f"q=11 provider=AP0 bchunk=request{bc}-effective{bc} symbol={symbol} "
            f"raw_bad=0 repeats={repeats} status=PASS")
    result.append(
        "A02_GROUPED_EVIDENCE classification=EXTERNAL_A01_PRODUCT_GATE_REFERENCE "
        "metadata=SeparateHalfPlanes execution_claim=NONE")
    return result


def self_test() -> None:
    def log(q, symbols):
        head = f"FQ_SHARD q={q} selected_rows=2 correctness_repeats=4096"
        cells = [f"FQ_TC_CELL symbol={s} S=1 scope=FULL_OUTPUT state=MEASURED raw_bad=0 failure_step=NONE"
                 for s in symbols]
        return "\n".join([head, *cells, f"FQ_SHAPE_DONE selected_rows=2 status=PASS"])
    q4, q3 = log(12, Q4), log(11, Q3)
    assert len(check_text(q4, q3, 4096)) == 5
    for planted in (q4.replace("raw_bad=0", "raw_bad=1", 1),
                    q4.replace(Q4[1], Q4[0]),
                    q4.replace("selected_rows=2", "selected_rows=3", 1)):
        try:
            check_text(planted, q3, 4096)
        except CheckError:
            pass
        else:
            raise AssertionError("A02 planted regression stayed green")
    print("[fq-a02-check:self-test] PASS exact-four; raw/duplicate/denominator plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--q4-log", type=Path)
    parser.add_argument("--q3-log", type=Path)
    parser.add_argument("--repeats", type=int, default=4096)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if not args.q4_log or not args.q3_log or args.repeats < 1024:
                raise CheckError("logs and high-cadence repeats>=1024 are required")
            for line in check_text(args.q4_log.read_text(), args.q3_log.read_text(), args.repeats):
                print(line)
        return 0
    except (OSError, CheckError, AssertionError) as error:
        print(f"[fq-a02-check] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
