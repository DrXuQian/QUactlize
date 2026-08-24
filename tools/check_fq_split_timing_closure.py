#!/usr/bin/env python3
"""Adjudicate legacy-host-gap versus ordered-close Split-K timing."""

from __future__ import annotations

import argparse
import pathlib
import shlex
import sys

from select_fq_split_timing_closure import EXPECTED


AP0 = next(symbol for symbol in EXPECTED if symbol.endswith("_ap0"))


def cells(text: str) -> list[dict[str, str]]:
    result = []
    for line in text.splitlines():
        if not line.startswith("FQ_TC_CELL "):
            continue
        fields = {}
        for token in shlex.split(line.removeprefix("FQ_TC_CELL ")):
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        result.append(fields)
    return result


def exact_census(rows: list[dict[str, str]]) -> None:
    expected = {(symbol, str(split)) for symbol in EXPECTED
                for split in (1, 2, 4, 8)}
    observed = [(row.get("symbol"), row.get("S")) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("AP0/AP1 x S1/S2/S4/S8 census differs")


def check(legacy: str, candidate: str) -> None:
    old, new = cells(legacy), cells(candidate)
    exact_census(old)
    exact_census(new)
    old_failures = {row["S"]: row for row in old
                    if row.get("symbol") == AP0 and row.get("S") in ("2", "4")}
    if set(old_failures) != {"2", "4"}:
        raise ValueError("legacy AP0 S2/S4 denominator differs")
    for split, row in old_failures.items():
        first = int(row.get("first_bad", "-1"))
        if not (row.get("state") == "RAW_FP16_MISMATCH" and
                row.get("failure_step") == "POST_TIMING_RAW_FP16_MISMATCH" and
                row.get("raw_bad") == "32" and first % 64 == 32 and
                row.get("first_want") == "0x4e80" and
                row.get("first_got") == "0x4fc0"):
            raise ValueError(f"legacy S{split} signature differs: {row}")
    measured = [row for row in new if row.get("S") in ("1", "2", "4")]
    if len(measured) != 6:
        raise ValueError("candidate measured denominator differs")
    for row in measured:
        if not (row.get("state") == "MEASURED" and
                row.get("failure_step") == "NONE" and
                row.get("raw_bad") == "0"):
            raise ValueError(f"ordered-close candidate did not close: {row}")
    for row in new:
        if row.get("S") == "8" and row.get("state") != "SPLIT_PARTITION":
            raise ValueError("candidate S8 terminal differs")
    if "split_timing=legacy-host-gap" not in legacy or \
            "status=FAIL" not in legacy:
        raise ValueError("legacy marker did not bind the negative arm")
    if "split_timing=ordered-close" not in candidate or \
            "status=PASS" not in candidate:
        raise ValueError("candidate marker did not close PASS")
    print("[fq-split-timing-check] PASS legacy=AP0/S2+S4/32-bad "
          "candidate=AP0+AP1/S1+S2+S4/raw-bit-exact S8=PARTITION_TERMINAL")


def self_test() -> None:
    def make(legacy: bool) -> str:
        lines = [f"FQ_SHARD split_timing={'legacy-host-gap' if legacy else 'ordered-close'}"]
        for symbol in sorted(EXPECTED):
            for split in (1, 2, 4, 8):
                state, step, bad, first = "MEASURED", "NONE", "0", str(2**64-1)
                want = got = "0x0000"
                if split == 8:
                    state = "SPLIT_PARTITION"
                elif legacy and symbol == AP0 and split in (2, 4):
                    state, step, bad = ("RAW_FP16_MISMATCH",
                                        "POST_TIMING_RAW_FP16_MISMATCH", "32")
                    first, want, got = "32", "0x4e80", "0x4fc0"
                lines.append(
                    f"FQ_TC_CELL symbol={symbol} S={split} state={state} "
                    f"failure_step={step} raw_bad={bad} first_bad={first} "
                    f"first_want={want} first_got={got}")
        lines.append(f"FQ_SHAPE_DONE status={'FAIL' if legacy else 'PASS'}")
        return "\n".join(lines)
    legacy, candidate = make(True), make(False)
    check(legacy, candidate)
    for broken_legacy, broken_candidate in (
            (legacy.replace("raw_bad=32", "raw_bad=31", 1), candidate),
            (legacy, candidate.replace("state=MEASURED", "state=TIMING", 1)),
            (legacy, candidate.replace("split_timing=ordered-close", "split_timing=legacy-host-gap"))):
        try:
            check(broken_legacy, broken_candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("closure negative stayed green")
    print("[fq-split-timing-check:self-test] PASS raw-count, candidate-state, "
          "and protocol-marker negatives RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=pathlib.Path)
    parser.add_argument("--candidate", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if args.legacy is None or args.candidate is None:
                parser.error("--legacy and --candidate are required")
            check(args.legacy.read_text(), args.candidate.read_text())
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-split-timing-check] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
