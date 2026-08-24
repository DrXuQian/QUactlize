#!/usr/bin/env python3
"""Classify the exact Q4_K AP0/AP1 Split-K partial-workspace seam."""

from __future__ import annotations

import argparse
import pathlib
import shlex
import sys

from select_fq_split_timing_closure import EXPECTED


AP0 = next(symbol for symbol in EXPECTED if symbol.endswith("_ap0"))
AP1 = next(symbol for symbol in EXPECTED if symbol.endswith("_ap1"))


def records(text: str, prefix: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        fields: dict[str, str] = {}
        for token in shlex.split(line.removeprefix(prefix)):
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        result.append(fields)
    return result


def exact_cells(rows: list[dict[str, str]]) -> None:
    expected = {(symbol, str(split)) for symbol in EXPECTED
                for split in (1, 2, 4, 8)}
    observed = [(row.get("symbol"), row.get("S")) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("AP0/AP1 x S1/S2/S4/S8 cell census differs")


def nonnegative(row: dict[str, str], field: str) -> int:
    try:
        value = int(row[field], 0)
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid {field}: {row}") from error
    if value < 0:
        raise ValueError(f"negative {field}: {row}")
    return value


def direct_failure_signature(row: dict[str, str]) -> None:
    raw_bad = nonnegative(row, "raw_bad")
    first_bad = nonnegative(row, "first_bad")
    # The stable signature is the first affected 32-output stripe.  The
    # number of unequal lanes inside one or more affected stripes is race
    # dependent (the device has produced 32, 56 and 64), so raw_bad is not
    # itself required to be a multiple of 32.
    if not raw_bad or raw_bad > 1024 or first_bad >= 1024 or first_bad % 32:
        raise ValueError(
            f"direct failure lost aligned stripe-origin signature: {row}")


def classify(samples: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
    totals = {
        field: sum(nonnegative(row, field) for row in samples)
        for field in ("canary_words", "host_reduce_raw_bad",
                      "sync_only_raw_bad", "observed_reducer_raw_bad")
    }
    if totals["canary_words"]:
        verdict = "UNWRITTEN_PARTIAL_WORDS"
    elif totals["host_reduce_raw_bad"]:
        verdict = "PRODUCER_OR_PARTIAL_STORE_BAD"
    elif totals["observed_reducer_raw_bad"]:
        verdict = "REDUCER_LOAD_OR_INDEX_BAD"
    elif totals["sync_only_raw_bad"]:
        verdict = "D2H_VISIBILITY_BRIDGE_REQUIRED"
    else:
        verdict = "SAME_STREAM_PUBLICATION_GAP"
    return verdict, totals


def check(direct: str, probe: str) -> str:
    direct_cells = records(direct, "FQ_TC_CELL ")
    probe_cells = records(probe, "FQ_TC_CELL ")
    exact_cells(direct_cells)
    exact_cells(probe_cells)

    direct_ap0_fail = [row for row in direct_cells
                       if row.get("symbol") == AP0 and
                       row.get("S") in ("2", "4") and
                       row.get("state") == "RAW_FP16_MISMATCH" and
                       nonnegative(row, "raw_bad") > 0]
    failed_splits = {row.get("S") for row in direct_ap0_fail}
    if failed_splits != {"2", "4"}:
        raise ValueError(
            "direct AP0 S2/S4 failure denominator was not reproduced: "
            f"{sorted(failed_splits)}")
    direct_failures: dict[str, list[str]] = {AP0: [], AP1: []}
    for row in direct_cells:
        symbol, split = row.get("symbol", ""), row.get("S", "")
        state, raw_bad = row.get("state"), nonnegative(row, "raw_bad")
        if split == "1":
            if state != "MEASURED" or raw_bad:
                raise ValueError(f"S1 control is not raw-bit exact: {row}")
        elif split == "8":
            if state != "SPLIT_PARTITION" or raw_bad:
                raise ValueError(f"S8 partition control changed: {row}")
        elif state == "RAW_FP16_MISMATCH":
            direct_failure_signature(row)
            direct_failures[symbol].append(split)
        elif state != "MEASURED" or raw_bad:
            raise ValueError(f"unexpected direct Split-K state: {row}")

    expected_probe_cells = {(symbol, str(split)) for symbol in EXPECTED
                            for split in (2, 4)}
    completed = {(row.get("symbol"), row.get("S")) for row in probe_cells
                 if row.get("state") == "WORKSPACE_PROBE_COMPLETE"}
    if completed != expected_probe_cells:
        raise ValueError("workspace-probe completion denominator differs")
    if "split_workspace_probe=1" not in probe:
        raise ValueError("probe-mode marker missing")

    rows = records(probe, "FQ_WORKSPACE_PROBE ")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("symbol", ""), row.get("S", ""))
        if key not in expected_probe_cells:
            raise ValueError(f"unexpected workspace-probe key: {key}")
        grouped.setdefault(key, []).append(row)
    if set(grouped) != expected_probe_cells:
        raise ValueError("workspace-probe sample denominator is incomplete")
    counts = {len(value) for value in grouped.values()}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("workspace-probe repeats differ or are too small")
    for key, samples in grouped.items():
        repeats = sorted(nonnegative(row, "repeat") for row in samples)
        if repeats != list(range(len(samples))):
            raise ValueError(f"workspace-probe repeat denominator differs: {key}")

    provider_verdicts: dict[str, str] = {}
    provider_totals: dict[str, dict[str, int]] = {}
    for symbol, provider in ((AP0, "standard-aiu"), (AP1, "packed-row")):
        samples = grouped[(symbol, "2")] + grouped[(symbol, "4")]
        provider_verdicts[symbol], provider_totals[symbol] = classify(samples)
        totals = provider_totals[symbol]
        print("FQ_SPLIT_WORKSPACE_PROVIDER "
              f"provider={provider} symbol={symbol} "
              f"verdict={provider_verdicts[symbol]} "
              f"direct_failure_splits={','.join(direct_failures[symbol]) or 'none'} "
              f"samples={len(samples)} canary_words={totals['canary_words']} "
              f"host_reduce_raw_bad={totals['host_reduce_raw_bad']} "
              f"sync_only_raw_bad={totals['sync_only_raw_bad']} "
              f"observed_reducer_raw_bad={totals['observed_reducer_raw_bad']}")
    unique_verdicts = set(provider_verdicts.values())
    verdict = (next(iter(unique_verdicts)) if len(unique_verdicts) == 1
               else "MIXED_PROVIDER_VERDICTS")
    totals = {
        field: sum(provider_totals[symbol][field] for symbol in (AP0, AP1))
        for field in ("canary_words", "host_reduce_raw_bad",
                      "sync_only_raw_bad", "observed_reducer_raw_bad")
    }
    print("FQ_SPLIT_WORKSPACE_VERDICT "
          f"verdict={verdict} ap0={provider_verdicts[AP0]} "
          f"ap1={provider_verdicts[AP1]} "
          f"direct_failures={sum(map(len, direct_failures.values()))} "
          f"samples_per_cell={next(iter(counts))} "
          f"canary_words={totals['canary_words']} "
          f"host_reduce_raw_bad={totals['host_reduce_raw_bad']} "
          f"sync_only_raw_bad={totals['sync_only_raw_bad']} "
          f"observed_reducer_raw_bad={totals['observed_reducer_raw_bad']}")
    return verdict


def self_test() -> None:
    def cell(symbol: str, split: int, fail: bool = False,
             probe: bool = False) -> str:
        state = "SPLIT_PARTITION" if split == 8 else \
            "WORKSPACE_PROBE_COMPLETE" if probe and split in (2, 4) else \
            "RAW_FP16_MISMATCH" if fail else "MEASURED"
        # Preserve the observed non-integral stripe population: the first bad
        # index is stripe-aligned even when only 56 outputs compare unequal.
        bad = (56 if split == 4 else 32) if fail else 0
        first = 32 if fail else 2**64 - 1
        return (f"FQ_TC_CELL symbol={symbol} S={split} state={state} "
                f"raw_bad={bad} first_bad={first}")

    direct_lines, probe_lines = [], ["FQ_SHARD split_workspace_probe=1"]
    for symbol in sorted(EXPECTED):
        for split in (1, 2, 4, 8):
            direct_lines.append(cell(symbol, split,
                                     (symbol == AP0 and split in (2, 4)) or
                                     (symbol == AP1 and split == 2)))
            probe_lines.append(cell(symbol, split, probe=True))
        for split in (2, 4):
            for repeat in range(2):
                probe_lines.append(
                    f"FQ_WORKSPACE_PROBE symbol={symbol} S={split} "
                    f"repeat={repeat} sync_only_raw_bad=0 canary_words=0 "
                    f"host_reduce_raw_bad=0 observed_reducer_raw_bad=0")
    direct = "\n".join(direct_lines)
    probe = "\n".join(probe_lines)
    assert check(direct, probe) == "SAME_STREAM_PUBLICATION_GAP"
    variants = (
        probe.replace("canary_words=0", "canary_words=32"),
        probe.replace("host_reduce_raw_bad=0", "host_reduce_raw_bad=32"),
        probe.replace("sync_only_raw_bad=0", "sync_only_raw_bad=32"),
        probe.replace("observed_reducer_raw_bad=0",
                      "observed_reducer_raw_bad=32"),
    )
    assert check(direct, variants[0]) == "UNWRITTEN_PARTIAL_WORDS"
    assert check(direct, variants[1]) == "PRODUCER_OR_PARTIAL_STORE_BAD"
    assert check(direct, variants[2]) == "D2H_VISIBILITY_BRIDGE_REQUIRED"
    assert check(direct, variants[3]) == "REDUCER_LOAD_OR_INDEX_BAD"
    mixed = probe.replace(
        "host_reduce_raw_bad=0", "host_reduce_raw_bad=32", 1)
    assert check(direct, mixed) == "MIXED_PROVIDER_VERDICTS"
    for broken_direct, broken_probe in (
            (direct.replace("first_bad=32", "first_bad=33", 1), probe),
            (direct, probe.replace("split_workspace_probe=1", "split_workspace_probe=0")),
            (direct, "\n".join(probe.splitlines()[:-1]))):
        try:
            check(broken_direct, broken_probe)
        except ValueError:
            pass
        else:
            raise AssertionError("workspace-probe negative stayed green")
    print("[fq-split-workspace-check:self-test] PASS five seam verdicts plus "
          "mixed-provider; intermittent AP1, aligned stripe-origin, marker "
          "and denominator negatives RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=pathlib.Path)
    parser.add_argument("--probe", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if args.direct is None or args.probe is None:
                parser.error("--direct and --probe are required")
            check(args.direct.read_text(), args.probe.read_text())
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-split-workspace-check] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
