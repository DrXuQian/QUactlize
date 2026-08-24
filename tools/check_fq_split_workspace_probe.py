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
    if not direct_ap0_fail:
        raise ValueError("direct AP0 S2/S4 failure was not reproduced")
    for row in direct_ap0_fail:
        if nonnegative(row, "raw_bad") % 32 or \
                nonnegative(row, "first_bad") % 32:
            raise ValueError(f"direct failure lost 32-output stripe signature: {row}")

    for row in direct_cells:
        if row.get("symbol") == AP1 and row.get("S") in ("1", "2", "4"):
            if row.get("state") != "MEASURED" or nonnegative(row, "raw_bad"):
                raise ValueError(f"AP1 control is not raw-bit exact: {row}")

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

    # AP1 is the exact same B/layout/partition control with only the A provider
    # changed.  It must prove that the oracle and observed reducer are sound.
    for split in ("2", "4"):
        for row in grouped[(AP1, split)]:
            if nonnegative(row, "canary_words") or \
                    nonnegative(row, "host_reduce_raw_bad") or \
                    nonnegative(row, "observed_reducer_raw_bad"):
                raise ValueError(f"AP1 probe control failed: {row}")

    ap0 = grouped[(AP0, "2")] + grouped[(AP0, "4")]
    canary = sum(nonnegative(row, "canary_words") for row in ap0)
    host_bad = sum(nonnegative(row, "host_reduce_raw_bad") for row in ap0)
    sync_bad = sum(nonnegative(row, "sync_only_raw_bad") for row in ap0)
    observed_bad = sum(nonnegative(row, "observed_reducer_raw_bad") for row in ap0)
    if canary:
        verdict = "UNWRITTEN_PARTIAL_WORDS"
    elif host_bad:
        verdict = "PRODUCER_OR_PARTIAL_STORE_BAD"
    elif observed_bad:
        verdict = "REDUCER_LOAD_OR_INDEX_BAD"
    elif sync_bad:
        verdict = "D2H_VISIBILITY_BRIDGE_REQUIRED"
    else:
        verdict = "SAME_STREAM_PUBLICATION_GAP"
    print("FQ_SPLIT_WORKSPACE_VERDICT "
          f"verdict={verdict} direct_failures={len(direct_ap0_fail)} "
          f"samples_per_cell={next(iter(counts))} canary_words={canary} "
          f"host_reduce_raw_bad={host_bad} sync_only_raw_bad={sync_bad} "
          f"observed_reducer_raw_bad={observed_bad}")
    return verdict


def self_test() -> None:
    def cell(symbol: str, split: int, fail: bool = False,
             probe: bool = False) -> str:
        state = "SPLIT_PARTITION" if split == 8 else \
            "WORKSPACE_PROBE_COMPLETE" if probe and split in (2, 4) else \
            "RAW_FP16_MISMATCH" if fail else "MEASURED"
        bad = 32 if fail else 0
        first = 32 if fail else 2**64 - 1
        return (f"FQ_TC_CELL symbol={symbol} S={split} state={state} "
                f"raw_bad={bad} first_bad={first}")

    direct_lines, probe_lines = [], ["FQ_SHARD split_workspace_probe=1"]
    for symbol in sorted(EXPECTED):
        for split in (1, 2, 4, 8):
            direct_lines.append(cell(symbol, split,
                                     symbol == AP0 and split == 2))
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
        probe.replace("host_reduce_raw_bad=0", "host_reduce_raw_bad=32", 1),
        probe.replace("sync_only_raw_bad=0", "sync_only_raw_bad=32", 1),
        probe.replace("observed_reducer_raw_bad=0",
                      "observed_reducer_raw_bad=32", 1),
    )
    assert check(direct, variants[0]) == "PRODUCER_OR_PARTIAL_STORE_BAD"
    assert check(direct, variants[1]) == "D2H_VISIBILITY_BRIDGE_REQUIRED"
    assert check(direct, variants[2]) == "REDUCER_LOAD_OR_INDEX_BAD"
    for broken_direct, broken_probe in (
            (direct.replace("raw_bad=32", "raw_bad=31", 1), probe),
            (direct, probe.replace("split_workspace_probe=1", "split_workspace_probe=0")),
            (direct, "\n".join(probe.splitlines()[:-1]))):
        try:
            check(broken_direct, broken_probe)
        except ValueError:
            pass
        else:
            raise AssertionError("workspace-probe negative stayed green")
    print("[fq-split-workspace-check:self-test] PASS five verdicts; "
          "stripe, marker and denominator negatives RED")


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
