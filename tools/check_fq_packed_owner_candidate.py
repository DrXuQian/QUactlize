#!/usr/bin/env python3
"""Adjudicate legacy modulo publishers versus packed-metadata owner-only copy."""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import sys

from check_fq_split_workspace_probe import (
    AP0,
    AP1,
    EXPECTED,
    exact_cells,
    nonnegative,
    records,
)


VALID_SPLITS = (2, 4)


def probe_groups(text: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    cells = records(text, "FQ_TC_CELL ")
    exact_cells(cells)
    expected = {(symbol, str(split)) for symbol in EXPECTED
                for split in VALID_SPLITS}
    completed = {(row.get("symbol"), row.get("S")) for row in cells
                 if row.get("state") == "WORKSPACE_PROBE_COMPLETE"}
    if completed != expected:
        raise ValueError("workspace completion denominator differs")
    if "split_workspace_probe=1" not in text:
        raise ValueError("workspace probe marker missing")
    oracle = records(text, "FQ_WORKSPACE_ORACLE ")
    if len(oracle) != 1 or oracle[0].get("exact") != "1":
        raise ValueError("independent partial oracle marker differs")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in records(text, "FQ_WORKSPACE_PROBE "):
        key = (row.get("symbol", ""), row.get("S", ""))
        if key not in expected:
            raise ValueError(f"unexpected workspace key: {key}")
        grouped.setdefault(key, []).append(row)
    if set(grouped) != expected:
        raise ValueError("workspace sample denominator differs")
    counts = {len(samples) for samples in grouped.values()}
    if len(counts) != 1 or next(iter(counts)) < 8:
        raise ValueError("workspace repeat denominator differs or is too small")
    for key, samples in grouped.items():
        repeats = sorted(nonnegative(row, "repeat") for row in samples)
        if repeats != list(range(len(samples))):
            raise ValueError(f"workspace repeat identity differs: {key}")
    return grouped


def direct_summary(text: str) -> tuple[int, int]:
    cells = records(text, "FQ_TC_CELL ")
    exact_cells(cells)
    failures = 0
    valid = 0
    for row in cells:
        split = int(row["S"])
        state = row.get("state")
        bad = nonnegative(row, "raw_bad")
        if split == 8:
            if state != "SPLIT_PARTITION" or bad:
                raise ValueError(f"S8 partition control changed: {row}")
            continue
        valid += 1
        if state == "MEASURED" and bad == 0:
            continue
        if state == "RAW_FP16_MISMATCH" and bad > 0:
            failures += 1
            first = nonnegative(row, "first_bad")
            if first >= 1024 or first % 32:
                raise ValueError(f"direct mismatch lost stripe origin: {row}")
            continue
        raise ValueError(f"unexpected direct state: {row}")
    return valid, failures


def probe_summary(groups: dict[tuple[str, str], list[dict[str, str]]],
                  require_legacy_boundary: bool) -> dict[str, int]:
    result = {
        "samples": 0,
        "bad_samples": 0,
        "partial_bad": 0,
        "canary": 0,
        "host_bad": 0,
        "sync_bad": 0,
        "observed_bad": 0,
    }
    for samples in groups.values():
        for row in samples:
            result["samples"] += 1
            partial_bad = nonnegative(row, "partial_value_raw_bad")
            result["partial_bad"] += partial_bad
            result["canary"] += nonnegative(row, "canary_words")
            result["host_bad"] += nonnegative(row, "host_reduce_raw_bad")
            result["sync_bad"] += nonnegative(row, "sync_only_raw_bad")
            result["observed_bad"] += nonnegative(
                row, "observed_reducer_raw_bad")
            if partial_bad:
                result["bad_samples"] += 1
                index = nonnegative(row, "partial_first_bad_index")
                if require_legacy_boundary and index % 64 != 32:
                    raise ValueError(
                        f"legacy partial mismatch left local-N=32 boundary: {row}")
    return result


def check(legacy_direct: str, legacy_probe: str,
          candidate_direct: str, candidate_probe: str) -> str:
    legacy_groups = probe_groups(legacy_probe)
    candidate_groups = probe_groups(candidate_probe)
    legacy = probe_summary(legacy_groups, True)
    candidate = probe_summary(candidate_groups, False)
    legacy_valid, legacy_direct_failures = direct_summary(legacy_direct)
    candidate_valid, candidate_direct_failures = direct_summary(candidate_direct)
    if legacy_valid != 6 or candidate_valid != 6:
        raise ValueError("direct valid-cell denominator differs")
    if not legacy["partial_bad"] or not legacy["bad_samples"]:
        verdict = "UNADJUDICATED_LEGACY_DID_NOT_REPRODUCE"
    elif candidate["canary"]:
        verdict = "OWNER_ONLY_UNWRITTEN_PARTIALS"
    elif candidate["partial_bad"]:
        verdict = "OWNER_ONLY_REFUTED"
    elif candidate["host_bad"] or candidate["observed_bad"]:
        verdict = "OWNER_ONLY_PARTIALS_EXACT_REDUCER_SEAM_REMAINS"
    elif candidate_direct_failures:
        verdict = "OWNER_RACE_CLOSED_DIRECT_GAP_REMAINS"
    else:
        verdict = "OWNER_RACE_CLOSED_ALL_EXACT"
    print(
        "FQ_PACKED_OWNER_ARM variant=legacy-modulo-all "
        f"samples={legacy['samples']} bad_samples={legacy['bad_samples']} "
        f"partial_value_raw_bad={legacy['partial_bad']} "
        f"canary_words={legacy['canary']} "
        f"direct_failures={legacy_direct_failures}")
    print(
        "FQ_PACKED_OWNER_ARM variant=owner-only "
        f"samples={candidate['samples']} bad_samples={candidate['bad_samples']} "
        f"partial_value_raw_bad={candidate['partial_bad']} "
        f"canary_words={candidate['canary']} "
        f"host_reduce_raw_bad={candidate['host_bad']} "
        f"sync_only_raw_bad={candidate['sync_bad']} "
        f"observed_reducer_raw_bad={candidate['observed_bad']} "
        f"direct_failures={candidate_direct_failures}")
    print(
        "FQ_PACKED_OWNER_VERDICT "
        f"verdict={verdict} legacy_boundary=LOCAL_N32/PASS "
        "changed_semantic=PHYSICAL_PACKED_METADATA_PUBLISHERS_ONLY")
    return verdict


def self_test() -> None:
    def cell(symbol: str, split: int, fail: bool = False,
             probe: bool = False) -> str:
        if split == 8:
            state = "SPLIT_PARTITION"
        elif probe and split in VALID_SPLITS:
            state = "WORKSPACE_PROBE_COMPLETE"
        elif fail:
            state = "RAW_FP16_MISMATCH"
        else:
            state = "MEASURED"
        return (f"FQ_TC_CELL symbol={symbol} S={split} state={state} "
                f"raw_bad={32 if fail else 0} "
                f"first_bad={32 if fail else 18446744073709551615}")

    def logs(partial_bad: bool, direct_bad: bool = False,
             wrong_boundary: bool = False) -> tuple[str, str]:
        direct: list[str] = []
        probe: list[str] = [
            "FQ_SHARD split_workspace_probe=1",
            "FQ_WORKSPACE_ORACLE exact=1 S1=0x1 S2=0x2 S4=0x4 S8=0x8",
        ]
        for symbol in sorted(EXPECTED):
            for split in (1, 2, 4, 8):
                fail = direct_bad and symbol == AP0 and split == 2
                direct.append(cell(symbol, split, fail=fail))
                probe.append(cell(symbol, split, probe=True))
            for split in VALID_SPLITS:
                for repeat in range(8):
                    bad = partial_bad and symbol == AP0 and split == 2 and repeat == 0
                    index = 33 if wrong_boundary and bad else 32
                    probe.append(
                        f"FQ_WORKSPACE_PROBE symbol={symbol} S={split} "
                        f"repeat={repeat} canary_words=0 "
                        f"partial_value_raw_bad={32 if bad else 0} "
                        f"partial_first_bad_index={index if bad else 18446744073709551615} "
                        "host_reduce_raw_bad=0 sync_only_raw_bad=0 "
                        "observed_reducer_raw_bad=0")
        return "\n".join(direct), "\n".join(probe)

    legacy_direct, legacy_probe = logs(True, True)
    clean_direct, clean_probe = logs(False)
    with contextlib.redirect_stdout(io.StringIO()):
        assert check(legacy_direct, legacy_probe,
                     clean_direct, clean_probe) == "OWNER_RACE_CLOSED_ALL_EXACT"
        gap_direct, gap_probe = logs(False, True)
        assert check(legacy_direct, legacy_probe,
                     gap_direct, gap_probe) == "OWNER_RACE_CLOSED_DIRECT_GAP_REMAINS"
        bad_direct, bad_probe = logs(True, True)
        assert check(legacy_direct, legacy_probe,
                     bad_direct, bad_probe) == "OWNER_ONLY_REFUTED"
    _, wrong_probe = logs(True, True, True)
    try:
        check(legacy_direct, wrong_probe, clean_direct, clean_probe)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong local-N boundary stayed green")
    try:
        check(legacy_direct, "\n".join(legacy_probe.splitlines()[:-1]),
              clean_direct, clean_probe)
    except ValueError:
        pass
    else:
        raise AssertionError("missing sample stayed green")
    print("[fq-packed-owner-check:self-test] PASS causal, remaining-gap and "
          "refuted verdicts; local-N boundary and denominator negatives RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-direct", type=pathlib.Path)
    parser.add_argument("--legacy-probe", type=pathlib.Path)
    parser.add_argument("--candidate-direct", type=pathlib.Path)
    parser.add_argument("--candidate-probe", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            required = (args.legacy_direct, args.legacy_probe,
                        args.candidate_direct, args.candidate_probe)
            if any(path is None for path in required):
                parser.error("all four logs are required")
            check(*(path.read_text() for path in required))
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-packed-owner-check] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
