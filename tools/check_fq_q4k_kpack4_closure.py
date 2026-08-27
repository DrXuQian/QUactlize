#!/usr/bin/env python3
"""Fail-closed validator for Q4_K K-pack4 S=1 and Split-K closures."""

from __future__ import annotations

import argparse
import pathlib
import shlex
import sys

from select_fq_q4k_kpack4_closure import MAPPING_ID, SYMBOL


SHAPE = "1x1024x5120"


def fields(line: str, prefix: str) -> dict[str, str]:
    out = {}
    for token in shlex.split(line.removeprefix(prefix)):
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def one_line(text: str, prefix: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError(f"{prefix.strip()} denominator is {len(lines)}, expected 1")
    return fields(lines[0], prefix)


def check(text: str) -> None:
    fixture_lines = [fields(line, "FQ_KPACK4_FIXTURE ")
                     for line in text.splitlines()
                     if line.startswith("FQ_KPACK4_FIXTURE ")]
    fixture = {row.get("phase"): row for row in fixture_lines}
    if len(fixture_lines) != 2 or set(fixture) != {"prepare", "recover"}:
        raise ValueError("fixture prepare/recover denominator is not exactly two")
    prepare = fixture["prepare"]
    required_prepare = {
        "q": "12", "shape": SHAPE, "version": "2", "layout": "1",
        "bits": "4", "high_bits": "0", "artifact_tile_k": "0",
        "transport_tile_k": "64", "group_size": "32", "reserved": "0",
        "mapping_id": MAPPING_ID, "direct_rc": "0", "abi_rc": "0",
        "direct_equal": "1",
    }
    if any(prepare.get(key) != value
           for key, value in required_prepare.items()):
        raise ValueError(f"fixture prepare contract failed: {prepare}")
    recover = fixture["recover"]
    required_recover = {
        "q": "12", "shape": SHAPE, "mapping_id": MAPPING_ID,
        "direct_rc": "0", "abi_rc": "0", "direct_equal": "1",
        "native_equal": "1",
    }
    if any(recover.get(key) != value
           for key, value in required_recover.items()):
        raise ValueError(f"fixture recover contract failed: {recover}")

    shard = one_line(text, "FQ_SHARD ")
    required_shard = {
        "q": "12", "A": "0", "bchunk": "0", "shape": SHAPE,
        "weight_layout": "1", "weight_mapping_id": MAPPING_ID,
        "typed_rows": "1", "selected_rows": "1", "only_split": "1",
        "bc_mode": "skip",
    }
    if any(shard.get(key) != value for key, value in required_shard.items()):
        raise ValueError(f"shard identity mismatch: {shard}")

    cell = one_line(text, "FQ_TC_CELL ")
    required_cell = {
        "q": "12", "A": "0", "bchunk": "0", "shape": SHAPE,
        "symbol": SYMBOL, "tm": "8", "tn": "64", "tk": "256",
        "wm": "8", "wn": "16", "stages": "2",
        "provider": "standard-aiu", "provider_capacity_rows": "0",
        "S": "1", "scope": "FULL_OUTPUT", "state": "MEASURED",
        "raw_bad": "0", "reducer_untimed": "0",
        "failure_step": "NONE", "failure_repeat": "-1",
        "partial_bytes": "0",
    }
    if any(cell.get(key) != value for key, value in required_cell.items()):
        raise ValueError(f"S1 row did not close raw-bit exact: {cell}")
    if float(cell.get("us", "0")) <= 0:
        raise ValueError("S1 timing is not positive")
    if int(cell.get("shipping_smem", "0")) <= 0 or \
            int(cell.get("split_smem", "0")) <= 0:
        raise ValueError("compiled shared-storage witnesses are missing")
    samples = cell.get("samples", "")
    if not (samples.startswith("[") and samples.endswith("]") and
            samples != "[]"):
        raise ValueError("S1 measured row has no samples")

    done = one_line(text, "FQ_SHAPE_DONE ")
    required_done = {
        "q": "12", "A": "0", "bchunk": "0", "shape": SHAPE,
        "weight_layout": "1", "weight_mapping_id": MAPPING_ID,
        "typed_rows": "1", "selected_rows": "1", "only_split": "1",
        "bc_mode": "skip", "status": "PASS",
    }
    if any(done.get(key) != value for key, value in required_done.items()):
        raise ValueError(f"shape-level closure is not exact: {done}")
    print("[fq-q4k-kpack4-check] PASS layout=1 "
          f"mapping={MAPPING_ID} shape={SHAPE} tactics=1 cells=1 "
          "S1=RAW-BIT/PASS")


def check_all_splits(text: str) -> None:
    lines = text.splitlines()
    cells = [fields(line, "FQ_TC_CELL ") for line in lines
             if line.startswith("FQ_TC_CELL ")]
    by_split = {cell.get("S"): cell for cell in cells}
    if len(cells) != 4 or set(by_split) != {"1", "2", "4", "8"}:
        raise ValueError(
            f"all-split denominator changed: rows={len(cells)} "
            f"splits={sorted(by_split)}")

    # Reuse the exact S1 adjudicator rather than restating its contract.  The
    # synthetic log changes only the requested split marker and retains the
    # real fixture, cell and shape records byte-for-byte.
    s1_lines = [line for line in lines
                if line.startswith("FQ_KPACK4_FIXTURE ")]
    shard_line = next((line for line in lines if line.startswith("FQ_SHARD ")), "")
    done_line = next((line for line in lines if line.startswith("FQ_SHAPE_DONE ")), "")
    s1_cell = next((line for line in lines
                    if line.startswith("FQ_TC_CELL ") and " S=1 " in line), "")
    if not shard_line or not done_line or not s1_cell:
        raise ValueError("all-split log is missing its S1/shard/done authority")
    check("\n".join([
        *s1_lines,
        shard_line.replace("only_split=0", "only_split=1"),
        s1_cell,
        done_line.replace("only_split=0", "only_split=1"),
    ]))

    common = {
        "q": "12", "A": "0", "bchunk": "0", "shape": SHAPE,
        "symbol": SYMBOL, "tm": "8", "tn": "64", "tk": "256",
        "wm": "8", "wn": "16", "stages": "2",
        "provider": "standard-aiu", "provider_capacity_rows": "0",
        "raw_bad": "0", "failure_step": "NONE", "failure_repeat": "-1",
    }
    for split, cell in by_split.items():
        if any(cell.get(key) != value for key, value in common.items()):
            raise ValueError(f"S{split} identity/raw-bit contract failed: {cell}")
        if int(cell.get("shipping_smem", "0")) <= 0 or \
                int(cell.get("split_smem", "0")) <= 0:
            raise ValueError(f"S{split} shared-storage witness is missing")

    for split, partial_bytes in (("2", "8192"), ("4", "16384")):
        cell = by_split[split]
        required = {
            "scope": "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS",
            "state": "MEASURED", "reducer_untimed": "1",
            "partial_bytes": partial_bytes,
        }
        if any(cell.get(key) != value for key, value in required.items()) or \
                float(cell.get("us", "0")) <= 0 or cell.get("samples") == "[]":
            raise ValueError(f"S{split} producer/reducer closure failed: {cell}")

    s8 = by_split["8"]
    required_s8 = {
        "scope": "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS",
        "state": "SPLIT_PARTITION", "reducer_untimed": "0",
        "partial_bytes": "0", "us": "0.000000000", "samples": "[]",
    }
    if any(s8.get(key) != value for key, value in required_s8.items()):
        raise ValueError(f"S8 must be structurally unavailable for 20 K tiles: {s8}")

    shard = fields(shard_line, "FQ_SHARD ")
    done = fields(done_line, "FQ_SHAPE_DONE ")
    if shard.get("only_split") != "0" or done.get("only_split") != "0" or \
            done.get("status") != "PASS":
        raise ValueError("all-split shard/done marker is not a complete PASS")
    print("[fq-q4k-kpack4-split-check] PASS layout=1 "
          f"mapping={MAPPING_ID} shape={SHAPE} tactics=1 cells=4 "
          "S1/S2/S4=RAW-BIT/PASS S8=STRUCTURAL-SPLIT-PARTITION")


def fixture() -> str:
    return "\n".join([
        f"FQ_KPACK4_FIXTURE phase=prepare q=12 shape={SHAPE} version=2 "
        "layout=1 bits=4 high_bits=0 artifact_tile_k=0 "
        "transport_tile_k=64 group_size=32 reserved=0 "
        f"mapping_id={MAPPING_ID} direct_rc=0 abi_rc=0 direct_equal=1",
        f"FQ_KPACK4_FIXTURE phase=recover q=12 shape={SHAPE} "
        f"mapping_id={MAPPING_ID} direct_rc=0 abi_rc=0 direct_equal=1 "
        "native_equal=1",
        f"FQ_SHARD q=12 A=0 bchunk=0 shape={SHAPE} weight_layout=1 "
        f"weight_mapping_id={MAPPING_ID} typed_rows=1 selected_rows=1 "
        "only_split=1 bc_mode=skip iterations=1 correctness_repeats=64",
        f"FQ_TC_CELL q=12 A=0 bchunk=0 shape={SHAPE} symbol={SYMBOL} "
        "tm=8 tn=64 tk=256 wm=8 wn=16 stages=2 provider=standard-aiu "
        "S=1 scope=FULL_OUTPUT provider_capacity_rows=0 state=MEASURED "
        "us=27.000000000 raw_bad=0 reducer_untimed=0 failure_step=NONE "
        "failure_repeat=-1 first_bad=18446744073709551615 first_want=0x0000 "
        "first_got=0x0000 shipping_smem=38912 split_smem=38912 "
        "partial_bytes=0 samples=[26.9,27.0,27.1]",
        f"FQ_SHAPE_DONE q=12 A=0 bchunk=0 shape={SHAPE} weight_layout=1 "
        f"weight_mapping_id={MAPPING_ID} typed_rows=1 selected_rows=1 "
        "only_split=1 bc_mode=skip iterations=3 status=PASS",
    ])


def all_splits_fixture() -> str:
    base = fixture().replace("only_split=1", "only_split=0")
    done = next(line for line in base.splitlines()
                if line.startswith("FQ_SHAPE_DONE "))
    prefix = base.replace("\n" + done, "")
    common = (
        f"FQ_TC_CELL q=12 A=0 bchunk=0 shape={SHAPE} symbol={SYMBOL} "
        "tm=8 tn=64 tk=256 wm=8 wn=16 stages=2 provider=standard-aiu "
        "scope=PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS "
        "provider_capacity_rows=0 raw_bad=0 failure_step=NONE "
        "failure_repeat=-1 first_bad=18446744073709551615 "
        "first_want=0x0000 first_got=0x0000 shipping_smem=38912 "
        "split_smem=38912 ")
    return "\n".join([
        prefix,
        common + "S=2 state=MEASURED us=17.000000000 reducer_untimed=1 "
        "partial_bytes=8192 samples=[16.9,17.0,17.1]",
        common + "S=4 state=MEASURED us=11.000000000 reducer_untimed=1 "
        "partial_bytes=16384 samples=[10.9,11.0,11.1]",
        common + "S=8 state=SPLIT_PARTITION us=0.000000000 "
        "reducer_untimed=0 partial_bytes=0 samples=[]",
        done,
    ])


def self_test() -> None:
    good = fixture()
    check(good)
    negatives = (
        good.replace("direct_rc=0", "direct_rc=24", 1),
        good.replace("native_equal=1", "native_equal=0", 1),
        good.replace("weight_layout=1", "weight_layout=0", 1),
        good.replace(MAPPING_ID, "0x51344b5034540000", 1),
        good.replace("raw_bad=0", "raw_bad=32", 1),
        good.replace("state=MEASURED", "state=RAW_FP16_MISMATCH", 1),
        good.replace("A=0", "A=64", 1),
        good.replace("status=PASS", "status=FAIL", 1),
        good + "\n" + next(line for line in good.splitlines()
                          if line.startswith("FQ_TC_CELL ")).replace("S=1", "S=2"),
    )
    for broken in negatives:
        try:
            check(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("K-pack4 closure RED control stayed green")
    all_good = all_splits_fixture()
    check_all_splits(all_good)
    all_negatives = (
        all_good.replace("S=4 state=MEASURED", "S=16 state=MEASURED", 1),
        all_good.replace("S=2 state=MEASURED", "S=2 state=RAW_FP16_MISMATCH", 1),
        all_good.replace("partial_bytes=16384", "partial_bytes=8192", 1),
        all_good.replace("S=8 state=SPLIT_PARTITION", "S=8 state=MEASURED", 1),
    )
    for broken in all_negatives:
        try:
            check_all_splits(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("K-pack4 Split-K RED control stayed green")
    print("[fq-q4k-kpack4-check:self-test] PASS exact denominator; "
          "fixture rc/roundtrip, layout, mapping, raw-bit, state, A, status "
          "and extra-cell plants RED; S2/S4 partial/reducer and S8 structural "
          "contracts RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    logs = parser.add_mutually_exclusive_group()
    logs.add_argument("--log", type=pathlib.Path)
    logs.add_argument("--all-splits-log", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if args.log is not None:
                check(args.log.read_text())
            elif args.all_splits_log is not None:
                check_all_splits(args.all_splits_log.read_text())
            else:
                parser.error("--log or --all-splits-log is required")
        return 0
    except (OSError, ValueError, AssertionError) as exc:
        print(f"[fq-q4k-kpack4-check] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
