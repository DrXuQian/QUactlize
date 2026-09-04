#!/usr/bin/env python3
"""Contract and result checker for the grouped TM8 epilogue post-fix gate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_m8_grouped_epilogue_postfix_box.sh"
sys.path.insert(0, str(ROOT / "tools"))

import fully_quantized_kpack_discovery_matrix as matrix  # noqa: E402


@dataclass(frozen=True)
class Arm:
    label: str
    family: str
    profile: str
    experts: int
    n: int
    total: int
    maximum: int
    active: int
    empty: int
    symbol: str
    algorithm: str
    config: str


ARMS = (
    Arm("tm8-p-e0-9-n64", "tm8", "e0-9", 2, 64, 9, 9, 1, 1,
        "fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_persistent",
        "GROUPED_PERSISTENT", "8x64x64_w8x16_s2"),
    Arm("tm8-p-e0-9-n3072", "tm8", "e0-9", 2, 3072, 9, 9, 1, 1,
        "fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_persistent",
        "GROUPED_PERSISTENT", "8x64x64_w8x16_s2"),
    Arm("tm8-np-tilem-boundary-n3072", "tm8", "tilem-boundary", 256,
        3072, 528, 129, 9, 247,
        "fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_nonpersistent",
        "GROUPED_NONPERSISTENT", "8x64x64_w8x16_s2"),
    Arm("tm16-p-e0-9-n64", "tm16", "e0-9", 2, 64, 9, 9, 1, 1,
        "fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_persistent",
        "GROUPED_PERSISTENT", "16x64x64_w16x16_s2"),
    Arm("tm16-p-e0-9-n3072", "tm16", "e0-9", 2, 3072, 9, 9, 1, 1,
        "fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_persistent",
        "GROUPED_PERSISTENT", "16x64x64_w16x16_s2"),
    Arm("tm16-np-tilem-boundary-n3072", "tm16", "tilem-boundary", 256,
        3072, 528, 129, 9, 247,
        "fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_nonpersistent",
        "GROUPED_NONPERSISTENT", "16x64x64_w16x16_s2"),
)

EXPECTED_PARENTS = {
    "tm8": (60, (
        "fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_nonpersistent",
        "fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_persistent",
    )),
    "tm16": (516, (
        "fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_nonpersistent",
        "fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_persistent",
    )),
}


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("\\\n", ""))


def authority_errors() -> list[str]:
    rows = list(matrix.grouped_rows(12))
    bad: list[str] = []
    for family, (begin, symbols) in EXPECTED_PARENTS.items():
        actual: list[str] = []
        for ordinal, (row, delivery_n, algorithm) in enumerate(
                rows[begin:begin + 2], start=begin):
            persistent = int(algorithm == "GROUPED_PERSISTENT")
            mode = "persistent" if persistent else "nonpersistent"
            actual.append(
                f"fqg_q12_l{matrix.layout_for(12)}_tm{row.tile_m}_"
                f"tn{row.tile_n}_tk{row.tactic_tile_k}_wm{row.warp_m}_"
                f"wn{row.warp_n}_s{row.stages}_ap0_dn{delivery_n}_{mode}")
            expected_algorithm = ("GROUPED_NONPERSISTENT" if ordinal == begin
                                  else "GROUPED_PERSISTENT")
            if algorithm != expected_algorithm:
                bad.append(f"{family} parent {ordinal} algorithm differs")
        if tuple(actual) != symbols:
            bad.append(f"{family} parent [{begin},{begin + 2}) identity differs")
    return bad


def audit_runner(text: str) -> list[str]:
    shell = compact(text)
    bad = authority_errors()
    tokens = (
        ("set-u-opipefail", 1),
        ("check_m8_grouped_epilogue_postfix.py", 2),
        ("--parent-begin60--parent-count2--per-unit2", 1),
        ("--parent-begin516--parent-count2--per-unit2", 1),
        ("PPU_BUILD_RESUME=0", 1),
        ("FQ_GROUPED_KPACK_QTYPE=12", 1),
        ("FQ_GROUPED_KPACK_WEIGHT_LAYOUT=1", 1),
        ("FQ_GROUPED_KPACK_PACKED_FORMAT=0", 1),
        ("build_shardtm8\"$RUN_DIR/generated/tm8\"\"$RUN_DIR/build/tm8\""
         "\"$RUN_DIR/results/build-tm8.log\"&tm8_pid=$!", 1),
        ("build_shardtm16\"$RUN_DIR/generated/tm16\"\"$RUN_DIR/build/tm16\""
         "\"$RUN_DIR/results/build-tm16.log\"&tm16_pid=$!", 1),
        ("wait\"$tm8_pid\"||tm8_build_rc=$?", 1),
        ("wait\"$tm16_pid\"||tm16_build_rc=$?", 1),
        ("--correctness-repeats=7", 1),
        ("--iterations=1", 1),
        ("--warmups=1", 1),
        ("--validate-run-dir\"$RUN_DIR\"", 1),
        ("FQ_M8_GROUPED_EPILOGUE_POSTFIXverdict=PASSarms=6repeats=7", 1),
    )
    for token, count in tokens:
        if shell.count(token) != count:
            bad.append(f"runner must contain exactly {count} {token!r}")

    arm_tokens = (
        "run_armtm8-p-e0-9-n64tm8\"$TM8_BIN\"\"$E0_ROWS\"264e0-9"
        "fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_persistent"
        "GROUPED_PERSISTENT",
        "run_armtm8-p-e0-9-n3072tm8\"$TM8_BIN\"\"$E0_ROWS\"23072e0-9"
        "fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_persistent"
        "GROUPED_PERSISTENT",
        "run_armtm8-np-tilem-boundary-n3072tm8\"$TM8_BIN\"\"$BOUNDARY_ROWS\""
        "2563072tilem-boundary"
        "fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_nonpersistent"
        "GROUPED_NONPERSISTENT",
        "run_armtm16-p-e0-9-n64tm16\"$TM16_BIN\"\"$E0_ROWS\"264e0-9"
        "fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_persistent"
        "GROUPED_PERSISTENT",
        "run_armtm16-p-e0-9-n3072tm16\"$TM16_BIN\"\"$E0_ROWS\"23072e0-9"
        "fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_persistent"
        "GROUPED_PERSISTENT",
        "run_armtm16-np-tilem-boundary-n3072tm16\"$TM16_BIN\""
        "\"$BOUNDARY_ROWS\"2563072tilem-boundary"
        "fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_nonpersistent"
        "GROUPED_NONPERSISTENT",
    )
    for token in arm_tokens:
        if shell.count(token) != 1:
            bad.append(f"runner exact arm differs: {token[:72]!r}")
    if len(re.findall(r"^\s*run_arm\s", text, flags=re.M)) != 6:
        bad.append("runner must invoke exactly six full-production arms")
    if "m8-c24" in text or "PREBUILT" in text or "PPU_BUILD_RESUME=1" in text:
        bad.append("runner must not reuse the pre-fix binary or build tree")
    return bad


def fields(line: str) -> dict[str, str]:
    return {token.split("=", 1)[0]: token.split("=", 1)[1]
            for token in line.strip().split()[1:] if "=" in token}


def validate_arm_text(arm: Arm, text: str) -> list[str]:
    bad: list[str] = []
    lines = text.splitlines()
    shards = [line for line in lines if line.startswith("FQ_GROUPED_KPACK_SHARD ")]
    cells = [line for line in lines if line.startswith("FQ_GROUPED_KPACK_CELL ")]
    completes = [line for line in lines if line.startswith("FQ_GROUPED_KPACK_COMPLETE ")]
    mismatches = [line for line in lines
                  if line.startswith("FQ_GROUPED_KPACK_MISMATCH_MAP ")]
    if len(shards) != 1:
        bad.append(f"{arm.label}: shard rows={len(shards)}/1")
    if not cells:
        bad.append(f"{arm.label}: no cell rows")
    if len(completes) != 1:
        bad.append(f"{arm.label}: complete rows={len(completes)}/1")
    if mismatches:
        bad.append(f"{arm.label}: emitted mismatch map")
    if len(shards) == 1:
        got = fields(shards[0])
        expected = {
            "q": "12", "layout": "1", "mapping_id": "0x51344b5034540001",
            "type_rows": "2", "selected_rows": "1", "router": "exact-rows-v1",
            "experts": str(arm.experts),
            "total_rows": str(arm.total), "max_rows": str(arm.maximum),
            "active": str(arm.active), "empty": str(arm.empty),
            "workload": arm.label, "router_profile": arm.profile,
            "iterations": "1", "warmups": "1", "correctness_repeats": "7",
            "roundtrip": "PASS", "metadata": "PACKED_UNITS",
        }
        for key, value in expected.items():
            if got.get(key) != value:
                bad.append(f"{arm.label}: shard {key}={got.get(key)!r}, want {value!r}")
    for index, line in enumerate(cells):
        got = fields(line)
        expected = {
            "q": "12", "layout": "1", "symbol": arm.symbol,
            "config": arm.config, "algorithm": arm.algorithm,
            "state": "MEASURED", "raw_bad": "0", "first_bad": str(2**64 - 1),
            "want": "0x0000", "got": "0x0000", "failure_repeat": "-1",
            "execution_ordinal": "0",
        }
        for key, value in expected.items():
            if got.get(key) != value:
                bad.append(
                    f"{arm.label}: cell[{index}] {key}={got.get(key)!r}, want {value!r}")
        if arm.algorithm == "GROUPED_NONPERSISTENT":
            for key, value in {"policy": "ordinary", "grid": "0",
                               "occupancy": "0", "capacity_b_mask": "0x0",
                               "balanced_b_mask": "0x0"}.items():
                if got.get(key) != value:
                    bad.append(f"{arm.label}: nonpersistent {key} differs")
        else:
            try:
                if int(got.get("grid", "0")) <= 0 or int(got.get("occupancy", "0")) <= 0:
                    bad.append(f"{arm.label}: persistent grid/occupancy is not positive")
            except ValueError:
                bad.append(f"{arm.label}: persistent grid/occupancy is not numeric")
        if got.get("samples") in (None, "[]"):
            bad.append(f"{arm.label}: measured cell has no timing sample")
    if len(completes) == 1:
        got = fields(completes[0])
        expected = {
            "q": "12", "status": "PASS", "rows": "1",
            "cells": str(len(cells)), "measured": str(len(cells)),
            "structural": "0", "correctness": "RAW_FP16",
            "timing": "AFTER_CORRECTNESS", "top_n": "NONE",
        }
        for key, value in expected.items():
            if got.get(key) != value:
                bad.append(f"{arm.label}: complete {key}={got.get(key)!r}, want {value!r}")
    return bad


def synthetic(arm: Arm) -> str:
    policy = "ordinary" if arm.algorithm == "GROUPED_NONPERSISTENT" else "capacity+balanced"
    grid = 0 if arm.algorithm == "GROUPED_NONPERSISTENT" else 72
    occupancy = 0 if arm.algorithm == "GROUPED_NONPERSISTENT" else 12
    return (
        "FQ_GROUPED_KPACK_SHARD q=12 layout=1 mapping_id=0x51344b5034540001 "
        f"type_rows=2 selected_rows=1 router=exact-rows-v1 tokens=4 topk=2 "
        f"experts={arm.experts} total_rows={arm.total} max_rows={arm.maximum} "
        f"active={arm.active} empty={arm.empty} workload={arm.label} "
        f"router_profile={arm.profile} rows_hash=0x1 iterations=1 warmups=1 "
        "correctness_repeats=7 schedule_seed=0x1 roundtrip=PASS metadata=PACKED_UNITS\n"
        f"FQ_GROUPED_KPACK_CELL q=12 layout=1 symbol={arm.symbol} "
        f"config={arm.config} algorithm={arm.algorithm} policy={policy} grid={grid} "
        f"occupancy={occupancy} capacity_b_mask=0x0 balanced_b_mask=0x0 "
        "state=MEASURED raw_bad=0 first_bad=18446744073709551615 want=0x0000 "
        "got=0x0000 failure_repeat=-1 median_us=1.0 min_us=1.0 max_us=1.0 "
        "execution_ordinal=0 samples=[1.0]\n"
        "FQ_GROUPED_KPACK_COMPLETE q=12 status=PASS rows=1 cells=1 measured=1 "
        "structural=0 correctness=RAW_FP16 timing=AFTER_CORRECTNESS top_n=NONE\n")


def validate_generated(run_dir: Path) -> list[str]:
    bad: list[str] = []
    for family, (begin, symbols) in EXPECTED_PARENTS.items():
        path = run_dir / "generated" / family / "manifest.json"
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            bad.append(f"{family}: cannot read generated manifest: {exc}")
            continue
        if doc.get("parent_range") != {
                "begin": begin, "end": begin + 2, "count": 2,
                "authority_count": len(matrix.grouped_rows(12))}:
            bad.append(f"{family}: generated parent range differs")
        parents = doc.get("grouped_parents") or []
        if tuple(row.get("symbol") for row in parents) != symbols:
            bad.append(f"{family}: generated symbols differ")
    return bad


def validate_run_dir(run_dir: Path) -> list[str]:
    bad = validate_generated(run_dir)
    arms_dir = run_dir / "results" / "arms"
    expected_logs = {f"{arm.label}.run.log" for arm in ARMS}
    actual_logs = {path.name for path in arms_dir.glob("*.run.log")}
    if actual_logs != expected_logs:
        bad.append(f"arm log set differs: {sorted(actual_logs)}")
    for arm in ARMS:
        log = arms_dir / f"{arm.label}.run.log"
        rc_path = arms_dir / f"{arm.label}.rc"
        try:
            rc = rc_path.read_text().strip()
        except OSError as exc:
            bad.append(f"{arm.label}: cannot read rc: {exc}")
            continue
        if rc != "0":
            bad.append(f"{arm.label}: process rc={rc}")
        try:
            bad.extend(validate_arm_text(arm, log.read_text()))
        except OSError as exc:
            bad.append(f"{arm.label}: cannot read log: {exc}")

    try:
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as exc:
        bad.append(f"cannot resolve source HEAD: {exc}")
        source = ""
    for family in EXPECTED_PARENTS:
        authority = run_dir / "build" / family / ".quactlize-source-head"
        binary = (run_dir / "build" / family / "ppu_targets" /
                  "test_fully_quantized_grouped_kpack_discovery")
        try:
            if authority.read_text().strip() != source:
                bad.append(f"{family}: build source authority differs")
        except OSError as exc:
            bad.append(f"{family}: cannot read build source authority: {exc}")
        if not binary.is_file():
            bad.append(f"{family}: built binary is missing")
    return bad


def self_test() -> list[str]:
    bad: list[str] = []
    for arm in ARMS:
        clean = synthetic(arm)
        if errors := validate_arm_text(arm, clean):
            bad.append(f"synthetic clean {arm.label} rejected: {errors}")
    arm = ARMS[0]
    clean = synthetic(arm)
    plants = (
        ("symbol", arm.symbol, arm.symbol + "_wrong"),
        ("algorithm", arm.algorithm, "GROUPED_NONPERSISTENT"),
        ("state", "state=MEASURED", "state=RAW_FP16_MISMATCH"),
        ("raw denominator", "raw_bad=0", "raw_bad=1"),
        ("failure repeat", "failure_repeat=-1", "failure_repeat=0"),
        ("selected row", "selected_rows=1", "selected_rows=2"),
        ("workload", f"workload={arm.label}", "workload=wrong"),
        ("complete", "FQ_GROUPED_KPACK_COMPLETE", "FQ_GROUPED_KPACK_MISSING"),
    )
    for label, old, new in plants:
        if clean.count(old) != 1:
            bad.append(f"cannot plant {label}")
        elif not validate_arm_text(arm, clean.replace(old, new, 1)):
            bad.append(f"checker accepted planted {label}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-run-dir", type=Path)
    args = parser.parse_args()
    if args.validate_run_dir is not None:
        bad = validate_run_dir(args.validate_run_dir.resolve())
        if bad:
            print("[m8-grouped-epilogue-postfix] FAIL: " + "; ".join(bad))
            return 1
        print("[m8-grouped-epilogue-postfix] validated six exact production arms")
        return 0

    runner = RUNNER.read_text()
    bad = audit_runner(runner) + self_test()
    if bad:
        print("[m8-grouped-epilogue-postfix] FAIL: " + "; ".join(bad))
        return 1

    source_plants = (
        ("TM8 parent range", "--parent-begin 60 --parent-count 2 --per-unit 2",
         "--parent-begin 59 --parent-count 2 --per-unit 2"),
        ("TM16 parent range", "--parent-begin 516 --parent-count 2 --per-unit 2",
         "--parent-begin 515 --parent-count 2 --per-unit 2"),
        ("current build", "PPU_BUILD_RESUME=0", "PPU_BUILD_RESUME=1"),
        ("final verdict", "FQ_M8_GROUPED_EPILOGUE_POSTFIX verdict=PASS arms=6 repeats=7",
         "FQ_M8_GROUPED_EPILOGUE_POSTFIX verdict=UNPROVEN"),
    )
    for label, old, new in source_plants:
        if runner.count(old) != 1:
            print(f"[m8-grouped-epilogue-postfix] FAIL: cannot plant {label}")
            return 1
        if not audit_runner(runner.replace(old, new, 1)):
            print(f"[m8-grouped-epilogue-postfix] FAIL: accepted planted {label}")
            return 1

    print("[m8-grouped-epilogue-postfix] PASS authority=parents-60/61+516/517 "
          "arms=3-TM8+3-TM16 repeats=7 exact-log-contract; "
          "twelve negatives RED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
