#!/usr/bin/env python3
"""Contract and result checker for the dense TM8 epilogue post-fix gate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_m8_dense_epilogue_postfix_box.sh"
sys.path.insert(0, str(ROOT / "tools"))

import fully_quantized_kpack_discovery_matrix as matrix  # noqa: E402
import gen_fully_quantized_kpack_discovery_units as generator  # noqa: E402


MAPPING_ID = "0x51344b5034540001"
SHAPES = ((1, 64, 512), (8, 64, 512), (9, 64, 512),
          (15, 64, 512), (16, 64, 512), (17, 64, 512))


@dataclass(frozen=True)
class Family:
    name: str
    ordinal: int
    symbol: str
    tm: int
    wm: int


FAMILIES = (
    Family("tm8", 4809,
           "fqk_tc_q12_l1_a0_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0_dn16",
           8, 8),
    Family("tm16", 5139,
           "fqk_tc_q12_l1_a0_tm16_tn64_tk256_wm16_wn16_s2_bc0_ap0_dn16",
           16, 16),
)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("\\\n", ""))


def authority_errors() -> list[str]:
    rows = list(matrix.provider_rows(12))
    bad: list[str] = []
    if len(rows) != 6102:
        bad.append(f"Q4 dense provider denominator is {len(rows)}/6102")
        return bad
    for family in FAMILIES:
        row, provider, delivery_n = rows[family.ordinal]
        actual = generator.symbol(12, row, provider, delivery_n)
        if actual != family.symbol:
            bad.append(f"{family.name} ordinal {family.ordinal} symbol differs: {actual}")
        expected = (family.tm, 64, 256, family.wm, 16, 2, 0, 16)
        got = (row.tile_m, row.tile_n, row.tactic_tile_k, row.warp_m,
               row.warp_n, row.stages, provider, delivery_n)
        if got != expected:
            bad.append(f"{family.name} ordinal {family.ordinal} axes differ: {got}")
    return bad


def audit_runner(text: str) -> list[str]:
    shell = compact(text)
    bad = authority_errors()
    tokens = (
        ("set-u-opipefail", 1),
        ("check_m8_dense_epilogue_postfix.py", 2),
        ("gen_fully_quantized_kpack_discovery_units.py", 2),
        ("--parent-begin4809--parent-count1", 1),
        ("--parent-begin5139--parent-count1", 1),
        ("--per-unit1", 2),
        ("PPU_BUILD_RESUME=0", 1),
        ("FQ_SWEEP_QTYPE=12", 1),
        ("FQ_SWEEP_ARTIFACT_TK=0", 1),
        ("FQ_SWEEP_BCHUNK=0", 1),
        ("FQ_SWEEP_PACKED_FORMAT=0", 1),
        ("FQ_SWEEP_WEIGHT_LAYOUT=1", 1),
        ("build_shardtm8\"$RUN_DIR/generated/tm8\"\"$RUN_DIR/build/tm8\""
         "\"$RUN_DIR/results/build-tm8.log\"&tm8_pid=$!", 1),
        ("build_shardtm16\"$RUN_DIR/generated/tm16\"\"$RUN_DIR/build/tm16\""
         "\"$RUN_DIR/results/build-tm16.log\"&tm16_pid=$!", 1),
        ("wait\"$tm8_pid\"||tm8_build_rc=$?", 1),
        ("wait\"$tm16_pid\"||tm16_build_rc=$?", 1),
        ("--shape=1x64x512--shape=8x64x512--shape=9x64x512"
         "--shape=15x64x512--shape=16x64x512--shape=17x64x512", 1),
        ("--iterations=1--correctness-repeats=7--only-split=1", 1),
        ("--tm8-max-m=17--bc-mode=skip", 1),
        ("run_familytm8\"$TM8_BIN\"", 1),
        ("run_familytm16\"$TM16_BIN\"", 1),
        ("--validate-run-dir\"$RUN_DIR\"", 1),
        ("FQ_M8_DENSE_EPILOGUE_POSTFIXverdict=PASSfamilies=2shapes=6"
         "cells=12repeats=7split=S1", 1),
    )
    for token, count in tokens:
        if shell.count(token) != count:
            bad.append(f"runner must contain exactly {count} {token!r}")
    forbidden = ("--only-split=0", "--only-split=2", "--only-split=4",
                 "--only-split=8", "PPU_BUILD_RESUME=1", "PREBUILT")
    for token in forbidden:
        if token in text:
            bad.append(f"runner contains forbidden broad/reused path {token!r}")
    return bad


def fields(line: str) -> dict[str, str]:
    return {token.split("=", 1)[0]: token.split("=", 1)[1]
            for token in line.strip().split()[1:] if "=" in token}


def shape_text(shape: tuple[int, int, int]) -> str:
    return "x".join(map(str, shape))


def validate_family_text(family: Family, text: str) -> list[str]:
    bad: list[str] = []
    lines = text.splitlines()
    fixtures = [line for line in lines if line.startswith("FQ_KPACK4_FIXTURE ")]
    shards = [line for line in lines if line.startswith("FQ_SHARD ")]
    cells = [line for line in lines if line.startswith("FQ_TC_CELL ")]
    done = [line for line in lines if line.startswith("FQ_SHAPE_DONE ")]
    if len(fixtures) != 12:
        bad.append(f"{family.name}: fixture rows={len(fixtures)}/12")
    if len(shards) != 6:
        bad.append(f"{family.name}: shard rows={len(shards)}/6")
    if len(cells) != 6:
        bad.append(f"{family.name}: cell rows={len(cells)}/6")
    if len(done) != 6:
        bad.append(f"{family.name}: done rows={len(done)}/6")
    if bad:
        return bad

    fixture_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for line in fixtures:
        got = fields(line)
        key = (got.get("shape", ""), got.get("phase", ""))
        if key in fixture_by_key:
            bad.append(f"{family.name}: duplicate fixture {key}")
        fixture_by_key[key] = got
    shard_by_shape = {fields(line).get("shape", ""): fields(line)
                      for line in shards}
    cell_by_shape = {fields(line).get("shape", ""): fields(line)
                     for line in cells}
    done_by_shape = {fields(line).get("shape", ""): fields(line)
                     for line in done}

    for shape in SHAPES:
        name = shape_text(shape)
        prepare = fixture_by_key.get((name, "prepare"), {})
        recover = fixture_by_key.get((name, "recover"), {})
        for key, value in {
                "q": "12", "shape": name, "version": "2", "layout": "1",
                "bits": "4", "high_bits": "0", "artifact_tile_k": "0",
                "transport_tile_k": "64", "group_size": "32", "reserved": "0",
                "mapping_id": MAPPING_ID, "direct_rc": "0", "abi_rc": "0",
                "direct_equal": "1"}.items():
            if prepare.get(key) != value:
                bad.append(f"{family.name}/{name}: prepare {key} differs")
        for key, value in {
                "q": "12", "shape": name, "mapping_id": MAPPING_ID,
                "direct_rc": "0", "abi_rc": "0", "direct_equal": "1",
                "native_equal": "1"}.items():
            if recover.get(key) != value:
                bad.append(f"{family.name}/{name}: recover {key} differs")

        shard = shard_by_shape.get(name, {})
        for key, value in {
                "q": "12", "A": "0", "bchunk": "0", "shape": name,
                "weight_layout": "1", "weight_mapping_id": MAPPING_ID,
                "typed_rows": "1", "selected_rows": "1", "only_split": "1",
                "bc_mode": "skip", "iterations": "1",
                "correctness_repeats": "7"}.items():
            if shard.get(key) != value:
                bad.append(f"{family.name}/{name}: shard {key} differs")

        cell = cell_by_shape.get(name, {})
        expected_cell = {
            "q": "12", "A": "0", "bchunk": "0", "shape": name,
            "symbol": family.symbol, "tm": str(family.tm), "tn": "64",
            "tk": "256", "wm": str(family.wm), "wn": "16", "stages": "2",
            "provider": "standard-aiu", "S": "1", "scope": "FULL_OUTPUT",
            "resolved_delivery_n": "16", "provider_capacity_rows": "0",
            "scalezero_fused": "1", "state": "MEASURED", "raw_bad": "0",
            "reducer_untimed": "0", "failure_step": "NONE",
            "failure_cutlass_status": "0", "failure_runtime_status": "0",
            "failure_repeat": "-1", "first_bad": str(2**64 - 1),
            "first_want": "0x0000", "first_got": "0x0000", "partial_bytes": "0",
        }
        for key, value in expected_cell.items():
            if cell.get(key) != value:
                bad.append(f"{family.name}/{name}: cell {key}={cell.get(key)!r}, want {value!r}")
        try:
            if float(cell.get("us", "0")) <= 0:
                bad.append(f"{family.name}/{name}: timing is not positive")
            if int(cell.get("shipping_smem", "0")) <= 0 or \
                    int(cell.get("split_smem", "0")) <= 0:
                bad.append(f"{family.name}/{name}: shared storage witness missing")
        except ValueError:
            bad.append(f"{family.name}/{name}: numeric witness malformed")
        if cell.get("samples") in (None, "[]"):
            bad.append(f"{family.name}/{name}: timing sample missing")

        final = done_by_shape.get(name, {})
        for key, value in {
                "q": "12", "A": "0", "bchunk": "0", "shape": name,
                "weight_layout": "1", "weight_mapping_id": MAPPING_ID,
                "typed_rows": "1", "selected_rows": "1", "only_split": "1",
                "bc_mode": "skip", "iterations": "1", "status": "PASS"}.items():
            if final.get(key) != value:
                bad.append(f"{family.name}/{name}: done {key} differs")
    return bad


def synthetic(family: Family) -> str:
    out: list[str] = []
    for shape in SHAPES:
        name = shape_text(shape)
        out += [
            f"FQ_KPACK4_FIXTURE phase=prepare q=12 shape={name} version=2 layout=1 "
            "bits=4 high_bits=0 artifact_tile_k=0 transport_tile_k=64 group_size=32 "
            f"reserved=0 mapping_id={MAPPING_ID} direct_rc=0 abi_rc=0 direct_equal=1",
            f"FQ_KPACK4_FIXTURE phase=recover q=12 shape={name} mapping_id={MAPPING_ID} "
            "direct_rc=0 abi_rc=0 direct_equal=1 native_equal=1",
            f"FQ_SHARD q=12 A=0 bchunk=0 shape={name} weight_layout=1 "
            f"weight_mapping_id={MAPPING_ID} weight_delivery_n=0 typed_rows=1 "
            "selected_rows=1 only_split=1 bc_mode=skip iterations=1 "
            "correctness_repeats=7 schedule_seed=0x1",
            f"FQ_TC_CELL q=12 A=0 bchunk=0 shape={name} symbol={family.symbol} "
            f"tm={family.tm} tn=64 tk=256 wm={family.wm} wn=16 stages=2 "
            "provider=standard-aiu S=1 scope=FULL_OUTPUT resolved_delivery_n=16 "
            "provider_capacity_rows=0 scalezero_fused=1 state=MEASURED us=1.0 "
            "raw_bad=0 reducer_untimed=0 failure_step=NONE failure_cutlass_status=0 "
            "failure_runtime_status=0 failure_repeat=-1 first_bad=18446744073709551615 "
            "first_want=0x0000 first_got=0x0000 shipping_smem=1 split_smem=1 "
            "partial_bytes=0 samples=[1.0]",
            f"FQ_SHAPE_DONE q=12 A=0 bchunk=0 shape={name} weight_layout=1 "
            f"weight_mapping_id={MAPPING_ID} weight_delivery_n=0 typed_rows=1 "
            "selected_rows=1 only_split=1 bc_mode=skip iterations=1 status=PASS",
        ]
    return "\n".join(out) + "\n"


def validate_generated(run_dir: Path) -> list[str]:
    bad: list[str] = []
    for family in FAMILIES:
        path = run_dir / "generated" / family.name / "manifest.json"
        try:
            document = json.loads(path.read_text())
            generator.validate_manifest(document)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            bad.append(f"{family.name}: invalid generated manifest: {error}")
            continue
        rows = document.get("dense_tc_parents") or []
        expected_range = {"begin": family.ordinal, "end": family.ordinal + 1,
                          "count": 1, "authority_count": 6102}
        if document.get("parent_range") != expected_range:
            bad.append(f"{family.name}: parent range differs")
        if len(rows) != 1 or rows[0].get("symbol") != family.symbol:
            bad.append(f"{family.name}: generated executable symbol differs")
    return bad


def validate_run_dir(run_dir: Path) -> list[str]:
    bad = validate_generated(run_dir)
    try:
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True).strip()
    except subprocess.CalledProcessError as error:
        bad.append(f"cannot resolve source HEAD: {error}")
        source = ""
    for family in FAMILIES:
        rc_path = run_dir / "results" / f"{family.name}.rc"
        log_path = run_dir / "results" / f"{family.name}.run.log"
        authority = run_dir / "build" / family.name / ".quactlize-source-head"
        binary = (run_dir / "build" / family.name / "ppu_targets" /
                  "test_fully_quantized_internal_sweep")
        try:
            if authority.read_text().strip() != source:
                bad.append(f"{family.name}: build source authority differs")
        except OSError as error:
            bad.append(f"{family.name}: cannot read build authority: {error}")
        if not binary.is_file():
            bad.append(f"{family.name}: built binary is missing")
        try:
            rc = rc_path.read_text().strip()
            text = log_path.read_text()
        except OSError as error:
            bad.append(f"{family.name}: missing result: {error}")
            continue
        if rc != "0":
            bad.append(f"{family.name}: runner rc={rc!r}")
        bad += validate_family_text(family, text)
    return bad


def self_test() -> None:
    bad = audit_runner(RUNNER.read_text())
    if bad:
        raise AssertionError("; ".join(bad))
    for family in FAMILIES:
        good = synthetic(family)
        if validate_family_text(family, good):
            raise AssertionError(f"{family.name} synthetic positive failed")
        plants = (
            good.replace("correctness_repeats=7", "correctness_repeats=1", 1),
            good.replace("only_split=1", "only_split=4", 1),
            good.replace("state=MEASURED", "state=RAW_FP16_MISMATCH", 1),
            good.replace("raw_bad=0", "raw_bad=64", 1),
            good.replace("resolved_delivery_n=16", "resolved_delivery_n=64", 1),
            good.replace("scalezero_fused=1", "scalezero_fused=0", 1),
            good.replace("shape=9x64x512", "shape=7x64x512", 1),
            good.replace("status=PASS", "status=FAIL", 1),
            good + next(line for line in good.splitlines()
                        if line.startswith("FQ_TC_CELL ")) + "\n",
        )
        for planted in plants:
            if not validate_family_text(family, planted):
                raise AssertionError(f"{family.name} result plant stayed green")
    runner = RUNNER.read_text()
    runner_plants = (
        runner.replace("--tm8-max-m=17", "--tm8-max-m=8", 1),
        runner.replace("--only-split=1", "--only-split=0", 1),
        runner.replace("--parent-begin 4809", "--parent-begin 4810", 1),
        runner.replace("PPU_BUILD_RESUME=0", "PPU_BUILD_RESUME=1", 1),
    )
    for planted in runner_plants:
        if not audit_runner(planted):
            raise AssertionError("runner plant stayed green")
    print("[m8-dense-epilogue-postfix:self-test] PASS canonical parents=4809/5139 "
          "fresh two-binary S1-only M=1/8/9/15/16/17; 22 plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-run-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.validate_run_dir is None:
            self_test()
        else:
            bad = validate_run_dir(args.validate_run_dir.resolve())
            if bad:
                raise ValueError("; ".join(bad))
            print("[m8-dense-epilogue-postfix] PASS families=2 shapes=6 "
                  "cells=12 repeats=7 split=S1 layout=1 delivery=16")
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[m8-dense-epilogue-postfix] FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
