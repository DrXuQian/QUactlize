#!/usr/bin/env python3
"""Static authority and device-result checker for the focused A09 blockers."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_a09_remaining_blockers_box.sh"
sys.path.insert(0, str(ROOT / "tools"))

import fully_quantized_kpack_discovery_matrix as fq_matrix  # noqa: E402
import gen_fully_quantized_kpack_discovery_units as fq_gen  # noqa: E402
import gen_scalefirst_internal_units as sf_gen  # noqa: E402
import scalefirst_internal_matrix as sf_matrix  # noqa: E402


MAPPING_ID = "0x51344b5034540001"
FQ_SHAPE = "7x8192x5120"
SF_SHAPE = "16x4096x2048"
FQ_SEED = "0x740e23673ecf70c7"
SF_SEED = "0x6a09e667f3bcc909"


@dataclass(frozen=True)
class Arm:
    name: str
    family: str
    ordinal: int
    symbol: str
    tm: int
    tn: int
    tk: int
    wm: int
    wn: int
    stages: int
    delivery_n: int
    cta_threads: int

    @property
    def config(self) -> str:
        return (f"{self.tm}x{self.tn}x{self.tk}_w{self.wm}x{self.wn}_"
                f"s{self.stages}_bc0_ap0_dn{self.delivery_n}")


ARMS = (
    Arm("fq-dense", "fq", 1417,
        "fqk_tc_q12_l1_a0_tm64_tn128_tk64_wm16_wn32_s4_bc0_ap0_dn32",
        64, 128, 64, 16, 32, 4, 32, 512),
    Arm("sf-subject", "sf", 4738,
        "sf_q12_a0_tm256_tn256_tk128_wm16_wn128_s2_bc0_ap0_dn32",
        256, 256, 128, 16, 128, 2, 32, 1024),
    Arm("sf-control", "sf", 4750,
        "sf_q12_a0_tm256_tn256_tk128_wm32_wn128_s2_bc0_ap0_dn32",
        256, 256, 128, 32, 128, 2, 32, 512),
)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("\\\n", ""))


def fields(line: str) -> dict[str, str]:
    return {item.split("=", 1)[0]: item.split("=", 1)[1]
            for item in line.strip().split()[1:] if "=" in item}


def exact_lines(text: str, prefix: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(prefix)]


def authority_errors() -> list[str]:
    bad: list[str] = []
    if len(fq_matrix.provider_rows(12)) != 6120:
        bad.append("Q4 FQ provider denominator differs from 6120")
        return bad
    fq_arm = ARMS[0]
    row, provider, delivery_n = fq_matrix.provider_rows(12)[fq_arm.ordinal]
    got = (row.tile_m, row.tile_n, row.tactic_tile_k, row.warp_m,
           row.warp_n, row.stages, provider, delivery_n)
    want = (fq_arm.tm, fq_arm.tn, fq_arm.tk, fq_arm.wm, fq_arm.wn,
            fq_arm.stages, 0, fq_arm.delivery_n)
    if got != want or fq_gen.symbol(12, row, provider, delivery_n) != fq_arm.symbol:
        bad.append(f"FQ parent 1417 authority differs: {got}")

    sf_rows = sf_matrix.kpack_dense_candidates(12)
    if len(sf_rows) != 6120:
        bad.append("Q4 ScaleFirst provider denominator differs from 6120")
        return bad
    for arm in ARMS[1:]:
        row, provider, delivery_n = sf_rows[arm.ordinal]
        got = (row.tile_m, row.tile_n, row.tactic_tile_k, row.warp_m,
               row.warp_n, row.stages, provider, delivery_n)
        want = (arm.tm, arm.tn, arm.tk, arm.wm, arm.wn, arm.stages,
                0, arm.delivery_n)
        symbol = sf_gen.symbol(12, 0, row, provider, delivery_n)
        if got != want or symbol != arm.symbol:
            bad.append(f"{arm.name} parent {arm.ordinal} authority differs: {got}")
        calculated_threads = (row.tile_m // row.warp_m) * \
            (row.tile_n // row.warp_n) * 32
        if calculated_threads != arm.cta_threads:
            bad.append(f"{arm.name} CTA threads differ: {calculated_threads}")
    return bad


def audit_runner(text: str) -> list[str]:
    shell = compact(text)
    bad = authority_errors()
    tokens = (
        ("set-u-opipefail", 1),
        ("check_a09_remaining_blockers.py", 4),
        ("a09_device_limits_probe.cpp", 2),
        ("maxThreadsPerBlock", 0),
        ("gen_fully_quantized_kpack_discovery_units.py", 1),
        ("--parent-begin1417--parent-count1--per-unit1", 1),
        ("gen_scalefirst_internal_units.py", 2),
        ("--artifact-tk0--bchunk0--weight-layout1", 2),
        ("--select-symbol\"$SF_SUBJECT_SYMBOL\"--per-unit1", 1),
        ("--select-symbol\"$SF_CONTROL_SYMBOL\"--per-unit1", 1),
        ("PPU_BUILD_DIR=\"$build\"PPU_BUILD_RESUME=0", 1),
        ("build_armfq-densefqtest_fully_quantized_internal_sweep&", 1),
        ("build_armsf-subjectsftest_scalefirst_internal_sweep&", 1),
        ("build_armsf-controlsftest_scalefirst_internal_sweep&", 1),
        ("--shape=7x8192x5120--iterations=1--correctness-repeats=1"
         "--only-split=1--bc-mode=skip--schedule-seed=\"$FQ_SEED\"", 1),
        ("--shape=16x4096x2048--algorithm=nonpersistent--fixture=exact"
         "--fixture-binding--iterations=1--correctness-repeats=1"
         "--schedule-seed=\"$SF_SEED\"", 1),
        ("run_fqrun_sfsf-subjectrun_sfsf-control", 1),
        ("A09_REMAINING_BLOCKERS_DIAGNOSTICverdict=DIAGNOSTIC_COMPLETE", 1),
    )
    for token, count in tokens:
        if shell.count(token) != count:
            bad.append(f"runner must contain exactly {count} {token!r}")
    forbidden = ("PPU_BUILD_RESUME=1", "--only-split=0", "--algorithm=all",
                 "FQ_SWEEP_WEIGHT_LAYOUT=0", "SCALEFIRST_SWEEP_WEIGHT_LAYOUT=0",
                 "run_fully_quantized_kpack_discovery_box.sh",
                 "run_scalefirst_kpack_discovery_box.sh")
    for token in forbidden:
        if token in text:
            bad.append(f"runner contains broad/reused/Xplane token {token!r}")
    return bad


def validate_generated(run_dir: Path) -> list[str]:
    bad: list[str] = []
    for arm in ARMS:
        path = run_dir / "generated" / arm.name / "manifest.json"
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            bad.append(f"{arm.name}: cannot read manifest: {error}")
            continue
        if arm.family == "fq":
            try:
                fq_gen.validate_manifest(document)
            except ValueError as error:
                bad.append(f"{arm.name}: FQ manifest rejected: {error}")
                continue
            rows = document.get("dense_tc_parents") or []
            if len(rows) != 1 or rows[0].get("symbol") != arm.symbol or \
                    rows[0].get("parent_ordinal") != arm.ordinal:
                bad.append(f"{arm.name}: exact executable row differs")
        else:
            rows = document.get("typed_rows") or []
            selection = document.get("selection") or {}
            if document.get("schema") != "quactlize.scalefirst.generated_shard.v3":
                bad.append(f"{arm.name}: ScaleFirst schema differs")
            if len(rows) != 1 or rows[0].get("symbol") != arm.symbol or \
                    rows[0].get("parent_id") != arm.ordinal:
                bad.append(f"{arm.name}: exact executable row differs")
            if selection != {
                    "mode": "exact-symbol", "begin": arm.ordinal,
                    "end": arm.ordinal + 1, "authority_typed_rows": 6120,
                    "compiled_rows": 1, "symbol": arm.symbol}:
                bad.append(f"{arm.name}: exact selection differs")
        if document.get("identity") != {
                "qtype": 12, "format": "Q4_K", "artifact_tile_k": 0,
                "bchunk": 0, "weight_layout": 1,
                "weight_layout_name": "q4-kpack4-transpose-v1",
                **({"metadata": "PACKED_UNITS_SCALE_AND_ZERO",
                    "packed_scale": 1} if arm.family == "fq" else {})}:
            bad.append(f"{arm.name}: layout identity differs")
        if document.get("parent_range") != {
                "begin": arm.ordinal, "end": arm.ordinal + 1,
                "count": 1, "authority_count": 6120}:
            bad.append(f"{arm.name}: parent range differs")
    return bad


def validate_device_limits(text: str) -> tuple[list[str], dict[str, object]]:
    bad: list[str] = []
    rows = exact_lines(text, "A09_DEVICE_LIMITS ")
    if len(rows) != 1:
        return [f"device limit rows={len(rows)}/1"], {}
    got = fields(rows[0])
    try:
        ordinal = int(got.get("ordinal", "-1"))
        compute_units = int(got.get("compute_units", "0"))
        maximum = int(got.get("max_threads_per_block", "0"))
        dims = [int(item) for item in got.get("max_threads_dim", "").split(",")]
    except ValueError:
        return ["device limit row has malformed integers"], {}
    if ordinal != 0:
        bad.append(f"selected device ordinal={ordinal}/0")
    if not got.get("name") or compute_units <= 0:
        bad.append("device identity is incomplete")
    if maximum <= 0 or len(dims) != 3 or dims[0] < maximum or any(x <= 0 for x in dims):
        bad.append("device thread limits are not concrete/consistent")
    return bad, {"ordinal": ordinal, "name": got.get("name", ""),
                 "compute_units": compute_units,
                 "max_threads_per_block": maximum,
                 "max_threads_dim": dims}


def validate_fq(text: str, rc: int) -> tuple[list[str], dict[str, object]]:
    arm = ARMS[0]
    bad: list[str] = []
    fixtures = exact_lines(text, "FQ_KPACK4_FIXTURE ")
    shards = exact_lines(text, "FQ_SHARD ")
    cells = exact_lines(text, "FQ_TC_CELL ")
    done = exact_lines(text, "FQ_SHAPE_DONE ")
    if len(fixtures) != 2:
        bad.append(f"fq-dense: fixture rows={len(fixtures)}/2")
    if len(shards) != 1 or len(cells) != 1:
        bad.append(f"fq-dense: shard/cell rows={len(shards)}/{len(cells)}, want 1/1")
        return bad, {}
    phases = {fields(line).get("phase"): fields(line) for line in fixtures}
    prepare, recover = phases.get("prepare", {}), phases.get("recover", {})
    for key, value in {
            "q": "12", "shape": FQ_SHAPE, "layout": "1", "bits": "4",
            "high_bits": "0", "artifact_tile_k": "0",
            "transport_tile_k": "64", "group_size": "32",
            "mapping_id": MAPPING_ID, "direct_rc": "0", "abi_rc": "0",
            "direct_equal": "1"}.items():
        if prepare.get(key) != value:
            bad.append(f"fq-dense: prepare {key} differs")
    for key, value in {"q": "12", "shape": FQ_SHAPE,
                       "mapping_id": MAPPING_ID, "direct_rc": "0",
                       "abi_rc": "0", "direct_equal": "1",
                       "native_equal": "1"}.items():
        if recover.get(key) != value:
            bad.append(f"fq-dense: recover {key} differs")

    shard = fields(shards[0])
    for key, value in {"q": "12", "A": "0", "bchunk": "0",
                       "shape": FQ_SHAPE, "weight_layout": "1",
                       "weight_mapping_id": MAPPING_ID, "typed_rows": "1",
                       "selected_rows": "1", "only_split": "1",
                       "bc_mode": "skip", "iterations": "1",
                       "correctness_repeats": "1", "schedule_seed": FQ_SEED}.items():
        if shard.get(key) != value:
            bad.append(f"fq-dense: shard {key}={shard.get(key)!r}, want {value!r}")
    cell = fields(cells[0])
    expected = {
        "q": "12", "A": "0", "bchunk": "0", "shape": FQ_SHAPE,
        "symbol": arm.symbol, "tm": "64", "tn": "128", "tk": "64",
        "wm": "16", "wn": "32", "stages": "4",
        "provider": "standard-aiu", "S": "1", "scope": "FULL_OUTPUT",
        "resolved_delivery_n": "32", "provider_capacity_rows": "0",
        "scalezero_fused": "1", "reducer_untimed": "0",
        "partial_bytes": "0",
    }
    for key, value in expected.items():
        if cell.get(key) != value:
            bad.append(f"fq-dense: cell {key}={cell.get(key)!r}, want {value!r}")
    try:
        cutlass_status = int(cell.get("failure_cutlass_status", "-1"))
        runtime_status = int(cell.get("failure_runtime_status", "-1"))
        raw_bad = int(cell.get("raw_bad", "-1"))
    except ValueError:
        bad.append("fq-dense: malformed status/count")
        return bad, {}
    state = cell.get("state", "MISSING")
    step = cell.get("failure_step", "MISSING")
    structural = {"SHIPPING_SHARED_STORAGE", "SPLIT_SHARED_STORAGE",
                  "SPLIT_PARTITION", "INADMISSIBLE_PIPELINE_DEPTH",
                  "REAL_CAN_IMPLEMENT"}
    if state == "MEASURED":
        classification = "MEASURED"
        if rc != 0 or cutlass_status or runtime_status or raw_bad or step != "NONE":
            bad.append("fq-dense: measured row carries failure evidence")
        if len(done) != 1 or fields(done[0]).get("status") != "PASS":
            bad.append("fq-dense: measured row lacks PASS completion")
        try:
            if float(cell.get("us", "0")) <= 0 or cell.get("samples") in (None, "[]"):
                bad.append("fq-dense: measured row lacks positive sample")
        except ValueError:
            bad.append("fq-dense: malformed timing")
    elif state in structural:
        classification = f"STRUCTURAL_{state}"
        if rc != 0 or cutlass_status or runtime_status or raw_bad or step != "NONE":
            bad.append("fq-dense: structural row carries hard failure evidence")
        if len(done) != 1 or fields(done[0]).get("status") != "PASS":
            bad.append("fq-dense: structural row lacks completion")
    elif state in {"INITIALIZE", "LAUNCH", "TIMING"}:
        classification = f"HARD_{state}"
        if rc != 1 or len(done) != 0 or step == "NONE" or \
                not (cutlass_status or runtime_status) or raw_bad:
            bad.append("fq-dense: hard failure evidence is inconsistent")
    elif state == "RAW_FP16_MISMATCH":
        classification = "NUMERIC_MISMATCH"
        if rc != 1 or len(done) != 0 or raw_bad <= 0 or \
                cutlass_status or runtime_status:
            bad.append("fq-dense: numeric failure evidence is inconsistent")
    else:
        classification = "UNKNOWN"
        bad.append(f"fq-dense: unclassified state {state!r}")
    return bad, {"arm": arm.name, "parent": arm.ordinal,
                 "symbol": arm.symbol, "cta_threads": arm.cta_threads,
                 "process_rc": rc, "classification": classification,
                 "state": state, "step": step,
                 "cutlass_status": cutlass_status,
                 "runtime_status": runtime_status, "raw_bad": raw_bad}


def validate_sf(arm: Arm, text: str, rc: int) -> tuple[list[str], dict[str, object]]:
    bad: list[str] = []
    shards = exact_lines(text, "SF_SHARD ")
    fixtures = exact_lines(text, "SF_FIXTURE ")
    attempts = exact_lines(text, "SF_ATTEMPT ")
    fatals = exact_lines(text, "SF_FATAL ")
    cells = exact_lines(text, "SF_CELL ")
    completes = exact_lines(text, "SF_COMPLETE ")
    if len(shards) != 1 or len(fixtures) != 1 or len(attempts) != 1:
        bad.append(f"{arm.name}: shard/fixture/attempt rows="
                   f"{len(shards)}/{len(fixtures)}/{len(attempts)}, want 1/1/1")
        return bad, {}
    shard = fields(shards[0])
    for key, value in {"qtype": "12", "artifact_tile_k": "0", "bchunk": "0",
                       "typed_rows": "1", "weight_layout": "1",
                       "weight_mapping_id": MAPPING_ID, "selected_rows": "1",
                       "algorithm_mask": "0x1", "iterations": "1",
                       "correctness_repeats": "1", "schedule_seed": SF_SEED}.items():
        if shard.get(key) != value:
            bad.append(f"{arm.name}: shard {key}={shard.get(key)!r}, want {value!r}")
    fixture = fields(fixtures[0])
    for key, value in {"mode": "exact", "tag_round": "none",
                       "roundtrip": "1", "exact": "1", "isolation": "1"}.items():
        if fixture.get(key) != value:
            bad.append(f"{arm.name}: fixture {key} differs")
    attempt = fields(attempts[0])
    if attempt.get("shape") != SF_SHAPE or attempt.get("symbol") != arm.symbol:
        bad.append(f"{arm.name}: exact attempt differs")

    if len(fatals) == 1 and not cells and not completes:
        fatal = fields(fatals[0])
        for key, value in {"symbol": arm.symbol, "shape": SF_SHAPE,
                           "algorithm": "NONPERSISTENT",
                           "cta_threads": str(arm.cta_threads)}.items():
            if fatal.get(key) != value:
                bad.append(f"{arm.name}: fatal {key}={fatal.get(key)!r}, want {value!r}")
        try:
            cutlass_status = int(fatal.get("cutlass_status", "-1"))
            runtime_status = int(fatal.get("runtime_status", "-1"))
            raw_bad = int(fatal.get("raw_bad", "-1"))
        except ValueError:
            bad.append(f"{arm.name}: malformed fatal status/count")
            return bad, {}
        state, step = fatal.get("state", "MISSING"), fatal.get("step", "MISSING")
        if state == "INITIALIZE_FAIL":
            classification = "INITIALIZE_REJECTED"
            if step != "NONPERSISTENT_INITIALIZE" or cutlass_status == 0 or \
                    runtime_status or raw_bad:
                bad.append(f"{arm.name}: initialize rejection evidence inconsistent")
        elif state == "LAUNCH_FAIL":
            classification = "LAUNCH_REJECTED"
            if step == "NONE" or not (cutlass_status or runtime_status) or raw_bad:
                bad.append(f"{arm.name}: launch rejection evidence inconsistent")
        elif state == "RAW_FP16_MISMATCH":
            classification = "NUMERIC_MISMATCH"
            if raw_bad <= 0 or cutlass_status or runtime_status:
                bad.append(f"{arm.name}: numeric mismatch evidence inconsistent")
        elif state == "TIMING_FAIL":
            classification = "TIMING_FAILURE"
            if step == "NONE" or not (cutlass_status or runtime_status):
                bad.append(f"{arm.name}: timing failure evidence inconsistent")
        else:
            classification = "UNKNOWN"
            bad.append(f"{arm.name}: unclassified fatal state {state!r}")
        if rc != 1:
            bad.append(f"{arm.name}: fatal process rc={rc}/1")
    elif not fatals and len(cells) == 1 and len(completes) == 1:
        try:
            cell = json.loads(cells[0][len("SF_CELL "):])
        except json.JSONDecodeError as error:
            return [f"{arm.name}: malformed SF_CELL JSON: {error}"], {}
        complete = fields(completes[0])
        for key, value in {"shape": SF_SHAPE, "qtype": 12,
                           "symbol": arm.symbol, "a_provider": 0,
                           "resolved_delivery_n": 32, "config": arm.config,
                           "algorithm": "NONPERSISTENT",
                           "metric_scope": "FULL_OUTPUT", "split": 1,
                           "raw_bad": 0, "execution_ordinal": 0}.items():
            if cell.get(key) != value:
                bad.append(f"{arm.name}: cell {key}={cell.get(key)!r}, want {value!r}")
        status, reason = cell.get("status", "MISSING"), cell.get("reason", "MISSING")
        if status == "MEASURED":
            classification = "MEASURED"
            if reason != "MEASURED" or float(cell.get("sample_us", 0)) <= 0:
                bad.append(f"{arm.name}: measured result lacks positive timing")
            want_counts = {"runtime_cells": "1", "measured_cells": "1",
                           "records": "1"}
        elif status == "INADMISSIBLE":
            classification = f"STRUCTURAL_{reason}"
            if not str(reason).startswith("INADMISSIBLE_"):
                bad.append(f"{arm.name}: structural reason is not named")
            want_counts = {"runtime_cells": "1", "measured_cells": "0",
                           "records": "1"}
        else:
            classification = "UNKNOWN"
            bad.append(f"{arm.name}: unknown cell status {status!r}")
            want_counts = {}
        for key, value in {"status": "COMPLETE", "shape": SF_SHAPE,
                           "typed_rows": "1", "iterations": "1",
                           "fixture_mode": "exact", "roundtrip": "PASS",
                           **want_counts}.items():
            if complete.get(key) != value:
                bad.append(f"{arm.name}: complete {key} differs")
        if rc != 0:
            bad.append(f"{arm.name}: completed process rc={rc}/0")
        state, step, cutlass_status, runtime_status, raw_bad = \
            reason, "NONE", 0, 0, int(cell.get("raw_bad", -1))
    else:
        return [f"{arm.name}: fatal/cell/complete rows="
                f"{len(fatals)}/{len(cells)}/{len(completes)} are inconsistent"], {}
    return bad, {"arm": arm.name, "parent": arm.ordinal,
                 "symbol": arm.symbol, "cta_threads": arm.cta_threads,
                 "process_rc": rc, "classification": classification,
                 "state": state, "step": step,
                 "cutlass_status": cutlass_status,
                 "runtime_status": runtime_status, "raw_bad": raw_bad}


def conclusions(observations: list[dict[str, object]],
                device: dict[str, object]) -> dict[str, str]:
    by_arm = {row["arm"]: row for row in observations}
    fq = by_arm["fq-dense"]
    subject = by_arm["sf-subject"]
    control = by_arm["sf-control"]
    maximum = int(device["max_threads_per_block"])
    fq_class = str(fq["classification"])
    if fq_class == "MEASURED":
        fq_verdict = "FRESH_PROCESS_BLOCKER_NOT_REPRODUCED"
    elif fq_class.startswith("STRUCTURAL_"):
        fq_verdict = "STRUCTURALLY_EXCLUDABLE"
    elif fq_class in {"HARD_INITIALIZE", "HARD_LAUNCH"} and maximum >= 512:
        fq_verdict = "NOT_GLOBAL_THREAD_LIMIT"
    else:
        fq_verdict = fq_class

    subject_class = str(subject["classification"])
    control_class = str(control["classification"])
    if subject_class == "MEASURED" and control_class == "MEASURED":
        sf_verdict = "FRESH_PROCESS_BLOCKER_NOT_REPRODUCED"
    elif subject_class == "INITIALIZE_REJECTED" and control_class == "MEASURED":
        sf_verdict = ("DEVICE_THREAD_LIMIT_CAUSAL" if maximum < 1024
                      else "SUBJECT_ONLY_INIT_REJECTION_NOT_DEVICE_THREAD_LIMIT")
    elif subject_class.startswith("STRUCTURAL_") and control_class == "MEASURED":
        sf_verdict = "SUBJECT_STRUCTURALLY_EXCLUDABLE_CONTROL_MEASURED"
    else:
        sf_verdict = f"SUBJECT_{subject_class}__CONTROL_{control_class}"
    return {"fq": fq_verdict, "scalefirst": sf_verdict}


def read_rc(path: Path) -> int:
    value = path.read_text().strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{path.name}: process rc={value!r}, want 0 or 1")
    return int(value)


def validate_run(run_dir: Path, write_results: bool = True) -> list[str]:
    bad = validate_generated(run_dir)
    try:
        device_text = (run_dir / "results/device-limits.log").read_text()
        device_bad, device = validate_device_limits(device_text)
        bad += device_bad
    except OSError as error:
        bad.append(f"cannot read device limits: {error}")
        device = {}
    observations: list[dict[str, object]] = []
    for arm in ARMS:
        try:
            rc = read_rc(run_dir / "results" / f"{arm.name}.rc")
            text = (run_dir / "results" / f"{arm.name}.run.log").read_text()
        except (OSError, ValueError) as error:
            bad.append(f"{arm.name}: cannot read result: {error}")
            continue
        arm_bad, observation = (validate_fq(text, rc) if arm.family == "fq"
                                else validate_sf(arm, text, rc))
        bad += arm_bad
        if observation:
            observations.append(observation)

        try:
            authority = (run_dir / "build" / arm.name /
                         ".quactlize-source-head").read_text().strip()
            source = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
            if authority != source:
                bad.append(f"{arm.name}: build source differs")
            binary_name = ("test_fully_quantized_internal_sweep"
                           if arm.family == "fq" else
                           "test_scalefirst_internal_sweep")
            if not (run_dir / "build" / arm.name / "ppu_targets" /
                    binary_name).is_file():
                bad.append(f"{arm.name}: binary is missing")
        except (OSError, subprocess.CalledProcessError) as error:
            bad.append(f"{arm.name}: cannot validate build authority: {error}")

    if len(observations) != len(ARMS) or not device:
        return bad
    verdicts = conclusions(observations, device)
    if write_results and not bad:
        results = run_dir / "results"
        header = ("arm\tparent\tsymbol\tcta_threads\tprocess_rc\tclassification\t"
                  "state\tstep\tcutlass_status\truntime_status\traw_bad\t"
                  "device_max_threads_per_block\n")
        rows = "".join(
            "\t".join(str(item[key]) for key in (
                "arm", "parent", "symbol", "cta_threads", "process_rc",
                "classification", "state", "step", "cutlass_status",
                "runtime_status", "raw_bad")) +
            f"\t{device['max_threads_per_block']}\n" for item in observations)
        (results / "summary.tsv").write_text(header + rows)
        document = {"schema": "quactlize.a09.remaining-blockers.v1",
                    "device": device, "observations": observations,
                    "verdicts": verdicts}
        (results / "verdict.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n")
    if not bad:
        for item in observations:
            print("A09_BLOCKER_OBSERVATION " + " ".join(
                f"{key}={item[key]}" for key in (
                    "arm", "parent", "cta_threads", "process_rc",
                    "classification", "state", "step", "cutlass_status",
                    "runtime_status", "raw_bad")))
        print(f"A09_BLOCKER_VERDICT fq={verdicts['fq']} "
              f"scalefirst={verdicts['scalefirst']} "
              f"max_threads_per_block={device['max_threads_per_block']}")
    return bad


def synthetic_fq(measured: bool = False) -> str:
    state = "MEASURED" if measured else "LAUNCH"
    step = "NONE" if measured else "CORRECTNESS_LAUNCH"
    runtime = 0 if measured else 9
    us = "1.0" if measured else "0.0"
    samples = "[1.0]" if measured else "[]"
    out = [
        f"FQ_KPACK4_FIXTURE phase=prepare q=12 shape={FQ_SHAPE} version=2 layout=1 "
        "bits=4 high_bits=0 artifact_tile_k=0 transport_tile_k=64 group_size=32 "
        f"reserved=0 mapping_id={MAPPING_ID} direct_rc=0 abi_rc=0 direct_equal=1",
        f"FQ_KPACK4_FIXTURE phase=recover q=12 shape={FQ_SHAPE} mapping_id={MAPPING_ID} "
        "direct_rc=0 abi_rc=0 direct_equal=1 native_equal=1",
        f"FQ_SHARD q=12 A=0 bchunk=0 shape={FQ_SHAPE} weight_layout=1 "
        f"weight_mapping_id={MAPPING_ID} weight_delivery_n=0 typed_rows=1 "
        "selected_rows=1 only_split=1 bc_mode=skip iterations=1 correctness_repeats=1 "
        f"schedule_seed={FQ_SEED}",
        f"FQ_TC_CELL q=12 A=0 bchunk=0 shape={FQ_SHAPE} symbol={ARMS[0].symbol} "
        "tm=64 tn=128 tk=64 wm=16 wn=32 stages=4 provider=standard-aiu S=1 "
        "scope=FULL_OUTPUT resolved_delivery_n=32 provider_capacity_rows=0 "
        f"scalezero_fused=1 state={state} us={us} raw_bad=0 reducer_untimed=0 "
        f"failure_step={step} failure_cutlass_status=0 failure_runtime_status={runtime} "
        "failure_repeat=0 first_bad=18446744073709551615 first_want=0x0000 "
        f"first_got=0x0000 shipping_smem=1 split_smem=1 partial_bytes=0 samples={samples}",
    ]
    if measured:
        out.append(f"FQ_SHAPE_DONE q=12 A=0 bchunk=0 shape={FQ_SHAPE} "
                   f"weight_layout=1 weight_mapping_id={MAPPING_ID} "
                   "weight_delivery_n=0 typed_rows=1 selected_rows=1 only_split=1 "
                   "bc_mode=skip iterations=1 status=PASS")
    return "\n".join(out) + "\n"


def synthetic_sf(arm: Arm, failure: bool) -> str:
    prefix = [
        f"SF_SHARD qtype=12 artifact_tile_k=0 bchunk=0 typed_rows=1 weight_layout=1 "
        f"weight_mapping_id={MAPPING_ID} selected_rows=1 algorithm_mask=0x1 "
        f"device=0 cu=72 iterations=1 correctness_repeats=1 schedule_seed={SF_SEED}",
        "SF_FIXTURE mode=exact first_golden=0x3c00 tag_round=none probe_count=8 "
        "probe_fnv=1 a_fnv=2 low_native_fnv=3 low_placed_fnv=4 scale_fnv=5 "
        "zero_fnv=6 golden_fnv=7 roundtrip=1 exact=1 isolation=1",
        f"SF_ATTEMPT shape={SF_SHAPE} ordinal=1/1 symbol={arm.symbol}",
    ]
    if failure:
        prefix.append(
            f"SF_FATAL symbol={arm.symbol} shape={SF_SHAPE} algorithm=NONPERSISTENT "
            "state=INITIALIZE_FAIL step=NONPERSISTENT_INITIALIZE cutlass_status=3 "
            f"runtime_status=0 cta_threads={arm.cta_threads} shipping_smem=1 "
            "persistent_smem=1 split_smem=1 repeat=-1 raw_bad=0 "
            "first_bad=18446744073709551615 want=0x0000 got=0x0000")
    else:
        cell = {"shape": SF_SHAPE, "qtype": 12, "artifact_tile_k": 0,
                "bchunk": 0, "symbol": arm.symbol, "a_provider": 0,
                "resolved_delivery_n": 32, "config": arm.config,
                "algorithm": "NONPERSISTENT", "metric_scope": "FULL_OUTPUT",
                "policy": "ordinary", "split": 1, "grid": 128,
                "occupancy": 0, "capacity_b_mask": "0x0",
                "balanced_b_mask": "0x0", "status": "MEASURED",
                "reason": "MEASURED", "sample": 0, "sample_us": 1.0,
                "MFU_pct": 1.0, "distinct_MBU_model_pct": 1.0,
                "raw_bad": 0, "fingerprint": "0x1",
                "reducer_correctness_untimed": 0, "partial_bytes": 0,
                "shipping_smem": 1, "persistent_smem": 1,
                "split_smem": 1, "execution_ordinal": 0}
        prefix.append("SF_CELL " + json.dumps(cell, separators=(",", ":")))
        prefix.append(f"SF_COMPLETE status=COMPLETE shape={SF_SHAPE} typed_rows=1 "
                      "runtime_cells=1 measured_cells=1 records=1 iterations=1 "
                      "fixture=ORDER-INDEPENDENT+FP16-EXACT fixture_mode=exact "
                      "roundtrip=PASS high_plane_coverage=PASS isolation_coverage=PASS")
    return "\n".join(prefix) + "\n"


def self_test() -> None:
    bad = audit_runner(RUNNER.read_text())
    if bad:
        raise AssertionError("; ".join(bad))
    device_bad, device = validate_device_limits(
        "A09_DEVICE_LIMITS ordinal=0 name=PPU-ZW810 compute_units=72 "
        "max_threads_per_block=1024 max_threads_dim=1024,1024,64\n")
    if device_bad:
        raise AssertionError("device synthetic positive failed")
    fq_bad, fq = validate_fq(synthetic_fq(False), 1)
    subject_bad, subject = validate_sf(ARMS[1], synthetic_sf(ARMS[1], True), 1)
    control_bad, control = validate_sf(ARMS[2], synthetic_sf(ARMS[2], False), 0)
    if fq_bad or subject_bad or control_bad:
        raise AssertionError("synthetic positive failed: " +
                             "; ".join(fq_bad + subject_bad + control_bad))
    result = conclusions([fq, subject, control], device)
    if result != {"fq": "NOT_GLOBAL_THREAD_LIMIT",
                  "scalefirst": "SUBJECT_ONLY_INIT_REJECTION_NOT_DEVICE_THREAD_LIMIT"}:
        raise AssertionError(f"synthetic conclusion differs: {result}")
    plants = 0
    for text, validator, rc in (
        (synthetic_fq(False).replace(FQ_SHAPE, "7x8192x5121", 1),
         lambda value, code: validate_fq(value, code)[0], 1),
        (synthetic_fq(False).replace("failure_runtime_status=9",
                                     "failure_runtime_status=0"),
         lambda value, code: validate_fq(value, code)[0], 1),
        (synthetic_sf(ARMS[1], True).replace("cta_threads=1024",
                                             "cta_threads=512"),
         lambda value, code: validate_sf(ARMS[1], value, code)[0], 1),
        (synthetic_sf(ARMS[2], False).replace('"raw_bad":0', '"raw_bad":1'),
         lambda value, code: validate_sf(ARMS[2], value, code)[0], 0),
    ):
        if validator(text, rc):
            plants += 1
    for planted in (
        RUNNER.read_text().replace("--parent-begin 1417", "--parent-begin 1418", 1),
        RUNNER.read_text().replace("PPU_BUILD_RESUME=0", "PPU_BUILD_RESUME=1", 1),
        RUNNER.read_text().replace("--algorithm=nonpersistent", "--algorithm=all", 1),
    ):
        if audit_runner(planted):
            plants += 1
    limits_bad, _ = validate_device_limits(
        "A09_DEVICE_LIMITS ordinal=0 name=PPU compute_units=72 "
        "max_threads_per_block=0 max_threads_dim=1024,1024,64\n")
    plants += bool(limits_bad)
    if plants != 8:
        raise AssertionError(f"negative plants red={plants}/8")
    with tempfile.TemporaryDirectory(prefix="a09-blockers-") as tmp:
        root = Path(tmp)
        fq_gen.generate(12, root / "generated/fq-dense", 1,
                        parent_begin=1417, parent_count=1)
        for arm in ARMS[1:]:
            sf_gen.generate(12, 0, 0, root / "generated" / arm.name, 1,
                            False, arm.symbol, 1)
        generated_bad = validate_generated(root)
        if generated_bad:
            raise AssertionError("generated synthetic failed: " + "; ".join(generated_bad))
    print("[a09-blockers:self-test] PASS exact FQ1417 SF4738/4750, "
          "fresh three-tree/process contract, runtime thread limit, 8 plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-generated-dir", type=Path)
    parser.add_argument("--validate-run-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.validate_generated_dir is not None and args.validate_run_dir is not None:
            raise ValueError("choose only one validation mode")
        if args.validate_generated_dir is not None:
            bad = validate_generated(args.validate_generated_dir.resolve())
            if bad:
                raise ValueError("; ".join(bad))
            print("[a09-blockers] generated PASS fq=1417 sf=4738/4750")
        elif args.validate_run_dir is not None:
            bad = validate_run(args.validate_run_dir.resolve())
            if bad:
                raise ValueError("; ".join(bad))
        else:
            self_test()
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[a09-blockers] FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
