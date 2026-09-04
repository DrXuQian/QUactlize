#!/usr/bin/env python3
"""Authority and result checker for the TM8/WN16 cross-format gate."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_m8n16_cross_format_correctness_box.sh"
sys.path.insert(0, str(ROOT / "tools"))

import fully_quantized_kpack_discovery_matrix as fq_matrix  # noqa: E402
import gen_fully_quantized_grouped_kpack_units as fqg_gen  # noqa: E402
import gen_fully_quantized_kpack_discovery_units as fqd_gen  # noqa: E402
import gen_scalefirst_grouped_kpack_units as sfg_gen  # noqa: E402
import gen_scalefirst_internal_units as sfd_gen  # noqa: E402
import scalefirst_grouped_kpack_matrix as sfg_matrix  # noqa: E402
import scalefirst_internal_matrix as sf_matrix  # noqa: E402


REPEATS = 7
MAPPING_Q4 = "0x51344b5034540001"
MAPPING_GENERIC = "0x514b504b54000001"
ROUTES = ("fq-dense", "fq-grouped", "sf-dense", "sf-grouped")
ORCHESTRATION_PATHS = {
    "ci/check_m8n16_cross_format_correctness.py",
    "tools/run_m8n16_cross_format_correctness_box.sh",
}
RELEVANT_SOURCE_PATHS = (
    ".gitmodules", "CMakeLists.txt", "build.sh", "benchmarks", "ci", "dev",
    "quactlize", "tests", "third_party", "tools",
)


@dataclass(frozen=True)
class Format:
    name: str
    qtype: int
    packed_format: int
    layout: int
    tile_k: int
    dense_begin: int
    low_bits: int
    high_bits: int
    group_size: int
    transport_k: int
    mapping_id: str

    @property
    def fqd_symbol(self) -> str:
        return (f"fqk_tc_q{self.qtype}_l{self.layout}_a0_tm8_tn64_"
                f"tk{self.tile_k}_wm8_wn16_s2_bc0_ap0_dn16")

    @property
    def fqg_symbols(self) -> tuple[str, str]:
        base = (f"fqg_q{self.qtype}_l{self.layout}_tm8_tn64_"
                f"tk{self.tile_k}_wm8_wn16_s2_ap0_dn16_")
        return base + "nonpersistent", base + "persistent"

    @property
    def sfd_symbol(self) -> str:
        return (f"sf_q{self.qtype}_a0_tm8_tn64_tk{self.tile_k}_"
                "wm8_wn16_s2_bc0_ap0_dn16")

    @property
    def sfg_symbol(self) -> str:
        return (f"sfg_q{self.qtype}_tm8_tn64_tk{self.tile_k}_"
                "wm8_wn16_s2_ap0_dn16")


FORMATS = (
    Format("Q2_K", 10, 2, 2, 128, 60, 2, 0, 16, 128, MAPPING_GENERIC),
    Format("Q3_K", 11, 3, 2, 256, 30, 2, 1, 16, 256, MAPPING_GENERIC),
    Format("Q4_K", 12, 0, 1, 64, 60, 4, 0, 32, 64, MAPPING_Q4),
    Format("Q5_K", 13, 1, 2, 256, 30, 4, 1, 32, 256, MAPPING_GENERIC),
    Format("Q6_K", 14, 4, 2, 128, 30, 4, 2, 16, 128, MAPPING_GENERIC),
)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("\\\n", ""))


def fields(line: str) -> dict[str, str]:
    return {part.split("=", 1)[0]: part.split("=", 1)[1]
            for part in line.strip().split()[1:] if "=" in part}


def one_line(text: str, prefix: str, bad: list[str], label: str) -> str:
    rows = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        bad.append(f"{label}: {prefix.strip()} rows={len(rows)}/1")
        return ""
    return rows[0]


def expected_axes(fmt: Format) -> tuple[int, ...]:
    return (8, 64, fmt.tile_k, 8, 16, 2, 0, 16)


def authority_errors() -> list[str]:
    bad: list[str] = []
    if tuple(row.qtype for row in FORMATS) != fq_matrix.QTYPES:
        bad.append("qtype order/set differs from FullyQuantized authority")
    for fmt in FORMATS:
        registered = fq_matrix.format_for(fmt.qtype)
        if (registered.name, registered.low_bits, registered.high_bits,
                registered.group_size) != (
                fmt.name, fmt.low_bits, fmt.high_bits, fmt.group_size):
            bad.append(f"{fmt.name}: format registry differs")
        if fq_matrix.layout_for(fmt.qtype) != fmt.layout:
            bad.append(f"{fmt.name}: canonical layout differs")

        dense = list(fq_matrix.provider_rows(fmt.qtype))
        if fmt.dense_begin >= len(dense):
            bad.append(f"{fmt.name}: dense ordinal outside authority")
        else:
            row, provider, delivery_n = dense[fmt.dense_begin]
            axes = (row.tile_m, row.tile_n, row.tactic_tile_k,
                    row.warp_m, row.warp_n, row.stages,
                    provider, delivery_n)
            if axes != expected_axes(fmt):
                bad.append(f"{fmt.name}: dense axes differ: {axes}")
            if fqd_gen.symbol(fmt.qtype, row, provider, delivery_n) != \
                    fmt.fqd_symbol:
                bad.append(f"{fmt.name}: dense symbol differs")

        grouped = list(fq_matrix.grouped_rows(fmt.qtype))[60:62]
        actual_grouped: list[str] = []
        for row, delivery_n, algorithm in grouped:
            persistent = int(algorithm == "GROUPED_PERSISTENT")
            axes = (row.tile_m, row.tile_n, row.tactic_tile_k,
                    row.warp_m, row.warp_n, row.stages, 0, delivery_n)
            if axes != expected_axes(fmt):
                bad.append(f"{fmt.name}: grouped axes differ: {axes}")
            actual_grouped.append(
                fqg_gen.symbol(fmt.qtype, row, delivery_n, persistent))
        if tuple(actual_grouped) != fmt.fqg_symbols:
            bad.append(f"{fmt.name}: grouped parent [60,62) differs")

        sf_dense = [item for item in sf_matrix.kpack_dense_candidates(fmt.qtype)
                    if sfd_gen.symbol(fmt.qtype, 0, *item) == fmt.sfd_symbol]
        if len(sf_dense) != 1:
            bad.append(f"{fmt.name}: ScaleFirst dense exact row count={len(sf_dense)}/1")
        else:
            row, provider, delivery_n = sf_dense[0]
            axes = (row.tile_m, row.tile_n, row.tactic_tile_k,
                    row.warp_m, row.warp_n, row.stages,
                    provider, delivery_n)
            if axes != expected_axes(fmt):
                bad.append(f"{fmt.name}: ScaleFirst dense axes differ: {axes}")

        sf_grouped = [item for item in sfg_matrix.candidate_rows(fmt.qtype)
                      if sfg_gen.symbol(fmt.qtype, *item) == fmt.sfg_symbol]
        if len(sf_grouped) != 1:
            bad.append(
                f"{fmt.name}: ScaleFirst grouped exact row count={len(sf_grouped)}/1")
        else:
            row, delivery_n = sf_grouped[0]
            axes = (row.tile_m, row.tile_n, row.tactic_tile_k,
                    row.warp_m, row.warp_n, row.stages, 0, delivery_n)
            if axes != expected_axes(fmt):
                bad.append(f"{fmt.name}: ScaleFirst grouped axes differ: {axes}")
    return bad


def audit_runner(text: str) -> list[str]:
    shell = compact(text)
    bad = authority_errors()
    tokens = (
        ("set-u-opipefail", 1),
        ("check_m8n16_cross_format_correctness.py", 5),
        ("gen_fully_quantized_kpack_discovery_units.py", 1),
        ("gen_fully_quantized_grouped_kpack_units.py", 1),
        ("gen_scalefirst_internal_units.py", 1),
        ("gen_scalefirst_grouped_kpack_units.py", 1),
        ("--parent-begin\"$DENSE_BEGIN\"--parent-count1", 1),
        ("--parent-begin60--parent-count2", 1),
        ("--select-symbol\"$sf_symbol\"", 1),
        ("--select-symbol\"$sfg_symbol\"", 1),
        ("forfamilyinfqsf", 1),
        ("build_family\"$q\"\"$family\"&", 1),
        ("localbuild=\"$RUN_DIR/build/q$q/$family\"", 1),
        ("PPU_BUILD_DIR=\"$build\"PPU_BUILD_RESUME=\"$resume\"", 1),
        ("PPU_PRESERVE_STALE_BUILD_TREES=1", 1),
        ("-uFQ_A02_Q3_GENERATED_DIR-uFQ_KQUANT_PERF_QTYPE", 1),
        ("build_target\"$q\"fq0test_fully_quantized_internal_sweepfq-dense", 1),
        ("build_target\"$q\"fq1test_fully_quantized_grouped_kpack_discoveryfq-grouped", 1),
        ("build_target\"$q\"sf0test_scalefirst_internal_sweepsf-dense", 1),
        ("build_target\"$q\"sf1test_scalefirst_grouped_kpack_discoverysf-grouped", 1),
        ("--validate-generated-dir\"$RUN_DIR\"", 1),
        ("--validate-build-dir\"$RUN_DIR\"", 1),
        ("RESUME=1reusingbuilds;rerunningall20routes", 1),
        ("case\"${RESUME:-0}\"in", 1),
        ("RESUME_requires_explicit_OUT", 1),
        ("mv\"$RUN_DIR/results\"\"$archived_results\"", 1),
        ("cmp-s\"$resume_snapshot\"", 1),
        ("--shape=7x64x512--shape=9x64x512", 1),
        ("--only-split=1--tm8-max-m=9--bc-mode=skip", 1),
        ("--shape=7x256x512--algorithm=full-output--fixture=exact", 1),
        ("--rows-file=\"$ROWS9\"--experts=2--n=64--k=512", 2),
        ("--correctness-repeats=7", 4),
        ("M8N16_CROSS_FORMAT_CORRECTNESSverdict=PASSformats=5routes=20"
         "cells=40measured=40structural=0repeats=7"
         "out_of_scope_structural=35", 1),
    )
    for token, count in tokens:
        if shell.count(token) != count:
            bad.append(f"runner must contain exactly {count} {token!r}")
    for fmt in FORMATS:
        axis = (f"{fmt.qtype})FORMAT_NAME={fmt.name};"
                f"PACKED_FORMAT={fmt.packed_format};WEIGHT_LAYOUT={fmt.layout};"
                f"TILE_K={fmt.tile_k};DENSE_BEGIN={fmt.dense_begin};;")
        if shell.count(axis) != 1:
            bad.append(f"runner {fmt.name} axis tuple differs")
    forbidden = (
        "run_fully_quantized_kpack_discovery_box.sh",
        "run_scalefirst_kpack_discovery_box.sh",
        "run_fq_kquant_kpack_perf_box.sh",
        "--only-split=0", "--only-split=2", "--only-split=4",
        "--only-split=8", "--algorithm=all", "--algorithm=split",
        "FQ_SWEEP_WEIGHT_LAYOUT=0", "SCALEFIRST_SWEEP_WEIGHT_LAYOUT=0",
        'build/q$q/ppu_targets',
    )
    for token in forbidden:
        if token in text:
            bad.append(f"runner contains forbidden broad/Xplane path {token!r}")
    return bad


def validate_manifest_set(fmt: Format, run_dir: Path) -> list[str]:
    bad: list[str] = []
    root = run_dir / "generated" / f"q{fmt.qtype}"
    paths = {route: root / route / "manifest.json" for route in ROUTES}
    docs: dict[str, dict] = {}
    for route, path in paths.items():
        if not path.is_file() or path.is_symlink():
            bad.append(f"{fmt.name}/{route}: manifest is not a regular file")
            continue
        try:
            docs[route] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            bad.append(f"{fmt.name}/{route}: cannot read manifest: {error}")
    if len(docs) != len(ROUTES):
        return bad

    try:
        fqd_gen.validate_manifest(docs["fq-dense"])
        fqg_gen.validate_manifest(docs["fq-grouped"])
    except ValueError as error:
        bad.append(f"{fmt.name}: FullyQuantized manifest rejected: {error}")
    fqd = docs["fq-dense"]
    expected_range = {
        "begin": fmt.dense_begin, "end": fmt.dense_begin + 1,
        "count": 1, "authority_count": len(fq_matrix.provider_rows(fmt.qtype)),
    }
    if fqd.get("parent_range") != expected_range:
        bad.append(f"{fmt.name}: FQ dense parent range differs")
    if [row.get("symbol") for row in fqd.get("dense_tc_parents", [])] != \
            [fmt.fqd_symbol]:
        bad.append(f"{fmt.name}: FQ dense executable symbol differs")

    fqg = docs["fq-grouped"]
    expected_grouped_range = {
        "begin": 60, "end": 62, "count": 2,
        "authority_count": len(fq_matrix.grouped_rows(fmt.qtype)),
    }
    if fqg.get("parent_range") != expected_grouped_range:
        bad.append(f"{fmt.name}: FQ grouped parent range differs")
    if tuple(row.get("symbol") for row in fqg.get("grouped_parents", [])) != \
            fmt.fqg_symbols:
        bad.append(f"{fmt.name}: FQ grouped executable symbols differ")

    sfd = docs["sf-dense"]
    if sfd.get("schema") != "quactlize.scalefirst.generated_shard.v3" or \
            sfd.get("identity") != {
                "qtype": fmt.qtype, "format": fmt.name,
                "artifact_tile_k": 0, "bchunk": 0,
                "weight_layout": fmt.layout,
                "weight_layout_name": sf_matrix.layout_name(fmt.layout),
            }:
        bad.append(f"{fmt.name}: ScaleFirst dense identity differs")
    if [row.get("symbol") for row in sfd.get("typed_rows", [])] != \
            [fmt.sfd_symbol] or sfd.get("selection", {}).get("symbol") != \
            fmt.sfd_symbol:
        bad.append(f"{fmt.name}: ScaleFirst dense exact selection differs")
    if sfd.get("selection", {}).get("mode") != "exact-symbol" or \
            sfd.get("selection", {}).get("compiled_rows") != 1:
        bad.append(f"{fmt.name}: ScaleFirst dense selection denominator differs")

    sfg = docs["sf-grouped"]
    expected_sfg_identity = {
        "qtype": fmt.qtype, "weight_layout": fmt.layout,
        "weight_layout_name": sf_matrix.layout_name(fmt.layout),
        "artifact_tile_k": 0, "quant_mode": "ScaleZero",
        "metadata_planes": 2,
    }
    if sfg.get("schema") != "quactlize.scalefirst_grouped_kpack_shard.v2" or \
            sfg.get("identity") != expected_sfg_identity:
        bad.append(f"{fmt.name}: ScaleFirst grouped identity differs")
    if [row.get("symbol") for row in sfg.get("typed_rows", [])] != \
            [fmt.sfg_symbol] or sfg.get("selection", {}).get("symbol") != \
            fmt.sfg_symbol:
        bad.append(f"{fmt.name}: ScaleFirst grouped exact selection differs")
    if sfg.get("selection", {}).get("mode") != "exact-symbol" or \
            sfg.get("selection", {}).get("compiled_rows") != 1:
        bad.append(f"{fmt.name}: ScaleFirst grouped selection denominator differs")
    return bad


def measured_common(got: dict[str, str], label: str, bad: list[str]) -> None:
    expected = {
        "state": "MEASURED", "raw_bad": "0", "failure_repeat": "-1",
        "first_bad": str(2**64 - 1), "want": "0x0000", "got": "0x0000",
    }
    for key, value in expected.items():
        if got.get(key) != value:
            bad.append(f"{label}: {key}={got.get(key)!r}, want {value!r}")
    if got.get("samples") in (None, "[]"):
        bad.append(f"{label}: measured cell has no completion sample")


def report_nonmeasured(fmt: Format, route: str, cells: list[dict[str, str]]) -> None:
    for cell in cells:
        state = cell.get("state", cell.get("status", "MISSING"))
        if state != "MEASURED":
            kind = "STRUCTURAL" if (
                state.startswith("INADMISSIBLE") or
                state in {"SHIPPING_SHARED_STORAGE", "SHARED_STORAGE", "OCCUPANCY"}
            ) else "NONMEASURED"
            print(f"M8N16_CROSS_FORMAT_{kind} q={fmt.qtype} route={route} "
                  f"symbol={cell.get('symbol', 'NONE')} "
                  f"algorithm={cell.get('algorithm', 'NONE')} state={state}")


def validate_fq_dense(fmt: Format, text: str) -> tuple[list[str], int, int]:
    label = f"{fmt.name}/fq-dense"
    bad: list[str] = []
    lines = text.splitlines()
    fixtures = [line for line in lines if line.startswith("FQ_KPACK4_FIXTURE ")]
    shards = [fields(line) for line in lines if line.startswith("FQ_SHARD ")]
    cells = [fields(line) for line in lines if line.startswith("FQ_TC_CELL ")]
    done = [fields(line) for line in lines if line.startswith("FQ_SHAPE_DONE ")]
    for what, got, want in (("fixture", len(fixtures), 4),
                            ("shard", len(shards), 2),
                            ("cell", len(cells), 2),
                            ("complete", len(done), 2)):
        if got != want:
            bad.append(f"{label}: {what} rows={got}/{want}")
    if any(len(rows) != 2 for rows in (shards, cells, done)):
        report_nonmeasured(fmt, "fq-dense", cells)
        return bad, sum(cell.get("state") == "MEASURED" for cell in cells), 0
    fixture_fields = [fields(line) for line in fixtures]
    fixture_by_key = {(row.get("shape"), row.get("phase")): row
                      for row in fixture_fields}
    by_shape = {row.get("shape"): row for row in cells}
    for m in (7, 9):
        shape = f"{m}x64x512"
        prepare = fixture_by_key.get((shape, "prepare"), {})
        for key, value in {
                "q": str(fmt.qtype), "shape": shape, "version": "2",
                "layout": str(fmt.layout), "bits": str(fmt.low_bits),
                "high_bits": str(fmt.high_bits), "artifact_tile_k": "0",
                "transport_tile_k": str(fmt.transport_k),
                "group_size": str(fmt.group_size), "reserved": "0",
                "mapping_id": fmt.mapping_id, "direct_rc": "0", "abi_rc": "0",
                "direct_equal": "1"}.items():
            if prepare.get(key) != value:
                bad.append(f"{label}/{shape}: prepare {key} differs")
        recover = fixture_by_key.get((shape, "recover"), {})
        for key, value in {
                "q": str(fmt.qtype), "shape": shape,
                "mapping_id": fmt.mapping_id, "direct_rc": "0", "abi_rc": "0",
                "direct_equal": "1", "native_equal": "1"}.items():
            if recover.get(key) != value:
                bad.append(f"{label}/{shape}: recover {key} differs")
        cell = by_shape.get(shape, {})
        expected = {
            "q": str(fmt.qtype), "A": "0", "bchunk": "0", "shape": shape,
            "symbol": fmt.fqd_symbol, "tm": "8", "tn": "64",
            "tk": str(fmt.tile_k), "wm": "8", "wn": "16", "stages": "2",
            "provider": "standard-aiu", "S": "1", "scope": "FULL_OUTPUT",
            "resolved_delivery_n": "16", "state": "MEASURED", "raw_bad": "0",
            "failure_step": "NONE", "failure_repeat": "-1",
            "first_bad": str(2**64 - 1), "first_want": "0x0000",
            "first_got": "0x0000", "partial_bytes": "0",
        }
        for key, value in expected.items():
            if cell.get(key) != value:
                bad.append(f"{label}/{shape}: {key}={cell.get(key)!r}, want {value!r}")
        if cell.get("samples") in (None, "[]"):
            bad.append(f"{label}/{shape}: completion sample missing")
    for shard in shards:
        expected = {
            "q": str(fmt.qtype), "A": "0", "bchunk": "0",
            "weight_layout": str(fmt.layout),
            "weight_mapping_id": fmt.mapping_id, "typed_rows": "1",
            "selected_rows": "1", "only_split": "1", "bc_mode": "skip",
            "iterations": "1", "correctness_repeats": str(REPEATS),
        }
        for key, value in expected.items():
            if shard.get(key) != value:
                bad.append(f"{label}: shard {key} differs")
    if any(row.get("status") != "PASS" for row in done):
        bad.append(f"{label}: shape completion is not PASS")
    report_nonmeasured(fmt, "fq-dense", cells)
    measured = sum(cell.get("state") == "MEASURED" for cell in cells)
    structural = sum(cell.get("state", "").startswith("INADMISSIBLE")
                     for cell in cells)
    return bad, measured, structural


def validate_fq_grouped(fmt: Format, text: str) -> tuple[list[str], int, int]:
    label = f"{fmt.name}/fq-grouped"
    bad: list[str] = []
    shard_line = one_line(text, "FQ_GROUPED_KPACK_SHARD ", bad, label)
    complete_line = one_line(text, "FQ_GROUPED_KPACK_COMPLETE ", bad, label)
    cells = [fields(line) for line in text.splitlines()
             if line.startswith("FQ_GROUPED_KPACK_CELL ")]
    structural_lines = [fields(line) for line in text.splitlines()
                        if line.startswith("FQ_GROUPED_KPACK_STRUCTURAL ")]
    if len(cells) != 2:
        bad.append(f"{label}: cells={len(cells)}/2")
    expected_structural = {
        ("SPLITK_S2", "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER"),
        ("SPLITK_S4", "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER"),
        ("SPLITK_S8", "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER"),
        ("BC_FULL_OUTPUT", "NO_CANONICAL_KPACK_BC_READER"),
    }
    got_structural = {(row.get("algorithm"), row.get("reason"))
                      for row in structural_lines
                      if row.get("q") == str(fmt.qtype) and
                      row.get("status") == "STRUCTURAL_UNAVAILABLE"}
    if len(structural_lines) != 4 or got_structural != expected_structural:
        bad.append(f"{label}: explicit out-of-scope structural set differs")
    shard = fields(shard_line) if shard_line else {}
    for key, value in {
            "q": str(fmt.qtype), "layout": str(fmt.layout),
            "mapping_id": fmt.mapping_id, "type_rows": "2",
            "selected_rows": "2", "router": "exact-rows-v1",
            "experts": "2", "total_rows": "9", "max_rows": "9",
            "active": "1", "empty": "1", "iterations": "1",
            "warmups": "1", "correctness_repeats": str(REPEATS),
            "roundtrip": "PASS", "metadata": "PACKED_UNITS"}.items():
        if shard.get(key) != value:
            bad.append(f"{label}: shard {key} differs")
    expected = dict(zip(("GROUPED_NONPERSISTENT", "GROUPED_PERSISTENT"),
                        fmt.fqg_symbols))
    for cell in cells:
        algorithm = cell.get("algorithm", "")
        for key, value in {
                "q": str(fmt.qtype), "layout": str(fmt.layout),
                "symbol": expected.get(algorithm, "INVALID"),
                "config": f"8x64x{fmt.tile_k}_w8x16_s2"}.items():
            if cell.get(key) != value:
                bad.append(f"{label}/{algorithm}: {key} differs")
        measured_common(cell, f"{label}/{algorithm}", bad)
    complete = fields(complete_line) if complete_line else {}
    for key, value in {"q": str(fmt.qtype), "status": "PASS", "rows": "2",
                       "cells": "2", "measured": "2", "structural": "0",
                       "correctness": "RAW_FP16"}.items():
        if complete.get(key) != value:
            bad.append(f"{label}: complete {key} differs")
    report_nonmeasured(fmt, "fq-grouped", cells)
    measured = sum(cell.get("state") == "MEASURED" for cell in cells)
    structural = sum(cell.get("state") in {"SHARED_STORAGE", "OCCUPANCY"}
                     for cell in cells)
    return bad, measured, structural


def validate_sf_dense(fmt: Format, text: str) -> tuple[list[str], int, int]:
    label = f"{fmt.name}/sf-dense"
    bad: list[str] = []
    shard_line = one_line(text, "SF_SHARD ", bad, label)
    complete_line = one_line(text, "SF_COMPLETE ", bad, label)
    records: list[dict] = []
    for line in text.splitlines():
        if line.startswith("SF_CELL "):
            try:
                records.append(json.loads(line[len("SF_CELL "):]))
            except json.JSONDecodeError as error:
                bad.append(f"{label}: malformed cell JSON: {error}")
    if len(records) != 2:
        bad.append(f"{label}: records={len(records)}/2")
    algorithms = {record.get("algorithm") for record in records}
    if algorithms != {"NONPERSISTENT", "PERSISTENT"}:
        bad.append(f"{label}: algorithm set differs: {algorithms}")
    for record in records:
        algorithm = str(record.get("algorithm", "NONE"))
        expected = {
            "shape": "7x256x512", "qtype": fmt.qtype,
            "artifact_tile_k": 0, "bchunk": 0, "symbol": fmt.sfd_symbol,
            "a_provider": 0, "resolved_delivery_n": 16,
            "config": f"8x64x{fmt.tile_k}_w8x16_s2_bc0_ap0_dn16",
            "metric_scope": "FULL_OUTPUT", "split": 1,
            "status": "MEASURED", "reason": "MEASURED", "sample": 0,
            "raw_bad": 0, "reducer_correctness_untimed": 0,
            "execution_ordinal": 0,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                bad.append(f"{label}/{algorithm}: {key} differs")
        if not isinstance(record.get("sample_us"), (int, float)) or \
                float(record.get("sample_us", 0)) <= 0:
            bad.append(f"{label}/{algorithm}: completion sample missing")
        if algorithm == "NONPERSISTENT":
            for key, value in {
                    "policy": "ordinary", "grid": 4, "occupancy": 0,
                    "capacity_b_mask": "0x0", "balanced_b_mask": "0x0"}.items():
                if record.get(key) != value:
                    bad.append(f"{label}/{algorithm}: {key} differs")
        elif algorithm == "PERSISTENT":
            occupancy = record.get("occupancy")
            if not isinstance(occupancy, int) or not 1 <= occupancy <= 63:
                bad.append(f"{label}/{algorithm}: occupancy differs")
            else:
                mask = hex((1 << (occupancy + 1)) - 2)
                for key, value in {
                        "policy": "capacity+balanced", "grid": 4,
                        "capacity_b_mask": mask,
                        "balanced_b_mask": mask}.items():
                    if record.get(key) != value:
                        bad.append(f"{label}/{algorithm}: {key} differs")
    shard = fields(shard_line) if shard_line else {}
    for key, value in {
            "qtype": str(fmt.qtype), "artifact_tile_k": "0", "bchunk": "0",
            "typed_rows": "1", "weight_layout": str(fmt.layout),
            "weight_mapping_id": fmt.mapping_id, "selected_rows": "1",
            "algorithm_mask": "0x3", "iterations": "1",
            "correctness_repeats": str(REPEATS)}.items():
        if shard.get(key) != value:
            bad.append(f"{label}: shard {key} differs")
    complete = fields(complete_line) if complete_line else {}
    for key, value in {"status": "COMPLETE", "shape": "7x256x512",
                       "typed_rows": "1", "runtime_cells": "2",
                       "measured_cells": "2", "records": "2",
                       "iterations": "1", "fixture_mode": "exact",
                       "roundtrip": "PASS"}.items():
        if complete.get(key) != value:
            bad.append(f"{label}: complete {key} differs")
    simplified = [{"symbol": str(row.get("symbol", "NONE")),
                   "algorithm": str(row.get("algorithm", "NONE")),
                   "status": str(row.get("status", "MISSING"))}
                  for row in records]
    report_nonmeasured(fmt, "sf-dense", simplified)
    measured = sum(row.get("status") == "MEASURED" for row in records)
    structural = sum(str(row.get("reason", "")).startswith("INADMISSIBLE")
                     for row in records)
    return bad, measured, structural


def validate_sf_grouped(fmt: Format, text: str) -> tuple[list[str], int, int]:
    label = f"{fmt.name}/sf-grouped"
    bad: list[str] = []
    shard_line = one_line(text, "SF_GROUPED_SHARD ", bad, label)
    complete_line = one_line(text, "SF_GROUPED_COMPLETE ", bad, label)
    cells = [fields(line) for line in text.splitlines()
             if line.startswith("SF_GROUPED_CELL ")]
    split_lines = [fields(line) for line in text.splitlines()
                   if line.startswith("SF_GROUPED_SPLITK ")]
    if len(cells) != 2:
        bad.append(f"{label}: cells={len(cells)}/2")
    expected_splits = {"2", "4", "8"}
    got_splits = {row.get("S") for row in split_lines
                  if row.get("q") == str(fmt.qtype) and
                  row.get("status") == "STRUCTURAL_UNAVAILABLE" and
                  row.get("reason") == "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER"}
    if len(split_lines) != 3 or got_splits != expected_splits:
        bad.append(f"{label}: explicit Split-K structural set differs")
    shard = fields(shard_line) if shard_line else {}
    for key, value in {
            "q": str(fmt.qtype), "layout": str(fmt.layout),
            "mapping_id": fmt.mapping_id, "typed_rows": "1",
            "selected_rows": "1", "router": "exact-rows-v1",
            "experts": "2", "total_rows": "9", "max_rows": "9",
            "active": "1", "empty": "1", "iterations": "1",
            "warmups": "1", "correctness_repeats": str(REPEATS),
            "roundtrip": "PASS"}.items():
        if shard.get(key) != value:
            bad.append(f"{label}: shard {key} differs")
    if {cell.get("algorithm") for cell in cells} != {
            "GROUPED_NONPERSISTENT", "GROUPED_PERSISTENT"}:
        bad.append(f"{label}: algorithm set differs")
    for cell in cells:
        algorithm = cell.get("algorithm", "NONE")
        for key, value in {
                "q": str(fmt.qtype), "layout": str(fmt.layout),
                "symbol": fmt.sfg_symbol,
                "config": f"8x64x{fmt.tile_k}_w8x16_s2",
                "scope": "FULL_OUTPUT"}.items():
            if cell.get(key) != value:
                bad.append(f"{label}/{algorithm}: {key} differs")
        measured_common(cell, f"{label}/{algorithm}", bad)
    complete = fields(complete_line) if complete_line else {}
    for key, value in {"q": str(fmt.qtype), "status": "PASS", "rows": "1",
                       "cells": "2", "measured": "2", "structural": "0",
                       "correctness": "RAW_FP16",
                       "splitk": "STRUCTURAL_UNAVAILABLE"}.items():
        if complete.get(key) != value:
            bad.append(f"{label}: complete {key} differs")
    report_nonmeasured(fmt, "sf-grouped", cells)
    measured = sum(cell.get("state") == "MEASURED" for cell in cells)
    structural = sum(cell.get("state") in {"SHARED_STORAGE", "OCCUPANCY"}
                     for cell in cells)
    return bad, measured, structural


VALIDATORS = {
    "fq-dense": validate_fq_dense,
    "fq-grouped": validate_fq_grouped,
    "sf-dense": validate_sf_dense,
    "sf-grouped": validate_sf_grouped,
}


def synthetic_fq_dense(fmt: Format) -> str:
    rows: list[str] = []
    for m in (7, 9):
        shape = f"{m}x64x512"
        rows += [
            f"FQ_KPACK4_FIXTURE phase=prepare q={fmt.qtype} shape={shape} "
            f"version=2 layout={fmt.layout} bits={fmt.low_bits} "
            f"high_bits={fmt.high_bits} artifact_tile_k=0 "
            f"transport_tile_k={fmt.transport_k} group_size={fmt.group_size} "
            f"reserved=0 mapping_id={fmt.mapping_id} direct_rc=0 abi_rc=0 "
            "direct_equal=1",
            f"FQ_KPACK4_FIXTURE phase=recover q={fmt.qtype} shape={shape} "
            f"mapping_id={fmt.mapping_id} direct_rc=0 abi_rc=0 direct_equal=1 "
            "native_equal=1",
            f"FQ_SHARD q={fmt.qtype} A=0 bchunk=0 shape={shape} "
            f"weight_layout={fmt.layout} weight_mapping_id={fmt.mapping_id} "
            "typed_rows=1 selected_rows=1 only_split=1 bc_mode=skip "
            "iterations=1 correctness_repeats=7",
            f"FQ_TC_CELL q={fmt.qtype} A=0 bchunk=0 shape={shape} "
            f"symbol={fmt.fqd_symbol} tm=8 tn=64 tk={fmt.tile_k} wm=8 wn=16 "
            "stages=2 provider=standard-aiu S=1 scope=FULL_OUTPUT "
            "resolved_delivery_n=16 state=MEASURED us=1 raw_bad=0 "
            "failure_step=NONE failure_repeat=-1 first_bad=18446744073709551615 "
            "first_want=0x0000 first_got=0x0000 partial_bytes=0 samples=[1]",
            f"FQ_SHAPE_DONE q={fmt.qtype} shape={shape} status=PASS",
        ]
    return "\n".join(rows) + "\n"


def synthetic_fq_grouped(fmt: Format) -> str:
    rows = [
        f"FQ_GROUPED_KPACK_SHARD q={fmt.qtype} layout={fmt.layout} "
        f"mapping_id={fmt.mapping_id} type_rows=2 selected_rows=2 "
        "router=exact-rows-v1 experts=2 total_rows=9 max_rows=9 active=1 empty=1 "
        "iterations=1 warmups=1 correctness_repeats=7 roundtrip=PASS metadata=PACKED_UNITS",
    ]
    for algorithm, reason in (
            ("SPLITK_S2", "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER"),
            ("SPLITK_S4", "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER"),
            ("SPLITK_S8", "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER"),
            ("BC_FULL_OUTPUT", "NO_CANONICAL_KPACK_BC_READER")):
        rows.append(f"FQ_GROUPED_KPACK_STRUCTURAL q={fmt.qtype} "
                    f"algorithm={algorithm} status=STRUCTURAL_UNAVAILABLE reason={reason}")
    for algorithm, symbol in zip(
            ("GROUPED_NONPERSISTENT", "GROUPED_PERSISTENT"), fmt.fqg_symbols):
        rows.append(
            f"FQ_GROUPED_KPACK_CELL q={fmt.qtype} layout={fmt.layout} "
            f"symbol={symbol} config=8x64x{fmt.tile_k}_w8x16_s2 "
            f"algorithm={algorithm} state=MEASURED raw_bad=0 "
            "first_bad=18446744073709551615 want=0x0000 got=0x0000 "
            "failure_repeat=-1 samples=[1]")
    rows.append(f"FQ_GROUPED_KPACK_COMPLETE q={fmt.qtype} status=PASS rows=2 "
                "cells=2 measured=2 structural=0 correctness=RAW_FP16")
    return "\n".join(rows) + "\n"


def synthetic_sf_dense(fmt: Format) -> str:
    rows = [
        f"SF_SHARD qtype={fmt.qtype} artifact_tile_k=0 bchunk=0 typed_rows=1 "
        f"weight_layout={fmt.layout} weight_mapping_id={fmt.mapping_id} "
        "selected_rows=1 algorithm_mask=0x3 iterations=1 correctness_repeats=7",
    ]
    for algorithm in ("NONPERSISTENT", "PERSISTENT"):
        occupancy = 0 if algorithm == "NONPERSISTENT" else 10
        mask = "0x0" if occupancy == 0 else hex((1 << (occupancy + 1)) - 2)
        record = {
            "shape": "7x256x512", "qtype": fmt.qtype, "artifact_tile_k": 0,
            "bchunk": 0, "symbol": fmt.sfd_symbol, "a_provider": 0,
            "resolved_delivery_n": 16,
            "config": f"8x64x{fmt.tile_k}_w8x16_s2_bc0_ap0_dn16",
            "algorithm": algorithm, "metric_scope": "FULL_OUTPUT", "split": 1,
            "policy": ("ordinary" if algorithm == "NONPERSISTENT" else
                       "capacity+balanced"),
            "grid": 4, "occupancy": occupancy,
            "capacity_b_mask": mask, "balanced_b_mask": mask,
            "status": "MEASURED", "reason": "MEASURED", "sample": 0,
            "sample_us": 1.0, "raw_bad": 0,
            "reducer_correctness_untimed": 0, "execution_ordinal": 0,
        }
        rows.append("SF_CELL " + json.dumps(record, separators=(",", ":")))
    rows.append("SF_COMPLETE status=COMPLETE shape=7x256x512 typed_rows=1 "
                "runtime_cells=2 measured_cells=2 records=2 iterations=1 "
                "fixture_mode=exact roundtrip=PASS")
    return "\n".join(rows) + "\n"


def synthetic_sf_grouped(fmt: Format) -> str:
    rows = [
        f"SF_GROUPED_SHARD q={fmt.qtype} layout={fmt.layout} "
        f"mapping_id={fmt.mapping_id} typed_rows=1 selected_rows=1 "
        "router=exact-rows-v1 experts=2 total_rows=9 max_rows=9 active=1 empty=1 "
        "iterations=1 warmups=1 correctness_repeats=7 roundtrip=PASS",
    ]
    for split in (2, 4, 8):
        rows.append(f"SF_GROUPED_SPLITK q={fmt.qtype} S={split} "
                    "status=STRUCTURAL_UNAVAILABLE "
                    "reason=NO_GROUPED_SPLITK_KERNEL_OR_REDUCER")
    for algorithm in ("GROUPED_NONPERSISTENT", "GROUPED_PERSISTENT"):
        rows.append(
            f"SF_GROUPED_CELL q={fmt.qtype} layout={fmt.layout} "
            f"symbol={fmt.sfg_symbol} config=8x64x{fmt.tile_k}_w8x16_s2 "
            f"scope=FULL_OUTPUT algorithm={algorithm} state=MEASURED raw_bad=0 "
            "first_bad=18446744073709551615 want=0x0000 got=0x0000 "
            "failure_repeat=-1 samples=[1]")
    rows.append(f"SF_GROUPED_COMPLETE q={fmt.qtype} status=PASS rows=1 cells=2 "
                "measured=2 structural=0 correctness=RAW_FP16 "
                "splitk=STRUCTURAL_UNAVAILABLE")
    return "\n".join(rows) + "\n"


SYNTHETIC = {
    "fq-dense": synthetic_fq_dense,
    "fq-grouped": synthetic_fq_grouped,
    "sf-dense": synthetic_sf_dense,
    "sf-grouped": synthetic_sf_grouped,
}


def validate_build(fmt: Format, run_dir: Path, source: str) -> list[str]:
    bad: list[str] = []
    generated = run_dir / "generated" / f"q{fmt.qtype}"
    families = {
        "fq": {
            "targets": (
                "test_fully_quantized_internal_sweep",
                "test_fully_quantized_grouped_kpack_discovery",
            ),
            "expected": {
                "FQ_SWEEP_GENERATED_DIR": generated / "fq-dense",
                "FQ_SWEEP_QTYPE": fmt.qtype,
                "FQ_SWEEP_PACKED_FORMAT": fmt.packed_format,
                "FQ_SWEEP_WEIGHT_LAYOUT": fmt.layout,
                "FQ_GROUPED_KPACK_GENERATED_DIR": generated / "fq-grouped",
                "FQ_GROUPED_KPACK_QTYPE": fmt.qtype,
                "FQ_GROUPED_KPACK_PACKED_FORMAT": fmt.packed_format,
                "FQ_GROUPED_KPACK_WEIGHT_LAYOUT": fmt.layout,
            },
            "forbidden": ("SCALEFIRST_SWEEP_GENERATED_DIR",
                          "SCALEFIRST_GROUPED_KPACK_GENERATED_DIR",
                          "FQ_A02_Q3_GENERATED_DIR", "FQ_KQUANT_PERF_QTYPE"),
        },
        "sf": {
            "targets": (
                "test_scalefirst_internal_sweep",
                "test_scalefirst_grouped_kpack_discovery",
            ),
            "expected": {
                "SCALEFIRST_SWEEP_GENERATED_DIR": generated / "sf-dense",
                "SCALEFIRST_SWEEP_QTYPE": fmt.qtype,
                "SCALEFIRST_SWEEP_WEIGHT_LAYOUT": fmt.layout,
                "SCALEFIRST_GROUPED_KPACK_GENERATED_DIR": generated / "sf-grouped",
                "SCALEFIRST_GROUPED_KPACK_QTYPE": fmt.qtype,
                "SCALEFIRST_GROUPED_KPACK_WEIGHT_LAYOUT": fmt.layout,
            },
            "forbidden": ("FQ_SWEEP_GENERATED_DIR",
                          "FQ_GROUPED_KPACK_GENERATED_DIR",
                          "FQ_A02_Q3_GENERATED_DIR", "FQ_KQUANT_PERF_QTYPE"),
        },
    }
    for family, contract in families.items():
        build = run_dir / "build" / f"q{fmt.qtype}" / family
        marker = build / ".quactlize-source-head"
        try:
            if not marker.is_file() or marker.is_symlink():
                bad.append(f"{fmt.name}/{family}: source authority is not regular")
            elif marker.read_text() != source + "\n":
                bad.append(f"{fmt.name}/{family}: build source authority differs")
        except OSError as error:
            bad.append(f"{fmt.name}/{family}: cannot read build authority: {error}")
        if (build / ".quactlize-source-dirty").exists():
            bad.append(f"{fmt.name}/{family}: build records tracked source dirt")
        for target in contract["targets"]:
            binary = build / "ppu_targets" / target
            if (not binary.is_file() or binary.is_symlink() or
                    binary.stat().st_size == 0 or
                    not binary.stat().st_mode & 0o111):
                bad.append(f"{fmt.name}/{family}: missing regular binary {target}")
        try:
            cache = (build / "CMakeCache.txt").read_text()
        except OSError as error:
            bad.append(f"{fmt.name}/{family}: cannot read CMake cache: {error}")
            continue
        for key, value in contract["expected"].items():
            expected = re.escape(str(value))
            if not re.search(rf"^{re.escape(key)}(?::[^=]+)?={expected}$", cache, re.M):
                bad.append(f"{fmt.name}/{family}: CMake cache lacks {key}={value}")
        for key in contract["forbidden"]:
            if re.search(rf"^{re.escape(key)}(?::[^=]+)?=.+$", cache, re.M):
                bad.append(f"{fmt.name}/{family}: foreign family cache key {key}")
    return bad


def build_source_authority(run_dir: Path) -> tuple[str, list[str]]:
    bad: list[str] = []
    values: dict[str, list[str]] = {}
    for fmt in FORMATS:
        for family in ("fq", "sf"):
            path = (run_dir / "build" / f"q{fmt.qtype}" / family /
                    ".quactlize-source-head")
            try:
                if not path.is_file() or path.is_symlink():
                    bad.append(
                        f"{fmt.name}/{family} source authority is not regular")
                    continue
                raw = path.read_text()
            except OSError as error:
                bad.append(f"cannot read {fmt.name}/{family} source authority: {error}")
                continue
            if not re.fullmatch(r"[0-9a-f]{40}\n", raw):
                bad.append(f"{fmt.name}/{family} source authority is malformed")
                continue
            value = raw[:-1]
            values.setdefault(value, []).append(f"{fmt.name}/{family}")
    if len(values) != 1:
        bad.append(f"build source authority set differs: {sorted(values)}")
        return "", bad
    source = next(iter(values))
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        bad.append(f"malformed build source authority: {source!r}")
        return "", bad
    return source, bad


def source_reuse_errors(build_source: str) -> list[str]:
    bad: list[str] = []
    try:
        current = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain",
             "--untracked-files=no", "--", *RELEVANT_SOURCE_PATHS],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return [f"cannot resolve runner source authority: {error}"]
    if dirty:
        bad.append(f"orchestration source is dirty: {dirty!r}")
    try:
        build_actlize = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse",
             f"{build_source}:third_party/actlize"], text=True).strip()
        current_actlize = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD:third_party/actlize"],
            text=True).strip()
        checkout_actlize = subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party/actlize"),
             "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        bad.append(f"cannot resolve actlize source authority: {error}")
        return bad
    if len({build_actlize, current_actlize, checkout_actlize}) != 1:
        bad.append("actlize build/current/checkout authorities differ")
    if build_source == current:
        return bad
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
         build_source, current], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False)
    if ancestor.returncode != 0:
        bad.append("build source is not an ancestor of runner source")
        return bad
    try:
        changed = set(filter(None, subprocess.check_output(
            ["git", "-C", str(ROOT), "diff", "--name-only",
             f"{build_source}..{current}"], text=True).splitlines()))
        touched = set(filter(None, subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "--format=", "--name-only",
             f"{build_source}..{current}"], text=True).splitlines()))
    except (OSError, subprocess.CalledProcessError) as error:
        bad.append(f"cannot resolve build-to-runner source delta: {error}")
        return bad
    bad += transition_path_errors(changed, touched)
    return bad


def transition_path_errors(changed: set[str], touched: set[str]) -> list[str]:
    bad: list[str] = []
    unexpected = touched - ORCHESTRATION_PATHS
    if unexpected:
        bad.append("build-to-runner commit range touched compiled inputs: " +
                   ",".join(sorted(unexpected)))
    if changed != ORCHESTRATION_PATHS:
        bad.append("build-to-runner final delta is not the exact orchestration pair: " +
                   ",".join(sorted(changed)))
    return bad


def validate_build_set(run_dir: Path) -> tuple[list[str], str]:
    source, bad = build_source_authority(run_dir)
    if not source:
        return bad, source
    bad += source_reuse_errors(source)
    for fmt in FORMATS:
        bad += validate_manifest_set(fmt, run_dir)
        bad += validate_build(fmt, run_dir, source)
    return bad, source


def validate_run_dir(run_dir: Path) -> list[str]:
    bad, _ = validate_build_set(run_dir)
    actual_logs = {path.name for path in (run_dir / "results").glob("q*-*.run.log")}
    expected_logs = {f"q{fmt.qtype}-{route}.run.log"
                     for fmt in FORMATS for route in ROUTES}
    if actual_logs != expected_logs:
        bad.append(f"run log set differs: got={sorted(actual_logs)}")

    total_measured = 0
    total_structural = 0
    declared_out_of_scope = 0
    for fmt in FORMATS:
        for route in ROUTES:
            rc_path = run_dir / "results" / f"q{fmt.qtype}-{route}.rc"
            log_path = run_dir / "results" / f"q{fmt.qtype}-{route}.run.log"
            try:
                rc = rc_path.read_text().strip()
                text = log_path.read_text()
            except OSError as error:
                bad.append(f"{fmt.name}/{route}: missing result: {error}")
                continue
            if rc != "0":
                bad.append(f"{fmt.name}/{route}: process rc={rc!r}")
            errors, measured, structural = VALIDATORS[route](fmt, text)
            bad += errors
            total_measured += measured
            total_structural += structural
            declared = 4 if route == "fq-grouped" else 3 if route == "sf-grouped" else 0
            declared_out_of_scope += declared
            print(f"M8N16_CROSS_FORMAT_ROUTE q={fmt.qtype} format={fmt.name} "
                  f"route={route} expected=2 measured={measured} "
                  f"structural={structural} out_of_scope_structural={declared} "
                  f"status={'PASS' if not errors and rc == '0' else 'FAIL'}")
    if total_measured != 40:
        bad.append(f"measured denominator={total_measured}/40")
    if total_structural != 0:
        bad.append(f"affected full-output structural cells={total_structural}/0")
    if declared_out_of_scope != 35:
        bad.append(f"declared out-of-scope structural cells={declared_out_of_scope}/35")
    return bad


def self_test() -> None:
    bad = audit_runner(RUNNER.read_text())
    if bad:
        raise AssertionError("; ".join(bad))
    for fmt in FORMATS:
        for route in ROUTES:
            text = SYNTHETIC[route](fmt)
            errors, measured, structural = VALIDATORS[route](fmt, text)
            if errors or measured != 2 or structural:
                raise AssertionError(
                    f"synthetic positive {fmt.name}/{route} failed: {errors}")

    fmt = FORMATS[2]
    plants = (
        ("fq dense raw", "fq-dense", "raw_bad=0", "raw_bad=64"),
        ("fq dense split", "fq-dense", "only_split=1", "only_split=4"),
        ("fq dense symbol", "fq-dense", fmt.fqd_symbol, fmt.fqd_symbol + "_bad"),
        ("fq grouped state", "fq-grouped", "state=MEASURED", "state=OCCUPANCY"),
        ("fq grouped structural", "fq-grouped", "status=STRUCTURAL_UNAVAILABLE",
         "status=AVAILABLE"),
        ("fq grouped symbol", "fq-grouped", fmt.fqg_symbols[0],
         fmt.fqg_symbols[0] + "_bad"),
        ("sf dense status", "sf-dense", '"status":"MEASURED"',
         '"status":"INADMISSIBLE"'),
        ("sf dense algorithm", "sf-dense", '"algorithm":"NONPERSISTENT"',
         '"algorithm":"PERSISTENT"'),
        ("sf dense policy", "sf-dense", '"policy":"capacity+balanced"',
         '"policy":"capacity"'),
        ("sf dense symbol", "sf-dense", fmt.sfd_symbol, fmt.sfd_symbol + "_bad"),
        ("sf grouped state", "sf-grouped", "state=MEASURED", "state=OCCUPANCY"),
        ("sf grouped structural", "sf-grouped", "status=STRUCTURAL_UNAVAILABLE",
         "status=AVAILABLE"),
        ("sf grouped symbol", "sf-grouped", fmt.sfg_symbol,
         fmt.sfg_symbol + "_bad"),
    )
    rejected = 0
    for name, route, old, new in plants:
        text = SYNTHETIC[route](fmt)
        if text.count(old) < 1:
            raise AssertionError(f"cannot plant {name}")
        with contextlib.redirect_stdout(io.StringIO()):
            errors, measured, structural = VALIDATORS[route](
                fmt, text.replace(old, new, 1))
        if not errors and measured == 2 and structural == 0:
            raise AssertionError(f"checker accepted planted {name}")
        rejected += 1

    runner = RUNNER.read_text()
    runner_plants = (
        ("repeat", "--correctness-repeats=7", "--correctness-repeats=1"),
        ("broad split", "--only-split=1", "--only-split=4"),
        ("layout", "12) FORMAT_NAME=Q4_K; PACKED_FORMAT=0; WEIGHT_LAYOUT=1;",
         "12) FORMAT_NAME=Q4_K; PACKED_FORMAT=0; WEIGHT_LAYOUT=2;"),
        ("parallel build", 'build_family "$q" "$family" &',
         'build_family "$q" "$family"'),
        ("family isolation", 'local build="$RUN_DIR/build/q$q/$family"',
         'local build="$RUN_DIR/build/q$q"'),
        ("ScaleFirst aligned shipping shape", "--shape=7x256x512",
         "--shape=7x64x512"),
        ("resume build authority", '--validate-build-dir "$RUN_DIR"',
         '--validate-generated-dir "$RUN_DIR"'),
        ("resume artifact snapshot", 'cmp -s "$resume_snapshot"',
         'test -s "$resume_snapshot"'),
        ("denominator", "cells=40 measured=40", "cells=39 measured=39"),
    )
    for name, old, new in runner_plants:
        if runner.count(old) < 1:
            raise AssertionError(f"cannot plant runner {name}")
        if not audit_runner(runner.replace(old, new, 1)):
            raise AssertionError(f"runner checker accepted planted {name}")
        rejected += 1

    if transition_path_errors(set(ORCHESTRATION_PATHS),
                              set(ORCHESTRATION_PATHS)):
        raise AssertionError("exact orchestration-only transition was rejected")
    for name, changed, touched in (
            ("compiled-history", set(ORCHESTRATION_PATHS),
             set(ORCHESTRATION_PATHS) | {"quactlize/include/kernel.cuh"}),
            ("incomplete-final-delta",
             {"tools/run_m8n16_cross_format_correctness_box.sh"},
             set(ORCHESTRATION_PATHS))):
        if not transition_path_errors(changed, touched):
            raise AssertionError(f"transition checker accepted {name}")
        rejected += 1

    # The runtime checker must bind each format to two independent CMake
    # caches.  A single-tree synthetic positive and runner text auditing are
    # not enough: either could drift while still accepting the duplicate-rule
    # topology that this gate is meant to exclude.
    with tempfile.TemporaryDirectory(prefix="m8n16-cross-format-") as tmp:
        run_dir = Path(tmp)
        source = "0123456789abcdef"
        cache_rows = {
            "fq": (
                f"FQ_SWEEP_GENERATED_DIR:UNINITIALIZED={run_dir}/generated/q{fmt.qtype}/fq-dense",
                f"FQ_SWEEP_QTYPE:UNINITIALIZED={fmt.qtype}",
                f"FQ_SWEEP_PACKED_FORMAT:UNINITIALIZED={fmt.packed_format}",
                f"FQ_SWEEP_WEIGHT_LAYOUT:UNINITIALIZED={fmt.layout}",
                f"FQ_GROUPED_KPACK_GENERATED_DIR:UNINITIALIZED={run_dir}/generated/q{fmt.qtype}/fq-grouped",
                f"FQ_GROUPED_KPACK_QTYPE:UNINITIALIZED={fmt.qtype}",
                f"FQ_GROUPED_KPACK_PACKED_FORMAT:UNINITIALIZED={fmt.packed_format}",
                f"FQ_GROUPED_KPACK_WEIGHT_LAYOUT:UNINITIALIZED={fmt.layout}",
            ),
            "sf": (
                f"SCALEFIRST_SWEEP_GENERATED_DIR:UNINITIALIZED={run_dir}/generated/q{fmt.qtype}/sf-dense",
                f"SCALEFIRST_SWEEP_QTYPE:UNINITIALIZED={fmt.qtype}",
                f"SCALEFIRST_SWEEP_WEIGHT_LAYOUT:UNINITIALIZED={fmt.layout}",
                f"SCALEFIRST_GROUPED_KPACK_GENERATED_DIR:UNINITIALIZED={run_dir}/generated/q{fmt.qtype}/sf-grouped",
                f"SCALEFIRST_GROUPED_KPACK_QTYPE:UNINITIALIZED={fmt.qtype}",
                f"SCALEFIRST_GROUPED_KPACK_WEIGHT_LAYOUT:UNINITIALIZED={fmt.layout}",
            ),
        }
        targets = {
            "fq": ("test_fully_quantized_internal_sweep",
                   "test_fully_quantized_grouped_kpack_discovery"),
            "sf": ("test_scalefirst_internal_sweep",
                   "test_scalefirst_grouped_kpack_discovery"),
        }
        for family in ("fq", "sf"):
            build = run_dir / "build" / f"q{fmt.qtype}" / family
            (build / "ppu_targets").mkdir(parents=True)
            (build / ".quactlize-source-head").write_text(source + "\n")
            (build / "CMakeCache.txt").write_text(
                "\n".join(cache_rows[family]) + "\n")
            for target in targets[family]:
                binary = build / "ppu_targets" / target
                binary.write_bytes(b"binary")
                binary.chmod(0o755)
        errors = validate_build(fmt, run_dir, source)
        if errors:
            raise AssertionError(f"isolated build-tree positive failed: {errors}")

        sf_cache = run_dir / "build" / f"q{fmt.qtype}" / "sf" / "CMakeCache.txt"
        sf_cache.write_text(sf_cache.read_text().replace(
            f"{run_dir}/generated/q{fmt.qtype}/sf-dense", "/wrong/sf-dense"))
        if not any("CMake cache lacks SCALEFIRST_SWEEP_GENERATED_DIR" in error
                   for error in validate_build(fmt, run_dir, source)):
            raise AssertionError("build checker accepted a wrong generated directory")
        rejected += 1

        fq_cache = run_dir / "build" / f"q{fmt.qtype}" / "fq" / "CMakeCache.txt"
        fq_cache.write_text(fq_cache.read_text() +
                            "SCALEFIRST_SWEEP_GENERATED_DIR:UNINITIALIZED=/wrong\n")
        if not any("foreign family cache key" in error
                   for error in validate_build(fmt, run_dir, source)):
            raise AssertionError("build checker accepted a foreign family cache key")
        rejected += 1

        missing = (run_dir / "build" / f"q{fmt.qtype}" / "sf" /
                   "ppu_targets" / "test_scalefirst_internal_sweep")
        missing.unlink()
        if not any("missing regular binary" in error
                   for error in validate_build(fmt, run_dir, source)):
            raise AssertionError("build checker accepted a missing family binary")
        rejected += 1

        dirty = run_dir / "build" / f"q{fmt.qtype}" / "fq" / ".quactlize-source-dirty"
        dirty.write_text("tracked change\n")
        if not any("tracked source dirt" in error
                   for error in validate_build(fmt, run_dir, source)):
            raise AssertionError("build checker accepted a dirty build authority")
        rejected += 1
    print("[m8n16-cross-format:self-test] PASS formats=5 routes=20 "
          f"cells=40 exact-TM8/WN16; structural fail-close; {rejected} negatives RED")


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-run-dir", type=Path)
    modes.add_argument("--validate-generated-dir", type=Path)
    modes.add_argument("--validate-build-dir", type=Path)
    args = parser.parse_args()
    try:
        if (args.validate_run_dir is None and
                args.validate_generated_dir is None and
                args.validate_build_dir is None):
            self_test()
            return 0
        if args.validate_generated_dir is not None:
            root = args.validate_generated_dir.resolve()
            bad = [error for fmt in FORMATS
                   for error in validate_manifest_set(fmt, root)]
            if bad:
                print("[m8n16-cross-format] FAIL: " + "; ".join(bad))
                return 1
            print("[m8n16-cross-format] validated 5 formats / 20 generated "
                  "manifests before isolated-family builds")
            return 0
        if args.validate_build_dir is not None:
            root = args.validate_build_dir.resolve()
            bad, source = validate_build_set(root)
            if bad:
                print("[m8n16-cross-format] FAIL: " + "; ".join(bad))
                return 1
            current = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                text=True).strip()
            print("[m8n16-cross-format] reusable builds validated "
                  f"formats=5 trees=10 build_source={source} "
                  f"runner_source={current}")
            return 0
        bad = validate_run_dir(args.validate_run_dir.resolve())
        if bad:
            print("[m8n16-cross-format] FAIL: " + "; ".join(bad))
            return 1
        print("[m8n16-cross-format] validated 5 formats / 20 routes / "
              "40 measured full-output cells; affected structural=0; "
              "out-of-scope structural=35")
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[m8n16-cross-format] FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
