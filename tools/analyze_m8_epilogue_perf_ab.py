#!/usr/bin/env python3
"""Fail-closed analysis for the TM8 epilogue pre/post performance A/B."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools/build_m8_epilogue_perf_ab_bundle.sh"
RUNNER = ROOT / "tools/run_m8_epilogue_perf_ab_prebuilt_box.sh"
sys.path.insert(0, str(ROOT / "tools"))

import gen_fully_quantized_kpack_discovery_units as generator  # noqa: E402


SCHEMA = "quactlize.m8-epilogue-perf-ab-bundle.v1"
RESULT_SCHEMA = "quactlize.m8-epilogue-perf-ab-result.v1"
BASELINE_SOURCE = "a0fa8d03013d3cd0bc340e876cb0d646f3cfb72d"
BASELINE_TREE = "1bdee8cc1386ca29e8116d893b63f705b118eecc"
BASELINE_PARENT = "6ec447ac25477a40a29be8c2809a933c84d0b7ad"
BASELINE_ACTLIZE = "9d063e4c5fde5119d4d68bfbe124aacd8ed2ec88"
CANDIDATE_SOURCE = "6ec447ac25477a40a29be8c2809a933c84d0b7ad"
CANDIDATE_ACTLIZE = "423253c00df333ead6fb72ea623d526f24f56b5a"
CUTLASS_SOURCE = "f94ec46f4f63f96003d6cfdf2014731e7672c281"
SYMBOL = "fqk_tc_q12_l1_a0_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0_dn16"
SHAPE = (8, 3072, 512)
SHAPE_TEXT = "8x3072x512"
ITERATIONS = 31
ROUNDS = 6
CORRECTNESS_REPEATS = 7
SCHEDULE_SEED = "0x6a09e667f3bcc909"
MAPPING_ID = "0x51344b5034540001"
ARMS = ("baseline", "candidate")


class AnalysisError(ValueError):
    pass


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exactly_one(text: str, prefix: str) -> dict[str, str]:
    rows = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise AnalysisError(f"{prefix.strip()} denominator is {len(rows)}/1")
    return {token.split("=", 1)[0]: token.split("=", 1)[1]
            for token in rows[0].split()[1:] if "=" in token}


def parse_samples(text: str) -> list[float]:
    if not text.startswith("[") or not text.endswith("]"):
        raise AnalysisError("sample vector is malformed")
    values = text[1:-1]
    if not values:
        return []
    try:
        result = [float(value) for value in values.split(",")]
    except ValueError as error:
        raise AnalysisError("sample vector is not numeric") from error
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise AnalysisError("sample vector contains non-finite/non-positive timing")
    return result


def expected_order(round_index: int) -> tuple[str, str]:
    return ARMS if round_index % 2 else tuple(reversed(ARMS))


def regular_file(path: pathlib.Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError(f"required regular file is missing or symlinked: {path}")


def validate_generated(path: pathlib.Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    generator.validate_manifest(document)
    expected_range = {"begin": 4827, "end": 4828, "count": 1,
                      "authority_count": 6120}
    rows = document.get("dense_tc_parents") or []
    if document.get("schema") != "quactlize.fully_quantized_kpack_dense_shard.v2" or \
            document.get("parent_range") != expected_range or len(rows) != 1:
        raise AnalysisError("generated one-parent authority differs")
    row = rows[0]
    expected = {
        "parent_ordinal": 4827, "symbol": SYMBOL, "qtype": 12,
        "weight_layout": 1, "artifact_tile_k": 0, "bchunk": 0,
        "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
        "warp_m": 8, "warp_n": 16, "stages": 2,
        "a_provider": 0, "a_provider_name": "standard-aiu",
        "resolved_delivery_n": 16,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise AnalysisError(f"generated canonical row differs: {row}")
    return document


def make_bundle_manifest(bundle: pathlib.Path, sdk: pathlib.Path,
                         baseline: pathlib.Path,
                         candidate: pathlib.Path) -> None:
    inputs = bundle / "inputs"
    generated = inputs / "generated-manifest.json"
    registry = inputs / "fq_tc_registry.inc"
    unit = inputs / "fq_kpack_dense_unit_00000.cu"
    symbol = inputs / "symbol.txt"
    for path in (generated, registry, unit, symbol, baseline, candidate,
                 sdk / "bin/hgcc", sdk / "bin/hgobjdump"):
        regular_file(path)
    validate_generated(generated)
    if symbol.read_text() != SYMBOL + "\n":
        raise AnalysisError("bundle symbol file differs")
    compiler = subprocess.check_output(
        [str(sdk / "bin/hgcc"), "--version"], text=True,
        stderr=subprocess.STDOUT).splitlines()[0]
    inspector = subprocess.check_output(
        [str(sdk / "bin/hgobjdump"), "--version"], text=True,
        stderr=subprocess.STDOUT).splitlines()[0]
    value = {
        "schema": SCHEMA,
        "experiment": {
            "shape": list(SHAPE), "canonical_symbol": SYMBOL,
            "qtype": 12, "layout": 1, "mapping_id": MAPPING_ID,
            "provider": "standard-aiu", "provider_id": 0,
            "split": 1, "artifact_tile_k": 0, "bchunk": 0,
            "delivery_n": 16, "iterations": ITERATIONS,
            "rounds": ROUNDS, "correctness_repeats": CORRECTNESS_REPEATS,
            "schedule_seed": SCHEDULE_SEED,
        },
        "arms": {
            "baseline": {
                "source_sha": BASELINE_SOURCE,
                "actlize_sha": BASELINE_ACTLIZE,
                "cutlass_sha": CUTLASS_SOURCE,
                "synthetic_source": {
                    "parent_sha": BASELINE_PARENT,
                    "tree_sha": BASELINE_TREE,
                    "only_change": "third_party/actlize",
                },
                "path": "bin/baseline",
                "size": baseline.stat().st_size,
                "sha256": sha256(baseline),
            },
            "candidate": {
                "source_sha": CANDIDATE_SOURCE,
                "actlize_sha": CANDIDATE_ACTLIZE,
                "cutlass_sha": CUTLASS_SOURCE,
                "path": "bin/candidate",
                "size": candidate.stat().st_size,
                "sha256": sha256(candidate),
            },
        },
        "inputs": {
            path.name: {"path": f"inputs/{path.name}",
                        "size": path.stat().st_size,
                        "sha256": sha256(path)}
            for path in (generated, registry, unit, symbol)
        },
        "build_sdk": {
            "root": str(sdk), "compiler_identity": compiler,
            "compiler_sha256": sha256(sdk / "bin/hgcc"),
            "inspector_identity": inspector,
            "inspector_sha256": sha256(sdk / "bin/hgobjdump"),
        },
    }
    (bundle / "manifest.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(f"[m8-epilogue-perf-ab-bundle] manifest={bundle / 'manifest.json'}")


def load_bundle(bundle: pathlib.Path) -> dict[str, Any]:
    manifest = bundle / "manifest.json"
    regular_file(manifest)
    value = json.loads(manifest.read_text())
    expected_experiment = {
        "shape": list(SHAPE), "canonical_symbol": SYMBOL,
        "qtype": 12, "layout": 1, "mapping_id": MAPPING_ID,
        "provider": "standard-aiu", "provider_id": 0,
        "split": 1, "artifact_tile_k": 0, "bchunk": 0,
        "delivery_n": 16, "iterations": ITERATIONS,
        "rounds": ROUNDS, "correctness_repeats": CORRECTNESS_REPEATS,
        "schedule_seed": SCHEDULE_SEED,
    }
    if value.get("schema") != SCHEMA or \
            value.get("experiment") != expected_experiment:
        raise AnalysisError("bundle experiment identity differs")
    expected_sources = {
        "baseline": (BASELINE_SOURCE, BASELINE_ACTLIZE),
        "candidate": (CANDIDATE_SOURCE, CANDIDATE_ACTLIZE),
    }
    arms = value.get("arms") or {}
    if set(arms) != set(ARMS):
        raise AnalysisError("bundle arm set differs")
    for name, (source, actlize) in expected_sources.items():
        record = arms[name]
        if (record.get("source_sha"), record.get("actlize_sha"),
                record.get("cutlass_sha")) != (source, actlize, CUTLASS_SOURCE):
            raise AnalysisError(f"{name} source authority differs")
        if name == "baseline" and record.get("synthetic_source") != {
                "parent_sha": BASELINE_PARENT, "tree_sha": BASELINE_TREE,
                "only_change": "third_party/actlize"}:
            raise AnalysisError("baseline synthetic source recipe differs")
        if name == "candidate" and "synthetic_source" in record:
            raise AnalysisError("candidate unexpectedly names a synthetic source")
        path = bundle / record.get("path", "")
        regular_file(path)
        if path.stat().st_size != record.get("size") or \
                sha256(path) != record.get("sha256"):
            raise AnalysisError(f"{name} binary identity differs")
    inputs = value.get("inputs") or {}
    expected_inputs = {"generated-manifest.json", "fq_tc_registry.inc",
                       "fq_kpack_dense_unit_00000.cu", "symbol.txt"}
    if set(inputs) != expected_inputs:
        raise AnalysisError("bundle input set differs")
    for name, record in inputs.items():
        path = bundle / record.get("path", "")
        regular_file(path)
        if path.stat().st_size != record.get("size") or \
                sha256(path) != record.get("sha256"):
            raise AnalysisError(f"bundle input identity differs: {name}")
    validate_generated(bundle / "inputs/generated-manifest.json")
    if (bundle / "inputs/symbol.txt").read_text() != SYMBOL + "\n":
        raise AnalysisError("bundle canonical symbol differs")
    return value


def parse_list_elf(text: str) -> list[str]:
    symbols = re.findall(r"^.*Func\s+\d+:\s*(\S+).*$", text, re.MULTILINE)
    if not symbols:
        raise AnalysisError("hgobjdump -lelf exposed no functions")
    return symbols


def choose_s1(symbols: list[str]) -> str:
    candidates = [symbol for symbol in symbols
                  if "device_kernel" in symbol and
                  "gemm6kernel13GemmUniversal" in symbol and
                  "GemmUniversalMixedInputSplitKParallel" not in symbol and
                  "KernelAiuQ4KPack4Transpose" in symbol and
                  "PPU0010_8x16x16_F32F16F16F32_TN" in symbol and
                  "SplitKSerialScheduler" in symbol]
    if len(candidates) != 1:
        raise AnalysisError(f"S1 kernel denominator is {len(candidates)}/1")
    return candidates[0]


def select_symbol(list_elf: pathlib.Path, symbol_out: pathlib.Path,
                  demangled_out: pathlib.Path) -> None:
    symbol = choose_s1(parse_list_elf(list_elf.read_text(errors="replace")))
    demangled = subprocess.check_output(
        ["c++filt", symbol], text=True, stderr=subprocess.STDOUT).strip()
    symbol_out.write_text(symbol + "\n")
    demangled_out.write_text(demangled + "\n")
    print(f"[m8-epilogue-perf-ab-symbol] S1=1 symbol={symbol}")


def parse_instructions(text: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    pattern = re.compile(
        r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+){8}\s*([^\s]+)(?:\s+(.*))?$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            result.append((match.group(1).lower(), (match.group(2) or "").lower()))
    if not result:
        raise AnalysisError("exact S1 disassembly has no instructions")
    return result


def one_resource(text: str, label: str) -> int:
    values = {int(value) for value in re.findall(
        rf"(?im)^\s*{re.escape(label)}\s*:\s*(\d+)\s*$", text)}
    if len(values) != 1:
        raise AnalysisError(f"resource {label} denominator is {len(values)}/1")
    return next(iter(values))


def count_like(counter: collections.Counter[str], needle: str) -> int:
    return sum(count for opcode, count in counter.items() if needle in opcode)


def codegen(arm: str, line_path: pathlib.Path, resource_path: pathlib.Path,
            symbol_path: pathlib.Path, binary: pathlib.Path,
            output: pathlib.Path) -> None:
    if arm not in ARMS:
        raise AnalysisError(f"unknown arm {arm}")
    symbol = symbol_path.read_text().strip()
    widths = [int(value) for value in re.findall(
        r"thread17LinearCombinationISO_Li(\d+)E", symbol)]
    alignments = [int(value) for value in re.findall(
        r"AutoVectorizingCopyWithAssumedAlignmentILi(\d+)E", symbol)]
    # The Itanium ABI substitutes the second copy-atom type, so its alignment
    # is not reliably adjacent to the type name.  Prove R2G width from ISA.
    if widths != [4] or alignments != [128]:
        raise AnalysisError(
            f"{arm} epilogue type differs: values={widths}, alignments={alignments}")
    instructions = parse_instructions(line_path.read_text(errors="replace"))
    counts = collections.Counter(opcode for opcode, _ in instructions)
    wait_tsm = sum(1 for opcode, operands in instructions
                   if opcode == "s.wait" and "tsmcnt" in operands)
    spill_ops = sum(count for opcode, count in counts.items()
                    if "spill" in opcode or "scratch" in opcode or
                    ".local" in opcode)
    focus = {
        "mma": count_like(counts, ".mma"),
        "aiu_load": count_like(counts, "aiu.ld"),
        "tsm_load_total": count_like(counts, "tsm.ld"),
        "tsm_store_total": count_like(counts, "tsm.st"),
        "tsm_ld_b32x4": counts.get("tsm.ld.b32x4", 0),
        "tsm_ld_swzl_b32x4": count_like(counts, "tsm.ld.swzl.b32x4"),
        "r2g_vmem_st_b32x2": counts.get("vmem.st.b32x2", 0),
        "r2g_vmem_st_b32x4": counts.get("vmem.st.b32x4", 0),
        "vmem_store_total": count_like(counts, "vmem.st"),
        "wait_tsm": wait_tsm,
        "explicit_spill_or_local_ops": spill_ops,
    }
    if focus["mma"] <= 0 or focus["aiu_load"] <= 0 or \
            focus["tsm_ld_b32x4"] <= 0:
        raise AnalysisError(f"{arm} required mainloop/S2R opcodes absent: {focus}")
    expected_r2g = (0, 2) if arm == "baseline" else (2, 0)
    observed_r2g = (focus["r2g_vmem_st_b32x2"],
                    focus["r2g_vmem_st_b32x4"])
    if observed_r2g != expected_r2g:
        raise AnalysisError(
            f"{arm} exact R2G ISA differs: x2/x4={observed_r2g}, "
            f"expected={expected_r2g}")
    r2g_bits = 128 if arm == "baseline" else 64
    resource = resource_path.read_text(errors="replace")
    value = {
        "arm": arm, "binary_sha256": sha256(binary), "symbol": symbol,
        "instruction_total": len(instructions),
        "thread_output_operator_values": widths[0],
        "epilogue_r2g_width_bits_from_isa": r2g_bits,
        "s2r_alignment_bits": alignments[0],
        "vregs": one_resource(resource, "vreg_number"),
        "sregs": one_resource(resource, "sreg_number"),
        "static_shared": one_resource(resource, "shared_memory_size"),
        "stack_bytes": one_resource(resource, "STACK SIZE"),
        "spill_status": "UNKNOWN_NO_AUTHORITATIVE_RESOURCE_FIELD",
        "focus": focus,
        "opcode_counts": dict(sorted(counts.items())),
        "line_sha256": sha256(line_path),
        "resource_sha256": sha256(resource_path),
    }
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print("FQ_M8_EPILOGUE_CODEGEN "
          f"arm={arm} instructions={value['instruction_total']} "
          f"vregs={value['vregs']} sregs={value['sregs']} "
          f"stack_bytes={value['stack_bytes']} spill=UNKNOWN "
          f"explicit_local_ops={spill_ops} "
          f"thread_op_values={widths[0]} r2g_bits={r2g_bits} "
          f"tsm_ld_b32x4={focus['tsm_ld_b32x4']} "
          f"r2g_b32x2={focus['r2g_vmem_st_b32x2']} "
          f"r2g_b32x4={focus['r2g_vmem_st_b32x4']}")


def runtime_linkage(arm: str, ldd_path: pathlib.Path,
                    output: pathlib.Path) -> None:
    if arm not in ARMS:
        raise AnalysisError(f"unknown arm {arm}")
    text = ldd_path.read_text(errors="replace")
    if "not found" in text:
        raise AnalysisError(f"{arm} runtime linkage has an unresolved library")
    matches = re.findall(
        r"(?m)^\s*(libhgg[^\s]+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)\s*$",
        text)
    if not matches or sum(name == "libhggc_wrapper.so" for name, _ in matches) != 1:
        raise AnalysisError(f"{arm} libhgg runtime denominator differs: {matches}")
    libraries: dict[str, dict[str, Any]] = {}
    for name, raw_path in matches:
        if name in libraries:
            raise AnalysisError(f"{arm} duplicate runtime soname: {name}")
        path = pathlib.Path(raw_path).resolve()
        regular_file(path)
        libraries[name] = {
            "path": str(path), "size": path.stat().st_size,
            "sha256": sha256(path),
        }
    value = {"arm": arm, "libraries": libraries, "ldd_sha256": sha256(ldd_path)}
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print("FQ_M8_EPILOGUE_RUNTIME "
          f"arm={arm} libraries={','.join(sorted(libraries))}")


def load_run(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    fixture_rows = [
        {token.split("=", 1)[0]: token.split("=", 1)[1]
         for token in line.split()[1:] if "=" in token}
        for line in text.splitlines()
        if line.startswith("FQ_KPACK4_FIXTURE ")]
    by_phase = {row.get("phase"): row for row in fixture_rows}
    if len(fixture_rows) != 2 or set(by_phase) != {"prepare", "recover"}:
        raise AnalysisError(f"{path}: fixture denominator differs")
    expected_prepare = {
        "phase": "prepare", "q": "12", "shape": SHAPE_TEXT,
        "version": "2", "layout": "1", "bits": "4", "high_bits": "0",
        "artifact_tile_k": "0", "transport_tile_k": "64",
        "group_size": "32", "reserved": "0", "mapping_id": MAPPING_ID,
        "direct_rc": "0", "abi_rc": "0", "direct_equal": "1",
    }
    expected_recover = {
        "phase": "recover", "q": "12", "shape": SHAPE_TEXT,
        "mapping_id": MAPPING_ID, "direct_rc": "0", "abi_rc": "0",
        "direct_equal": "1", "native_equal": "1",
    }
    if by_phase["prepare"] != expected_prepare or \
            by_phase["recover"] != expected_recover:
        raise AnalysisError(f"{path}: fixture contract differs: {fixture_rows}")
    shard = exactly_one(text, "FQ_SHARD ")
    cell = exactly_one(text, "FQ_TC_CELL ")
    done = exactly_one(text, "FQ_SHAPE_DONE ")
    marker = {
        "q": "12", "A": "0", "bchunk": "0", "shape": SHAPE_TEXT,
        "weight_layout": "1", "weight_mapping_id": MAPPING_ID,
        "weight_delivery_n": "0",
        "typed_rows": "1", "selected_rows": "1", "only_split": "1",
        "bc_mode": "skip", "bc_batch": "native-grid-y-m-lt8",
        "split_timing": "ordered-close", "iterations": str(ITERATIONS),
    }
    for name, row in (("shard", shard), ("done", done)):
        if any(row.get(key) != value for key, value in marker.items()):
            raise AnalysisError(f"{path}: {name} identity differs")
    if shard.get("correctness_repeats") != str(CORRECTNESS_REPEATS) or \
            shard.get("schedule_seed") != SCHEDULE_SEED or \
            done.get("status") != "PASS":
        raise AnalysisError(f"{path}: correctness/schedule/completion differs")
    expected_cell = {
        "q": "12", "A": "0", "bchunk": "0", "shape": SHAPE_TEXT,
        "symbol": SYMBOL, "tm": "8", "tn": "64", "tk": "256",
        "wm": "8", "wn": "16", "stages": "2",
        "provider": "standard-aiu", "S": "1", "scope": "FULL_OUTPUT",
        "resolved_delivery_n": "16", "provider_capacity_rows": "0",
        "scalezero_fused": "1", "state": "MEASURED", "raw_bad": "0",
        "reducer_untimed": "0", "failure_step": "NONE",
        "failure_cutlass_status": "0", "failure_runtime_status": "0",
        "failure_repeat": "-1", "first_bad": str(2**64 - 1),
        "first_want": "0x0000", "first_got": "0x0000",
        "partial_bytes": "0",
    }
    if any(cell.get(key) != value for key, value in expected_cell.items()):
        raise AnalysisError(f"{path}: exact S1 cell identity differs: {cell}")
    samples = parse_samples(cell.get("samples", ""))
    if len(samples) != ITERATIONS:
        raise AnalysisError(f"{path}: samples={len(samples)}/{ITERATIONS}")
    median = statistics.median(samples)
    printed = float(cell["us"])
    if not math.isfinite(printed) or printed <= 0 or abs(median - printed) > 1e-6:
        raise AnalysisError(f"{path}: printed median differs from samples")
    return {
        "samples": samples, "median_us": median,
        "shipping_smem": int(cell["shipping_smem"]),
        "split_smem": int(cell["split_smem"]),
        "partial_bytes": int(cell["partial_bytes"]),
    }


def validate_execution(path: pathlib.Path) -> None:
    rows = path.read_text().splitlines()
    expected = ["round\tslot\tarm"]
    for round_index in range(1, ROUNDS + 1):
        for slot, arm in enumerate(expected_order(round_index), start=1):
            expected.append(f"{round_index}\t{slot}\t{arm}")
    if rows != expected:
        raise AnalysisError("execution order is not exact AB/BA alternation")


def analyze(bundle: pathlib.Path, runs: pathlib.Path, codegen_root: pathlib.Path,
            execution: pathlib.Path, threshold: float,
            output_json: pathlib.Path, output_tsv: pathlib.Path) -> bool:
    bundle_manifest = load_bundle(bundle)
    validate_execution(execution)
    if not 0 < threshold < 0.2:
        raise AnalysisError("regression threshold is outside (0,0.2)")
    codegens: dict[str, dict[str, Any]] = {}
    linkages: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        cg = json.loads((codegen_root / f"{arm}.json").read_text())
        expected_binary_hash = bundle_manifest["arms"][arm]["sha256"]
        if cg.get("arm") != arm or cg.get("binary_sha256") != expected_binary_hash:
            raise AnalysisError(f"{arm} codegen identity differs")
        symbol = cg.get("symbol")
        if not isinstance(symbol, str) or choose_s1([symbol]) != symbol:
            raise AnalysisError(f"{arm} codegen S1 symbol differs")
        codegens[arm] = cg
        linkage = json.loads((codegen_root / f"{arm}-runtime.json").read_text())
        if linkage.get("arm") != arm or not linkage.get("libraries"):
            raise AnalysisError(f"{arm} runtime linkage identity differs")
        linkages[arm] = linkage
        samples: list[float] = []
        round_medians: list[float] = []
        resources: set[tuple[int, int, int]] = set()
        for round_index in range(1, ROUNDS + 1):
            slot = expected_order(round_index).index(arm) + 1
            run = load_run(runs / f"round-{round_index:02d}-slot-{slot}-{arm}.log")
            samples.extend(run["samples"])
            round_medians.append(run["median_us"])
            resources.add((run["shipping_smem"], run["split_smem"],
                           run["partial_bytes"]))
        if len(resources) != 1 or len(samples) != ITERATIONS * ROUNDS:
            raise AnalysisError(f"{arm} runtime resource/sample denominator differs")
        results[arm] = {
            "samples": len(samples), "median_us": statistics.median(samples),
            "min_us": min(samples), "max_us": max(samples),
            "round_medians_us": round_medians,
            "shipping_smem": next(iter(resources))[0],
            "split_smem": next(iter(resources))[1],
            "partial_bytes": next(iter(resources))[2],
        }

    normalized_linkage = []
    for arm in ARMS:
        normalized_linkage.append({
            name: (row.get("size"), row.get("sha256"))
            for name, row in linkages[arm]["libraries"].items()})
    if normalized_linkage[0] != normalized_linkage[1]:
        raise AnalysisError("baseline/candidate libhgg runtime identities differ")

    baseline, candidate = results["baseline"], results["candidate"]
    if baseline["shipping_smem"] <= 0 or baseline["split_smem"] <= 0 or \
            (baseline["shipping_smem"], baseline["split_smem"],
            baseline["partial_bytes"]) != (
            candidate["shipping_smem"], candidate["split_smem"],
            candidate["partial_bytes"]):
        raise AnalysisError("runtime shared/workspace ABI changed")
    if codegens["baseline"]["static_shared"] <= 0 or \
            codegens["baseline"]["static_shared"] != codegens["candidate"]["static_shared"]:
        raise AnalysisError("static shared resource changed or is non-positive")
    for key in ("mma", "aiu_load", "tsm_ld_swzl_b32x4"):
        values = {codegens[arm]["focus"][key] for arm in ARMS}
        if len(values) != 1:
            raise AnalysisError(f"mainloop codegen count changed: {key}={sorted(values)}")
    if codegens["baseline"]["thread_output_operator_values"] != 4 or \
            codegens["candidate"]["thread_output_operator_values"] != 4 or \
            codegens["baseline"]["epilogue_r2g_width_bits_from_isa"] != 128 or \
            codegens["candidate"]["epilogue_r2g_width_bits_from_isa"] != 64:
        raise AnalysisError("epilogue ISA transition is not exact x4/128 -> x2/64")

    delta = candidate["median_us"] / baseline["median_us"] - 1.0
    paired = [current / old - 1.0 for old, current in zip(
        baseline["round_medians_us"], candidate["round_medians_us"])]
    stable_regression = delta > threshold and sum(value > threshold for value in paired) >= 5
    noisy_regression = delta > threshold and not stable_regression
    verdict = ("MATERIAL_REGRESSION" if stable_regression else
               "UNRESOLVED_NOISY_REGRESSION" if noisy_regression else
               "NO_MATERIAL_REGRESSION")
    result = {
        "schema": RESULT_SCHEMA, "verdict": verdict,
        "shape": list(SHAPE), "symbol": SYMBOL,
        "iterations_per_round": ITERATIONS, "rounds": ROUNDS,
        "samples_per_arm": ITERATIONS * ROUNDS,
        "correctness_repeats_per_run": CORRECTNESS_REPEATS,
        "regression_threshold": threshold,
        "baseline": baseline, "candidate": candidate,
        "delta": delta, "paired_round_deltas": paired,
        "codegen": codegens, "runtime_linkage": linkages,
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    header = (
        "verdict\tshape\tsymbol\tbaseline_us\tcandidate_us\tdelta_pct\t"
        "paired_round_delta_pct\tbaseline_instructions\tcandidate_instructions\t"
        "baseline_vregs\tcandidate_vregs\tbaseline_sregs\tcandidate_sregs\t"
        "baseline_stack\tcandidate_stack\tbaseline_spill_status\tcandidate_spill_status\t"
        "baseline_explicit_local_ops\tcandidate_explicit_local_ops\t"
        "static_shared\tshipping_smem\tsplit_smem\tbaseline_thread_op_values\t"
        "candidate_thread_op_values\tbaseline_r2g_bits\tcandidate_r2g_bits\t"
        "baseline_tsm_ld_b32x4\tcandidate_tsm_ld_b32x4\t"
        "baseline_r2g_b32x2\tcandidate_r2g_b32x2\t"
        "baseline_r2g_b32x4\tcandidate_r2g_b32x4")
    row = "\t".join((
        verdict, SHAPE_TEXT, SYMBOL, f"{baseline['median_us']:.9f}",
        f"{candidate['median_us']:.9f}", f"{100 * delta:.6f}",
        ",".join(f"{100 * value:.6f}" for value in paired),
        str(codegens["baseline"]["instruction_total"]),
        str(codegens["candidate"]["instruction_total"]),
        str(codegens["baseline"]["vregs"]), str(codegens["candidate"]["vregs"]),
        str(codegens["baseline"]["sregs"]), str(codegens["candidate"]["sregs"]),
        str(codegens["baseline"]["stack_bytes"]),
        str(codegens["candidate"]["stack_bytes"]),
        codegens["baseline"]["spill_status"], codegens["candidate"]["spill_status"],
        str(codegens["baseline"]["focus"]["explicit_spill_or_local_ops"]),
        str(codegens["candidate"]["focus"]["explicit_spill_or_local_ops"]),
        str(codegens["baseline"]["static_shared"]),
        str(baseline["shipping_smem"]), str(baseline["split_smem"]),
        str(codegens["baseline"]["thread_output_operator_values"]),
        str(codegens["candidate"]["thread_output_operator_values"]),
        str(codegens["baseline"]["epilogue_r2g_width_bits_from_isa"]),
        str(codegens["candidate"]["epilogue_r2g_width_bits_from_isa"]),
        str(codegens["baseline"]["focus"]["tsm_ld_b32x4"]),
        str(codegens["candidate"]["focus"]["tsm_ld_b32x4"]),
        str(codegens["baseline"]["focus"]["r2g_vmem_st_b32x2"]),
        str(codegens["candidate"]["focus"]["r2g_vmem_st_b32x2"]),
        str(codegens["baseline"]["focus"]["r2g_vmem_st_b32x4"]),
        str(codegens["candidate"]["focus"]["r2g_vmem_st_b32x4"]),
    ))
    output_tsv.write_text(header + "\n" + row + "\n")
    print("FQ_M8_EPILOGUE_PERF_AB "
          f"verdict={verdict} shape={SHAPE_TEXT} samples={ITERATIONS * ROUNDS} "
          f"baseline_us={baseline['median_us']:.9f} "
          f"candidate_us={candidate['median_us']:.9f} delta_pct={100 * delta:.6f} "
          f"paired_pct={','.join(f'{100 * value:.6f}' for value in paired)}")
    print("FQ_M8_EPILOGUE_RESOURCE_AB "
          f"smem={baseline['shipping_smem']} "
          f"vregs={codegens['baseline']['vregs']}->{codegens['candidate']['vregs']} "
          f"stack={codegens['baseline']['stack_bytes']}->{codegens['candidate']['stack_bytes']} "
          "spill=UNKNOWN explicit_local_ops="
          f"{codegens['baseline']['focus']['explicit_spill_or_local_ops']}->"
          f"{codegens['candidate']['focus']['explicit_spill_or_local_ops']} "
          f"static_shared={codegens['baseline']['static_shared']} "
          "thread_op_values=4->4 r2g_bits=128->64 "
          f"tsm_ld_b32x4={codegens['baseline']['focus']['tsm_ld_b32x4']}->"
          f"{codegens['candidate']['focus']['tsm_ld_b32x4']} "
          f"r2g_b32x2={codegens['baseline']['focus']['r2g_vmem_st_b32x2']}->"
          f"{codegens['candidate']['focus']['r2g_vmem_st_b32x2']} "
          f"r2g_b32x4={codegens['baseline']['focus']['r2g_vmem_st_b32x4']}->"
          f"{codegens['candidate']['focus']['r2g_vmem_st_b32x4']}")
    return verdict == "NO_MATERIAL_REGRESSION"


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("\\\n", ""))


def audit_scripts() -> list[str]:
    bad: list[str] = []
    builder = compact(BUILDER.read_text())
    runner = compact(RUNNER.read_text())
    builder_tokens = (
        BASELINE_SOURCE, CANDIDATE_SOURCE, BASELINE_ACTLIZE,
        CANDIDATE_ACTLIZE, "--parent-begin4827--parent-count1--per-unit1",
        "FQ_SWEEP_QTYPE=12", "FQ_SWEEP_WEIGHT_LAYOUT=1",
        "FQ_SWEEP_ARTIFACT_TK=0", "FQ_SWEEP_BCHUNK=0",
    )
    runner_tokens = (
        "--shape=8x3072x512", "--iterations=31", "--correctness-repeats=7",
        "--schedule-seed=0x6a09e667f3bcc909", "--only-split=1",
        "--tm8-max-m=8", "--bc-mode=skip", "forroundin123456",
        'order=(baselinecandidate)', 'order=(candidatebaseline)',
        "FQ_M8_EPILOGUE_PERF_AB_GATEverdict=PASS",
    )
    for token in builder_tokens:
        if builder.count(compact(token)) != 1:
            bad.append(f"builder token differs: {token}")
    for token in runner_tokens:
        if runner.count(compact(token)) != 1:
            bad.append(f"runner token differs: {token}")
    for forbidden in ("--shape=9x", "--tm8-max-m=9", "--only-split=4"):
        if forbidden in RUNNER.read_text():
            bad.append(f"runner contains invalid timing path {forbidden}")
    return bad


def synthetic_run(us: float) -> str:
    samples = ",".join(f"{us:.9f}" for _ in range(ITERATIONS))
    return (
        f"FQ_KPACK4_FIXTURE phase=prepare q=12 shape={SHAPE_TEXT} version=2 "
        "layout=1 bits=4 high_bits=0 artifact_tile_k=0 transport_tile_k=64 "
        f"group_size=32 reserved=0 mapping_id={MAPPING_ID} direct_rc=0 "
        "abi_rc=0 direct_equal=1\n"
        f"FQ_KPACK4_FIXTURE phase=recover q=12 shape={SHAPE_TEXT} "
        f"mapping_id={MAPPING_ID} direct_rc=0 abi_rc=0 direct_equal=1 "
        "native_equal=1\n"
        f"FQ_SHARD q=12 A=0 bchunk=0 shape={SHAPE_TEXT} weight_layout=1 "
        f"weight_mapping_id={MAPPING_ID} weight_delivery_n=0 typed_rows=1 "
        "selected_rows=1 only_split=1 bc_mode=skip "
        "bc_batch=native-grid-y-m-lt8 split_timing=ordered-close "
        f"iterations={ITERATIONS} "
        f"correctness_repeats={CORRECTNESS_REPEATS} schedule_seed={SCHEDULE_SEED}\n"
        f"FQ_TC_CELL q=12 A=0 bchunk=0 shape={SHAPE_TEXT} symbol={SYMBOL} "
        "tm=8 tn=64 tk=256 wm=8 wn=16 stages=2 provider=standard-aiu S=1 "
        "scope=FULL_OUTPUT resolved_delivery_n=16 provider_capacity_rows=0 "
        f"scalezero_fused=1 state=MEASURED us={us:.9f} raw_bad=0 "
        "reducer_untimed=0 failure_step=NONE failure_cutlass_status=0 "
        "failure_runtime_status=0 failure_repeat=-1 first_bad=18446744073709551615 "
        "first_want=0x0000 first_got=0x0000 shipping_smem=100 split_smem=100 "
        f"partial_bytes=0 samples=[{samples}]\n"
        f"FQ_SHAPE_DONE q=12 A=0 bchunk=0 shape={SHAPE_TEXT} weight_layout=1 "
        f"weight_mapping_id={MAPPING_ID} weight_delivery_n=0 typed_rows=1 "
        "selected_rows=1 only_split=1 bc_mode=skip "
        "bc_batch=native-grid-y-m-lt8 split_timing=ordered-close "
        f"iterations={ITERATIONS} status=PASS\n")


def self_test() -> None:
    bad = audit_scripts()
    if bad:
        raise AnalysisError("; ".join(bad))
    good = synthetic_run(10.0)
    with tempfile.TemporaryDirectory(prefix="m8-epilogue-perf-ab-") as temp:
        load_run_path = pathlib.Path(temp) / "run.log"
        load_run_path.write_text(good)
        load_run(load_run_path)
        plants = (
            ("M9", f"FQ_SHARD q=12 A=0 bchunk=0 shape={SHAPE_TEXT}",
             "FQ_SHARD q=12 A=0 bchunk=0 shape=9x3072x512"),
            ("symbol", SYMBOL, SYMBOL + "_wrong"),
            ("provider", "provider=standard-aiu", "provider=packed-row"),
            ("split", "only_split=1", "only_split=4"),
            ("raw", "raw_bad=0", "raw_bad=1"),
            ("samples", "samples=[", "samples=[] # ["),
            ("nonfinite", "samples=[10.000000000,",
             "samples=[nan,"),
            ("fixture", "direct_rc=0", "direct_rc=9"),
        )
        for label, old, new in plants:
            planted = good.replace(old, new, 1)
            load_run_path.write_text(planted)
            try:
                load_run(load_run_path)
            except (AnalysisError, KeyError, ValueError):
                pass
            else:
                raise AnalysisError(f"result plant stayed green: {label}")
        ldd_path = pathlib.Path(temp) / "ldd.txt"
        runtime_json = pathlib.Path(temp) / "runtime.json"
        runtime_target = pathlib.Path(sys.executable).resolve()
        ldd_path.write_text(
            f"libhggc_wrapper.so => {runtime_target} (0x000000000001)\n")
        runtime_linkage("baseline", ldd_path, runtime_json)
        if set(json.loads(runtime_json.read_text())["libraries"]) != {
                "libhggc_wrapper.so"}:
            raise AnalysisError("runtime linkage positive differs")
        ldd_path.write_text("libhggc_wrapper.so => not found\n")
        try:
            runtime_linkage("baseline", ldd_path, runtime_json)
        except AnalysisError:
            pass
        else:
            raise AnalysisError("missing runtime library stayed green")

    pairs = [
        "_ZN_device_kernel_gemm6kernel13GemmUniversal_"
        "KernelAiuQ4KPack4Transpose_PPU0010_8x16x16_F32F16F16F32_TN_"
        "SplitKSerialScheduler",
        "_ZN_device_kernel_GemmUniversalMixedInputSplitKParallel_"
        "KernelAiuQ4KPack4Transpose_PPU0010_8x16x16_F32F16F16F32_TN",
    ]
    if choose_s1(pairs) != pairs[0]:
        raise AnalysisError("S1 selector rejected its exact positive")
    try:
        choose_s1(pairs[1:])
    except AnalysisError:
        pass
    else:
        raise AnalysisError("S1 selector accepted Split-K")
    print("[m8-epilogue-perf-ab:self-test] PASS exact M8/AP0/S1 row, "
          "ABBA 6x31 finite samples, exact fixture/S1/resource/ISA parsers; "
          "eleven negatives RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    create = sub.add_parser("make-bundle-manifest")
    create.add_argument("--bundle", type=pathlib.Path, required=True)
    create.add_argument("--sdk", type=pathlib.Path, required=True)
    create.add_argument("--baseline", type=pathlib.Path, required=True)
    create.add_argument("--candidate", type=pathlib.Path, required=True)
    verify = sub.add_parser("verify-bundle")
    verify.add_argument("--bundle", type=pathlib.Path, required=True)
    select = sub.add_parser("select-symbol")
    select.add_argument("--list-elf", type=pathlib.Path, required=True)
    select.add_argument("--symbol-output", type=pathlib.Path, required=True)
    select.add_argument("--demangled-output", type=pathlib.Path, required=True)
    cg = sub.add_parser("codegen")
    cg.add_argument("--arm", choices=ARMS, required=True)
    cg.add_argument("--line", type=pathlib.Path, required=True)
    cg.add_argument("--resource", type=pathlib.Path, required=True)
    cg.add_argument("--symbol", type=pathlib.Path, required=True)
    cg.add_argument("--binary", type=pathlib.Path, required=True)
    cg.add_argument("--output", type=pathlib.Path, required=True)
    linkage = sub.add_parser("runtime-linkage")
    linkage.add_argument("--arm", choices=ARMS, required=True)
    linkage.add_argument("--ldd", type=pathlib.Path, required=True)
    linkage.add_argument("--output", type=pathlib.Path, required=True)
    run = sub.add_parser("analyze")
    run.add_argument("--bundle", type=pathlib.Path, required=True)
    run.add_argument("--runs", type=pathlib.Path, required=True)
    run.add_argument("--codegen", type=pathlib.Path, required=True)
    run.add_argument("--execution", type=pathlib.Path, required=True)
    run.add_argument("--threshold", type=float, default=0.03)
    run.add_argument("--output-json", type=pathlib.Path, required=True)
    run.add_argument("--output-tsv", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "make-bundle-manifest":
            make_bundle_manifest(args.bundle.resolve(), args.sdk.resolve(),
                                 args.baseline.resolve(), args.candidate.resolve())
        elif args.command == "verify-bundle":
            value = load_bundle(args.bundle.resolve())
            print("[m8-epilogue-perf-ab-bundle] VERIFIED "
                  f"arms={','.join(value['arms'])} shape={SHAPE_TEXT}")
        elif args.command == "select-symbol":
            select_symbol(args.list_elf, args.symbol_output, args.demangled_output)
        elif args.command == "codegen":
            codegen(args.arm, args.line, args.resource, args.symbol,
                    args.binary, args.output)
        elif args.command == "runtime-linkage":
            runtime_linkage(args.arm, args.ldd, args.output)
        else:
            clean = analyze(args.bundle.resolve(), args.runs.resolve(),
                            args.codegen.resolve(), args.execution.resolve(),
                            args.threshold, args.output_json, args.output_tsv)
            return 0 if clean else 1
        return 0
    except (AnalysisError, json.JSONDecodeError, OSError, subprocess.SubprocessError,
            KeyError, ValueError) as error:
        print(f"[m8-epilogue-perf-ab] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
