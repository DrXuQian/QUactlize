#!/usr/bin/env python3
"""Fail-closed codegen and timing analysis for the Q4_K layout/provider A/B."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "quactlize.fq-q4k-kpack4-xplane-isomorphic-ab-result.v1"
ARM_SCHEMA = "quactlize.fq-q4k-kpack4-xplane-isomorphic-arm.v1"
MAPPING_ID = "0x51344b5034540001"
CONFIG = "8x64x256_w8x16_s2"
SPLIT = 4
CASES = (
    ("m1_n5120_k8192", (1, 5120, 8192), (0, 1)),
    ("m1_n5120_k25600", (1, 5120, 25600), (0, 1)),
    ("m1_n8192_k5120", (1, 8192, 5120), (0, 1)),
    ("m2_n5120_k25600", (2, 5120, 25600), (0,)),
    ("m4_n5120_k8192", (4, 5120, 8192), (0,)),
)
ARMS = ("xplane-ap0", "kpack4-ap0", "xplane-ap1", "kpack4-ap1")


class AnalysisError(ValueError):
    pass


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(line: str, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in shlex.split(line.removeprefix(prefix)):
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def exactly_one(text: str, prefix: str) -> dict[str, str]:
    rows = [fields(line, prefix) for line in text.splitlines()
            if line.startswith(prefix)]
    if len(rows) != 1:
        raise AnalysisError(
            f"{prefix.strip()} denominator is {len(rows)}, expected 1")
    return rows[0]


def parse_samples(value: str) -> list[float]:
    if not value.startswith("[") or not value.endswith("]"):
        raise AnalysisError(f"malformed samples: {value}")
    body = value[1:-1]
    samples = [] if not body else [float(item) for item in body.split(",")]
    if any(not (sample > 0) for sample in samples):
        raise AnalysisError("timing sample is not positive")
    return samples


def validate_arm_manifest(value: dict[str, Any], name: str) -> dict[str, Any]:
    if value.get("schema") != ARM_SCHEMA or value.get("name") != name or \
            value.get("selection_denominator") != 1 or \
            value.get("source_typed_denominator") != 144 or \
            value.get("source_global_typed_denominator") != 918:
        raise AnalysisError(f"arm manifest identity differs: {name}")
    layout = "q4-kpack4" if name.startswith("kpack4") else "xplane"
    ap = 1 if name.endswith("ap1") else 0
    artifact = 0 if layout == "q4-kpack4" else 64
    expected = {
        "layout": layout, "weight_layout": int(layout == "q4-kpack4"),
        "artifact_tile_k": artifact,
        "a_provider": "packed-row" if ap else "standard-aiu",
        "a_provider_id": ap,
    }
    if any(value.get(key) != want for key, want in expected.items()):
        raise AnalysisError(f"arm axes differ for {name}: {value}")
    row = value.get("row", {})
    axes = {
        "qtype": 12, "artifact_tile_k": artifact,
        "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
        "warp_m": 8, "warp_n": 16, "stages": 2, "bchunk": 0,
        "a_provider": expected["a_provider"],
    }
    if any(row.get(key) != want for key, want in axes.items()):
        raise AnalysisError(f"selected row axes differ for {name}: {row}")
    return row


def load_master(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text())
    if value.get("schema") != \
            "quactlize.fq-q4k-kpack4-xplane-isomorphic-ab.v1" or \
            value.get("axes") != {
                "qtype": 12, "tile_m": 8, "tile_n": 64,
                "tactic_tile_k": 256, "warp_m": 8, "warp_n": 16,
                "stages": 2, "bchunk": 0, "split": 4}:
        raise AnalysisError("master A/B manifest identity differs")
    arms = {arm.get("name"): arm for arm in value.get("arms", [])}
    if set(arms) != set(ARMS):
        raise AnalysisError(f"master arm denominator differs: {sorted(arms)}")
    for name, arm in arms.items():
        validate_arm_manifest(arm, name)
    return arms


def emit_plan(output: pathlib.Path) -> None:
    value = {
        "schema": "quactlize.fq-q4k-kpack4-xplane-isomorphic-ab-plan.v1",
        "config": CONFIG, "split": SPLIT,
        "cases": [
            {"shape_key": key, "shape": list(shape),
             "providers": list(providers)}
            for key, shape, providers in CASES
        ],
    }
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(f"[fq-kpack4-xplane-ab-plan] PASS comparisons=8 output={output}")


def validate_plan(value: dict[str, Any]) -> None:
    expected = {key: (shape, providers) for key, shape, providers in CASES}
    actual = {
        row["shape_key"]: (tuple(row["shape"]), tuple(row["providers"]))
        for row in value.get("cases", [])
    }
    if value.get("schema") != \
            "quactlize.fq-q4k-kpack4-xplane-isomorphic-ab-plan.v1" or \
            value.get("config") != CONFIG or value.get("split") != SPLIT or \
            actual != expected:
        raise AnalysisError("run plan differs from the exact five-shape authority")


def validate_inputs(master_path: pathlib.Path, plan_path: pathlib.Path) -> None:
    load_master(master_path)
    validate_plan(json.loads(plan_path.read_text()))
    print("[fq-kpack4-xplane-ab-inputs] PASS arms=4 comparisons=8 "
          f"master={master_path} plan={plan_path}")


def parse_list_elf(text: str) -> list[tuple[str, str]]:
    mangled = re.findall(r"^.*Func\s+\d+:\s*(\S+).*$", text, re.MULTILINE)
    if not mangled:
        raise AnalysisError("hgobjdump -lelf exposed no function symbols")
    proc = subprocess.run(
        ["c++filt", *mangled], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    demangled = proc.stdout.splitlines()
    if len(demangled) != len(mangled):
        raise AnalysisError("c++filt denominator differs")
    return list(zip(mangled, demangled))


def choose_producer(pairs: list[tuple[str, str]]) -> tuple[str, str]:
    candidates = [
        pair for pair in pairs
        if "device_kernel" in pair[1] and
        "GemmUniversalMixedInputSplitKParallel" in pair[1] and
        "LastArriverM1Fp16Completion" not in pair[1]
    ]
    if len(candidates) != 1:
        raise AnalysisError(
            f"Split-K producer symbol denominator is {len(candidates)}, expected 1: "
            + " | ".join(value for _, value in candidates))
    return candidates[0]


def select_symbol(list_elf: pathlib.Path, symbol_output: pathlib.Path,
                  demangled_output: pathlib.Path) -> None:
    symbol, demangled = choose_producer(parse_list_elf(
        list_elf.read_text(errors="replace")))
    symbol_output.write_text(symbol + "\n")
    demangled_output.write_text(demangled + "\n")
    print("[fq-kpack4-xplane-ab-symbol] PASS producer=1 "
          f"symbol={symbol}")


def parse_instructions(text: str) -> list[str]:
    """Parse both source mnemonics and hgobjdump's lowered dotted opcodes.

    PPU hgobjdump normally prints the source ``ppu.tc01.ldmatrix`` operation
    as a backend ``tsm.ld...`` instruction.  Requiring the source mnemonic in
    that output confuses an assembler lowering boundary with a missing load.
    This is the parser already proved by the historical shared-prefix codegen
    closure, extended here without interpreting the opcode.
    """
    result: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("//", "#", ".", "File ",
                                        "Function ")):
            continue
        match = re.search(
            r"(?:^|\s)([sv]?\.?[A-Za-z][A-Za-z0-9_]*"
            r"(?:\.[A-Za-z0-9_:]+)+)\s+", line)
        if match:
            result.append(match.group(1).lower())
    if not result:
        raise AnalysisError("exact-symbol disassembly contains no parsed instruction")
    return result


def count_like(counter: collections.Counter[str], needle: str) -> int:
    return sum(value for opcode, value in counter.items() if needle in opcode)


def parse_registers(text: str) -> int | None:
    patterns = (
        re.compile(r"(?i)\b(?:numregs|registers?|regs?|reg)\s*(?:per[- ]thread)?\s*[:=]\s*(\d+)"),
        re.compile(r"(?i)\b(?:numregs|registers?|regs?|reg)\s+(\d+)\b"),
        re.compile(r"(?i)\b(\d+)\s+(?:registers?|regs?)\b"),
    )
    values: set[int] = set()
    for line in text.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                values.add(int(match.group(1)))
                break
    return next(iter(values)) if len(values) == 1 else None


def parse_local_fields(text: str) -> dict[str, int]:
    pattern = re.compile(
        r"(?i)\b(?P<label>spill(?:[- _](?:loads?|stores?|bytes?))?"
        r"|stack(?:[- _]frame|[- _]size)?"
        r"|scratch(?:[- _](?:bytes?|size))?"
        r"|local(?:[- _]memory)?)\b\s*(?:per[- ]thread\s*)?[:=]?\s*(?P<value>\d+)")
    values: dict[str, set[int]] = {}
    for match in pattern.finditer(text):
        label = re.sub(r"[- _]+", "_", match.group("label").lower())
        values.setdefault(label, set()).add(int(match.group("value")))
    ambiguous = {key: sorted(value) for key, value in values.items()
                 if len(value) != 1}
    if ambiguous:
        raise AnalysisError(f"ambiguous resource fields: {ambiguous}")
    return {key: next(iter(value)) for key, value in values.items()}


def codegen(arm_manifest: pathlib.Path, line_path: pathlib.Path,
            resource_path: pathlib.Path, binary: pathlib.Path,
            symbol_path: pathlib.Path, demangled_path: pathlib.Path,
            output: pathlib.Path) -> None:
    arm = json.loads(arm_manifest.read_text())
    name = str(arm.get("name"))
    validate_arm_manifest(arm, name)
    line = line_path.read_text(errors="replace")
    resource = resource_path.read_text(errors="replace")
    demangled = demangled_path.read_text(errors="replace").strip()
    expects_kpack = name.startswith("kpack4")
    expects_ap1 = name.endswith("ap1")
    has_kpack = "KernelAiuQ4KPack4Transpose" in demangled
    has_ap1 = "KernelAiuPackedA<1" in demangled
    if has_kpack != expects_kpack or has_ap1 != expects_ap1:
        raise AnalysisError(
            f"{name} exact producer schedule differs: "
            f"kpack={int(has_kpack)} ap1={int(has_ap1)}")
    inst = parse_instructions(line)
    counts = collections.Counter(inst)
    local = parse_local_fields(resource)
    local_ops = sum(value for opcode, value in counts.items()
                    if "local" in opcode or "spill" in opcode)
    explicit_local = bool(local)
    local_total = sum(local.values())
    spill_status = ("NONZERO" if local_total or local_ops else
                    "ZERO" if explicit_local else "UNKNOWN")
    focus = {
        "ldmatrix_total": count_like(counts, "ldmatrix"),
        "m8n8_x4_swzl": count_like(counts, "m8n8.x4.swzl.shared.b16"),
        "m16n16_x1_swzl_trans": count_like(
            counts, "m16n16.x1.swzl.trans.shared.b16"),
        "mma": count_like(counts, ".mma"),
        "cp_async": count_like(counts, "cp.async"),
        "tsm_load": count_like(counts, "tsm.ld"),
        "tsm_store": count_like(counts, "tsm.st"),
        "local_or_spill_ops": local_ops,
    }
    source_reader_total = (focus["m8n8_x4_swzl"] +
                           focus["m16n16_x1_swzl_trans"])
    if source_reader_total:
        if expects_kpack and (focus["m16n16_x1_swzl_trans"] <= 0 or
                              focus["m8n8_x4_swzl"] != 0):
            raise AnalysisError(
                f"{name} source-mnemonic reader differs: focus={focus}")
        if not expects_kpack and (focus["m8n8_x4_swzl"] <= 0 or
                                  focus["m16n16_x1_swzl_trans"] != 0):
            raise AnalysisError(
                f"{name} source-mnemonic reader differs: focus={focus}")
        reader_lowering = "SOURCE_MNEMONIC"
    else:
        if focus["tsm_load"] <= 0:
            raise AnalysisError(
                f"{name} exact producer contains no source reader or lowered "
                f"tsm load: focus={focus} opcodes={dict(sorted(counts.items()))}")
        reader_lowering = "HGOBJDUMP_TSM_LOWERED"
    if focus["mma"] <= 0:
        raise AnalysisError(f"{name} producer contains no MMA instruction")
    value = {
        "schema": "quactlize.fq-q4k-kpack4-xplane-codegen.v1",
        "arm": name,
        "binary_sha256": sha256(binary),
        "symbol": symbol_path.read_text().strip(),
        "demangled": demangled,
        "instruction_total": len(inst),
        "focus_counts": focus,
        "registers": parse_registers(resource),
        "resource_local_fields": local,
        "spill_status": spill_status,
        "reader_lowering": reader_lowering,
        "opcode_counts": dict(sorted(counts.items())),
        "line_sha256": sha256(line_path),
        "resource_sha256": sha256(resource_path),
    }
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print("FQ_KPACK4_XPLANE_CODEGEN "
          f"arm={name} instructions={len(inst)} registers="
          f"{value['registers'] if value['registers'] is not None else 'UNKNOWN'} "
          f"spill={spill_status} ldmatrix={focus['ldmatrix_total']} "
          f"tsm_load={focus['tsm_load']} reader={reader_lowering} "
          f"m8x4={focus['m8n8_x4_swzl']} "
          f"m16x1_trans={focus['m16n16_x1_swzl_trans']} mma={focus['mma']}")


def load_run(path: pathlib.Path, arm: dict[str, Any], shape: tuple[int, int, int],
             iterations: int) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    shard = exactly_one(text, "FQ_SHARD ")
    cell = exactly_one(text, "FQ_TC_CELL ")
    done = exactly_one(text, "FQ_SHAPE_DONE ")
    name = str(arm["name"])
    row = validate_arm_manifest(arm, name)
    shape_text = "x".join(map(str, shape))
    expected_common = {
        "q": "12", "A": str(arm["artifact_tile_k"]), "bchunk": "0",
        "shape": shape_text,
    }
    expected_markers = {
        **expected_common, "weight_layout": str(arm["weight_layout"]),
        "weight_mapping_id": MAPPING_ID if arm["weight_layout"] else "0x0000000000000000",
        "typed_rows": "1", "selected_rows": "1", "only_split": "4",
        "bc_mode": "skip", "iterations": str(iterations),
    }
    for marker in (shard, done):
        if any(marker.get(key) != want for key, want in expected_markers.items()):
            raise AnalysisError(f"{path}: run marker identity differs: {marker}")
    if done.get("status") != "PASS":
        raise AnalysisError(f"{path}: run did not complete PASS")
    expected_cell = {
        **expected_common, "symbol": str(row["symbol"]),
        "tm": "8", "tn": "64", "tk": "256", "wm": "8", "wn": "16",
        "stages": "2", "provider": str(arm["a_provider"]), "S": "4",
        "scope": "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS",
        "provider_capacity_rows": str(arm["a_provider_id"]),
        "state": "MEASURED", "raw_bad": "0", "reducer_untimed": "1",
        "failure_step": "NONE", "failure_repeat": "-1",
    }
    if any(cell.get(key) != want for key, want in expected_cell.items()):
        raise AnalysisError(f"{path}: measured cell identity differs: {cell}")
    samples = parse_samples(cell.get("samples", ""))
    if len(samples) != iterations:
        raise AnalysisError(
            f"{path}: sample denominator is {len(samples)}, expected {iterations}")
    if abs(statistics.median(samples) - float(cell["us"])) > 1e-6:
        raise AnalysisError(f"{path}: printed median differs from samples")
    return {
        "samples": samples, "median_us": statistics.median(samples),
        "min_us": min(samples), "max_us": max(samples),
        "shipping_smem": int(cell["shipping_smem"]),
        "split_smem": int(cell["split_smem"]),
        "partial_bytes": int(cell["partial_bytes"]),
    }


def analyze(master_path: pathlib.Path, plan_path: pathlib.Path,
            runs_root: pathlib.Path, codegen_root: pathlib.Path,
            iterations: int, rounds: int, gap_threshold: float,
            output_json: pathlib.Path, output_tsv: pathlib.Path) -> None:
    arms = load_master(master_path)
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    codegens = {}
    for name in ARMS:
        path = codegen_root / f"{name}.json"
        value = json.loads(path.read_text())
        if value.get("arm") != name or value.get("schema") != \
                "quactlize.fq-q4k-kpack4-xplane-codegen.v1":
            raise AnalysisError(f"codegen identity differs for {name}")
        codegens[name] = value
    mma_counts = {value["focus_counts"]["mma"] for value in codegens.values()}
    if len(mma_counts) != 1:
        raise AnalysisError(f"same-tactic MMA counts differ: {sorted(mma_counts)}")

    comparisons = []
    acu_targets = []
    for shape_key, shape, providers in CASES:
        for ap in providers:
            rows: dict[str, dict[str, Any]] = {}
            for layout in ("xplane", "kpack4"):
                name = f"{layout}-ap{ap}"
                all_samples: list[float] = []
                medians = []
                smem = set()
                partial = set()
                for round_index in range(1, rounds + 1):
                    path = (runs_root / shape_key / f"ap{ap}" /
                            f"round-{round_index}-{name}.log")
                    run = load_run(path, arms[name], shape, iterations)
                    all_samples.extend(run["samples"])
                    medians.append(run["median_us"])
                    smem.add((run["shipping_smem"], run["split_smem"]))
                    partial.add(run["partial_bytes"])
                if len(smem) != 1 or len(partial) != 1:
                    raise AnalysisError(f"runtime resource identity drifted for {shape_key}/{name}")
                rows[layout] = {
                    "arm": name,
                    "samples": len(all_samples),
                    "median_us": statistics.median(all_samples),
                    "min_us": min(all_samples), "max_us": max(all_samples),
                    "run_medians_us": medians,
                    "shipping_smem": next(iter(smem))[0],
                    "split_smem": next(iter(smem))[1],
                    "partial_bytes": next(iter(partial)),
                }
            if rows["xplane"]["partial_bytes"] != rows["kpack4"]["partial_bytes"]:
                raise AnalysisError(f"partial ABI differs for {shape_key}/AP{ap}")
            delta = rows["kpack4"]["median_us"] / rows["xplane"]["median_us"] - 1.0
            paired = [
                k / x - 1.0 for x, k in zip(
                    rows["xplane"]["run_medians_us"],
                    rows["kpack4"]["run_medians_us"])
            ]
            same_sign = all(value > 0 for value in paired) or \
                all(value < 0 for value in paired)
            requires_acu = abs(delta) >= gap_threshold and same_sign
            comparison = {
                "shape_key": shape_key, "shape": list(shape),
                "a_provider": f"AP{ap}", "xplane": rows["xplane"],
                "kpack4": rows["kpack4"], "delta": delta,
                "codegen": {
                    "xplane": codegens[f"xplane-ap{ap}"],
                    "kpack4": codegens[f"kpack4-ap{ap}"],
                },
                "paired_run_deltas": paired,
                "requires_acu": requires_acu,
            }
            comparisons.append(comparison)
            if requires_acu:
                acu_targets.append({
                    "shape_key": shape_key, "shape": list(shape),
                    "a_provider": f"AP{ap}", "delta": delta,
                    "arms": [f"xplane-ap{ap}", f"kpack4-ap{ap}"],
                })

    result = {
        "schema": SCHEMA, "config": CONFIG, "split": SPLIT,
        "iterations_per_run": iterations, "rounds": rounds,
        "samples_per_arm_cell": iterations * rounds,
        "gap_threshold": gap_threshold,
        "comparisons": comparisons, "codegen": codegens,
        "acu_required": bool(acu_targets), "acu_targets": acu_targets,
        "authority": {
            "master_sha256": sha256(master_path),
            "plan_sha256": sha256(plan_path),
        },
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "shape\tprovider\txplane_us\tkpack4_us\tkpack4_delta_pct\t"
        "xplane_range\tkpack4_range\tpaired_deltas_pct\t"
        "xplane_instructions\tkpack4_instructions\tinstruction_delta\t"
        "xplane_registers\tkpack4_registers\txplane_spill\tkpack4_spill\t"
        "xplane_ldmatrix\tkpack4_ldmatrix\txplane_tsm_load\t"
        "kpack4_tsm_load\txplane_reader\tkpack4_reader\tmma\tacu_required"
    ]
    for row in comparisons:
        x, k = row["xplane"], row["kpack4"]
        xcg, kcg = row["codegen"]["xplane"], row["codegen"]["kpack4"]
        lines.append("\t".join((
            row["shape_key"], row["a_provider"], f"{x['median_us']:.9f}",
            f"{k['median_us']:.9f}", f"{100 * row['delta']:.6f}",
            f"[{x['min_us']:.9f},{x['max_us']:.9f}]",
            f"[{k['min_us']:.9f},{k['max_us']:.9f}]",
            ",".join(f"{100 * value:.6f}" for value in row["paired_run_deltas"]),
            str(xcg["instruction_total"]), str(kcg["instruction_total"]),
            str(kcg["instruction_total"] - xcg["instruction_total"]),
            str(xcg["registers"] if xcg["registers"] is not None else "UNKNOWN"),
            str(kcg["registers"] if kcg["registers"] is not None else "UNKNOWN"),
            str(xcg["spill_status"]), str(kcg["spill_status"]),
            str(xcg["focus_counts"]["ldmatrix_total"]),
            str(kcg["focus_counts"]["ldmatrix_total"]),
            str(xcg["focus_counts"]["tsm_load"]),
            str(kcg["focus_counts"]["tsm_load"]),
            str(xcg["reader_lowering"]), str(kcg["reader_lowering"]),
            str(xcg["focus_counts"]["mma"]),
            str(int(row["requires_acu"])),
        )))
    output_tsv.write_text("\n".join(lines) + "\n")
    for row in comparisons:
        print("FQ_KPACK4_XPLANE_AB "
              f"shape={row['shape_key']} provider={row['a_provider']} "
              f"xplane_us={row['xplane']['median_us']:.9f} "
              f"kpack4_us={row['kpack4']['median_us']:.9f} "
              f"delta_pct={100 * row['delta']:.6f} "
              f"acu_required={int(row['requires_acu'])}")
    print("FQ_KPACK4_XPLANE_AB_VERDICT "
          f"comparisons={len(comparisons)} acu_required={int(bool(acu_targets))} "
          f"acu_targets={len(acu_targets)} output={output_json}")


def self_test() -> None:
    pairs = [
        ("producer", "void device_kernel<cutlass::gemm::kernel::GemmUniversalMixedInputSplitKParallel<X>>()"),
        ("fused", "void device_kernel<cutlass::gemm::kernel::GemmUniversalMixedInputSplitKParallel<X, LastArriverM1Fp16Completion<2>>>()"),
        ("reducer", "void reduction_kernel<float>()"),
    ]
    assert choose_producer(pairs)[0] == "producer"
    try:
        choose_producer(pairs[1:])
    except AnalysisError:
        pass
    else:
        raise AssertionError("missing producer symbol stayed green")
    xinst = parse_instructions(
        "0000 ppu.tc01.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 r0\n"
        "0004 ppu.tc01.mma.f32.f16.m8n16k16 r1\n")
    kinst = parse_instructions(
        "0000 ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16 r0\n"
        "0004 ppu.tc01.mma.f32.f16.m8n16k16 r1\n")
    assert "m8n8.x4" in xinst[0] and "m16n16.x1" in kinst[0]
    assert parse_instructions(
        "0000 tsm.ld.swzl.b16 r0, [r1]\n0004 v.mma.f32.f16 r2, r3\n") == [
            "tsm.ld.swzl.b16", "v.mma.f32.f16"]
    assert parse_registers("Registers: 128\nSTACK SIZE: 0\n") == 128
    assert parse_local_fields("Registers: 128\nSTACK SIZE: 0\n") == {"stack_size": 0}
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-xplane-analysis-") as temp:
        root = pathlib.Path(temp)
        path = root / "plan.json"
        emit_plan(path)
        plan = json.loads(path.read_text())
        assert len(plan["cases"]) == 5 and sum(
            len(row["providers"]) for row in plan["cases"]) == 8
        arms = []
        for name in ARMS:
            kpack = name.startswith("kpack4")
            ap = int(name.endswith("ap1"))
            provider = "packed-row" if ap else "standard-aiu"
            artifact = 0 if kpack else 64
            arms.append({
                "schema": ARM_SCHEMA, "name": name,
                "layout": "q4-kpack4" if kpack else "xplane",
                "weight_layout": int(kpack), "artifact_tile_k": artifact,
                "a_provider": provider, "a_provider_id": ap,
                "selection_denominator": 1,
                "source_typed_denominator": 144,
                "source_global_typed_denominator": 918,
                "row": {
                    "qtype": 12, "artifact_tile_k": artifact,
                    "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
                    "warp_m": 8, "warp_n": 16, "stages": 2,
                    "bchunk": 0, "a_provider": provider,
                    "symbol": f"symbol_{name}",
                },
            })
        master = root / "master.json"
        master.write_text(json.dumps({
            "schema": "quactlize.fq-q4k-kpack4-xplane-isomorphic-ab.v1",
            "axes": {"qtype": 12, "tile_m": 8, "tile_n": 64,
                     "tactic_tile_k": 256, "warp_m": 8, "warp_n": 16,
                     "stages": 2, "bchunk": 0, "split": 4},
            "arms": arms,
        }, sort_keys=True) + "\n")
        runs, codegen_root = root / "runs", root / "codegen"
        codegen_root.mkdir()
        arm_by_name = {arm["name"]: arm for arm in arms}
        for name in ARMS:
            kpack = name.startswith("kpack4")
            (codegen_root / f"{name}.json").write_text(json.dumps({
                "schema": "quactlize.fq-q4k-kpack4-xplane-codegen.v1",
                "arm": name, "instruction_total": 102 if kpack else 100,
                "registers": 128, "spill_status": "ZERO",
                "reader_lowering": "SOURCE_MNEMONIC",
                "focus_counts": {
                    "mma": 16, "ldmatrix_total": 16, "tsm_load": 0,
                    "m8n8_x4_swzl": 0 if kpack else 16,
                    "m16n16_x1_swzl_trans": 16 if kpack else 0,
                },
            }, sort_keys=True) + "\n")
        for shape_key, shape, providers in CASES:
            shape_text = "x".join(map(str, shape))
            for ap in providers:
                directory = runs / shape_key / f"ap{ap}"
                directory.mkdir(parents=True)
                for round_index in (1, 2):
                    for layout in ("xplane", "kpack4"):
                        name = f"{layout}-ap{ap}"
                        arm = arm_by_name[name]
                        median = 10.0 if layout == "xplane" else 10.5
                        samples = [median - 0.1, median, median + 0.1]
                        mapping = MAPPING_ID if arm["weight_layout"] else \
                            "0x0000000000000000"
                        common = (
                            f"q=12 A={arm['artifact_tile_k']} bchunk=0 "
                            f"shape={shape_text}")
                        text = (
                            f"FQ_SHARD {common} weight_layout={arm['weight_layout']} "
                            f"weight_mapping_id={mapping} typed_rows=1 "
                            "selected_rows=1 only_split=4 bc_mode=skip "
                            "iterations=3 correctness_repeats=8\n"
                            f"FQ_TC_CELL {common} symbol={arm['row']['symbol']} "
                            "tm=8 tn=64 tk=256 wm=8 wn=16 stages=2 "
                            f"provider={arm['a_provider']} S=4 "
                            "scope=PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS "
                            f"provider_capacity_rows={ap} state=MEASURED "
                            f"us={median:.9f} raw_bad=0 reducer_untimed=1 "
                            "failure_step=NONE failure_repeat=-1 "
                            "shipping_smem=1024 split_smem=2048 "
                            f"partial_bytes={shape[0] * shape[1] * 16} "
                            f"samples=[{','.join(str(x) for x in samples)}]\n"
                            f"FQ_SHAPE_DONE {common} "
                            f"weight_layout={arm['weight_layout']} "
                            f"weight_mapping_id={mapping} typed_rows=1 "
                            "selected_rows=1 only_split=4 bc_mode=skip "
                            "iterations=3 status=PASS\n")
                        (directory / f"round-{round_index}-{name}.log").write_text(text)
        output_json, output_tsv = root / "summary.json", root / "summary.tsv"
        analyze(master, path, runs, codegen_root, 3, 2, 0.03,
                output_json, output_tsv)
        result = json.loads(output_json.read_text())
        if len(result["comparisons"]) != 8 or \
                len(result["acu_targets"]) != 8 or \
                "instruction_delta" not in output_tsv.read_text():
            raise AssertionError("synthetic timing/codegen comparison did not close")
        missing = runs / CASES[0][0] / "ap0" / "round-1-xplane-ap0.log"
        missing.unlink()
        try:
            analyze(master, path, runs, codegen_root, 3, 2, 0.03,
                    root / "red.json", root / "red.tsv")
        except OSError:
            pass
        else:
            raise AssertionError("missing runtime arm stayed green")
    print("[fq-kpack4-xplane-ab-analysis:self-test] PASS producer-symbol, "
          "opcode/resource parsers, exact 8-comparison timing/ACU plan and "
          "missing-arm negative; RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    plan = sub.add_parser("plan")
    plan.add_argument("--output", type=pathlib.Path, required=True)
    validate = sub.add_parser("validate-inputs")
    validate.add_argument("--master", type=pathlib.Path, required=True)
    validate.add_argument("--plan", type=pathlib.Path, required=True)
    symbol = sub.add_parser("select-symbol")
    symbol.add_argument("--list-elf", type=pathlib.Path, required=True)
    symbol.add_argument("--symbol-output", type=pathlib.Path, required=True)
    symbol.add_argument("--demangled-output", type=pathlib.Path, required=True)
    cg = sub.add_parser("codegen")
    for flag in ("arm-manifest", "line", "resource", "binary", "symbol",
                 "demangled", "output"):
        cg.add_argument("--" + flag, type=pathlib.Path, required=True)
    run = sub.add_parser("analyze")
    run.add_argument("--master", type=pathlib.Path, required=True)
    run.add_argument("--plan", type=pathlib.Path, required=True)
    run.add_argument("--runs-root", type=pathlib.Path, required=True)
    run.add_argument("--codegen-root", type=pathlib.Path, required=True)
    run.add_argument("--iterations", type=int, required=True)
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--gap-threshold", type=float, default=0.03)
    run.add_argument("--output-json", type=pathlib.Path, required=True)
    run.add_argument("--output-tsv", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "plan":
            emit_plan(args.output)
        elif args.command == "validate-inputs":
            validate_inputs(args.master, args.plan)
        elif args.command == "select-symbol":
            select_symbol(args.list_elf, args.symbol_output,
                          args.demangled_output)
        elif args.command == "codegen":
            codegen(args.arm_manifest, args.line, args.resource, args.binary,
                    args.symbol, args.demangled, args.output)
        else:
            if args.iterations <= 0 or args.rounds <= 0 or \
                    not (0 < args.gap_threshold < 1):
                raise AnalysisError("iterations/rounds/threshold must be positive")
            analyze(args.master, args.plan, args.runs_root, args.codegen_root,
                    args.iterations, args.rounds, args.gap_threshold,
                    args.output_json, args.output_tsv)
        return 0
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError,
            AnalysisError, AssertionError, KeyError, ValueError) as exc:
        print(f"[fq-kpack4-xplane-ab-analysis] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
