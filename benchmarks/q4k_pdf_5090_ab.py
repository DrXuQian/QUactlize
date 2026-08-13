#!/usr/bin/env python3
"""Build, run and summarize the INBOX 132B RTX5090 Q4_K A/B.

The PDF arm is explicitly a reconstruction, not an exact-paper reproduction;
the generated report keeps that source boundary beside every result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import pathlib
import statistics
import struct
import subprocess
import sys
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_EVEN

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "benchmarks" / "q4k_pdf_5090_ab.cu"
DEFAULT_BINARY = pathlib.Path("/tmp/quactlize_q4k_pdf_5090_ab")
DEFAULT_RAW = pathlib.Path("/tmp/quactlize_q4k_pdf_5090_ab_raw.csv")
DEFAULT_SUMMARY = pathlib.Path("/tmp/quactlize_q4k_pdf_5090_ab.md")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_clean(allow_dirty: bool) -> None:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status and not allow_dirty:
        raise SystemExit("refusing evidence from a dirty tree; commit first or pass --allow-dirty for development")


def build(binary: pathlib.Path) -> None:
    cap = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
    ).splitlines()[0].strip()
    if cap != "12.0":
        raise SystemExit(f"INBOX 132B evidence target is RTX5090/sm_120, got compute capability {cap}")
    binary.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_120",
        "--expt-relaxed-constexpr",
        f"-I{ROOT / 'quactlize' / 'include'}",
        f"-I{ROOT / 'third_party' / 'cutlass' / 'include'}",
        f"-I{ROOT / 'third_party' / 'actlize' / 'include'}",
        str(SOURCE), "-lnvidia-ml", "-o", str(binary),
    ]
    run(cmd, cwd=ROOT)


def decoded_event_us(row: dict[str, str]) -> Decimal:
    bits = int(row["event_ms_bits"], 16)
    ms = struct.unpack("<f", struct.pack("<I", bits))[0]
    return Decimal.from_float(ms) * Decimal(1000)


def timing_us(row: dict[str, str]) -> float:
    batch = int(row["batch"])
    if batch <= 0:
        raise ValueError("event batch must be positive")
    return float(decoded_event_us(row) / Decimal(batch))


def infer_quantum(rows: list[dict[str, str]]) -> tuple[Decimal | None, Decimal | None, str]:
    values = sorted({decoded_event_us(r) for r in rows})
    if len(values) < 2:
        return None, None, "fewer than two distinct raw event values"
    ns = Decimal("0.001")
    ints = [int(v.quantize(ns, rounding=ROUND_HALF_EVEN) / ns) for v in values]
    gcd = 0
    for left, right in zip(ints, ints[1:]):
        gcd = math.gcd(gcd, right - left)
    quantum = Decimal(gcd) * ns
    if quantum < Decimal("0.5"):
        return None, quantum, f"adjacent-difference GCD {quantum} us rejected by the predeclared 0.5-us floor"
    return quantum, quantum, f"integer-nanosecond adjacent-difference GCD over {len(values)} distinct binary32 event values"


def fnum(value: float) -> str:
    return f"{value:.4f}"


EXPECTED_ARMS = {
    "D-EXT-O": ["pdf_scalar_dense1", "pdf_pair_dense1", "ours_native_dense1"],
    "D-EXT-K1024": ["pdf_scalar_dense1", "pdf_pair_dense1", "ours_native_dense1"],
    "D-EXT-Q": ["pdf_scalar_dense1", "pdf_pair_dense1", "ours_native_dense1"],
    "D-144-K1024-N1024": ["pdf_scalar_dense1", "pdf_pair_dense1", "ours_native_dense1"],
    "D-144-K5120-N1024": ["pdf_scalar_dense1", "pdf_pair_dense1", "ours_native_dense1"],
    "D-144-K5120-N5120": ["pdf_scalar_dense1", "pdf_pair_dense1", "ours_native_dense1"],
    "H-G8-2048": [
        "pdf_scalar_dense8", "pdf_pair_dense8", "ours_native_dense8", "ours_native_grouped1",
    ],
}


def validate_rows(rows: list[dict[str, str]], expected_binary_sha: str | None = None) -> None:
    if not rows:
        raise ValueError("raw CSV is empty")
    required = {
        "schema", "git_sha", "binary_sha", "device_pci", "nvidia_driver_version",
        "event_ms_bits", "event_total_us", "event_us_per_workload", "correctness_hash",
        "representation_bytes", "distinct_bytes", "event_pending_after_clock_query",
        "device_name", "samples_requested", "warmup_rounds_per_arm",
        "precondition_host_enqueue_ms", "cold_budget_mib", "timing_scope", "clock_scope",
    }
    if not required.issubset(rows[0]):
        raise ValueError(f"raw schema missing {sorted(required - set(rows[0]))}")
    protocol_columns = [
        "schema", "git_sha", "binary_sha", "device_pci", "nvidia_driver_version", "device_name",
        "samples_requested", "warmup_rounds_per_arm", "precondition_host_enqueue_ms",
        "cold_budget_mib", "timing_scope", "clock_scope",
    ]
    for column in protocol_columns:
        values = {row[column] for row in rows}
        if len(values) != 1:
            raise ValueError(f"raw CSV mixes protocol field {column}: {sorted(values)}")
    if rows[0]["schema"] != "q4k-pdf-ab-raw-v2":
        raise ValueError(f"unknown raw schema {rows[0]['schema']!r}")
    if rows[0]["timing_scope"] != "cuda_event_gpu_span":
        raise ValueError("raw timing scope is not the declared CUDA event GPU span")
    if rows[0]["clock_scope"] != "nvml_adjacent_snapshot":
        raise ValueError("raw clock scope is not the declared adjacent NVML snapshot")
    if expected_binary_sha is not None and rows[0]["binary_sha"] != expected_binary_sha:
        raise ValueError("raw binary SHA does not match the executable being summarized")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    states: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    fixed_group_columns = [
        "m", "n", "k", "l", "kernels_per_workload", "representation",
        "representation_bytes", "distinct_bytes", "cold_copy_count", "l2_bytes", "flush_bytes",
        "correctness_hash", "pdf_config", "pdf_config_authority",
    ]
    for row in rows:
        shape, state, arm = row["shape_id"], row["cache_state"], row["arm"]
        if shape not in EXPECTED_ARMS:
            raise ValueError(f"raw CSV contains unknown shape {shape!r}")
        if state not in {"weight_metadata_cold", "warm"}:
            raise ValueError(f"raw CSV contains unknown cache state {state!r}")
        if arm not in EXPECTED_ARMS[shape]:
            raise ValueError(f"raw CSV contains unexpected arm {shape}/{arm}")
        batch = int(row["batch"])
        if batch <= 0 or int(row["logical_workloads"]) != batch:
            raise ValueError(f"raw row {shape}/{state}/{arm} has invalid logical workload batch")
        total = float(decoded_event_us(row))
        per = total / batch
        emitted_total = float(row["event_total_us"])
        emitted_per = float(row["event_us_per_workload"])
        if not all(math.isfinite(x) and x > 0 for x in (total, per, emitted_total, emitted_per)):
            raise ValueError(f"raw row {shape}/{state}/{arm} has non-positive/non-finite timing")
        if not math.isclose(emitted_total, total, rel_tol=2e-7, abs_tol=1e-7):
            raise ValueError(f"raw row {shape}/{state}/{arm} decimal total disagrees with event bits")
        if not math.isclose(emitted_per, per, rel_tol=2e-7, abs_tol=1e-7):
            raise ValueError(f"raw row {shape}/{state}/{arm} decimal per-workload time disagrees with bits")
        if int(row["sm_clock_mhz"]) <= 0:
            raise ValueError(f"raw row {shape}/{state}/{arm} has invalid SM clock")
        groups[(shape, state, arm)].append(row)
        states[(shape, state)].append(row)

    requested = int(rows[0]["samples_requested"])
    if requested <= 0:
        raise ValueError("samples_requested must be positive")
    for key, group in groups.items():
        passes = sorted(int(row["pass"]) for row in group)
        if len(group) != requested or passes != list(range(requested)):
            raise ValueError(f"raw group {key} has {len(group)} samples/passes {passes}, requested {requested}")
        for column in fixed_group_columns:
            values = {row[column] for row in group}
            if len(values) != 1:
                raise ValueError(f"raw group {key} mixes {column}: {sorted(values)}")

    shapes = {row["shape_id"] for row in rows}
    for shape in shapes:
        present_states = {state for (candidate, state) in states if candidate == shape}
        if present_states != {"weight_metadata_cold", "warm"}:
            raise ValueError(f"raw shape {shape} lacks a complete cold/warm pair: {sorted(present_states)}")
    for (shape, state), state_rows in states.items():
        batches = {int(row["batch"]) for row in state_rows}
        if len(batches) != 1:
            raise ValueError(f"raw state {shape}/{state} mixes arm batches: {sorted(batches)}")
        batch = next(iter(batches))
        if state == "weight_metadata_cold":
            copies = {int(row["cold_copy_count"]) for row in state_rows}
            if copies != {batch}:
                raise ValueError(f"raw cold state {shape} batch {batch} disagrees with copies {sorted(copies)}")
        canonical = EXPECTED_ARMS[shape]
        for pass_id in range(requested):
            pass_rows = [row for row in state_rows if int(row["pass"]) == pass_id]
            expected = canonical if pass_id % 2 == 0 else list(reversed(canonical))
            actual = [row["arm"] for row in sorted(pass_rows, key=lambda row: int(row["arm_order"]))]
            if actual != expected:
                raise ValueError(f"raw pass {shape}/{state}/{pass_id} order={actual}, expected={expected}")


def protocol_selftest() -> None:
    rows: list[dict[str, str]] = []
    arms = EXPECTED_ARMS["D-EXT-O"]
    for state, batch in (("weight_metadata_cold", 2), ("warm", 4)):
        for pass_id in range(2):
            order = arms if pass_id == 0 else list(reversed(arms))
            for rank, arm in enumerate(order):
                ms = 0.001 + 0.0001 * (pass_id * len(arms) + rank)
                bits = struct.unpack("<I", struct.pack("<f", ms))[0]
                total = struct.unpack("<f", struct.pack("<I", bits))[0] * 1000
                pdf = arm.startswith("pdf_")
                rows.append({
                    "schema": "q4k-pdf-ab-raw-v2", "git_sha": "g", "binary_sha": "b",
                    "device_pci": "p", "nvidia_driver_version": "d", "device_name": "dev",
                    "shape_id": "D-EXT-O", "m": "1", "n": "5120", "k": "8192", "l": "1",
                    "arm": arm, "cache_state": state, "pass": str(pass_id), "arm_order": str(rank),
                    "logical_workloads": str(batch), "batch": str(batch), "kernels_per_workload": "1",
                    "representation": "pdf" if pdf else "ours", "representation_bytes": "10",
                    "distinct_bytes": "20", "cold_copy_count": "2", "l2_bytes": "30",
                    "flush_bytes": "40", "event_ms_bits": f"0x{bits:08x}",
                    "event_total_us": f"{total:.9g}", "event_us_per_workload": f"{total/batch:.9g}",
                    "sm_clock_mhz": "2407", "event_pending_after_clock_query": "1",
                    "correctness_hash": "1" if pdf else "2", "pdf_config": "2x8x1",
                    "pdf_config_authority": "pdf_p22_winner", "timing_scope": "cuda_event_gpu_span",
                    "clock_scope": "nvml_adjacent_snapshot", "samples_requested": "2",
                    "warmup_rounds_per_arm": "100", "precondition_host_enqueue_ms": "50",
                    "cold_budget_mib": "512",
                })
    validate_rows(rows, "b")
    controls = [
        (lambda rs: rs.pop(0), "missing arm/pass"),
        (lambda rs: rs[0].update(batch="3", logical_workloads="3"), "mixed arm batch"),
        (lambda rs: rs[0].update(event_us_per_workload="999"), "decimal/bits disagreement"),
        (lambda rs: rs[0].update(schema="q4k-pdf-ab-raw-v1"), "old schema"),
        (lambda rs: rs[0].update(cache_state="mystery"), "unknown cache state"),
    ]
    import copy
    for mutate, label in controls:
        planted = copy.deepcopy(rows)
        mutate(planted)
        try:
            validate_rows(planted, "b")
        except ValueError:
            continue
        raise AssertionError(f"protocol selftest fault escaped: {label}")
    tiny = [dict(rows[0]), dict(rows[0])]
    tiny[0]["event_ms_bits"] = f"0x{struct.unpack('<I', struct.pack('<f', 0.000032))[0]:08x}"
    tiny[1]["event_ms_bits"] = f"0x{struct.unpack('<I', struct.pack('<f', 0.000064))[0]:08x}"
    admissible, demonstrated, _ = infer_quantum(tiny)
    if admissible is not None or demonstrated is None or demonstrated >= Decimal("0.5"):
        raise AssertionError("sub-floor observed GCD was admitted as timer resolution")
    print(f"PASS: protocol synthetic controls={len(controls)}; sub-floor GCD remains diagnostic-only")


def summarize(raw: pathlib.Path, output: pathlib.Path, binary: pathlib.Path) -> None:
    with raw.open(newline="") as f:
        rows = list(csv.DictReader(f))
    expected_binary = sha256(binary)
    try:
        validate_rows(rows, expected_binary)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    states: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["shape_id"], row["cache_state"], row["arm"])].append(row)
        states[(row["shape_id"], row["cache_state"])].append(row)

    stats = {}
    for key, group in groups.items():
        values = sorted(timing_us(r) for r in group)
        clocks = sorted(int(r["sm_clock_mhz"]) for r in group)
        stats[key] = {
            "median": statistics.median(values), "min": min(values), "max": max(values),
            "clock": statistics.median(clocks), "clock_min": min(clocks), "clock_max": max(clocks),
            "pending": sum(int(r["event_pending_after_clock_query"]) for r in group),
            "samples": len(group), "row": group[0],
            "by_pass": {int(r["pass"]): timing_us(r) for r in group},
        }
    canonical_shapes = [
        "D-EXT-O", "D-EXT-K1024", "D-EXT-Q",
        "D-144-K1024-N1024", "D-144-K5120-N1024", "D-144-K5120-N5120",
        "H-G8-2048",
    ]
    shape_order = [shape for shape in canonical_shapes if any(k[0] == shape for k in stats)]
    warm_batches = sorted({int(r["batch"]) for r in rows if r["cache_state"] == "warm"})
    warm_batch_text = "/".join(map(str, warm_batches))
    samples_requested = int(rows[0]["samples_requested"])
    warmup = int(rows[0]["warmup_rounds_per_arm"])
    precondition_ms = int(rows[0]["precondition_host_enqueue_ms"])
    cold_budget_mib = int(rows[0]["cold_budget_mib"])

    lines = []
    lines += [
        "# RTX 5090：Q4_K PDF 重建版与 gemv_lowbit 同机 A/B",
        "",
        "结论边界：这是方向性实验，不是 PPU 判决，也不是 PDF 原实现的逐字复现。PDF 缺失 launcher 尾、",
        "packer、golden 和 timer；launcher 由文档第 14 页的 grid/block/smem 伪码恢复。文档主 listing 的",
        "scalar metadata 转换与解释页的 pair 转换同时保留为两个 arm，避免把两份不同代码揉成一个“原版”。",
        "",
        f"- Raw CSV: `{raw}`",
        f"- Git: `{rows[0]['git_sha']}`",
        f"- Binary SHA-256: `{rows[0]['binary_sha']}`",
        f"- Device / PCI / driver: `{rows[0]['device_name']}` / `{rows[0]['device_pci']}` / `{rows[0]['nvidia_driver_version']}`",
        "",
        "## 输入与协议",
        "",
        "- 两臂来自同一份 logical Q4_K。PDF 臂直接读 144 B/256-weight block；ours 读 Native affine int4",
        "  code plane + fp16 scale/zero(gs=32)。CPU golden 独立从 raw block 解码；任一输出不满足固定",
        "  conditioned error `<=2^-7` 时整组拒绝计时。",
        "- `weight_metadata_cold`：先触碰 `max(2×L2,128 MiB)` flush buffer，再在一个 event 中逐份读取完整且",
        f"  不重叠的 representation。cold budget={cold_budget_mib} MiB，copies=`min(64,floor(budget/max_repr))`；",
        "  两臂 cold batch 相同，ours 的 S/Z 也逐份复制，不只冷 low plane。",
        f"- `warm`：计时前每个 arm warmup {warmup} rounds；每个 event 固定 {warm_batch_text} 个 logical workloads。",
        f"  每个 shape/state/arm 保留 {samples_requested} 个原始样本；两状态前均有 {precondition_ms} ms host enqueue window",
        "  交替提交各 arm，随后同步。该窗口不是 GPU 恰好运行同样时长的声明。",
        "  每个 pass 对完整 arm 列表采用 forward/reverse 顺序；三臂时中间 arm 的位置不变，因此不冒充完整的",
        "  positional counterbalance。初始化、pack、H2D、flush 与 NVML 查询均在目标 event 外。event span 包含 GPU launch",
        "  间隙，因此是 kernel-only 的上界而非 CUPTI kernel duration 同义词。",
        "- 每个 stop event 入队后采一次 NVML SM clock，并记录 event 当时是否仍 pending。它是 adjacent snapshot，",
        "  不是 time-integrated kernel clock。",
        "- L=8 点同时报 `ours_native_grouped1`（1 kernel/workload）与 `ours_native_dense8`（8 kernels/workload）；",
        "  PDF API 只有 dense，因此是 8 kernels/workload。不能把 1-vs-8 launch 差异藏起来。",
        "  该 H-G8 shape 使用 PDF 文档给出的默认 `2x8x1`，但不是第 22 页实测 winner；raw authority 明确标为",
        "  `pdf_documented_default_unmeasured_shape`。",
        "",
        "## 原始汇总",
        "",
        "| shape | state | arm | batch | kernels/work | repr MiB | median us | min..max us | GB/s* | % of 1792 nameplate* | SM MHz median[min,max] | pending | observed GCD grid/work | policy floor/work | admissible quantum/work |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for shape in shape_order:
        for state in ["weight_metadata_cold", "warm"]:
            q, demonstrated, qwhy = infer_quantum(states[(shape, state)])
            for key in sorted(k for k in stats if k[0] == shape and k[1] == state):
                s = stats[key]
                row = s["row"]
                batch = int(row["batch"])
                observed = "UNKNOWN" if demonstrated is None else f"{float(demonstrated) / batch:.6f} us"
                admissible = "UNKNOWN" if q is None else f"{float(q) / batch:.6f} us"
                policy_floor = float(Decimal("0.5") / Decimal(batch))
                gbs = float(row["distinct_bytes"]) / (s["median"] * 1e-6) / 1e9
                lines.append(
                    f"| {shape} | {state} | {key[2]} | {batch} | {row['kernels_per_workload']} | "
                    f"{int(row['representation_bytes']) / 2**20:.3f} | {s['median']:.4f} | "
                    f"{s['min']:.4f}..{s['max']:.4f} | {gbs:.1f} | {100.0 * gbs / 1792.0:.2f}% | "
                    f"{s['clock']:.0f}[{s['clock_min']},{s['clock_max']}] | "
                    f"{s['pending']}/{s['samples']} | {observed} | {policy_floor:.6f} us | {admissible} |"
                )
            lines.append(f"<!-- timer {shape}/{state}: {qwhy} -->")
    lines += [
        "",
        "`GB/s*` 用每个 arm 自己的 distinct representation + A + D；`% of 1792 nameplate*` 是该模型速率除以 RTX 5090 的 1792 GB/s 分母。warm 行是 cache-equivalent rate，不是硬件 DRAM counter 利用率。",
        "",
        "## 相对方向（事前判据）",
        "",
        "方向只有在两臂 raw `[min,max]` 不重叠，且 median 差大于一个有效 event quantum 时才判定；否则为",
        "`UNRESOLVED`。PDF 两个 metadata variant 先各自展示，比较时采用其中更快者并明确这是文档内部歧义，",
        "不是事后把两份实现冒充成一个确定原版。",
        "本协议事前规定：总 event 的 GCD 低于 0.5 us 时只作为 observed grid 展示，不准入为计时器分辨率；",
        "因此本次 admissible quantum 全为 `UNKNOWN`，正式 verdict 全部 fail-close 为 `UNRESOLVED`。",
        "",
        "| shape | state | comparison | ratio target/PDF | sampled bands | paired target/PDF/tie | resolution-qualified verdict |",
        "|---|---|---|---:|---|---:|---|",
    ]
    comparison_count = 0
    disjoint_count = 0
    unanimous_count = 0
    for shape in shape_order:
        for state in ["weight_metadata_cold", "warm"]:
            pool = [stats[k] for k in stats if k[0] == shape and k[1] == state and k[2].startswith("pdf_")]
            pdf = min(pool, key=lambda x: x["median"])
            pdf_name = pdf["row"]["arm"]
            q, _, _ = infer_quantum(states[(shape, state)])
            comparisons = ["ours_native_dense1"] if shape != "H-G8-2048" else [
                "ours_native_dense8", "ours_native_grouped1"]
            for target_name in comparisons:
                target = stats[(shape, state, target_name)]
                batch = int(target["row"]["batch"])
                effective = None if q is None else float(q) / batch
                overlap = not (target["max"] < pdf["min"] or pdf["max"] < target["min"])
                diff = abs(target["median"] - pdf["median"])
                sampled = ("bands overlap" if overlap else "target faster" if target["max"] < pdf["min"]
                           else "selected PDF variant faster")
                pair_delta = [target["by_pass"][p] - pdf["by_pass"][p]
                              for p in sorted(target["by_pass"])]
                target_wins = sum(delta < 0 for delta in pair_delta)
                pdf_wins = sum(delta > 0 for delta in pair_delta)
                ties = sum(delta == 0 for delta in pair_delta)
                comparison_count += 1
                disjoint_count += not overlap
                unanimous_count += max(target_wins, pdf_wins) == len(pair_delta)
                if overlap:
                    verdict = "UNRESOLVED: bands overlap"
                elif effective is None:
                    verdict = "UNRESOLVED: quantum rejected by policy"
                elif diff <= effective:
                    verdict = "UNRESOLVED: median gap <= 1 quantum"
                elif target["median"] < pdf["median"]:
                    verdict = "target faster on RTX5090"
                else:
                    verdict = "selected PDF variant faster on RTX5090"
                topology = " (topology-inclusive 1-vs-8)" if target_name.endswith("grouped1") else ""
                lines.append(
                    f"| {shape} | {state} | {target_name} / {pdf_name}{topology} | "
                    f"{target['median'] / pdf['median']:.4f} | {sampled} | "
                    f"{target_wins}/{pdf_wins}/{ties} | {verdict} |"
                )
    lines += [
        "",
        f"方向性证据：{comparison_count} 个比较中 raw bands 有 {disjoint_count} 个不重叠，按同 pass 配对有",
        f"{unanimous_count} 个呈 31/31 单向。它证明 sampled direction 稳定；它不把被事前政策拒绝的",
        "32 ns observed grid 升格成可准入 timer quantum，因此不会把方向证据冒充 resolution-qualified 判决。",
        "",
        "## 不可外推的部分",
        "",
        "1. 这只消除了机器差异。5090 与 PPU 在 gs=32 上出现过 config 排名反转，因此只能提出方向与待验假设。",
        "2. PDF 第 1 页的 15/15/4 us headline、instrumented profile、以及第 22 页 `warmup 3 + 20 iters`",
        "   throughput 不是同一精确 protocol；本表不把它们拼接成一个基线。",
        "3. 原生 Q4_K 是 0.5625 B/weight；ours 是 0.625 B/weight。绝对时间可比，GB/s 必须用各自行的分子。",
        "4. PDF 未提供 L=8 grouped kernel；该行的 8 次 dense launch 是 API 事实，不是 grouped 等价实现。",
        "5. scalar/pair 两版均来自 PDF，但无法从文档判定第 22 页时间对应哪版。本结果保留两行，不替作者选择。",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"summary: {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_RAW)
    ap.add_argument("--summary", type=pathlib.Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--samples", type=int, default=31)
    ap.add_argument("--warm-batch", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--precondition-ms", type=int, default=50)
    ap.add_argument("--cold-budget-mib", type=int, default=512)
    ap.add_argument("--shape", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        protocol_selftest()
        return
    require_clean(a.allow_dirty)
    build(a.binary)
    if a.build_only:
        print(f"binary: {a.binary} sha256={sha256(a.binary)}")
        return
    if a.quick:
        a.shape, a.samples, a.warm_batch, a.warmup = "D-EXT-K1024", 3, 4, 2
        a.precondition_ms, a.cold_budget_mib = 0, 32
    bsha = sha256(a.binary)
    cmd = [
        str(a.binary), "--output", str(a.output), "--samples", str(a.samples),
        "--warm-batch", str(a.warm_batch), "--warmup", str(a.warmup),
        "--precondition-ms", str(a.precondition_ms),
        "--cold-budget-mib", str(a.cold_budget_mib),
        "--git-sha", git_sha(), "--binary-sha", bsha,
    ]
    if a.shape:
        cmd += ["--shape", a.shape]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    run(cmd, cwd=ROOT)
    summarize(a.output, a.summary, a.binary)


if __name__ == "__main__":
    main()
