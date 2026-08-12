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


def infer_quantum(rows: list[dict[str, str]]) -> tuple[Decimal | None, str]:
    values = sorted({decoded_event_us(r) for r in rows})
    if len(values) < 2:
        return None, "fewer than two distinct raw event values"
    ns = Decimal("0.001")
    ints = [int(v.quantize(ns, rounding=ROUND_HALF_EVEN) / ns) for v in values]
    gcd = 0
    for value in ints:
        gcd = math.gcd(gcd, abs(value))
    quantum = Decimal(gcd) * ns
    if quantum < Decimal("0.5"):
        return None, f"GCD {quantum} us below the conservative 0.5-us floor"
    return quantum, f"integer-nanosecond GCD over {len(values)} distinct binary32 event values"


def fnum(value: float) -> str:
    return f"{value:.4f}"


def summarize(raw: pathlib.Path, output: pathlib.Path, binary: pathlib.Path) -> None:
    with raw.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("raw CSV is empty")
    required = {
        "event_ms_bits", "event_us_per_workload", "correctness_hash",
        "representation_bytes", "distinct_bytes", "event_pending_after_clock_query",
    }
    if not required.issubset(rows[0]):
        raise SystemExit(f"raw schema missing {sorted(required - set(rows[0]))}")
    identities = {(r["git_sha"], r["binary_sha"], r["device_pci"], r["driver"]) for r in rows}
    if len(identities) != 1:
        raise SystemExit(f"raw CSV merged incompatible runs: {identities}")
    expected_binary = sha256(binary)
    if rows[0]["binary_sha"] != expected_binary:
        raise SystemExit("raw binary SHA does not match the executable being summarized")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    states: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["shape_id"], row["cache_state"], row["arm"])].append(row)
        states[(row["shape_id"], row["cache_state"])].append(row)

    stats = {}
    for key, group in groups.items():
        values = sorted(float(r["event_us_per_workload"]) for r in group)
        clocks = sorted(int(r["sm_clock_mhz"]) for r in group)
        stats[key] = {
            "median": statistics.median(values), "min": min(values), "max": max(values),
            "clock": statistics.median(clocks), "clock_min": min(clocks), "clock_max": max(clocks),
            "pending": sum(int(r["event_pending_after_clock_query"]) for r in group),
            "samples": len(group), "row": group[0],
        }
    canonical_shapes = ["D-EXT-O", "D-EXT-K1024", "D-EXT-Q", "H-G8-2048"]
    shape_order = [shape for shape in canonical_shapes if any(k[0] == shape for k in stats)]
    warm_batches = sorted({int(r["batch"]) for r in rows if r["cache_state"] == "warm"})
    warm_batch_text = "/".join(map(str, warm_batches))

    lines = []
    lines += [
        "# RTX 5090：Q4_K PDF 重建版与 gemv_lowbit 同机 A/B",
        "",
        "结论边界：这是方向性实验，不是 PPU 判决，也不是 PDF 原实现的逐字复现。PDF 缺失 launcher 尾、",
        "packer、golden 和 timer；launcher 由文档第 14 页的 grid/block/smem 伪码恢复。文档主 listing 的",
        "scalar metadata 转换与解释页的 pair 转换同时保留为两个 arm，避免把两份不同代码揉成一个“原版”。",
        "",
        f"Raw CSV: `{raw}`  ",
        f"Git: `{rows[0]['git_sha']}`  ",
        f"Binary SHA-256: `{rows[0]['binary_sha']}`  ",
        f"Device PCI / CUDA driver integer: `{rows[0]['device_pci']}` / `{rows[0]['driver']}`",
        "",
        "## 输入与协议",
        "",
        "- 两臂来自同一份 logical Q4_K。PDF 臂直接读 144 B/256-weight block；ours 读 Native affine int4",
        "  code plane + fp16 scale/zero(gs=32)。CPU golden 独立从 raw block 解码；任一输出不满足固定",
        "  conditioned error `<=2^-7` 时整组拒绝计时。",
        "- `weight_metadata_cold`：先触碰 `max(2×L2,128 MiB)` flush buffer，再在一个 event 中逐份读取完整且",
        "  不重叠的 representation。两臂 cold batch 相同；ours 的 S/Z 也逐份复制，不只冷 low plane。",
        f"- `warm`：计时前 warmup；本次每个 event 固定 {warm_batch_text} 个 logical workloads。每种状态保留全部原始样本，",
        "  AB/BA 交替；初始化、pack、H2D、flush 与 NVML 查询均在目标 event 外。event span 包含 GPU launch",
        "  间隙，因此是 kernel-only 的上界而非 CUPTI kernel duration 同义词。",
        "- 每个 stop event 入队后采一次 NVML SM clock，并记录 event 当时是否仍 pending。它是 adjacent snapshot，",
        "  不是 time-integrated kernel clock。",
        "- L=8 点同时报 `ours_native_grouped1`（1 kernel/workload）与 `ours_native_dense8`（8 kernels/workload）；",
        "  PDF API 只有 dense，因此是 8 kernels/workload。不能把 1-vs-8 launch 差异藏起来。",
        "",
        "## 原始汇总",
        "",
        "| shape | state | arm | kernels/work | repr MiB | median us | min..max us | GB/s* | SM MHz median[min,max] | pending | quantum/work |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for shape in shape_order:
        for state in ["weight_metadata_cold", "warm"]:
            q, qwhy = infer_quantum(states[(shape, state)])
            for key in sorted(k for k in stats if k[0] == shape and k[1] == state):
                s = stats[key]
                row = s["row"]
                batch = int(row["batch"])
                qwork = "UNKNOWN" if q is None else f"{float(q) / batch:.4f} us"
                gbs = float(row["distinct_bytes"]) / (s["median"] * 1e-6) / 1e9
                lines.append(
                    f"| {shape} | {state} | {key[2]} | {row['kernels_per_workload']} | "
                    f"{int(row['representation_bytes']) / 2**20:.3f} | {s['median']:.4f} | "
                    f"{s['min']:.4f}..{s['max']:.4f} | {gbs:.1f} | "
                    f"{s['clock']:.0f}[{s['clock_min']},{s['clock_max']}] | "
                    f"{s['pending']}/{s['samples']} | {qwork} |"
                )
            lines.append(f"<!-- timer {shape}/{state}: {qwhy} -->")
    lines += [
        "",
        "`GB/s*` 用每个 arm 自己的 distinct representation + A + D。warm 行是 cache-equivalent rate，不是 DRAM 利用率。",
        "",
        "## 相对方向（事前判据）",
        "",
        "方向只有在两臂 raw `[min,max]` 不重叠，且 median 差大于一个有效 event quantum 时才判定；否则为",
        "`UNRESOLVED`。PDF 两个 metadata variant 先各自展示，比较时采用其中更快者并明确这是文档内部歧义，",
        "不是事后把两份实现冒充成一个确定原版。",
        "",
        "| shape | state | comparison | ratio target/PDF | verdict |",
        "|---|---|---|---:|---|",
    ]
    for shape in shape_order:
        for state in ["weight_metadata_cold", "warm"]:
            pool = [stats[k] for k in stats if k[0] == shape and k[1] == state and k[2].startswith("pdf_")]
            pdf = min(pool, key=lambda x: x["median"])
            pdf_name = pdf["row"]["arm"]
            q, _ = infer_quantum(states[(shape, state)])
            comparisons = ["ours_native_dense1"] if shape != "H-G8-2048" else [
                "ours_native_dense8", "ours_native_grouped1"]
            for target_name in comparisons:
                target = stats[(shape, state, target_name)]
                batch = int(target["row"]["batch"])
                effective = None if q is None else float(q) / batch
                overlap = not (target["max"] < pdf["min"] or pdf["max"] < target["min"])
                diff = abs(target["median"] - pdf["median"])
                if overlap or effective is None or diff <= effective:
                    verdict = "UNRESOLVED"
                elif target["median"] < pdf["median"]:
                    verdict = "target faster on RTX5090"
                else:
                    verdict = "PDF reconstruction faster on RTX5090"
                topology = " (topology-inclusive 1-vs-8)" if target_name.endswith("grouped1") else ""
                lines.append(
                    f"| {shape} | {state} | {target_name} / {pdf_name}{topology} | "
                    f"{target['median'] / pdf['median']:.4f} | {verdict} |"
                )
    lines += [
        "",
        "## 不可外推的部分",
        "",
        "1. 这只消除了机器差异。5090 与 PPU 在 gs=32 上出现过 config 排名反转，因此只能提出方向与待验假设。",
        "2. PDF 第 1 页的 15/15/4 us headline、instrumented profile、以及第 22 页 `warmup 3 + 20 iters`",
        "   throughput 不是同一精确 protocol；本表不把它们拼接成一个基线。",
        "3. 原生 Q4_K 是 0.5625 B/weight；ours 是 0.625 B/weight。绝对时间可比，GB/s 必须用各自行的分子。",
        "4. PDF 未提供 L=8 grouped kernel；该行的 8 次 dense launch 是 API 事实，不是 grouped 等价实现。",
        "5. scalar/pair 两版均来自 PDF，但无法从文档判定第 22 页时间对应哪版。本结果保留两行，不替作者选择。",
        "",
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
    a = ap.parse_args()
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
