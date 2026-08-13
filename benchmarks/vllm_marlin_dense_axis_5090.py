#!/usr/bin/env python3
"""Build, run, validate, and summarize the INBOX 144 dense Marlin axis.

The CUDA translation unit includes the pinned vLLM full-run source directly.
This runner verifies that source authority before building; it never writes to
the reference tree.  The older K=5120,N=1024 warm-only cell is retained as a
protocol-drift witness, not used as a substitute for the new four-cell run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import struct
import subprocess
import sys
from collections import defaultdict
from decimal import Decimal

from q4k_pdf_5090_ab import infer_quantum, require_clean, sha256


ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "benchmarks" / "vllm_marlin_dense_axis_5090.cu"
AUTHORITY = pathlib.Path("/root/ref5090/marlin/fullrun/marlin_fullrun.cu")
AUTHORITY_SHA = "686e72323967bbeea8739d28b0143f18b69c3e951375efa3e8b7fa6f96ea8cb8"
VLLM_SHA = "11ba93f3646d4c5476c3b3fd56835589701f0fb1"
ARCHIVE_JSONL = pathlib.Path("/root/ref5090/marlin/fullrun/results.jsonl")
DEFAULT_BINARY = pathlib.Path("/tmp/vllm_marlin_dense_axis_5090")
DEFAULT_RAW = ROOT / "dev" / "acu" / "vllm_marlin_dense_axis_5090_raw.csv"
DEFAULT_SUMMARY = ROOT / "dev" / "fold_derivation" / "VLLM_MARLIN_DENSE_AXIS_5090.md"
DEFAULT_SHAPES = [(1024, 1024), (1024, 5120), (5120, 1024), (5120, 5120)]
# The pinned fullrun has K=5120,N=1024, but only under an older warm-only
# single-launch protocol.  It is a drift witness, not a replacement for the
# four-cell run under this protocol.
ARCHIVED_SHAPE = (5120, 1024)
PEAK_GBS = 1792.0
TOTAL_EVENT_FLOOR_US = Decimal("0.5")
CORRECTNESS_FIXTURE = "exact_q9_a1_scale2m8_expectedKover256_fp16bits_v1"
TIMING_FIXTURE = "pinned_fullrun_seeds_b57a41d9b_s16334a2f_a91104f23_v1"


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(command, check=True, text=True, **kwargs)


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def check_authority() -> None:
    if not AUTHORITY.is_file():
        raise SystemExit(f"missing pinned source authority: {AUTHORITY}")
    actual = sha256(AUTHORITY)
    if actual != AUTHORITY_SHA:
        raise SystemExit(
            f"pinned source authority drift: expected {AUTHORITY_SHA}, got {actual}"
        )
    run(["sha256sum", "-c", "/root/ref5090/marlin/vllm-raw.sha256"],
        cwd="/root/ref5090/marlin", stdout=subprocess.DEVNULL)


def require_single_5090() -> None:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,pci.bus_id,driver_version",
         "--format=csv,noheader"], text=True
    ).splitlines()
    rows = [row.strip() for row in rows if row.strip()]
    if len(rows) != 1:
        raise SystemExit(f"requires exactly one visible GPU, found {len(rows)}: {rows}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4 or fields[1] != "12.0" or "RTX 5090" not in fields[0]:
        raise SystemExit(f"requires one RTX 5090 / sm_120, got {rows[0]!r}")


def build(binary: pathlib.Path) -> None:
    check_authority()
    require_single_5090()
    binary.parent.mkdir(parents=True, exist_ok=True)
    torch = pathlib.Path("/root/miniconda3/lib/python3.12/site-packages/torch/include")
    command = [
        "/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17",
        "--expt-relaxed-constexpr", "-lineinfo", "-arch=sm_120",
        "-I/root/ref5090/marlin/probe-shim",
        "-I/root/ref5090/marlin/vllm-raw/csrc",
        f"-I{torch}", f"-I{torch / 'torch/csrc/api/include'}",
        str(SOURCE), "-lnvidia-ml", "-o", str(binary),
    ]
    run(command, cwd=ROOT)


def decoded_total_us(row: dict[str, str]) -> Decimal:
    bits = int(row["event_ms_bits"], 16)
    ms = struct.unpack("<f", struct.pack("<I", bits))[0]
    return Decimal.from_float(ms) * Decimal(1000)


def per_workload_us(row: dict[str, str]) -> float:
    return float(decoded_total_us(row) / Decimal(int(row["batch"])))


def expected_distinct_bytes(k: int, n: int) -> int:
    return k * 2 + k * n // 2 + (k // 32) * n * 2 + n * 2


def validate_rows(
    rows: list[dict[str, str]], binary_sha: str, expected_shapes: set[tuple[int, int]]
) -> None:
    if not rows:
        raise ValueError("raw CSV is empty")
    required = {
        "schema", "repo_git_sha", "authority_source_sha", "binary_sha",
        "vllm_commit", "device_name", "device_pci", "driver_version",
        "shape_id", "m", "n", "k", "group_size", "cache_state", "pass",
        "arm_order", "protocol_order", "batch", "cold_copy_count", "l2_bytes",
        "flush_bytes", "event_ms_bits", "event_total_us", "event_us_per_workload",
        "sm_clock_mhz", "event_pending_after_clock_query", "correctness_hash",
        "correctness_expected_bits", "correctness_fixture", "timing_fixture",
        "distinct_bytes", "weight_bytes",
        "scale_bytes", "activation_bytes", "output_bytes", "kernel_config",
        "samples_requested", "warmup_rounds", "precondition_host_enqueue_ms",
        "cold_budget_mib", "timing_scope", "clock_scope",
    }
    if not required.issubset(rows[0]):
        raise ValueError(f"raw schema missing {sorted(required - set(rows[0]))}")
    fixed = [
        "schema", "repo_git_sha", "authority_source_sha", "binary_sha",
        "vllm_commit", "device_name", "device_pci", "driver_version",
        "samples_requested", "warmup_rounds", "precondition_host_enqueue_ms",
        "cold_budget_mib", "timing_scope", "clock_scope", "protocol_order",
        "correctness_fixture", "timing_fixture",
    ]
    for field in fixed:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(f"raw CSV mixes protocol field {field}: {sorted(values)}")
    first = rows[0]
    if first["schema"] != "vllm-marlin-dense-axis-raw-v1":
        raise ValueError(f"unknown schema {first['schema']!r}")
    if first["authority_source_sha"] != AUTHORITY_SHA:
        raise ValueError("raw source authority SHA does not match the pinned source")
    if first["binary_sha"] != binary_sha:
        raise ValueError("raw binary SHA does not match the executable")
    if first["vllm_commit"] != VLLM_SHA:
        raise ValueError("raw vLLM commit does not match the pinned source")
    if first["timing_scope"] != "cuda_event_gpu_span":
        raise ValueError("timing scope is not a CUDA event GPU span")
    if first["clock_scope"] != "nvml_adjacent_snapshot":
        raise ValueError("clock scope is not an adjacent NVML snapshot")
    if first["protocol_order"] != "single_arm_no_counterbalance":
        raise ValueError("single-arm protocol identity changed")
    if first["correctness_fixture"] != CORRECTNESS_FIXTURE:
        raise ValueError("raw correctness fixture identity changed")
    if first["timing_fixture"] != TIMING_FIXTURE:
        raise ValueError("raw timing fixture is not the pinned fullrun seed set")

    groups: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        k, n = int(row["k"]), int(row["n"])
        shape = (k, n)
        state = row["cache_state"]
        if shape not in expected_shapes:
            raise ValueError(f"unexpected shape {shape}")
        if row["shape_id"] != f"K{k}_N{n}":
            raise ValueError(f"shape ID/dimensions disagree for {shape}")
        if row["m"] != "1" or row["group_size"] != "32":
            raise ValueError(f"non-M1/gs32 row in dense axis: {shape}")
        if state not in {"weight_metadata_cold", "warm"}:
            raise ValueError(f"unknown cache state {state!r}")
        if int(row["arm_order"]) != 0:
            raise ValueError("single arm must have arm_order=0")
        batch = int(row["batch"])
        copies = int(row["cold_copy_count"])
        if batch <= 0 or (state == "weight_metadata_cold" and copies != batch):
            raise ValueError(f"cold batch/copy mismatch for {shape}")
        if state == "warm" and copies != 0:
            raise ValueError(f"warm row unexpectedly declares cold copies for {shape}")
        total = float(decoded_total_us(row))
        per = total / batch
        if not math.isclose(total, float(row["event_total_us"]), rel_tol=2e-7, abs_tol=1e-7):
            raise ValueError(f"decimal event total disagrees with event bits for {shape}")
        if not math.isclose(per, float(row["event_us_per_workload"]), rel_tol=2e-7, abs_tol=1e-7):
            raise ValueError(f"decimal event/workload disagrees with event bits for {shape}")
        if int(row["distinct_bytes"]) != expected_distinct_bytes(k, n):
            raise ValueError(f"Marlin distinct-byte model drift for {shape}")
        expected_bits = struct.unpack("<H", struct.pack("<e", k / 256.0))[0]
        if int(row["correctness_expected_bits"], 16) != expected_bits:
            raise ValueError(f"constant fixture oracle drift for {shape}")
        if int(row["sm_clock_mhz"]) <= 0:
            raise ValueError(f"invalid adjacent SM clock for {shape}")
        groups[(k, n, state)].append(row)

    requested = int(first["samples_requested"])
    expected_keys = {
        (k, n, state)
        for k, n in expected_shapes
        for state in ("weight_metadata_cold", "warm")
    }
    if set(groups) != expected_keys:
        raise ValueError(f"missing/extra shape-state groups: {set(groups) ^ expected_keys}")
    for key, group in groups.items():
        passes = sorted(int(row["pass"]) for row in group)
        if len(group) != requested or passes != list(range(requested)):
            raise ValueError(f"group {key} has incomplete samples: {passes}")
        for field in (
            "batch", "correctness_hash", "correctness_expected_bits", "distinct_bytes",
            "correctness_fixture", "timing_fixture", "weight_bytes", "scale_bytes",
            "activation_bytes", "output_bytes",
            "kernel_config",
        ):
            if len({row[field] for row in group}) != 1:
                raise ValueError(f"group {key} mixes {field}")


def protocol_selftest() -> None:
    # The negative controls target the authority boundaries, not arithmetic
    # copied out of validate_rows: a raw row cannot choose its own binary or
    # source identity, and a single-arm run cannot masquerade as counterbalanced.
    base = {
        "schema": "vllm-marlin-dense-axis-raw-v1", "repo_git_sha": "g",
        "authority_source_sha": AUTHORITY_SHA, "binary_sha": "b",
        "vllm_commit": VLLM_SHA, "device_name": "NVIDIA GeForce RTX 5090",
        "device_pci": "0000:01:00.0", "driver_version": "d",
        "shape_id": "K1024_N1024", "m": "1", "n": "1024", "k": "1024",
        "group_size": "32", "arm_order": "0",
        "protocol_order": "single_arm_no_counterbalance", "l2_bytes": "1",
        "flush_bytes": "2", "event_ms_bits": "0x3c23d70a",
        "event_total_us": "10.0000004749745", "event_us_per_workload": "1.00000004749745",
        "sm_clock_mhz": "2400", "event_pending_after_clock_query": "1",
        "correctness_hash": "1", "correctness_expected_bits": "0x4400",
        "correctness_fixture": CORRECTNESS_FIXTURE, "timing_fixture": TIMING_FIXTURE,
        "distinct_bytes": str(expected_distinct_bytes(1024, 1024)),
        "weight_bytes": str(1024 * 1024 // 2), "scale_bytes": str(1024 // 32 * 1024 * 2),
        "activation_bytes": str(1024 * 2), "output_bytes": str(1024 * 2),
        "kernel_config": "cfg", "samples_requested": "2", "warmup_rounds": "1",
        "precondition_host_enqueue_ms": "1", "cold_budget_mib": "1",
        "timing_scope": "cuda_event_gpu_span", "clock_scope": "nvml_adjacent_snapshot",
    }
    rows: list[dict[str, str]] = []
    for state, batch, copies in (("weight_metadata_cold", 10, 10), ("warm", 10, 0)):
        for pass_id in range(2):
            row = dict(base)
            row.update(cache_state=state, batch=str(batch),
                       cold_copy_count=str(copies))
            row["pass"] = str(pass_id)
            rows.append(row)
    validate_rows(rows, "b", {(1024, 1024)})
    import copy
    controls = [
        ("authority", lambda r: r[0].update(authority_source_sha="wrong")),
        ("binary", lambda r: r[0].update(binary_sha="wrong")),
        ("single-arm identity", lambda r: r[0].update(protocol_order="forward_reverse")),
        ("missing sample", lambda r: r.pop()),
        ("cold/copy mismatch", lambda r: r[0].update(cold_copy_count="9")),
        ("byte-model drift", lambda r: r[0].update(distinct_bytes="1")),
        ("timing fixture substitution", lambda r: r[0].update(timing_fixture="constant")),
    ]
    for label, mutate in controls:
        planted = copy.deepcopy(rows)
        mutate(planted)
        try:
            validate_rows(planted, "b", {(1024, 1024)})
        except ValueError:
            continue
        raise AssertionError(f"negative control escaped: {label}")
    print(f"PASS: dense Marlin axis protocol controls={len(controls)}")


def archived_dense(k: int, n: int) -> dict:
    found = []
    with ARCHIVE_JSONL.open() as stream:
        for line in stream:
            item = json.loads(line)
            case = item.get("case", {})
            if (case.get("kind"), case.get("M"), case.get("K"), case.get("N")) == (
                "dense", 1, k, n
            ):
                found.append(item)
    if len(found) != 1:
        raise ValueError(f"expected one archived K{k}/N{n} result, found {len(found)}")
    item = found[0]
    if item["row"]["vllm_commit"] != VLLM_SHA:
        raise ValueError("archived reference is not from the pinned vLLM commit")
    return item


def summarize(raw: pathlib.Path, output: pathlib.Path, binary: pathlib.Path,
              expected_shapes: set[tuple[int, int]]) -> None:
    with raw.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    validate_rows(rows, sha256(binary), expected_shapes)
    groups: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["k"]), int(row["n"]), row["cache_state"])].append(row)

    lines = [
        "# RTX 5090 dense vLLM Marlin：K/N 轴（INBOX 144）",
        "",
        "范围：这是本机 RTX 5090 / SM120 的 vLLM dense Marlin W4A16、M=1、gs=32 测量，",
        "不是 PPU 结论。分母固定为该 5090 的 1792 GB/s；仓库已有 gs=32 在两台机器上",
        "config 排名反转的记录，因此不能把这里的排序外推到 PPU。",
        "",
        f"- Raw CSV: `{raw}`",
        f"- Repository SHA: `{rows[0]['repo_git_sha']}`",
        f"- Measurement source authority: `{AUTHORITY}` / `{rows[0]['authority_source_sha']}`",
        f"- vLLM commit: `{rows[0]['vllm_commit']}`",
        f"- Binary SHA-256: `{rows[0]['binary_sha']}`",
        f"- Device / PCI / driver: `{rows[0]['device_name']}` / `{rows[0]['device_pci']}` / `{rows[0]['driver_version']}`",
        "",
        "## 协议",
        "",
        "- 直接 include pinned `marlin_fullrun.cu`（rename `main`），复用原 `Config`、",
        "  `select_dense_config`、`DenseLaunch` 和 kernel 实例化；没有复制 selector/launch ABI。",
        "- 计时前先做非零、可精确预测的 correctness gate：A=1，所有 int4 code=9，",
        "  fp16 scale=1/256，故每个输出严格为 K/256；所有 fp16 bits 必须逐位相等。",
        f"  correctness fixture identity=`{CORRECTNESS_FIXTURE}`。通过后才把 A/B/scale 全量重填为",
        f"  pinned fullrun 的三个原始 seed；timing fixture identity=`{TIMING_FIXTURE}`。因此常量",
        "  correctness 数据不会进入计时，也不会把压缩性带入 cold HBM 结果。",
        "- cold：event 外先触碰 `max(2×L2,128 MiB)`，event 内依次消费互不重叠的完整",
        "  B+scale replica。warm：同一 B/scale 上批量调用。两者均 31 个独立 event 样本；",
        "  初始化、correctness、warmup、precondition、flush 与 NVML 查询均在 event 外。",
        "- 单 arm 不存在 forward/reverse；raw 明确记录 `single_arm_no_counterbalance`，不冒充",
        "  positional counterbalance。stop event 后紧邻采 NVML clock，并保留 binary32 event bits。",
        "- 事前分辨率政策沿用 q4k：总 event 小于 0.5 us 的 observed GCD 不准入。表中同时",
        "  给 `policy-min/work = 0.5 us / batch` 与由 event bits 推出的 admissible resolution；",
        "  后者拿不到即标 `QUANTUM UNKNOWN`；拿到则标 `QUANTUM ESTABLISHED`。单臂绝对时间",
        "  没有“赢家”判决；quantum 只约束后续差值。",
        "",
        "## 新测四格",
        "",
        "| K | N | state | batch | median us | min..max us | distinct MiB | GB/s* | % of 1792 HBM | policy-min/work us | observed GCD/work | admissible resolution/work | resolution status | clock MHz median[min,max] | pending |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for key in sorted(groups):
        k, n, state = key
        group = groups[key]
        values = sorted(per_workload_us(row) for row in group)
        median = statistics.median(values)
        row = group[0]
        batch = int(row["batch"])
        distinct = int(row["distinct_bytes"])
        gbs = distinct / (median * 1e-6) / 1e9
        floor = float(TOTAL_EVENT_FLOOR_US / Decimal(batch))
        admissible, observed, why = infer_quantum(group)
        observed_text = "UNKNOWN" if observed is None else f"{float(observed) / batch:.6f}"
        resolution_text = (
            "UNKNOWN" if admissible is None else f"{float(admissible) / batch:.6f}"
        )
        resolution_status = (
            "QUANTUM UNKNOWN" if admissible is None else "QUANTUM ESTABLISHED"
        )
        clocks = sorted(int(item["sm_clock_mhz"]) for item in group)
        pending = sum(int(item["event_pending_after_clock_query"]) for item in group)
        lines.append(
            f"| {k} | {n} | {state} | {batch} | {median:.4f} | "
            f"{min(values):.4f}..{max(values):.4f} | {distinct / 2**20:.4f} | "
            f"{gbs:.2f} | {100 * gbs / PEAK_GBS:.3f}% | {floor:.6f} | "
            f"{observed_text} | {resolution_text} | {resolution_status} | "
            f"{statistics.median(clocks):.0f}[{min(clocks)},{max(clocks)}] | "
            f"{pending}/{len(group)} |"
        )
        lines.append(f"<!-- K{k}/N{n}/{state}: {why}; admissible={admissible} -->")
    lines += [
        "",
        "`GB/s*` 使用 Marlin 自己的 distinct bytes：A + biased-int4 B + fp16 scale(gs32) + D。",
        "warm 超过 100% 也只表示 cache-equivalent rate，不是 DRAM counter。cold event 每次读取互不",
        "重叠的 B+scale replica；A 与 D 是同一 buffer（flush 后重用），所以 distinct 分子按一次 A/D",
        "计费，并不冒充四个 operand 都逐 batch 复制。",
    ]
    if (5120, 1024, "warm") in groups:
        archived = archived_dense(5120, 1024)
        result, row = archived["result"], archived["row"]
        group = groups[(5120, 1024, "warm")]
        values = sorted(per_workload_us(item) for item in group)
        median = statistics.median(values)
        batch = int(group[0]["batch"])
        admissible, _, _ = infer_quantum(group)
        floor = None if admissible is None else float(admissible / Decimal(batch))
        delta = abs(median - float(result["median_us"]))
        overlap = not (
            max(values) < float(result["min_us"])
            or float(result["max_us"]) < min(values)
        )
        verdict = (
            "DRIFT UNRESOLVED: sampled bands overlap"
            if overlap
            else "DRIFT UNRESOLVED: current admissible quantum unavailable"
            if floor is None
            else "DRIFT UNRESOLVED: delta <= current admissible quantum"
            if delta <= floor
            else "PROTOCOL DRIFT OBSERVED (not a kernel regression verdict)"
        )
        lines += [
            "",
            "## 旧协议交叉检查：K=5120, N=1024",
            "",
            f"Pinned fullrun case `{row['case_id']}`: {result['median_us']:.6f} us, "
            f"{float(row['mbu_pct']):.6f}% of 1792 GB/s, raw band "
            f"{result['min_us']:.6f}..{result['max_us']:.6f} us, 31 samples。",
            "",
            "两次使用相同的 pinned random fill seeds，故 fixture matched；但该旧协议是",
            "`warm same buffers; no explicit L2 flush`、每个 event 只发一次，且当时没有",
            "事前注册 0.5-us event-floor。因此它不替代本次四格中的任何一格，只用来检查口径漂移；",
            "旧值自身的分辨率状态为 **QUANTUM UNKNOWN (legacy protocol had no registered floor)**。",
            "",
            f"本次 warm: {median:.6f} us vs archived {float(result['median_us']):.6f} us; "
            f"delta={delta:.6f} us, current admissible resolution/work="
            f"{'UNKNOWN' if floor is None else f'{floor:.6f} us'}; **{verdict}**。",
        ]
    anchor_rows = [archived_dense(8192, 5120), archived_dense(5120, 8192)]
    lines += [
        "",
        "## 旧 warm anchors（引用，不重测）",
        "",
        "| K | N | archived median us | archived MBU | scope |",
        "|---:|---:|---:|---:|---|",
    ]
    for item in anchor_rows:
        result, row = item["result"], item["row"]
        lines.append(
            f"| {int(row['K'])} | {int(row['N'])} | {float(result['median_us']):.6f} | "
            f"{float(row['mbu_pct']):.6f}% | cache-equivalent; old warm-only single-launch protocol |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "1. K=1024 的低利用率事前视为小工作量/setup 负控；不能仅凭它判实现有墙。",
        "2. K=5120 若仍低，才是需要解释的主信号；解释前必须按每权重/每输出元素归一，",
        "   不能用总指令或总字节直接比较。",
        "3. 本报告只含 M=1，因此只报 %HBM；没有把 M>1 MFU 混进同一列。",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"summary: {output}")


def parse_shape(text: str) -> tuple[int, int]:
    try:
        k, n = (int(value) for value in text.lower().split("x", 1))
    except Exception as exc:
        raise argparse.ArgumentTypeError("shape must be KxN") from exc
    if k <= 0 or n <= 0 or k % 32 or n % 64:
        raise argparse.ArgumentTypeError("shape must be positive, K%32=0, N%64=0")
    return k, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_RAW)
    parser.add_argument("--summary", type=pathlib.Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--shape", action="append", type=parse_shape)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--warm-batch", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--precondition-ms", type=int, default=50)
    parser.add_argument("--cold-budget-mib", type=int, default=512)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        protocol_selftest()
        return
    require_clean(args.allow_dirty)
    shapes = list(args.shape or DEFAULT_SHAPES)
    if len(set(shapes)) != len(shapes):
        raise SystemExit("duplicate shape requested")
    if args.quick:
        shapes = [(1024, 1024)]
        args.samples, args.warm_batch, args.warmups = 3, 4, 2
        args.precondition_ms, args.cold_budget_mib = 0, 32
    elif shapes != DEFAULT_SHAPES:
        raise SystemExit(
            "formal axis shape list must equal DEFAULT_SHAPES exactly; use --quick for a development subset"
        )
    build(args.binary)
    if args.build_only:
        print(f"binary: {args.binary} sha256={sha256(args.binary)}")
        return
    binary_sha = sha256(args.binary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.binary), "--output", str(args.output),
        "--shapes", ",".join(f"{k}x{n}" for k, n in shapes),
        "--repo-sha", git_sha(), "--authority-sha", AUTHORITY_SHA,
        "--binary-sha", binary_sha, "--samples", str(args.samples),
        "--warm-batch", str(args.warm_batch), "--warmups", str(args.warmups),
        "--precondition-ms", str(args.precondition_ms),
        "--cold-budget-mib", str(args.cold_budget_mib),
    ]
    run(command, cwd=ROOT)
    summarize(args.output, args.summary, args.binary, set(shapes))


if __name__ == "__main__":
    main()
