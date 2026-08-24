#!/usr/bin/env python3
"""Fail-closed local contract for Q4_K native SIMT batching and decode sweep."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "quactlize/include/gguf_bc_vecdot.hpp"
BACKEND = ROOT / "quactlize/csrc/device/ppu_backend.cu"
THOP = ROOT / "quactlize/csrc/preprocess/thop/gguf_prepass_ops.cpp"
BENCH = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
POLICY = ROOT / "benchmarks/fq_q4k_decode_real_shapes_policy.json"
PLAN = ROOT / "tools/plan_fq_q4k_decode_real_shapes.py"
ANALYZE = ROOT / "tools/analyze_fq_q4k_decode_real_shapes.py"
RUNNER = ROOT / "tools/run_fq_q4k_decode_real_shapes_box.sh"


class CheckError(ValueError):
    pass


def check(kernel: str, backend: str, thop: str, bench: str,
          policy: str, plan: str, analyze: str, runner: str) -> None:
    kernel_needles = (
        "int const dense_batch_row = Grouped ? 0 : int(blockIdx.y);",
        "int const activation_row = active ? gathered_row : 0;",
        "float* row_out = out + (active ? int64_t(gathered_row) * rows : 0);",
        "dim3 grid(vecdot::vecdot_grid_size<T,RowsPerWarp>(n,Threads),max_rows,Grouped?experts:1);",
    )
    if any(token not in kernel for token in kernel_needles):
        raise CheckError("dense SIMT grid-y activation/output ownership is incomplete")
    if "for (int batch" in kernel or "for (int row_launch" in kernel:
        raise CheckError("dense SIMT batch regressed to a host launch loop")
    if backend.count("(experts == 0 && total_rows >= 8)") != 4 or \
            "nullptr, out, n, bpr, 1, total_rows, stream" not in backend:
        raise CheckError("device ABI does not admit exactly dense M=1..7")
    if "a.size(0) > 0 && a.size(0) < 8" not in thop or \
            "int experts = 0, max_rows = int(a.size(0));" not in thop:
        raise CheckError("THOP ABI does not forward the native dense batch")
    bench_needles = (
        "bool const supported = shape.m < 8;",
        "dBcOut(std::size_t(shape.m) * shape.n)",
        "nullptr, output, shape.n, bpr, 1, shape.m, nullptr",
        '"native-grid-y-m-lt8"',
        '"UNSUPPORTED_M_GE_8"',
    )
    if any(token not in bench for token in bench_needles):
        raise CheckError("benchmark does not measure native M<8 SIMT output")
    if '"decode_m": [1, 2, 4, 8, 16]' not in policy or \
            '"bc_batch_policy": "native-grid-y-m-lt8"' not in policy:
        raise CheckError("policy lost the exact decode/SIMT denominator")
    if "DECODE_M = (1, 2, 4, 8, 16)" not in plan:
        raise CheckError("planner lost one decode M")
    analyze_needles = (
        "m * n * split * int(model[\"partial_element_bytes\"])",
        "float(model[\"bandwidth_fraction\"])",
        '"SIMT_BC"',
        '"PRODUCER_PLUS_MODELED_REDUCER"',
        "if shape[0] >= 8:",
    )
    if any(token not in analyze for token in analyze_needles):
        raise CheckError("analysis mixes producer-only, reducer, or SIMT scope")
    runner_needles = (
        "--only-split=1 --bc-mode=all",
        "phase=scheduler",
        "--bc-mode=skip",
        "--bc-mode=only",
        "committed phase log changed",
        'atomic_text "$commit" "$actual"',
        ".uncommitted.",
        ".failed.",
        "analyze_fq_q4k_decode_real_shapes.py\" finalize",
    )
    if any(token not in runner for token in runner_needles):
        raise CheckError("runner does not execute screen/scheduler/TC/SIMT phases")


def main() -> int:
    texts = [path.read_text() for path in
             (KERNEL, BACKEND, THOP, BENCH, POLICY, PLAN, ANALYZE, RUNNER)]
    check(*texts)
    plants = [
        (0, ",max_rows,Grouped?experts:1);", ",1,Grouped?experts:1);"),
        (0, "out + (active ? int64_t(gathered_row) * rows : 0)", "out"),
        (3, "shape.m < 8", "shape.m <= 8"),
        (4, '"decode_m": [1, 2, 4, 8, 16]', '"decode_m": [1, 2, 4, 16]'),
        (6, '"PRODUCER_PLUS_MODELED_REDUCER"', '"PRODUCER_ONLY"'),
    ]
    for index, old, new in plants:
        planted = list(texts)
        if old not in planted[index]:
            raise CheckError(f"negative-control seam missing: {old}")
        planted[index] = planted[index].replace(old, new, 1)
        try:
            check(*planted)
        except CheckError:
            pass
        else:
            raise CheckError(f"negative control stayed green: {old} -> {new}")
    print("[fq-q4k-decode:self-test] PASS: native one-launch M<8 SIMT, "
          "exact M denominator, phase separation, and five negative plants")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, OSError) as error:
        print(f"[fq-q4k-decode:self-test] FAIL: {error}")
        raise SystemExit(2)
