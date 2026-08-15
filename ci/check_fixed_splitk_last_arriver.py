#!/usr/bin/env python3
"""Fail-closed source/ABI contract for the fixed Split-K fused completion path."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_fixed_splitk_last_arriver.hpp"
KERNEL = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp"
DENSE = ROOT / "quactlize/include/dense_splitk_parallel_ppu.cuh"
REDUCER = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"


def valid_policy(text: str, reducer: str) -> bool:
    required_once = (
        "__threadfence();",
        "shared.old_count = atomicAdd(",
        "completion_arrival_is_last(old_count, work.peer_count)",
        "ppu.ld.global.acquire.gpu.b32",
        "shared.published_count = load_acquire(",
        "uint32_t(shared.published_count) != work.peer_count",
        "reduce_fp32_volatile_fixed_partition_order<",
        "atomicExch(",
        "case 2: reduce_tile<2>",
        "case 4: reduce_tile<4>",
        "case 8: reduce_tile<8>",
    )
    if any(text.count(token) != 1 for token in required_once):
        return False
    if text.count("work.completion_slot()") != 3 or text.count("__syncthreads();") < 5:
        return False
    if "work.is_final_peer()" in text or "while (" in text or "while(" in text:
        return False
    for token in (
        "int64_t const tile_column = int64_t(work.q) * int64_t(tile_n)",
        "params.partition_stride, column",
        "*reinterpret_cast<FragmentOutput*>(params.destination + column) = output",
        "static_cast<void const*>(args.partials) !=",
        "static_cast<void const*>(partial.ptr_D)",
    ):
        if token not in text:
            return False
    if text.count("disjoint_ranges(") != 4:
        return False
    volatile_begin = reducer.find("reduce_fp32_volatile_fixed_partition_order(")
    volatile_end = reducer.find(
        "template <int ElementsPerAccess>\nclass PpuMixedInputSplitKParallelReductionKernel",
        volatile_begin,
    )
    if volatile_begin < 0 or volatile_end <= volatile_begin:
        return False
    volatile_body = reducer[volatile_begin:volatile_end]
    reducer_tokens = (
        "reduce_fp32_volatile_fixed_partition_order(",
        "for (int split = 0; split < Partitions; ++split)",
        "accumulator[lane] += partial[lane]",
    )
    if any(token not in volatile_body for token in reducer_tokens):
        return False
    after = text.find("static void after_partial(")
    fence = text.find("__threadfence();", after)
    arrive = text.find("shared.old_count = atomicAdd(", fence)
    decide = text.find("completion_arrival_is_last", arrive)
    acquire = text.find("shared.published_count = load_acquire(", decide)
    published = text.find(
        "uint32_t(shared.published_count) != work.peer_count", acquire)
    reduce = text.find("switch (work.peer_count)", published)
    reset = text.find("atomicExch(", reduce)
    return -1 not in (
        after, fence, arrive, decide, acquire, published, reduce, reset
    ) and (
        after < fence < arrive < decide < acquire < published < reduce < reset
    )


def main() -> int:
    policy = POLICY.read_text()
    kernel = KERNEL.read_text()
    dense = DENSE.read_text()
    reducer = REDUCER.read_text()
    if not valid_policy(policy, reducer):
        print("[l196:source] FAIL: production last-arriver sequence is incomplete")
        return 1
    partial_call = kernel.find("partial_epilogue(partial_shape")
    completion_call = kernel.find("CompletionPolicy::after_partial(")
    if partial_call < 0 or completion_call <= partial_call:
        print("[l196:source] FAIL: completion is not sequenced after the partial epilogue")
        return 1
    dense_tokens = (
        "using FusedCompletion =",
        "LastArriverM1Fp16Completion<2>",
        "run_fused_last_arriver",
        "reset_fused_counters_for_diagnostics",
        "if (splits_ == 1) return shipping_.run(stream);",
        "query_fused_workspace_plan",
    )
    if any(dense.count(token) == 0 for token in dense_tokens):
        print("[l196:source] FAIL: dense launcher lost S1/fused/workspace authority")
        return 1

    plants = {
        "missing-fence": policy.replace("__threadfence();", "/* planted missing fence */", 1),
        "q-alias": policy.replace("work.completion_slot()", "work.peer_idx", 2),
        "logical-final": policy.replace(
            "completion_arrival_is_last(old_count, work.peer_count)",
            "work.is_final_peer()", 1),
        "nonvolatile-read": policy.replace(
            "reduce_fp32_volatile_fixed_partition_order<",
            "reduce_fp32_aligned_fixed_partition_order<", 1),
        "missing-acquire": policy.replace(
            "shared.published_count = load_acquire(",
            "shared.published_count = atomicAdd(", 1),
        "missing-reset": policy.replace("atomicExch(", "/* planted */ atomicAdd(", 1),
        "missing-S4": policy.replace("case 4: reduce_tile<4>", "case 4: reduce_tile<2>", 1),
        "missing-cta-sync": policy.replace("__syncthreads();", "/* planted */", 5),
        "output-q-alias": policy.replace(
            "int64_t const tile_column = int64_t(work.q) * int64_t(tile_n)",
            "int64_t const tile_column = int64_t(work.peer_idx) * int64_t(tile_n)",
            1),
        "partial-stride-zero": policy.replace(
            "params.partition_stride, column", "int64_t(0), column", 1),
        "missing-output-store": policy.replace(
            "*reinterpret_cast<FragmentOutput*>(params.destination + column) = output",
            "/* planted missing D store */", 1),
        "unbound-partials": policy.replace(
            "static_cast<void const*>(partial.ptr_D)",
            "static_cast<void const*>(args.partials)", 1),
    }
    escaped = [
        name for name, planted in plants.items()
        if valid_policy(planted, reducer)
    ]
    loop = "for (int split = 0; split < Partitions; ++split)"
    loop_pos = reducer.find(
        loop, reducer.find("reduce_fp32_volatile_fixed_partition_order("))
    reverse_reducer = reducer[:loop_pos] + (
        "for (int split = Partitions - 1; split >= 0; --split)"
    ) + reducer[loop_pos + len(loop):]
    reducer_plants = {"reverse-numeric-order": reverse_reducer}
    escaped += [
        name for name, planted in reducer_plants.items()
        if valid_policy(policy, planted)
    ]
    if escaped:
        print(f"[l196:source] FAIL: planted defects escaped: {','.join(escaped)}")
        return 1
    print(
        "[l196:source] PASS: partial->fence->fetch-old->actual-last->"
        "acquire->volatile fixed-order reduce->reset; plants=13 EXPECTED_RED"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
