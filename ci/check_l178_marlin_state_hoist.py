#!/usr/bin/env python3
"""Bind standalone Marlin's final address state to its production device body."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COLLECTIVE = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/"
    "marlin_collective_ppu.hpp"
)
KERNEL = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/kernel/"
    "marlin_kernel_ppu.hpp"
)


class ContractError(RuntimeError):
    pass


def body(text: str, needle: str) -> str:
    start = text.find(needle)
    if start < 0:
        raise ContractError(f"function/struct anchor missing: {needle}")
    brace = text.find("{", start)
    if brace < 0:
        raise ContractError(f"opening brace missing after: {needle}")
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise ContractError(f"unbalanced body: {needle}")


def apply_plant(plant: str, collective: str, kernel: str) -> tuple[str, str]:
    if plant == "none":
        return collective, kernel
    if plant == "runtime-topology":
        needle = "    int const tid = state.tid;"
        if collective.count(needle) != 1:
            raise ContractError("runtime-topology plant seam drifted")
        collective = collective.replace(
            needle,
            needle + "\n    int const warp_n = (tid / 32) % 2; (void)warp_n;",
            1,
        )
    elif plant == "shared-per-segment":
        needle = """    auto shared_bases = CollectiveMainloop::make_shared_bases(
        shared.tensors.mainloop);
    while (work.is_valid()) {"""
        replacement = """    while (work.is_valid()) {
      auto shared_bases = CollectiveMainloop::make_shared_bases(
          shared.tensors.mainloop);"""
        if kernel.count(needle) != 1:
            raise ContractError("shared-per-segment plant seam drifted")
        kernel = kernel.replace(needle, replacement, 1)
    elif plant == "integer-segment-offset":
        segment = body(collective, "struct SegmentState")
        needle = "marlin_ppu_detail::Vector128 const* a = nullptr;"
        if segment.count(needle) != 1:
            raise ContractError("integer-segment-offset plant seam drifted")
        changed = segment.replace(needle, "int a = 0;", 1)
        collective = collective.replace(segment, changed, 1)
    else:
        raise ContractError(f"unknown plant {plant}")
    return collective, kernel


def validate(collective: str, kernel: str) -> None:
    cta = body(collective, "struct CtaState")
    segment = body(collective, "struct SegmentState")
    shared = body(collective, "struct SharedBases")
    rebase = body(collective, "static SegmentState rebase_segment(")
    run = body(collective, "static void run_segment(")
    make_shared = body(collective, "static SharedBases make_shared_bases(")

    cta_required = (
        "Vector128 const* a_thread_base",
        "Vector128 const* b_thread_base",
        "Vector128 const* scale_thread_base",
        "int tid",
        "int b_inner_delta", "int b_k_delta", "int scale_k_delta",
        "int a_smem_write", "int a_smem_read[BInnerIters]",
        "int scale_smem_read", "bool a_copy_pred",
        "bool scale_copy_pred",
    )
    for token in cta_required:
        if cta.count(token) != 1:
            raise ContractError(f"CtaState final invariant is not unique: {token}")
    for forbidden in (
        "ptr_A", "ptr_B", "ptr_S", "problem_n", "problem_k",
        "a_global_stride", "a_global_inner", "b_global_stride",
        "b_global_outer", "b_global_inner", "scale_global_stride",
        "a_predicate", "a_write_transformed", "a_read_transformed",
        "n_tiles", "k_tiles", "bool valid",
    ):
        if forbidden in cta:
            raise ContractError(f"CtaState retained pre-hoist field: {forbidden}")

    for token in (
        "Vector128 const* a = nullptr",
        "Vector128 const* b[BInnerIters]",
        "Vector128 const* scale = nullptr",
        "int k_tiles_remaining",
    ):
        if segment.count(token) != 1:
            raise ContractError(f"SegmentState rebased pointer is not unique: {token}")
    for forbidden in (
        "n_tile", "k_tile_begin", "a_global_read", "b_global_read",
        "scale_global_read", "bool valid",
    ):
        if forbidden in segment:
            raise ContractError(f"SegmentState retained an integer rebase field: {forbidden}")

    for token in (
        "Vector128* a = nullptr", "Vector128* b = nullptr",
        "Vector128* scale = nullptr",
    ):
        if shared.count(token) != 1:
            raise ContractError(f"SharedBases field is not unique: {token}")
    for token in (
        "bases.a = shared.storage",
        "bases.b = bases.a + Stages * ASharedStage",
        "bases.scale = bases.b + Stages * BSharedStage",
    ):
        if token not in make_shared:
            raise ContractError(f"SharedBases production offset drifted: {token}")

    for token in (
        "segment.a = state.a_thread_base + AGlobalOuter * k_tile_begin",
        "state.b_thread_base + BSharedStride * n_tile",
        "state.b_k_delta * k_tile_begin",
        "segment.b[i] = b + state.b_inner_delta * i",
        "segment.scale = state.scale_thread_base + ScaleSharedStride * n_tile",
        "state.scale_k_delta * k_tile_begin",
    ):
        if token not in rebase:
            raise ContractError(f"pointer rebase seam drifted: {token}")

    for forbidden in (
        "warp_n", "lane", "tid / 32", "tid % 32", "/ 32", "% 32",
        "switch (", "state.ptr_", "global_read",
    ):
        if forbidden in run:
            raise ContractError(f"run_segment recomputes runtime topology/address: {forbidden}")
    for token in (
        "Vector128 const* a_pointer = segment.a",
        "b_pointer[i] = segment.b[i]",
        "Vector128 const* scale_pointer = segment.scale",
        "shared.a + ASharedStage * pipe",
        "shared.b + BSharedStage * pipe",
        "shared.scale + ScaleSharedStage * pipe",
        "state.a_smem_write", "state.a_smem_read[",
        "state.scale_smem_read", "state.a_copy_pred",
        "state.scale_copy_pred",
    ):
        if token not in run:
            raise ContractError(f"run_segment does not consume hoisted state: {token}")

    make_call = "CollectiveMainloop::make_shared_bases("
    init_call = "CollectiveMainloop::init_cta_state("
    loop = "while (work.is_valid())"
    run_call = "CollectiveMainloop::run_segment("
    if kernel.count(make_call) != 1 or kernel.count(init_call) != 1 or \
       kernel.count(loop) != 1 or kernel.count(run_call) != 1:
        raise ContractError("kernel state/shared callsite cardinality drifted")
    if not (kernel.index(init_call) < kernel.index(make_call) < kernel.index(loop)):
        raise ContractError("SharedBases is not constructed once before the segment loop")
    call_scope = kernel[kernel.index(run_call) : kernel.index(run_call) + 180]
    if "cta_state, segment, shared_bases, accum" not in call_scope:
        raise ContractError("kernel does not pass the one CTA SharedBases object")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", default="none")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    try:
        collective = COLLECTIVE.read_text(encoding="utf-8")
        kernel = KERNEL.read_text(encoding="utf-8")
        collective, kernel = apply_plant(args.plant, collective, kernel)
        validate(collective, kernel)
    except (OSError, ContractError) as exc:
        if args.plant == "none":
            print(f"[l178-source] FAIL: {exc}", file=sys.stderr)
        else:
            print(
                f"[l178-source:red] plant={args.plant} caught=1 "
                f"reason={exc} result=RED", file=sys.stderr,
            )
        return 1
    if args.plant != "none":
        print(
            f"[l178-source:red] plant={args.plant} caught=0 "
            "reason=named plant survived result=FAIL", file=sys.stderr,
        )
        return 1
    if args.source_only:
        print(
            "[l178-source] PASS: final CtaState + pointer SegmentState + "
            "one-CTA SharedBases feed run_segment without topology recomputation"
        )
        return 0

    run = subprocess.run(
        ("bash", "dev/fold_derivation/run_l178_marlin_state_hoist.sh"),
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    witness = (
        "[l178:runner] positive=exhaustive+source "
        "negative_controls=12/12_RED result=PASS"
    )
    if run.returncode != 0 or witness not in run.stdout:
        print(
            f"[l178-contract] FAIL: runner rc={run.returncode}\n{run.stdout[-4000:]}",
            file=sys.stderr,
        )
        return 1
    print(run.stdout, end="")
    print(
        "[l178-contract] PASS: all 4,325,376 legal fixed-target segments "
        "match independent classic/Awesome-CuTe pointer equations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
