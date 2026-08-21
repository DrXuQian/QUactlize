#!/usr/bin/env python3
"""Bind standalone Marlin's per-CTA init/rebase seam to production source."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COLLECTIVE = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
KERNEL = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/marlin_kernel_ppu.hpp"


def fail(message: str) -> int:
    print(f"[l173-contract] FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    collective = COLLECTIVE.read_text()
    kernel = KERNEL.read_text()
    init_head = collective.split("init_cta_state(", 1)[1].split(") {", 1)[0]
    if "Work" in init_head or "work" in init_head:
        return fail("init_cta_state accepts a work descriptor")
    required = (
        "struct CtaState", "struct SegmentState", "init_cta_state(",
        "rebase_segment(", "AGlobalOuter * k_tile_begin",
        "BSharedStride * n_tile", "ScaleSharedStride * n_tile",
    )
    for token in required:
        if token not in collective:
            return fail(f"collective lacks {token!r}")
    if "if (!work.is_valid()) {\n      return;" not in kernel:
        return fail("invalid CTA is not rejected before state init")
    if kernel.count("CollectiveMainloop::init_cta_state(") != 1:
        return fail("kernel must expose exactly one CTA-state init callsite")
    if kernel.count("CollectiveMainloop::rebase_segment(") != 1:
        return fail("kernel must expose exactly one per-segment rebase callsite")
    if kernel.index("init_cta_state(") > kernel.index("while (work.is_valid())"):
        return fail("CTA-state init remains inside the segment loop")
    segment_scope = kernel.split("auto segment =", 1)[1].split("thread_block_reduce", 1)[0]
    if "run_segment" not in segment_scope or "}" not in segment_scope:
        return fail("SegmentState lifetime is not closed before reduction")

    run = subprocess.run(
        ("bash", "dev/fold_derivation/run_l173_marlin_cta_state.sh"),
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    witness = "[l173:runner] positive=PASS negative_controls=9/9_RED result=PASS"
    if run.returncode != 0 or witness not in run.stdout:
        return fail(f"production oracle failed rc={run.returncode}\n{run.stdout[-3000:]}")
    print(run.stdout, end="")
    print("[l173-contract] PASS: production CTA state initializes only valid CTAs once, rebases every reverse-q segment with absolute q/K, and dies before reduction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
