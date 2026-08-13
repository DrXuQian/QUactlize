#!/usr/bin/env python3
"""Fail-closed source contract for standalone Marlin's native C fragment."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MMA = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_mma_ppu.hpp"
COLLECTIVE = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
KERNEL = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/marlin_kernel_ppu.hpp"


def die(plant: str, reason: str) -> "NoReturn":
    prefix = "[l175] FAIL" if plant == "none" else f"[l175:red] plant={plant} caught=1"
    print(f"{prefix} reason={reason}", file=sys.stderr)
    raise SystemExit(1)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def load(path: Path, plant: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        die(plant, f"cannot read {path}: {exc}")


def one_replace(text: str, old: str, new: str, plant: str) -> str:
    if text.count(old) != 1:
        die(plant, f"plant seam is not unique: {old!r}")
    return text.replace(old, new, 1)


def apply_plant(plant: str, mma: str, collective: str, kernel: str) -> tuple[str, str, str]:
    if plant == "none":
        return mma, collective, kernel
    if plant == "generic-fragment":
        needle = "Accumulator accum;"
        replacement = """auto accum = cute::make_fragment_like<ElementAccumulator>(
          cute::partition_fragment_C(TiledMma{}, cute::take<0, 2>(TileShape{})));"""
        kernel = one_replace(kernel, needle, replacement, plant)
    elif plant == "whole-accum-reinterpret":
        needle = "CollectiveMainloop::run_segment("
        replacement = """auto* l175_bad_flat = reinterpret_cast<float*>(&accum);
        (void)l175_bad_flat;
        CollectiveMainloop::run_segment("""
        kernel = one_replace(kernel, needle, replacement, plant)
    elif plant == "wrong-4x8-layout":
        # Preserve the total 128 bytes.  A size-only checker would accept this
        # transposition, while the native m16n16 register association is wrong.
        mma = one_replace(mma, "float value[8];", "float value[4];", plant)
        mma = one_replace(
            mma, "FragmentC fragments[4];", "FragmentC fragments[8];", plant
        )
    else:
        die(plant, "unknown plant")
    return mma, collective, kernel


def extract_struct(text: str, name: str, plant: str) -> str:
    match = re.search(rf"\bstruct\s+{re.escape(name)}\s*\{{", text)
    if match is None:
        die(plant, f"missing struct {name}")
    start = text.find("{", match.start())
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos]
    die(plant, f"unbalanced struct {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", default="none")
    args = parser.parse_args()
    plant = args.plant

    mma = load(MMA, plant)
    collective = load(COLLECTIVE, plant)
    kernel = load(KERNEL, plant)
    mma, collective, kernel = apply_plant(plant, mma, collective, kernel)

    fragment = compact(extract_struct(mma, "FragmentC", plant))
    accumulator = compact(extract_struct(mma, "MarlinAccumulatorPPU", plant))
    if fragment != "floatvalue[8];":
        die(plant, f"FragmentC is not exactly float value[8]: {fragment!r}")
    if accumulator != "FragmentCfragments[4];":
        die(plant, f"Accumulator is not exactly FragmentC fragments[4]: {accumulator!r}")

    c = compact(collective)
    for token in (
        "usingFragmentC=marlin_ppu_detail::FragmentC;",
        "usingAccumulator=marlin_ppu_detail::MarlinAccumulatorPPU;",
        "FragmentC&accum",
        "mma_n16<NBlock>(fragment_a,b0,b1,accum)",
        "fragment_scale[inner&1],accum.fragments[0])",
        "fragment_scale[inner&1],accum.fragments[1])",
        "fragment_scale[inner&1],accum.fragments[2])",
        "fragment_scale[inner&1],accum.fragments[3])",
        "SharedBasesconst&shared,Accumulator&accum",
    ):
        if token not in c:
            die(plant, f"collective exact-accumulator seam lacks {token!r}")
    for forbidden in (
        "template<intNBlock,classAccumulator>",
        "cute::clear(accum)",
        "partition_fragment_C",
        "make_fragment_like",
    ):
        if forbidden in c:
            die(plant, f"collective retained generic C-fragment seam {forbidden!r}")

    m = compact(mma)
    if "voidmma_n16(" not in m or "FragmentC&accum" not in m:
        die(plant, "MMA does not accept exactly one native FragmentC")
    operands = re.findall(r'"\+f"\(accum\.value\[(\d)\]\)', mma)
    if operands != [str(i) for i in range(8)]:
        # Normal source spells the constraint before the expression.
        operands = re.findall(r'"\+f"\s*\(accum\.value\[(\d)\]\)', mma)
    if operands != [str(i) for i in range(8)]:
        die(plant, f"native MMA operands are not value[0..7]: {operands}")

    k = compact(kernel)
    for token in (
        "usingAccumulator=typenameCollectiveMainloop::Accumulator;",
        "Accumulatoraccum;",
        "CollectiveMainloop::run_segment(cta_state,segment,shared_bases,accum);",
        "thread_block_reduce(accum,shared);",
        "global_handoff(accum,params,work,first,final,problem_m,problem_n);",
        "write_result(accum,params,work,problem_m,problem_n,shared);",
    ):
        if token not in k:
            die(plant, f"kernel exact accumulator path lacks {token!r}")
    for forbidden in ("partition_fragment_C", "make_fragment_like"):
        if forbidden in k:
            die(plant, f"kernel rebuilt a generic C fragment via {forbidden}")

    joined = "\n".join((mma, collective, kernel))
    whole_flat = re.compile(
        r"reinterpret_cast\s*<\s*(?:const\s+)?float\s*\*\s*>\s*\(\s*&?\s*accum\b"
    )
    if whole_flat.search(joined):
        die(plant, "whole accumulator escaped through a flat float pointer")
    if re.search(r"\baccum\s*\.\s*(?:data|begin)\s*\(", joined):
        die(plant, "whole accumulator escaped through a container pointer API")

    print(
        "[l175:source] fragment=4x8-fp32 bytes=128 aliases=exact "
        "generic-fragment=ABSENT flat-address-escape=ABSENT result=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
