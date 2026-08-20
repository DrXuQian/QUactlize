#!/usr/bin/env python3
"""Bind the exact Q4/A32 typed-scatter and cadence candidate to its source.

This is a source seam check, not a PPU numeric oracle.  L219 independently
checks the layout and a constructive lifetime plant; the exact box row remains
the only device verdict.  The purpose here is to make it impossible for the
type witness or the old/new A/B switches to disappear while those other gates
continue to print green.
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
CONVERT = ROOT / (
    "quactlize/include/quactlize_extensions/cutlass/"
    "quactlize_mix_gemm_convert.h"
)
DRIVER = ROOT / (
    "quactlize/include/quactlize_extensions/cutlass/gemm/collective/"
    "detail/ppu_mixed_pipeline.hpp"
)
FOLD = ROOT / (
    "quactlize/include/quactlize_extensions/cutlass/gemm/collective/"
    "ppu_mma_aiu_fold.hpp"
)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def audit(convert: str, driver: str, fold: str) -> list[str]:
    bad: list[str] = []
    c, d, f = compact(convert), compact(driver), compact(fold)

    convert_needles = (
        "using Half2Layout = cute::Layout<",
        "cute::Shape<cute::Int<Group / 2>, cute::Int<Deliveries>, cute::Int<Fold>, cute::Int<Instances>>",
        "cute::Stride<cute::_1, cute::Int<Group / 2>, cute::Int<Deliveries * Group / 2>, cute::Int<Deliveries * VEC / 2>>",
        "static void emit_to(uint32_t const* s, Store&& store_arg)",
        "store(cute::Int<at(T, V)>{}, x)",
    )
    for needle in convert_needles:
        if needle not in c:
            bad.append(f"converter lost {needle}")

    driver_needles = (
        "template <int Stages, bool ConsumeBeforePrepare = false,",
        "if constexpr (ConsumeBeforePrepare) { consume(k_block, consume_stage); }",
        "if constexpr (!ConsumeBeforePrepare) { consume(k_block, consume_stage); }",
        "prepare(k_block_next, smem_pipe_read, cute::Int<0>{});",
    )
    for needle in driver_needles:
        if needle not in d:
            bad.append(f"pipeline driver lost {needle}")

    fold_needles = (
        "PPU_Q4_A32_LEGACY_RAW_SCATTER",
        "PPU_Q4_A32_LEGACY_PREPARE_ORDER",
        "constexpr bool kQ4A32FoldedMultiDelivery =",
        "cutlass::sizeof_bits<RealInternalElementB>::value == 4",
        "FoldF == 2",
        "decltype(K_BLOCK_MAX)::value == 4",
        "the exact Q4/A32 row must select its folded cadence policy",
        "the exact Q4/A32 row must select its typed scatter",
        "using H2Layout = typename Scatter::template Half2Layout<NumIterations>",
        "tCrB_h2(group_h2, cute::Int<KBlockIndex>{}, cute::Int<FG>{}, cute::Int<II>{}) = value",
        "MixGemmChunkEmit<Bits, FG, FoldF, true, EmitLayout>::emit_to(",
        "detail::run_mixed_pipeline<DispatchPolicy::Stages, kConsumeBeforePrepare>(",
    )
    for needle in fold_needles:
        if needle not in f:
            bad.append(f"folded collective lost {needle}")

    # The two counterfactual controls must be independently selectable.  A
    # shared `legacy` macro would prevent the one box invocation from telling
    # typed-scatter and cadence apart.
    if f.count("PPU_Q4_A32_LEGACY_RAW_SCATTER") != 2:
        bad.append("raw-scatter counterfactual is not one independently guarded seam")
    if f.count("PPU_Q4_A32_LEGACY_PREPARE_ORDER") != 2:
        bad.append("prepare-order counterfactual is not one independently guarded seam")
    return bad


def main() -> int:
    convert = CONVERT.read_text(encoding="utf-8")
    driver = DRIVER.read_text(encoding="utf-8")
    fold = FOLD.read_text(encoding="utf-8")
    bad = audit(convert, driver, fold)
    if bad:
        print("[q4-a32-cadence] FAIL: " + "; ".join(bad), file=sys.stderr)
        return 1

    plants = (
        (convert.replace(
            "cute::Int<Deliveries * Group / 2>", "cute::_0", 1),
         driver, fold, "zero delivery stride"),
        (convert, driver.replace(
            "if constexpr (ConsumeBeforePrepare)", "if constexpr (false)", 1),
         fold, "disabled consume-first arm"),
        (convert, driver, fold.replace(
            "tCrB_h2(group_h2, cute::Int<KBlockIndex>{},",
            "removed_typed_store(group_h2, cute::Int<KBlockIndex>{},", 1),
         "typed store removed"),
        (convert, driver, fold.replace(
            "detail::run_mixed_pipeline<DispatchPolicy::Stages, kConsumeBeforePrepare>",
            "detail::run_mixed_pipeline<DispatchPolicy::Stages, false>", 1),
         "collective bypasses cadence policy"),
        (convert, driver, fold.replace(
            "PPU_Q4_A32_LEGACY_RAW_SCATTER", "PPU_Q4_A32_REMOVED", 2),
         "raw-scatter counterfactual removed"),
    )
    for planted_convert, planted_driver, planted_fold, label in plants:
        if not audit(planted_convert, planted_driver, planted_fold):
            print(f"[q4-a32-cadence] FAIL: {label} plant was false-green",
                  file=sys.stderr)
            return 1

    print(
        "[q4-a32-cadence] PASS: typed half2 destination, exact Q4/A32 "
        "consume-first policy, and two independent legacy arms are source-bound; "
        "five seam plants red"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
