/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cute/algorithm/tuple_algorithms.hpp"

namespace cutlass::gemm::collective::detail {

// Relate two independently retiled register-copy views through their common
// MMA-K atom space.  ACopyBlocks and BCopyBlocks are CPY_K extents; neither is
// a coordinate in the other view.
//
// Every A block is loaded immediately before the first B delivery that uses
// one of its atoms.  The sole prepare-ahead collision is a one-block A view
// paired with multiple B deliveries: next-tile A0 covers the whole physical A
// fragment while the current tile's final B delivery still consumes its tail.
// In that case B keeps its look-ahead cadence and A alone moves to the
// post-consume hook.
template <int MmaKAtoms, int ACopyBlocks, int BCopyBlocks>
struct MixedARegisterSchedule {
  static_assert(MmaKAtoms > 0 && ACopyBlocks > 0 && BCopyBlocks > 0,
                "mixed A schedule needs nonzero static extents");
  static_assert(MmaKAtoms % ACopyBlocks == 0,
                "A copy blocks must partition MMA-K atoms exactly");
  static_assert(MmaKAtoms % BCopyBlocks == 0,
                "B copy blocks must partition MMA-K atoms exactly");

  static constexpr int AAtomsPerCopy = MmaKAtoms / ACopyBlocks;
  static constexpr int BAtomsPerCopy = MmaKAtoms / BCopyBlocks;
  static constexpr int ABlocks = ACopyBlocks;
  static constexpr int BBlocks = BCopyBlocks;

  template <int ABlock>
  static constexpr int first_b_for_a() {
    static_assert(ABlock >= 0 && ABlock < ACopyBlocks,
                  "A copy block is outside its CuTe view");
    return (ABlock * AAtomsPerCopy) / BAtomsPerCopy;
  }

  template <int BBlock, int ABlock>
  static constexpr bool prepare_loads() {
    static_assert(BBlock >= 0 && BBlock < BCopyBlocks,
                  "B delivery is outside its CuTe view");
    return first_b_for_a<ABlock>() == BBlock;
  }

  template <int BBlock, bool Prime>
  static constexpr bool delay_prepare() {
    return !Prime && ACopyBlocks == 1 && BCopyBlocks > 1 && BBlock == 0;
  }

  template <int BBlock>
  static constexpr bool load_after_consume() {
    return ACopyBlocks == 1 && BCopyBlocks > 1 &&
           BBlock == BCopyBlocks - 1;
  }

  static constexpr int loads_per_k_tile() { return ACopyBlocks; }
};

template <class Schedule, class SmemTiledCopyA, class TCsA, class TCrA,
          class BBlock, class Prime>
CUTLASS_DEVICE void prepare_mixed_a_for_b(
    SmemTiledCopyA const& smem_tiled_copy_A,
    TCsA const& tCsA_p,
    TCrA& tCrA_copy_view,
    BBlock const& b_block,
    Prime const&) {
  constexpr int B = BBlock::value;
  cute::for_each(cute::make_int_sequence<Schedule::ABlocks>{},
                 [&] (auto a_block) {
    constexpr int A = decltype(a_block)::value;
    if constexpr (Schedule::template prepare_loads<B, A>() &&
                  !Schedule::template delay_prepare<
                      B, (Prime::value != 0)>()) {
      cute::copy(smem_tiled_copy_A,
                 tCsA_p(cute::_, cute::_, a_block),
                 tCrA_copy_view(cute::_, cute::_, a_block));
    }
  });
}

template <class Schedule, class SmemTiledCopyA, class TCsA, class TCrA,
          class BBlock>
CUTLASS_DEVICE void finish_mixed_a_after_consume(
    SmemTiledCopyA const& smem_tiled_copy_A,
    TCsA const& tCsA_p,
    TCrA& tCrA_copy_view,
    BBlock const&) {
  constexpr int B = BBlock::value;
  if constexpr (Schedule::template load_after_consume<B>()) {
    cute::copy(smem_tiled_copy_A,
               tCsA_p(cute::_, cute::_, cute::Int<0>{}),
               tCrA_copy_view(cute::_, cute::_, cute::Int<0>{}));
  }
}

}  // namespace cutlass::gemm::collective::detail
