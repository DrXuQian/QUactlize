/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone Marlin PPU MMA primitive.
 **************************************************************************************************/
#pragma once

#include <cstdint>
#include <type_traits>

#include "cutlass/cutlass.h"

#include "quactlize_extensions/cutlass/gemm/collective/marlin_dequant_ppu.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/marlin_load_ppu.hpp"

namespace cutlass::gemm::collective::marlin_ppu_detail {

struct FragmentC {
  float value[8];
};

struct MarlinAccumulatorPPU {
  FragmentC fragments[4];
};

struct FragmentC8 {
  float value[4];
};

struct MarlinAccumulatorM8PPU {
  FragmentC8 fragments[4];
};

// Keep the shipping four-n16 aggregates as named concrete types.  They are
// part of the compiled-row ABI and several compile gates intentionally compare
// them by type, not merely by sizeof.  Wider WarpN topologies need more native
// n16 fragments per output warp, so only the non-shipping counts use this
// parameterized aggregate.
template <class Fragment, int NBlocks>
struct MarlinAccumulatorNPPU {
  static_assert(NBlocks > 0,
                "a standalone Marlin output warp must own an n16 block");
  Fragment fragments[NBlocks];
};

template <int InstructionM>
using FragmentCFor = std::conditional_t<InstructionM == 8, FragmentC8, FragmentC>;

template <int InstructionM>
using MarlinAccumulatorFor = std::conditional_t<
    InstructionM == 8, MarlinAccumulatorM8PPU, MarlinAccumulatorPPU>;

template <int InstructionM, int NBlocks>
using MarlinAccumulatorForN = std::conditional_t<
    NBlocks == 4,
    MarlinAccumulatorFor<InstructionM>,
    MarlinAccumulatorNPPU<FragmentCFor<InstructionM>, NBlocks>>;

static_assert(sizeof(FragmentC8) == 4 * sizeof(float));
static_assert(sizeof(FragmentC) == 8 * sizeof(float));
static_assert(sizeof(MarlinAccumulatorM8PPU) == 16 * sizeof(float));
static_assert(sizeof(MarlinAccumulatorPPU) == 32 * sizeof(float));
static_assert(std::is_same_v<
                  MarlinAccumulatorForN<8, 4>, MarlinAccumulatorM8PPU> &&
              std::is_same_v<
                  MarlinAccumulatorForN<16, 4>, MarlinAccumulatorPPU>,
              "WN64 must retain the exact shipping accumulator types");
static_assert(sizeof(MarlinAccumulatorForN<8, 8>) == 32 * sizeof(float));
static_assert(sizeof(MarlinAccumulatorForN<16, 8>) == 64 * sizeof(float));

template <int InstructionM, int NBlock>
CUTLASS_DEVICE void mma_n16(
    FragmentAFor<InstructionM> const& a,
    FragmentB const& b0, FragmentB const& b1,
    FragmentCFor<InstructionM>& accum) {
  static_assert(InstructionM == 8 || InstructionM == 16,
                "standalone Marlin supports the real PPU m8/m16 atoms only");
  static_assert(NBlock >= 0 && NBlock < 8,
                "the proved Marlin WarpN domain owns four or eight n16 blocks");
  uint32_t const* av = reinterpret_cast<uint32_t const*>(&a);
  uint32_t const b[4] = {
      *reinterpret_cast<uint32_t const*>(&b0.value[0]),
      *reinterpret_cast<uint32_t const*>(&b0.value[1]),
      *reinterpret_cast<uint32_t const*>(&b1.value[0]),
      *reinterpret_cast<uint32_t const*>(&b1.value[1]),
  };
  if constexpr (InstructionM == 8) {
    asm volatile(
        "ppu.mma.sync.aligned.m8n16k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, {%0,%1,%2,%3};"
        : "+f"(accum.value[0]), "+f"(accum.value[1]),
          "+f"(accum.value[2]), "+f"(accum.value[3])
        : "r"(av[0]), "r"(av[1]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
  } else {
    asm volatile(
        "ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, "
        "{%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(accum.value[0]), "+f"(accum.value[1]),
          "+f"(accum.value[2]), "+f"(accum.value[3]),
          "+f"(accum.value[4]), "+f"(accum.value[5]),
          "+f"(accum.value[6]), "+f"(accum.value[7])
        : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
  }
}

}  // namespace cutlass::gemm::collective::marlin_ppu_detail
