/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone Marlin PPU MMA primitive.
 **************************************************************************************************/
#pragma once

#include <cstdint>

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

static_assert(sizeof(FragmentC) == 8 * sizeof(float));
static_assert(sizeof(MarlinAccumulatorPPU) == 32 * sizeof(float));

template <int NBlock>
CUTLASS_DEVICE void mma_n16(
    FragmentA const& a, FragmentB const& b0, FragmentB const& b1,
    FragmentC& accum) {
  static_assert(NBlock >= 0 && NBlock < 4,
                "the fixed Marlin warp owns exactly four n16 blocks");
  uint32_t const* av = reinterpret_cast<uint32_t const*>(&a);
  uint32_t const b[4] = {
      *reinterpret_cast<uint32_t const*>(&b0.value[0]),
      *reinterpret_cast<uint32_t const*>(&b0.value[1]),
      *reinterpret_cast<uint32_t const*>(&b1.value[0]),
      *reinterpret_cast<uint32_t const*>(&b1.value[1]),
  };
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

}  // namespace cutlass::gemm::collective::marlin_ppu_detail
