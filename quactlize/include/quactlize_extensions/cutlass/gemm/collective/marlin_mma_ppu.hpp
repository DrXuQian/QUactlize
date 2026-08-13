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

template <int NBlock, class Accumulator>
CUTLASS_DEVICE void mma_n16(
    FragmentA const& a, FragmentB const& b0, FragmentB const& b1,
    Accumulator& accum) {
  uint32_t const* av = reinterpret_cast<uint32_t const*>(&a);
  uint32_t const b[4] = {
      *reinterpret_cast<uint32_t const*>(&b0.value[0]),
      *reinterpret_cast<uint32_t const*>(&b0.value[1]),
      *reinterpret_cast<uint32_t const*>(&b1.value[0]),
      *reinterpret_cast<uint32_t const*>(&b1.value[1]),
  };
  constexpr int base = 8 * NBlock;
  asm volatile(
      "ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, "
      "{%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(accum(base + 0)), "+f"(accum(base + 1)),
        "+f"(accum(base + 2)), "+f"(accum(base + 3)),
        "+f"(accum(base + 4)), "+f"(accum(base + 5)),
        "+f"(accum(base + 6)), "+f"(accum(base + 7))
      : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]),
        "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
}

}  // namespace cutlass::gemm::collective::marlin_ppu_detail
