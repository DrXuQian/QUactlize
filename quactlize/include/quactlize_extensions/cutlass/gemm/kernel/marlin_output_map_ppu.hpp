/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Authoritative output-coordinate map for the standalone Marlin PPU kernel.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"

namespace cutlass::gemm::kernel::marlin_ppu_detail {

// The K0 cohort is the only cohort that hands off or writes output.  Keep its
// map host/device constexpr so production and the exhaustive L179 proof share
// one seam rather than maintaining two mutually-consistent implementations.
CUTLASS_HOST_DEVICE constexpr int output_row(int lane, int value) {
  return lane / 4 + (((value >> 2) & 1) << 3);
}

CUTLASS_HOST_DEVICE constexpr int output_n_base(
    int n_tile, int output_thread, int n_block) {
  int const warp_n = output_thread / 32;
  return (n_tile * 8 + warp_n * 4 + n_block) * 16;
}

CUTLASS_HOST_DEVICE constexpr int output_col_offset(int lane, int value) {
  return lane % 4 + ((value % 4) << 2);
}

}  // namespace cutlass::gemm::kernel::marlin_ppu_detail
