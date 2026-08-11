/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace cutlass::gemm::collective::detail {

// Resolve the outer A base from the stride the caller actually supplied.  The
// ragged ABI flattens expert rows and publishes their cumulative row offsets;
// the uniform ABI retains the rank-3 tensor's L pitch.  Multiplying either by
// the logical K extent silently substitutes a compact layout for dA.
template <class Element, class Stride>
CUTE_HOST_DEVICE Element const* mixed_a_expert_base(
    Element const* base, Stride const& dA, int const* group_row_offsets,
    int l_coord) {
  int64_t const element_offset = group_row_offsets
      ? int64_t(group_row_offsets[l_coord]) * int64_t(cute::get<0>(dA))
      : int64_t(l_coord) * int64_t(cute::get<2>(dA));
  return base + element_offset;
}

// Metadata is indexed in logical output columns even when the resident B
// artifact folds several logical N columns into one physical row.  Using
// size<0>(gB) charges the physical TileN/Fold and can admit OOB metadata on an
// N residue.
CUTE_HOST_DEVICE constexpr int64_t mixed_logical_n_residue(
    int64_t N, int logical_tile_n, int n_coord) {
  return N - int64_t(logical_tile_n) * int64_t(n_coord);
}

}  // namespace cutlass::gemm::collective::detail
