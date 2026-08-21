// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Logical-output predicate shared by Stream-K and Marlin-style cooperative
// reductions.  The accumulator's register order is not a lane formula: derive
// it from the exact TiledMma partition so m8 and every shipped topology follow
// the same source of truth.
#pragma once

#include <cstdint>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel::detail {

template <int Count>
struct AccumulatorResidueMask {
  cutlass::Array<uint8_t, Count> valid{};

  CUTLASS_HOST_DEVICE
  bool operator()(int register_idx) const {
    return valid[register_idx] != 0;
  }
};

template <class TiledMma, class AccumulatorTensor, class TileShapeMN,
          class ResidueMN>
CUTLASS_HOST_DEVICE auto
make_accumulator_residue_mask(
    TiledMma const& tiled_mma,
    AccumulatorTensor const& accumulators,
    TileShapeMN const& tile_shape_mn,
    ResidueMN const& residue_mn,
    int thread_idx) {
  using namespace cute;
  constexpr int Count = int(size(AccumulatorTensor{}));
  static_assert(cosize(typename AccumulatorTensor::layout_type{}) == Count,
                "accumulator storage must be a compact bijection");

  auto identity = make_identity_tensor(tile_shape_mn);
  auto coordinates = tiled_mma.get_thread_slice(thread_idx).partition_C(identity);
  static_assert(size(decltype(coordinates){}) == Count,
                "coordinate and accumulator fragments must have identical slots");
  auto physical_to_fragment = right_inverse(accumulators.layout());

  AccumulatorResidueMask<Count> mask;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Count; ++i) {
    auto mn = coordinates(physical_to_fragment(i));
    mask.valid[i] = uint8_t(
        int(get<0>(mn)) < int(get<0>(residue_mn)) &&
        int(get<1>(mn)) < int(get<1>(residue_mn)));
  }
  return mask;
}

}  // namespace cutlass::gemm::kernel::detail
