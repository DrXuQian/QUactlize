/***************************************************************************************************
 * Copyright (c) 2026 Quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Direct FP32 partial store for fixed Split-K.
 *
 * This runs strictly after the shipping mainloop.  It bypasses the shared-memory
 * R2S/barrier/S2R/vectorized partial epilogue and maps each accumulator register
 * to its logical output through the production TiledMma partition_C view.  The
 * partial workspace needs neither output conversion nor an output-layout
 * redistribution, so the accumulator ownership is already the desired store
 * ownership.  Keeping the path direct removes an otherwise unnecessary
 * cross-thread shared-memory handoff and its two CTA barriers.
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel::detail {

template <class PartialParams, class ProblemShapeMNKL, class BlockShapeMNK,
          class BlockCoordMNKL, class AccumulatorTensor, class TiledMma,
          class ResidueMNK>
CUTLASS_DEVICE void
store_splitk_accumulators_direct(
    PartialParams const& params,
    ProblemShapeMNKL problem_shape_mnkl,
    BlockShapeMNK block_shape_mnk,
    BlockCoordMNKL block_coord_mnkl,
    AccumulatorTensor const& accumulators,
    TiledMma tiled_mma,
    ResidueMNK residue_mnk,
    int partial_plane,
    int thread_idx) {
  using namespace cute;
  using X = Underscore;

  static_assert(rank(ProblemShapeMNKL{}) == 4,
                "direct split-K store expects [M,N,K,S]");
  static_assert(rank(BlockShapeMNK{}) == 3,
                "direct split-K store expects a static MNK tile");
  static_assert(rank(BlockCoordMNKL{}) == 4,
                "direct split-K store expects an MNKL block coordinate");
  static_assert(is_static<BlockShapeMNK>::value,
                "direct split-K store needs a static CTA tile");
  static_assert(is_rmem<AccumulatorTensor>::value,
                "direct split-K store must observe the mainloop register fragment");

  auto M = get<0>(problem_shape_mnkl);
  auto N = get<1>(problem_shape_mnkl);
  auto S = get<3>(problem_shape_mnkl);
  auto [m_coord, n_coord, k_coord, l_coord] = block_coord_mnkl;
  (void)k_coord;
  (void)l_coord;

  auto mD_mns = make_tensor(make_gmem_ptr(params.ptr_D),
                            make_shape(M, N, S), params.dD);
  auto gD_mns = local_tile(mD_mns, block_shape_mnk, make_coord(_, _, _),
                           Step<_1, _1, X>{});
  auto gD = gD_mns(_, _, m_coord, n_coord, partial_plane);

  auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
  auto tD = thread_mma.partition_C(gD);
  auto identity = make_identity_tensor(
      make_shape(size<0>(gD), size<1>(gD)));
  auto coordinates = thread_mma.partition_C(identity);

  static_assert(size(decltype(tD){}) == size(AccumulatorTensor{}),
                "direct destination and accumulator fragments must agree");
  static_assert(size(decltype(coordinates){}) == size(AccumulatorTensor{}),
                "direct coordinates and accumulator fragments must agree");

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(accumulators); ++i) {
    if (elem_less(coordinates(i),
                  make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
      tD(i) = accumulators(i);
    }
  }
}

}  // namespace cutlass::gemm::kernel::detail
