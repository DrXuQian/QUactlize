/***************************************************************************************************
 * Copyright (c) 2026 Quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Diagnostic-only clone of the legacy PPU Split-K shared partial epilogue.
 *
 * The production path stores FP32 accumulators directly.  This clone exists
 * solely to distinguish the historical EpilogueParallel R2S/S2R mapping from
 * its synchronization primitive.  Every operation and tensor partition below
 * matches ppu_epilogue_vectorized_parallel.hpp; the compile-time policy changes
 * only the two synchronization calls.
 **************************************************************************************************/

#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#define PPU_SPLITK_SHARED_SYNC_LEGACY_USER0 1
#define PPU_SPLITK_SHARED_SYNC_EPILOGUE_ID1 2
#define PPU_SPLITK_SHARED_SYNC_CTA 3

#ifndef PPU_SPLITK_SHARED_PROBE_PRE_R2S_CTA
#define PPU_SPLITK_SHARED_PROBE_PRE_R2S_CTA 0
#endif
#ifndef PPU_SPLITK_SHARED_PROBE_IDENTITY_CONVERT
#define PPU_SPLITK_SHARED_PROBE_IDENTITY_CONVERT 0
#endif
#ifndef PPU_SPLITK_SHARED_PROBE_SCALAR_R2S
#define PPU_SPLITK_SHARED_PROBE_SCALAR_R2S 0
#endif
#ifndef PPU_SPLITK_SHARED_PROBE_SCALAR_S2R
#define PPU_SPLITK_SHARED_PROBE_SCALAR_S2R 0
#endif
#ifndef PPU_SPLITK_SHARED_PROBE_DISCARD_GMEM
#define PPU_SPLITK_SHARED_PROBE_DISCARD_GMEM 0
#endif

namespace cutlass::gemm::kernel::detail {

template <int SyncPolicy, class TiledCopyS2R>
CUTLASS_DEVICE void splitk_shared_epilogue_sync() {
  static_assert(
      SyncPolicy == PPU_SPLITK_SHARED_SYNC_LEGACY_USER0 ||
          SyncPolicy == PPU_SPLITK_SHARED_SYNC_EPILOGUE_ID1 ||
          SyncPolicy == PPU_SPLITK_SHARED_SYNC_CTA,
      "unknown Split-K shared epilogue synchronization probe");
  if constexpr (SyncPolicy == PPU_SPLITK_SHARED_SYNC_LEGACY_USER0) {
    // Historical source spelling.  The integer overload adds
    // ReservedNamedBarrierCount, so this is hardware barrier ID 6.
    cutlass::arch::NamedBarrier::sync(
        typename TiledCopyS2R::TiledNumThr{}, uint32_t(0));
  } else if constexpr (SyncPolicy == PPU_SPLITK_SHARED_SYNC_EPILOGUE_ID1) {
    // The reserved-enum overload does not add the user-barrier base.  This is
    // hardware barrier ID 1, used by the ordinary PPU vectorized epilogues.
    cutlass::arch::NamedBarrier::sync(
        typename TiledCopyS2R::TiledNumThr{},
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  } else {
    __syncthreads();
  }
}

template <
    int SyncPolicy,
    class PartialEpilogue,
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine,
    class FrgLayout,
    class TiledMma,
    class ResidueMNK>
CUTLASS_DEVICE void store_splitk_accumulators_shared_sync_probe(
    typename PartialEpilogue::Params params,
    int partial_plane,
    ProblemShapeMNKL problem_shape_mnkl,
    BlockShapeMNK blk_shape_MNK,
    BlockCoordMNKL blk_coord_mnkl,
    cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
    TiledMma tiled_mma,
    ResidueMNK residue_mnk,
    int thread_idx,
    char* smem_buf) {
  using namespace cute;
  using X = Underscore;
  using ThreadEpilogueOp = typename PartialEpilogue::ThreadEpilogueOp;
  using ElementAccumulator = typename PartialEpilogue::ElementAccumulator;
  using ElementOutput = typename PartialEpilogue::ElementOutput;
  using SmemLayout = typename PartialEpilogue::SmemLayout;
  using CopyAtomR2S = typename PartialEpilogue::CopyAtomR2S;
  using TiledCopyS2R = typename PartialEpilogue::TiledCopyS2R;
  using CopyAtomR2G = typename PartialEpilogue::CopyAtomR2G;

  static_assert(rank(ProblemShapeMNKL{}) == 4);
  static_assert(is_static<BlockShapeMNK>::value);
  static_assert(rank(BlockShapeMNK{}) == 3);
  static_assert(rank(BlockCoordMNKL{}) == 4);

  // Match EpilogueParallel's constructor-side plane offset exactly.
  params.ptr_C += partial_plane * get<2>(params.dC);
  params.ptr_D += partial_plane * get<2>(params.dD);
  ThreadEpilogueOp epilogue_op{};

  auto M = get<0>(problem_shape_mnkl);
  auto N = get<1>(problem_shape_mnkl);
  auto L = get<3>(problem_shape_mnkl);
  Tensor mC_mnl = make_tensor(
      make_gmem_ptr(params.ptr_C), make_shape(M, N, L), params.dC);
  Tensor mD_mnl = make_tensor(
      make_gmem_ptr(params.ptr_D), make_shape(M, N, L), params.dD);
  Tensor gC_mnl = local_tile(
      mC_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});
  Tensor gD_mnl = local_tile(
      mD_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});

  auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
  (void)k_coord;
  (void)l_coord;
  Tensor gC = gC_mnl(_, _, m_coord, n_coord, 0);
  Tensor gD = gD_mnl(_, _, m_coord, n_coord, 0);

  using SharedStorage = typename PartialEpilogue::SharedStorage;
  SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
  Tensor sC = make_tensor(
      make_smem_ptr(storage.smem_epilogue.data()), SmemLayout{});

  auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
  auto tC = tiled_r2s.get_thread_slice(thread_idx);
  Tensor tCaC = tC.retile_S(accumulators);
  Tensor tCsC = tC.partition_D(sC);

  auto tile = make_shape(size<0>(sC), size<1>(sC));
  Tensor gCt = flat_divide(gC, tile);
  Tensor gDt = flat_divide(gD, tile);

  auto tiled_s2r = TiledCopyS2R{};
  auto tD = tiled_s2r.get_thread_slice(thread_idx);
  Tensor tDsC = tD.partition_S(sC);
  Tensor tDgC = tD.partition_D(gCt);
  Tensor tDgD = tD.partition_D(gDt);
  Tensor tDrC = make_tensor<ElementAccumulator>(take<0, 3>(shape(tDgC)));
  Tensor tDrD = make_tensor<ElementOutput>(shape(tDrC));

  Tensor cD = make_identity_tensor(make_shape(size<0>(gD), size<1>(gD)));
  Tensor cDt = flat_divide(cD, tile);
  Tensor tDcD = tD.partition_D(cDt);

  CUTE_STATIC_ASSERT(size<1>(tCaC) % size<3>(tDgC) == 0);
  CUTE_STATIC_ASSERT(size<2>(tCaC) % size<4>(tDgC) == 0);

#if PPU_SPLITK_SHARED_PROBE_PRE_R2S_CTA
  // A causal control for reuse of the mainloop/epilogue shared-storage union.
  // It is deliberately before the first R2S store; the historical barriers
  // occur only after that store and cannot protect a prior lifetime.
  __syncthreads();
#endif

  CUTLASS_PRAGMA_UNROLL
  for (int step_m = 0; step_m < size<2>(cDt); ++step_m) {
    CUTLASS_PRAGMA_UNROLL
    for (int step_n = 0; step_n < size<3>(cDt); ++step_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
          int mma_m = step_m * size<1>(tCsC) + pipe_m;
          int mma_n = step_n * size<2>(tCsC) + pipe_n;
#if PPU_SPLITK_SHARED_PROBE_SCALAR_R2S
          Tensor source = tCaC(_, mma_m, mma_n);
          Tensor destination = tCsC(_, pipe_m, pipe_n);
          CUTE_STATIC_ASSERT(size(source) == size(destination));
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(source); ++i) {
            destination(i) = source(i);
          }
#else
          copy(tiled_r2s, tCaC(_, mma_m, mma_n),
               tCsC(_, pipe_m, pipe_n));
#endif
        }
      }

      splitk_shared_epilogue_sync<SyncPolicy, TiledCopyS2R>();
#if PPU_SPLITK_SHARED_PROBE_SCALAR_S2R
      CUTE_STATIC_ASSERT(size(tDsC) == size(tDrC));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tDsC); ++i) {
        tDrC(i) = tDsC(i);
      }
#else
      copy(tiled_s2r, tDsC, tDrC);
#endif
      splitk_shared_epilogue_sync<SyncPolicy, TiledCopyS2R>();

      Tensor tDgDmn = tDgD(_, _, _, step_m, step_n);
      Tensor tDcDmn = tDcD(_, _, _, step_m, step_n);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tDrC); ++i) {
#if PPU_SPLITK_SHARED_PROBE_IDENTITY_CONVERT
        tDrD(i) = tDrC(i);
#else
        tDrD(i) = epilogue_op(tDrC(i));
#endif
      }

      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<1>(tDgDmn); ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size<2>(tDgDmn); ++n) {
          if (get<0>(tDcDmn(0, m, n)) < get<0>(residue_mnk) &&
              get<1>(tDcDmn(0, m, n)) < get<1>(residue_mnk)) {
#if !PPU_SPLITK_SHARED_PROBE_DISCARD_GMEM
            copy(CopyAtomR2G{}, tDrD(_, m, n), tDgDmn(_, m, n));
#endif
          }
        }
      }
    }
  }
}

}  // namespace cutlass::gemm::kernel::detail
