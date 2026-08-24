/***************************************************************************************************
 * Copyright (c) 2026 Quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Diagnostic-only prefixes for the historical PPU Split-K shared epilogue.
 *
 * Each policy executes before the production direct accumulator store.  The
 * policies are separate compile-time binaries: this is intentionally not a
 * runtime selector, because a runtime branch would give every arm the same
 * register-allocation footprint and destroy the causal experiment.
 **************************************************************************************************/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#define PPU_SPLITK_SHARED_PREFIX_ACCUMULATOR_OPAQUE 1
#define PPU_SPLITK_SHARED_PREFIX_CLONE_OPAQUE 2
#define PPU_SPLITK_SHARED_PREFIX_CTA_ONLY 3
#define PPU_SPLITK_SHARED_PREFIX_FLAT_CONSTANT 4
#define PPU_SPLITK_SHARED_PREFIX_FLAT_ACCUMULATOR 5
#define PPU_SPLITK_SHARED_PREFIX_R2S_VECTOR 6
#define PPU_SPLITK_SHARED_PREFIX_R2S_SCALAR 7
#define PPU_SPLITK_SHARED_PREFIX_R2S_SCALAR_SNAPSHOT 8
#define PPU_SPLITK_SHARED_PREFIX_R2S_S2R_VECTOR 9
#define PPU_SPLITK_SHARED_PREFIX_R2S_S2R_SCALAR 10

namespace cutlass::gemm::kernel::detail {

CUTLASS_DEVICE void splitk_shared_prefix_keep_float(float const& value) {
  // An empty asm input is a compiler barrier, not a hardware instruction.  It
  // keeps the selected register value live without adding a global-memory ABI
  // or perturbing the partial workspace being checked.
  uint32_t const bits = reinterpret_cast<uint32_t const&>(value);
  asm volatile("" : : "r"(bits) : "memory");
}

template <class Tensor>
CUTLASS_DEVICE void splitk_shared_prefix_keep_tensor(Tensor const& tensor) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < cute::size(tensor); ++i) {
    splitk_shared_prefix_keep_float(tensor(i));
  }
}

template <class Tensor>
CUTLASS_DEVICE auto splitk_shared_prefix_scalar_snapshot(
    Tensor const& source) {
  auto snapshot = cute::make_fragment_like(source);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < cute::size(source); ++i) {
    snapshot(i) = source(i);
  }
  return snapshot;
}

template <bool ScalarR2S, bool Readback, class PartialEpilogue,
          class AccumulatorTensor, class TiledMma>
CUTLASS_DEVICE void splitk_shared_prefix_r2s(
    AccumulatorTensor const& source_accumulators,
    TiledMma tiled_mma,
    int thread_idx,
    char* smem_buf) {
  using namespace cute;
  using ElementAccumulator = typename PartialEpilogue::ElementAccumulator;
  using SmemLayout = typename PartialEpilogue::SmemLayout;
  using CopyAtomR2S = typename PartialEpilogue::CopyAtomR2S;
  using TiledCopyS2R = typename PartialEpilogue::TiledCopyS2R;
  using SharedStorage = typename PartialEpilogue::SharedStorage;

  static_assert(is_same_v<ElementAccumulator, float>,
                "the frozen Split-K prefix probe requires fp32 accumulators");
  static_assert(is_rmem<AccumulatorTensor>::value,
                "R2S prefix must read a register fragment");

  SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
  Tensor sC = make_tensor(
      make_smem_ptr(storage.smem_epilogue.data()), SmemLayout{});
  auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
  auto thread_r2s = tiled_r2s.get_thread_slice(thread_idx);
  Tensor tCaC = thread_r2s.retile_S(source_accumulators);
  Tensor tCsC = thread_r2s.partition_D(sC);

  CUTE_STATIC_ASSERT(size<1>(tCaC) % size<1>(tCsC) == 0);
  CUTE_STATIC_ASSERT(size<2>(tCaC) % size<2>(tCsC) == 0);

  CUTLASS_PRAGMA_UNROLL
  for (int step_m = 0; step_m < size<1>(tCaC) / size<1>(tCsC);
       ++step_m) {
    CUTLASS_PRAGMA_UNROLL
    for (int step_n = 0; step_n < size<2>(tCaC) / size<2>(tCsC);
         ++step_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
          int const mma_m = step_m * size<1>(tCsC) + pipe_m;
          int const mma_n = step_n * size<2>(tCsC) + pipe_n;
          if constexpr (ScalarR2S) {
            Tensor source = tCaC(_, mma_m, mma_n);
            Tensor destination = tCsC(_, pipe_m, pipe_n);
            CUTE_STATIC_ASSERT(size(source) == size(destination));
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(source); ++i) {
              destination(i) = source(i);
            }
          } else {
            copy(tiled_r2s, tCaC(_, mma_m, mma_n),
                 tCsC(_, pipe_m, pipe_n));
          }
        }
      }

      __syncthreads();
      if constexpr (Readback) {
        auto tiled_s2r = TiledCopyS2R{};
        auto thread_s2r = tiled_s2r.get_thread_slice(thread_idx);
        Tensor tDsC = thread_s2r.partition_S(sC);
        Tensor tDrC = make_fragment_like(tDsC);
        copy(tiled_s2r, tDsC, tDrC);
        splitk_shared_prefix_keep_tensor(tDrC);
      }
      __syncthreads();
    }
  }
}

template <int PrefixPolicy, class PartialEpilogue, class FrgEngine,
          class FrgLayout, class TiledMma>
CUTLASS_DEVICE void run_splitk_shared_prefix_probe(
    cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
    TiledMma tiled_mma,
    int thread_idx,
    char* smem_buf) {
  using namespace cute;
  using ElementAccumulator = typename PartialEpilogue::ElementAccumulator;
  using SharedStorage = typename PartialEpilogue::SharedStorage;
  using SmemLayout = typename PartialEpilogue::SmemLayout;

  static_assert(
      PrefixPolicy >= PPU_SPLITK_SHARED_PREFIX_ACCUMULATOR_OPAQUE &&
          PrefixPolicy <= PPU_SPLITK_SHARED_PREFIX_R2S_S2R_SCALAR,
      "unknown Split-K shared prefix policy");
  static_assert(is_same_v<ElementAccumulator, float>,
                "the frozen Split-K prefix probe requires fp32 accumulators");
  static_assert(is_rmem<Tensor<FrgEngine, FrgLayout>>::value,
                "prefix probe must observe the mainloop register fragment");

  if constexpr (
      PrefixPolicy == PPU_SPLITK_SHARED_PREFIX_ACCUMULATOR_OPAQUE) {
    splitk_shared_prefix_keep_tensor(accumulators);
  } else if constexpr (
      PrefixPolicy == PPU_SPLITK_SHARED_PREFIX_CLONE_OPAQUE) {
    auto snapshot = splitk_shared_prefix_scalar_snapshot(accumulators);
    splitk_shared_prefix_keep_tensor(snapshot);
  } else if constexpr (PrefixPolicy == PPU_SPLITK_SHARED_PREFIX_CTA_ONLY) {
    __syncthreads();
    __syncthreads();
  } else if constexpr (
      PrefixPolicy == PPU_SPLITK_SHARED_PREFIX_FLAT_CONSTANT ||
      PrefixPolicy == PPU_SPLITK_SHARED_PREFIX_FLAT_ACCUMULATOR) {
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto* flat = storage.smem_epilogue.data();
    constexpr int ValuesPerThread = size(FrgLayout{});
    constexpr int Threads = size(TiledMma{});
    static_assert(ValuesPerThread * Threads == cosize_v<SmemLayout>,
                  "flat control must cover the exact shared epilogue once");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ValuesPerThread; ++i) {
      int const offset = thread_idx * ValuesPerThread + i;
      if constexpr (PrefixPolicy ==
                    PPU_SPLITK_SHARED_PREFIX_FLAT_CONSTANT) {
        flat[offset] = float(offset);
      } else {
        flat[offset] = accumulators(i);
      }
    }
    __syncthreads();
    __syncthreads();
  } else if constexpr (
      PrefixPolicy == PPU_SPLITK_SHARED_PREFIX_R2S_VECTOR) {
    splitk_shared_prefix_r2s<false, false, PartialEpilogue>(
        accumulators, tiled_mma, thread_idx, smem_buf);
  } else if constexpr (
      PrefixPolicy == PPU_SPLITK_SHARED_PREFIX_R2S_SCALAR) {
    splitk_shared_prefix_r2s<true, false, PartialEpilogue>(
        accumulators, tiled_mma, thread_idx, smem_buf);
  } else if constexpr (
      PrefixPolicy == PPU_SPLITK_SHARED_PREFIX_R2S_SCALAR_SNAPSHOT) {
    auto snapshot = splitk_shared_prefix_scalar_snapshot(accumulators);
    splitk_shared_prefix_r2s<true, false, PartialEpilogue>(
        snapshot, tiled_mma, thread_idx, smem_buf);
  } else if constexpr (
      PrefixPolicy == PPU_SPLITK_SHARED_PREFIX_R2S_S2R_VECTOR) {
    splitk_shared_prefix_r2s<false, true, PartialEpilogue>(
        accumulators, tiled_mma, thread_idx, smem_buf);
  } else {
    splitk_shared_prefix_r2s<true, true, PartialEpilogue>(
        accumulators, tiled_mma, thread_idx, smem_buf);
  }
}

}  // namespace cutlass::gemm::kernel::detail
