/***************************************************************************************************
 * Copyright (c) 2026 quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * PPU0010 AIU/MMA atoms used by the fixed-shape chunked-GDN forward path.
 *
 * This file is intentionally below the GDN algebra.  It proves the exact BF16
 * transport and FP32-accumulating tensor-core operations that the collective
 * is allowed to use; it does not claim that the complete recurrence is wired.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cute/arch/copy_ppu.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/arch/mma_ppu0010.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"

namespace cutlass::linear_attention::detail {

// PPU0010 moves a b16 AIU operand in 128-byte contiguous runs.  A 64x64 cube
// is therefore the natural reusable unit for C64/D128: a complete D128 operand
// is represented by two adjacent cubes, never by an invented plain-LDSM path.
struct PpuChunkedGdnBf16Aiu {
  using Element = cutlass::bfloat16_t;
  using MmaOperation = cute::PPU0010_16x16x16_F32BF16BF16F32_TN;
  using MmaTraits = cute::MMA_Traits<MmaOperation>;

  static constexpr int kWarpSize = 32;
  static constexpr int kAtomM = 16;
  static constexpr int kAtomN = 16;
  static constexpr int kAtomK = 16;
  static constexpr int kCubeRows = 64;
  static constexpr int kCubeColumns = 64;
  static constexpr int kCubeElements = kCubeRows * kCubeColumns;
  static constexpr int kCubeBits = kCubeElements * 16;
  static constexpr int kCubeBytes = kCubeElements * int(sizeof(Element));
  static constexpr int kHeadDimension = 128;
  static constexpr int kCubesPerHead = kHeadDimension / kCubeColumns;

  using AiuLoad = cute::PPU0010_AIU_LOAD<
      cute::C<kCubeBits>, Element, false, true>;
  using AiuLoadTransposed = cute::PPU0010_AIU_LOAD<
      cute::C<kCubeBits>, Element, true, true>;
  // Keep A/B and N/T forms distinct even though the current ppu0010 asm body
  // does not consume Swap directly.  The vendor CollectiveBuilder selects
  // Swap=false for A and Swap=true for B, and its partition/retile mapping is
  // part of the operand ABI.  These raw forms are opcode probes; production
  // GEMM coordinates are owned by the builder-backed collective below them.
  using AReadN = cute::PPU0010_TSM_LD_SWZL<
      Element, kCubeRows, kCubeColumns, false, false, 1>;
  using BReadN = cute::PPU0010_TSM_LD_SWZL<
      Element, kCubeRows, kCubeColumns, true, false, 1>;
  using AReadT = cute::PPU0010_TSM_LD_SWZL<
      Element, kCubeRows, kCubeColumns, false, true, 1>;
  using BReadT = cute::PPU0010_TSM_LD_SWZL<
      Element, kCubeRows, kCubeColumns, true, true, 1>;

  struct alignas(16) OperandFragment {
    std::uint32_t value[4]{};
  };

  struct alignas(16) AccumulatorFragment {
    float value[8]{};

    CUTLASS_HOST_DEVICE void clear() {
#pragma unroll
      for (int i = 0; i < 8; ++i) value[i] = 0.0f;
    }
  };

  static_assert(sizeof(Element) == 2, "GDN AIU path requires a 16-bit BF16 element");
  static_assert(kCubeBytes == 8192 && kCubesPerHead == 2,
                "C64/D128 must use two physical 64x64 BF16 AIU cubes");
  static_assert(std::extent_v<typename MmaOperation::ARegisters> == 4 &&
                    std::extent_v<typename MmaOperation::BRegisters> == 4 &&
                    std::extent_v<typename MmaOperation::CRegisters> == 8 &&
                    std::extent_v<typename MmaOperation::DRegisters> == 8,
                "PPU0010 BF16 MMA register ABI changed");
  static_assert(cute::size(typename cute::Copy_Traits<AiuLoad>::ThrID{}) == 1,
                "AIU bulk transfer must remain one opaque logical copy; physical issue is warp-synchronous");
  static_assert(cute::size(typename MmaTraits::ThrID{}) == kWarpSize,
                "PPU0010 BF16 MMA must be a 32-thread operation");

  // The descriptor is kept explicit because ppu0010 encodes the physical row
  // pitch in dim_w.  In particular, token-major Q/K/V may have heads between
  // consecutive rows; callers must pass that real leading dimension.
  CUTLASS_HOST_DEVICE static cute::AiuDesc make_nontransposed_desc(
      Element const* base, int valid_rows, int leading_dimension) {
    cute::AiuDesc desc{};
    desc.gmem_ptr = reinterpret_cast<std::uint8_t const*>(base);
    desc.dim_h = valid_rows;
    desc.dim_w = leading_dimension;
    desc.cube_h = kCubeRows;
    desc.cube_w = kCubeColumns;
    desc.offset_w = 0;
    return desc;
  }

  CUTLASS_HOST_DEVICE static cute::AiuDesc make_transposed_desc(
      Element const* base, int reduction_extent, int leading_dimension) {
    cute::AiuDesc desc{};
    desc.gmem_ptr = reinterpret_cast<std::uint8_t const*>(base);
    // For a transposed operand dim_h is GEMM K, not the number of output rows.
    desc.dim_h = reduction_extent;
    desc.dim_w = leading_dimension;
    desc.cube_h = kCubeRows;
    desc.cube_w = kCubeColumns;
    desc.offset_w = 0;
    return desc;
  }

  // coord0/coord1 deliberately mirror the raw operation.  They are
  // (column,row) for the ordinary load and (row,column) for the transposed
  // load.  Giving the two modes different named entry points prevents an
  // accidental coordinate swap from hiding in a bool argument.
  CUTLASS_DEVICE static void issue_nontransposed(
      void* shared_cube, Element const* global_base, cute::AiuDesc const& desc,
      int column, int row, int thread_idx) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    // AIU issue is warp-synchronous on ppu0010.  Copy_Traits::ThrID=_1
    // describes one opaque logical copy, not permission for one physical lane
    // to execute the instruction divergently.
    if ((thread_idx >> 5) == 0) {
      AiuLoad::copy(shared_cube, global_base, desc, column, row, 0);
    }
#elif defined(__HGGC_ARCH__)
    (void)shared_cube;
    (void)global_base;
    (void)desc;
    (void)column;
    (void)row;
    (void)thread_idx;
    CUTE_INVALID_CONTROL_PATH("PPU chunked GDN AIU load requires ppu0010");
#else
    (void)shared_cube;
    (void)global_base;
    (void)desc;
    (void)column;
    (void)row;
    (void)thread_idx;
#endif
  }

  CUTLASS_DEVICE static void issue_transposed(
      void* shared_cube, Element const* global_base, cute::AiuDesc const& desc,
      int row, int column, int thread_idx) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    if ((thread_idx >> 5) == 0) {
      AiuLoadTransposed::copy(shared_cube, global_base, desc, row, column, 0);
    }
#elif defined(__HGGC_ARCH__)
    (void)shared_cube;
    (void)global_base;
    (void)desc;
    (void)row;
    (void)column;
    (void)thread_idx;
    CUTE_INVALID_CONTROL_PATH("PPU chunked GDN transposed AIU load requires ppu0010");
#else
    (void)shared_cube;
    (void)global_base;
    (void)desc;
    (void)row;
    (void)column;
    (void)thread_idx;
#endif
  }

  // Every CTA thread must execute this method.  The CTA barrier is required
  // after wait_all: completion of the DMA does not by itself publish the
  // shared-memory payload to peer threads.
  CUTLASS_DEVICE static void commit_wait_and_sync() {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
#elif defined(__HGGC_ARCH__)
    CUTE_INVALID_CONTROL_PATH("PPU chunked GDN AIU wait requires ppu0010");
#endif
  }

  CUTLASS_DEVICE static OperandFragment read_a_nontransposed(
      void* shared_cube, int column, int row) {
    OperandFragment fragment{};
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    AReadN::copy(fragment.value, shared_cube, column, row, 0, 0);
#else
    (void)shared_cube;
    (void)column;
    (void)row;
#if defined(__HGGC_ARCH__)
    CUTE_INVALID_CONTROL_PATH("PPU chunked GDN swizzle read requires ppu0010");
#endif
#endif
    return fragment;
  }

  CUTLASS_DEVICE static OperandFragment read_b_nontransposed(
      void* shared_cube, int column, int row) {
    OperandFragment fragment{};
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    BReadN::copy(fragment.value, shared_cube, column, row, 0, 0);
#else
    (void)shared_cube;
    (void)column;
    (void)row;
#if defined(__HGGC_ARCH__)
    CUTE_INVALID_CONTROL_PATH("PPU chunked GDN B swizzle read requires ppu0010");
#endif
#endif
    return fragment;
  }

  CUTLASS_DEVICE static OperandFragment read_a_transposed(
      void* shared_cube, int row, int column) {
    OperandFragment fragment{};
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    AReadT::copy(fragment.value, shared_cube, row, column, 0, 0);
#else
    (void)shared_cube;
    (void)row;
    (void)column;
#if defined(__HGGC_ARCH__)
    CUTE_INVALID_CONTROL_PATH("PPU chunked GDN transposed swizzle read requires ppu0010");
#endif
#endif
    return fragment;
  }

  CUTLASS_DEVICE static OperandFragment read_b_transposed(
      void* shared_cube, int row, int column) {
    OperandFragment fragment{};
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    BReadT::copy(fragment.value, shared_cube, row, column, 0, 0);
#else
    (void)shared_cube;
    (void)row;
    (void)column;
#if defined(__HGGC_ARCH__)
    CUTE_INVALID_CONTROL_PATH("PPU chunked GDN transposed B swizzle read requires ppu0010");
#endif
#endif
    return fragment;
  }

  // This is the only tensor-core atom admitted by the v1 collective.  Keeping
  // the call here makes it impossible for a later phase to silently select the
  // PPU F16-accumulator atom, whose ppu0010 implementation is fail-closed.
  CUTLASS_DEVICE static void mma(
      AccumulatorFragment& accum, OperandFragment const& a,
      OperandFragment const& b) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    MmaOperation::fma(
        accum.value[0], accum.value[1], accum.value[2], accum.value[3],
        accum.value[4], accum.value[5], accum.value[6], accum.value[7],
        a.value[0], a.value[1], a.value[2], a.value[3],
        b.value[0], b.value[1], b.value[2], b.value[3],
        accum.value[0], accum.value[1], accum.value[2], accum.value[3],
        accum.value[4], accum.value[5], accum.value[6], accum.value[7]);
#else
    (void)accum;
    (void)a;
    (void)b;
#if defined(__HGGC_ARCH__)
    CUTE_INVALID_CONTROL_PATH("PPU chunked GDN BF16 MMA requires ppu0010");
#endif
#endif
  }
};

}  // namespace cutlass::linear_attention::detail
