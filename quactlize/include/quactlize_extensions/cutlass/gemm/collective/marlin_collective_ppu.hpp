/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Marlin's W4A16 mainloop, expressed as an independent collective.
 *
 * This is intentionally not a mode of quactlize_mma_mixed_input.hpp.  The reference cadence, packed
 * B representation and grouped-scale permutation are one contract; putting them behind branches in
 * the generic collective was both instruction-expensive and impossible to audit against Marlin.
 *
 * The first admitted specialization is the decode reference point
 *   Tile=16x128x128, Warp=16x64x32, 1M x 2N x 4K, 256 threads, Stages=4, W4 gs128.
 * The template surface is retained so proven shapes can later become sweep axes.  Until each one has
 * a byte-map and instruction-cadence oracle it fails at compile time instead of silently selecting a
 * generic fallback.
 **************************************************************************************************/
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__HGGCCC__)
#include <hggc_fp16.h>
#else
#include <cuda_fp16.h>
#endif

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "quactlize_extensions/cutlass/gemm/collective/marlin_dequant_ppu.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/marlin_load_ppu.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/marlin_mma_ppu.hpp"

namespace cutlass::gemm::collective {

namespace marlin_ppu_detail {

CUTLASS_HOST_DEVICE constexpr int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

}  // namespace marlin_ppu_detail

template <
    class TileShape_, class WarpShape_, int Stages_, int GroupSize_,
    class StrideA_, class StrideB_, class StrideScale_,
    class LoadPolicy_ = MarlinCpAsyncLoadPolicyPPU>
class MarlinCollectivePPU {
 public:
  using TileShape = TileShape_;
  using WarpShape = WarpShape_;
  using StrideA = StrideA_;
  using StrideB = StrideB_;
  using StrideScale = StrideScale_;
  using LoadPolicy = LoadPolicy_;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::int4b_t;
  using ElementScale = cutlass::half_t;
  using ElementAccumulator = float;
  using FragmentC = marlin_ppu_detail::FragmentC;
  using Accumulator = marlin_ppu_detail::MarlinAccumulatorPPU;
  using ArchTag = cutlass::arch::PPU0010;
  using DispatchPolicy = MainloopPPUAiu<Stages_, KernelAiuMultistageMixedInput>;
  using TransformA = cute::identity;
  using TransformB = cute::identity;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr bool SwapAB = false;

  static constexpr int Stages = Stages_;
  static constexpr int GroupSize = GroupSize_;
  static constexpr int TileM = int(cute::size<0>(TileShape{}));
  static constexpr int TileN = int(cute::size<1>(TileShape{}));
  static constexpr int TileK = int(cute::size<2>(TileShape{}));
  static constexpr int WarpM = int(cute::size<0>(WarpShape{}));
  static constexpr int WarpN = int(cute::size<1>(WarpShape{}));
  static constexpr int WarpK = int(cute::size<2>(WarpShape{}));
  static constexpr int WarpOnM = TileM / WarpM;
  static constexpr int WarpOnN = TileN / WarpN;
  static constexpr int WarpOnK = TileK / WarpK;
  static constexpr int Threads = 32 * WarpOnM * WarpOnN * WarpOnK;

  static_assert(cute::is_same_v<LoadPolicy, MarlinCpAsyncLoadPolicyPPU>,
                "the first Marlin PPU baseline admits only classic cp.async loads");
  static_assert(TileM == 16 && TileN == 128 && TileK == 128,
                "MarlinCollectivePPU shape is not yet proved by the fixed-target oracle");
  static_assert(WarpM == 16 && WarpN == 64 && WarpK == 32,
                "MarlinCollectivePPU warp shape is not yet proved by the fixed-target oracle");
  static_assert(Stages == 4 && GroupSize == 128 && Threads == 256,
                "the first Marlin PPU baseline is s4, gs128 and 256 threads");

  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<cute::PPU0010_16x16x16_F32F16F16F32_TN>,
      cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_4>>,
      cute::Tile<cute::_16, cute::_32, cute::_64>>;
  // TiledMma supplies the real 1M x 2N x 4K thread topology and the 32-value
  // register extent.  The single PPU n16 instruction's C register map is the
  // classic acc_i/acc_j map, not CuTe's NVIDIA two-n8 logical C partition;
  // MarlinKernelPPU therefore owns that explicit output map.
  static_assert(cute::size(TiledMma{}) == Threads);

  static constexpr int KBlocks = TileK / 16;
  static constexpr int NBlocks = TileN / 16;
  static constexpr int ASharedStride = 16 * KBlocks / 8;
  static constexpr int ASharedStage = ASharedStride * TileM;
  static constexpr int BSharedStride = 32 * NBlocks / 4;
  static constexpr int BSharedStage = BSharedStride * KBlocks;
  static constexpr int BInnerIters = BSharedStage / Threads;
  static constexpr int ScaleSharedStride = 16 * NBlocks / 8;
  static constexpr int ScaleSharedStage = ScaleSharedStride;
  static constexpr int AGlobalOuter = TileK / 8;
  static constexpr int ASharedWriteDelta =
      ASharedStride * (Threads / AGlobalOuter);
  static constexpr int ASharedReadOuter =
      2 * ((Threads / 32) / (NBlocks / 4));
  static constexpr int ASharedWriteIters =
      marlin_ppu_detail::ceil_div(ASharedStage, ASharedWriteDelta);
  static_assert(ASharedStage == 256 && BSharedStage == 512 &&
                    BInnerIters == 2 && ScaleSharedStage == 16 &&
                    AGlobalOuter == 16 && ASharedWriteDelta == 256 &&
                    ASharedReadOuter == 8 && ASharedWriteIters == 1);

  struct SharedStorage {
    alignas(16) marlin_ppu_detail::Vector128 storage[
        Stages * (ASharedStage + BSharedStage + ScaleSharedStage)];
  };
  static_assert(sizeof(SharedStorage) == 50176,
                "fixed Marlin mainloop shared-memory ledger drifted");

  struct Arguments {
    ElementA const* ptr_A = nullptr;
    ElementB const* ptr_B = nullptr;
    ElementScale const* ptr_S = nullptr;
    int group_size = 0;
  };

  struct Params {
    ElementA const* ptr_A = nullptr;
    ElementB const* ptr_B = nullptr;
    ElementScale const* ptr_S = nullptr;
  };

  // Classic computes the thread topology and all fixed source/destination
  // coordinates once, then only rebases three source pointers when its stripe
  // crosses an output tile.  Keep those lifetimes explicit: CtaState contains
  // final per-thread invariants, SegmentState contains already-rebased source
  // pointers, and SharedBases names the stage allocation exactly once per CTA.
  // Device numRegs is a post-build observation, not something this
  // source-level separation pretends to prove.
  struct CtaState {
    marlin_ppu_detail::Vector128 const* a_thread_base = nullptr;
    marlin_ppu_detail::Vector128 const* b_thread_base = nullptr;
    marlin_ppu_detail::Vector128 const* scale_thread_base = nullptr;
    int tid = 0;
    int b_inner_delta = 0;
    int b_k_delta = 0;
    int scale_k_delta = 0;
    int a_smem_write = 0;
    int a_smem_read[BInnerIters]{};
    int scale_smem_read = 0;
    bool a_copy_pred = false;
    bool scale_copy_pred = false;
  };

  struct SegmentState {
    marlin_ppu_detail::Vector128 const* a = nullptr;
    marlin_ppu_detail::Vector128 const* b[BInnerIters]{};
    marlin_ppu_detail::Vector128 const* scale = nullptr;
    int k_tiles_remaining = 0;
  };

  struct SharedBases {
    marlin_ppu_detail::Vector128* a = nullptr;
    marlin_ppu_detail::Vector128* b = nullptr;
    marlin_ppu_detail::Vector128* scale = nullptr;
  };

  static_assert(std::is_standard_layout_v<CtaState> &&
                    std::is_trivially_copyable_v<CtaState> &&
                    std::is_standard_layout_v<SegmentState> &&
                    std::is_trivially_copyable_v<SegmentState> &&
                    std::is_standard_layout_v<SharedBases> &&
                    std::is_trivially_copyable_v<SharedBases>,
                "standalone Marlin address state must stay register-local");

  CUTLASS_HOST_DEVICE static SharedBases make_shared_bases(
      SharedStorage& shared) {
    SharedBases bases;
    bases.a = shared.storage;
    bases.b = bases.a + Stages * ASharedStage;
    bases.scale = bases.b + Stages * BSharedStage;
    return bases;
  }

  CUTLASS_HOST_DEVICE static constexpr int transform_a_index(int index) {
    int const row = index / AGlobalOuter;
    return AGlobalOuter * row + ((index % AGlobalOuter) ^ row);
  }

  // No WorkTileInfo is accepted here by design.  A valid CTA initializes this
  // state exactly once; all q/K-dependent arithmetic belongs to rebase_segment.
  CUTLASS_HOST_DEVICE static CtaState init_cta_state(
      Params const& params, int problem_m, int problem_n, int problem_k,
      int tid) {
    CtaState state;
    state.tid = tid;
    int const a_global_stride = problem_k / 8;
    // N is admitted only when it is divisible by 256.  Spell this as N/2:
    // 16*N/32 has the same mathematical value but can overflow before the
    // division for an otherwise representable host shape.
    int const b_global_stride = problem_n / 2;
    state.b_inner_delta =
        b_global_stride * (Threads / BSharedStride);
    state.b_k_delta = b_global_stride * KBlocks;
    state.scale_k_delta = problem_n / 8;
    int const a_shared_write =
        ASharedStride * (tid / AGlobalOuter) + tid % AGlobalOuter;
    int const a_shared_read =
        ASharedStride * ((tid % 32) % 16) + (tid % 32) / 16 +
        2 * ((tid / 32) / (NBlocks / 4));
    state.a_smem_write = transform_a_index(a_shared_write);
    #pragma unroll
    for (int i = 0; i < BInnerIters; ++i) {
      state.a_smem_read[i] =
          transform_a_index(ASharedReadOuter * i + a_shared_read);
    }
    int const warp_n = (tid / 32) % (NBlocks / 4);
    int const lane = tid % 32;
    state.scale_smem_read = 8 * warp_n + lane / 4;
    state.a_copy_pred = a_shared_write < ASharedStride * problem_m;
    state.scale_copy_pred = tid < ScaleSharedStride;
    auto const* a = reinterpret_cast<marlin_ppu_detail::Vector128 const*>(
        params.ptr_A);
    auto const* b = reinterpret_cast<marlin_ppu_detail::Vector128 const*>(
        params.ptr_B);
    auto const* scale =
        reinterpret_cast<marlin_ppu_detail::Vector128 const*>(params.ptr_S);
    state.a_thread_base =
        a + a_global_stride * (tid / AGlobalOuter) + tid % AGlobalOuter;
    state.b_thread_base =
        b + b_global_stride * (tid / BSharedStride) + tid % BSharedStride;
    state.scale_thread_base = scale + tid;
    return state;
  }

  template <class WorkTile>
  CUTLASS_HOST_DEVICE static SegmentState rebase_segment(
      CtaState const& state, WorkTile const& work) {
    SegmentState segment;
    int const n_tile = int(work.N_idx);
    int const k_tile_begin = int(work.K_idx);
    int const k_tile_count = int(work.k_tile_count);
    // This is an assume-valid device seam.  Kernel Params are host-lowered
    // only after the mainloop and scheduler contracts pass, and L170/L178
    // exhaust every reachable q/K/count descriptor.  Repeating those bounds
    // in every thread and segment is neither a real fail-close (run_segment
    // cannot consume an invalid/empty SegmentState) nor part of classic.
    segment.a = state.a_thread_base + AGlobalOuter * k_tile_begin;
    marlin_ppu_detail::Vector128 const* b =
        state.b_thread_base + BSharedStride * n_tile +
        state.b_k_delta * k_tile_begin;
    #pragma unroll
    for (int i = 0; i < BInnerIters; ++i) {
      segment.b[i] = b + state.b_inner_delta * i;
    }
    segment.scale = state.scale_thread_base + ScaleSharedStride * n_tile +
                    state.scale_k_delta * k_tile_begin;
    segment.k_tiles_remaining = k_tile_count;
    return segment;
  }

  template <class ProblemShape>
  static bool address_arithmetic_supported(ProblemShape const& problem_shape) {
    auto mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
    int64_t const n_signed = int64_t(cute::get<1>(mnkl));
    int64_t const k_signed = int64_t(cute::get<2>(mnkl));
    if (n_signed <= 0 || k_signed <= 0) {
      return false;
    }
    uint64_t const n = uint64_t(n_signed);
    uint64_t const k = uint64_t(k_signed);
    uint64_t const int_max = uint64_t(std::numeric_limits<int>::max());
    auto const mul_fits_int = [int_max](uint64_t a, uint64_t b) {
      return a == 0 || b <= int_max / a;
    };
    auto const mul_add_fits_int = [int_max](
        uint64_t a, uint64_t b, uint64_t c) {
      return c <= int_max && (a == 0 || b <= (int_max - c) / a);
    };

    uint64_t const n_tiles = n / uint64_t(TileN);
    uint64_t const k_tiles = k / uint64_t(TileK);
    uint64_t const a_global_stride = k / 8;
    uint64_t const b_global_stride = n / 2;
    uint64_t const scale_k_delta = n / 8;
    uint64_t const last_n_tile = n_tiles == 0 ? 0 : n_tiles - 1;
    uint64_t const last_k_tile = k_tiles == 0 ? 0 : k_tiles - 1;

    if (!mul_fits_int(b_global_stride, KBlocks)) {
      return false;
    }
    uint64_t const b_k_delta = b_global_stride * uint64_t(KBlocks);

    // Every product below is evaluated as int in init_cta_state or
    // rebase_segment.  Prove each one before Arguments are lowered so the
    // assume-valid device path cannot acquire signed-overflow UB.
    return
        // Thread-local bases are formed by int multiply-add expressions.
        mul_add_fits_int(
            a_global_stride, Threads / AGlobalOuter - 1,
            AGlobalOuter - 1) &&
        mul_add_fits_int(
            b_global_stride, Threads / BSharedStride - 1,
            BSharedStride - 1) &&
        // These values are materialized as int deltas in CtaState.
        mul_fits_int(b_global_stride, Threads / BSharedStride) &&
        // Segment q/K products are also evaluated as int before rebasing a
        // pointer.  Checking their maxima covers every reachable descriptor.
        mul_fits_int(uint64_t(BSharedStride), last_n_tile) &&
        mul_fits_int(b_k_delta, last_k_tile) &&
        mul_fits_int(uint64_t(ScaleSharedStride), last_n_tile) &&
        mul_fits_int(scale_k_delta, last_k_tile) &&
        mul_fits_int(uint64_t(AGlobalOuter), last_k_tile);
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
    int64_t const m = int64_t(cute::get<0>(mnkl));
    int64_t const n = int64_t(cute::get<1>(mnkl));
    int64_t const k = int64_t(cute::get<2>(mnkl));
    int64_t const l = int64_t(cute::get<3>(mnkl));
    auto const aligned_16 = [](void const* ptr) {
      return (reinterpret_cast<uintptr_t>(ptr) & uintptr_t(15)) == 0;
    };
    return args.ptr_A != nullptr && args.ptr_B != nullptr && args.ptr_S != nullptr &&
           aligned_16(args.ptr_A) && aligned_16(args.ptr_B) && aligned_16(args.ptr_S) &&
           args.group_size == GroupSize && m > 0 && m <= TileM && l == 1 &&
           n > 0 && n % 256 == 0 && k > 0 && k % TileK == 0 &&
           address_arithmetic_supported(problem_shape);
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return {args.ptr_A, args.ptr_B, args.ptr_S};
  }

  template <int NBlock>
  CUTLASS_DEVICE static void multiply_n_block(
      marlin_ppu_detail::FragmentA const& fragment_a,
      marlin_ppu_detail::Vector128 const& fragment_b_quant,
      marlin_ppu_detail::FragmentScale const (&fragment_scale)[4],
      FragmentC& accum) {
    static_assert(NBlock >= 0 && NBlock < 4,
                  "the fixed Marlin warp owns exactly four n16 blocks");
    uint32_t const* quant =
        reinterpret_cast<uint32_t const*>(&fragment_b_quant);
    int const q = int(quant[NBlock]);
    marlin_ppu_detail::FragmentB b0 =
        marlin_ppu_detail::dequantize_biased_int4(q);
    marlin_ppu_detail::FragmentB b1 =
        marlin_ppu_detail::dequantize_biased_int4(q >> 8);
    marlin_ppu_detail::scale(b0, fragment_scale[NBlock], 0);
    marlin_ppu_detail::scale(b1, fragment_scale[NBlock], 1);
    marlin_ppu_detail::mma_n16<NBlock>(fragment_a, b0, b1, accum);
  }

  CUTLASS_DEVICE static void run_segment(
      CtaState const& state, SegmentState const& segment,
      SharedBases const& shared, Accumulator& accum) {
    using marlin_ppu_detail::FragmentA;
    using marlin_ppu_detail::FragmentScale;
    using marlin_ppu_detail::Vector128;

    int const tid = state.tid;
    int k_tiles_remaining = segment.k_tiles_remaining;

    Vector128 const* a_pointer = segment.a;
    Vector128 const* b_pointer[BInnerIters];
    #pragma unroll
    for (int i = 0; i < BInnerIters; ++i) {
      b_pointer[i] = segment.b[i];
    }
    Vector128 const* scale_pointer = segment.scale;

    FragmentA fragment_a[2];
    marlin_ppu_detail::Vector128 fragment_b_quant[2];
    FragmentScale fragment_scale[2][4];

    auto copy_stage = [&](int pipe, int a_offset, bool predicate) {
      if (predicate) {
        Vector128* a_stage = shared.a + ASharedStage * pipe;
        marlin_ppu_detail::cp_async_16_if(
            &a_stage[state.a_smem_write],
            &a_pointer[AGlobalOuter * a_offset], state.a_copy_pred);
        Vector128* b_stage = shared.b + BSharedStage * pipe;
        #pragma unroll
        for (int i = 0; i < BInnerIters; ++i) {
          marlin_ppu_detail::cp_async_16(
              &b_stage[Threads * i + tid], b_pointer[i]);
          b_pointer[i] += state.b_k_delta;
        }
        // TileK == GroupSize in the admitted target, hence every pipeline stage begins a group.
        if (state.scale_copy_pred) {
          marlin_ppu_detail::cp_async_16(
              &shared.scale[ScaleSharedStage * pipe + tid], scale_pointer);
        }
        scale_pointer += state.scale_k_delta;
      }
      marlin_ppu_detail::cp_async_commit();
    };

    auto wait_stage = [&]() {
      marlin_ppu_detail::cp_async_wait<Stages - 2>();
      __syncthreads();
    };

    auto load_registers = [&](int inner, int pipe) {
      Vector128 const* scale_stage =
          shared.scale + ScaleSharedStage * pipe;
      *reinterpret_cast<Vector128*>(&fragment_scale[inner & 1]) =
          scale_stage[state.scale_smem_read];

      Vector128 const* a_stage = shared.a + ASharedStage * pipe;
      marlin_ppu_detail::ldmatrix_a(
          fragment_a[inner & 1],
          &a_stage[state.a_smem_read[inner % BInnerIters]]);

      Vector128 const* b_stage = shared.b + BSharedStage * pipe;
      fragment_b_quant[inner & 1] =
          b_stage[Threads * (inner % BInnerIters) + tid];
    };

    auto multiply = [&](int inner) {
      // Spell the four compile-time N blocks explicitly.  The sweep remains a compile-time axis;
      // this fixed classic row must not pay a runtime dispatch or predicate for selecting a block.
      multiply_n_block<0>(fragment_a[inner & 1], fragment_b_quant[inner & 1],
                          fragment_scale[inner & 1], accum.fragments[0]);
      multiply_n_block<1>(fragment_a[inner & 1], fragment_b_quant[inner & 1],
                          fragment_scale[inner & 1], accum.fragments[1]);
      multiply_n_block<2>(fragment_a[inner & 1], fragment_b_quant[inner & 1],
                          fragment_scale[inner & 1], accum.fragments[2]);
      multiply_n_block<3>(fragment_a[inner & 1], fragment_b_quant[inner & 1],
                          fragment_scale[inner & 1], accum.fragments[3]);
    };

    #pragma unroll
    for (int i = 0; i < Stages - 1; ++i) {
      copy_stage(i, i, i < k_tiles_remaining);
    }
    wait_stage();
    load_registers(0, 0);
    #pragma unroll
    for (int n_block = 0; n_block < 4; ++n_block) {
      #pragma unroll
      for (int value = 0; value < 8; ++value) {
        accum.fragments[n_block].value[value] = 0.0f;
      }
    }
    a_pointer += AGlobalOuter * (Stages - 1);

    while (k_tiles_remaining > 0) {
      #pragma unroll
      for (int pipe = 0; pipe < Stages;) {
        #pragma unroll
        for (int inner = 0; inner < BInnerIters; ++inner) {
          load_registers(inner + 1, pipe % Stages);
          if (inner == BInnerIters - 2) {
            copy_stage(
                (pipe + Stages - 1) % Stages, pipe,
                k_tiles_remaining >= Stages);
            ++pipe;
            wait_stage();
          }
          multiply(inner);
        }
        --k_tiles_remaining;
        if (k_tiles_remaining == 0) {
          break;
        }
      }
      a_pointer += AGlobalOuter * Stages;
    }
    marlin_ppu_detail::cp_async_wait<0>();
  }
};

}  // namespace cutlass::gemm::collective
