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
 * The admitted classic-geometry family keeps WarpN64/WarpK32 while sweeping
 * the CTA N/K split.  In addition to 128x128 (2N x 4K), the first proved
 * extensions are 64x128 (1N x 4K), 128x64 (2N x 2K), and 256x64
 * (4N x 2K), for TileM={8,16} and Stages={2,3,4,5,6}.
 * The template surface is retained so proven shapes can later become sweep axes.  Until each one has
 * a byte-map and instruction-cadence oracle it fails at compile time instead of silently selecting a
 * generic fallback.
 **************************************************************************************************/
#pragma once

#include <cstddef>
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

// Same-SHA outer-pipeline footprint experiment for the exact standalone m8
// target.  Mode 0 is the shipping spelling and remains the default.  Mode 1
// rolls only the outer `pipe` loop; the register-indexed inner loop stays
// fully unrolled.  Mode 2 is a compile/disassembly-only causal control that
// rolls both loops and must violate the <=124-register/no-spill admission
// criterion before mode 1 is profiled.
//
//   PPU_DEFS=PPU_MARLIN_PIPE_ROLL=1 TARGET=test_lowbit_dense_marlin_m8_ab ./build.sh
#ifndef PPU_MARLIN_PIPE_ROLL
#define PPU_MARLIN_PIPE_ROLL 0
#endif
static_assert(PPU_MARLIN_PIPE_ROLL >= 0 && PPU_MARLIN_PIPE_ROLL <= 2,
              "PPU_MARLIN_PIPE_ROLL must be 0 (baseline), 1 (outer only), or 2 (outer+inner control)");

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
  static constexpr int NBlocksPerWarp = WarpN / 16;
  static constexpr int KInnerIters = WarpK / 16;
  static constexpr int BLoadsPerKInner = NBlocksPerWarp / 4;
  static constexpr int ComputeSteps = KInnerIters * BLoadsPerKInner;
  static constexpr int Threads = 32 * WarpOnM * WarpOnN * WarpOnK;
  static constexpr int PipeRollMode = PPU_MARLIN_PIPE_ROLL;
  static constexpr bool OuterPipeRolled = PipeRollMode != 0;
  static constexpr bool InnerLoopRolled = PipeRollMode == 2;
  static constexpr int InstructionM =
      TileM == 8 && WarpM == 8 ? 8 : 16;
  // The first m8 target is dense decode M=1.  Its ordinary shared-memory
  // reader can alias all masked output rows back onto the one resident A row,
  // so it stores neither the m16 physical tail nor seven logical padding rows.
  static constexpr int AStoredRows = InstructionM == 8 ? 1 : TileM;
  static constexpr int AccumulatorValues = InstructionM / 2;
  static constexpr int AccumulatorHalves = AccumulatorValues / 4;
  using FragmentC = marlin_ppu_detail::FragmentCFor<InstructionM>;
  using Accumulator = marlin_ppu_detail::MarlinAccumulatorForN<
      InstructionM, NBlocksPerWarp>;

  static_assert(cute::is_same_v<LoadPolicy, MarlinCpAsyncLoadPolicyPPU>,
                "the first Marlin PPU baseline admits only classic cp.async loads");
  static constexpr bool ProvedClassicGeometry =
      (TileN == 64 && TileK == 128) ||
      (TileN == 128 && (TileK == 64 || TileK == 128)) ||
      (TileN == 256 && TileK == 64);
  static_assert((TileM == 8 || TileM == 16) && ProvedClassicGeometry,
                "standalone Marlin N/K geometry lacks an exact cadence proof");
  static_assert(
      WarpM == TileM &&
          ((WarpN == 64 && WarpK == 32) ||
           (WarpN == 128 && WarpK == 16)),
      "standalone Marlin requires a proved WN64/WK32 or WN128/WK16 topology");
  static_assert(Stages >= 2 && Stages <= 6 &&
                    GroupSize == 128 && GroupSize % TileK == 0 &&
                    (Threads == 128 || Threads == 256),
                "standalone Marlin admits the proved s2..s6 ring, gs128 and 128/256 threads");

  using MmaAtom = std::conditional_t<
      InstructionM == 8,
      cute::MMA_Atom<cute::PPU0010_8x16x16_F32F16F16F32_TN>,
      cute::MMA_Atom<cute::PPU0010_16x16x16_F32F16F16F32_TN>>;
  using TiledMma = cute::TiledMMA<
      MmaAtom,
      cute::Layout<cute::Shape<
          cute::_1, cute::Int<WarpOnN>, cute::Int<WarpOnK>>>,
      cute::Tile<
          cute::Int<InstructionM>, cute::Int<16 * WarpOnN>,
          cute::Int<16 * WarpOnK>>>;
  // TiledMma supplies the real 1M x {1,2,4}N x {2,4}K thread topology.  The single PPU
  // n16 instruction's C register map is the
  // classic acc_i/acc_j map, not CuTe's NVIDIA two-n8 logical C partition;
  // MarlinKernelPPU therefore owns that explicit output map.
  static_assert(cute::size(TiledMma{}) == Threads);

  static constexpr int KBlocks = TileK / 16;
  static constexpr int NBlocks = TileN / 16;
  static constexpr int ASharedStride = 16 * KBlocks / 8;
  static constexpr int ASharedStage = ASharedStride * AStoredRows;
  static constexpr int BSharedStride = 32 * NBlocks / 4;
  static constexpr int BSharedStage = BSharedStride * KBlocks;
  static constexpr int BInnerIters = BSharedStage / Threads;
  static constexpr int ScaleSharedStride = 16 * NBlocks / 8;
  static constexpr int ScaleSharedStage = ScaleSharedStride;
  static constexpr int ScaleTilesPerGroup = GroupSize / TileK;
  static constexpr int AGlobalOuter = TileK / 8;
  static constexpr int ASharedWriteDelta =
      ASharedStride * (Threads / AGlobalOuter);
  static constexpr int ASharedReadOuter = 2 * WarpOnK;
  static constexpr int ASharedWriteIters =
      marlin_ppu_detail::ceil_div(ASharedStage, ASharedWriteDelta);
  static_assert(NBlocksPerWarp == 4 || NBlocksPerWarp == 8,
                "proved WarpN owns four or eight native n16 fragments");
  static_assert(BLoadsPerKInner >= 1 &&
                    NBlocksPerWarp == 4 * BLoadsPerKInner &&
                    BInnerIters == ComputeSteps && ComputeSteps == 2 &&
                    ASharedWriteIters >= 1 &&
                    (ScaleTilesPerGroup == 1 || ScaleTilesPerGroup == 2),
                "proved WN/WK geometry keeps two register-fed compute steps and one/two K tiles per gs128 group");
  static_assert(
      TileN != 128 || TileK != 128 || WarpN != 64 || WarpK != 32 ||
          ((InstructionM == 16 ? ASharedStage == 256 : ASharedStage == 16) &&
           BSharedStage == 512 && ScaleSharedStage == 16 &&
           AGlobalOuter == 16 && ASharedWriteDelta == 256 &&
           ASharedReadOuter == 8 && ASharedWriteIters == 1),
      "the original 128x128 WN64/WK32 copy ledger must remain byte-identical");

  CUTLASS_HOST_DEVICE static constexpr int a_producer_linear(
      int round, int tid) {
    return ASharedWriteDelta * round +
           ASharedStride * (tid / AGlobalOuter) + tid % AGlobalOuter;
  }

  CUTLASS_HOST_DEVICE static constexpr bool a_producer_active(
      int linear, int problem_m) {
    return linear >= 0 && linear < ASharedStage &&
           linear / AGlobalOuter < problem_m;
  }

  CUTLASS_HOST_DEVICE static constexpr int scale_group_index(
      int absolute_k_tile) {
    return absolute_k_tile / ScaleTilesPerGroup;
  }

  CUTLASS_HOST_DEVICE static constexpr int scale_group_offset(
      int phase, int tile_base, int tile_offset) {
    return (phase + tile_base + tile_offset) / ScaleTilesPerGroup;
  }

  struct SharedStorage {
    alignas(16) marlin_ppu_detail::Vector128 storage[
        Stages * (ASharedStage + BSharedStage + ScaleSharedStage)];
  };
  static_assert(
      sizeof(SharedStorage) ==
          std::size_t(Stages) *
              std::size_t(ASharedStage + BSharedStage + ScaleSharedStage) *
              sizeof(marlin_ppu_detail::Vector128),
      "standalone shared storage must scale exactly with the pipeline depth");
  static_assert(
      Stages != 4 || TileN != 128 || TileK != 128 ||
          sizeof(SharedStorage) == (InstructionM == 8 ? 34816 : 50176),
      "the shipping s4 shared ledger must remain byte-identical");

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
    marlin_ppu_detail::Vector128 const*
        a_thread_base[ASharedWriteIters]{};
    marlin_ppu_detail::Vector128 const* b_thread_base = nullptr;
    marlin_ppu_detail::Vector128 const* scale_thread_base = nullptr;
    int tid = 0;
    int b_inner_delta = 0;
    int b_k_delta = 0;
    int scale_k_delta = 0;
    int a_smem_write[ASharedWriteIters]{};
    // m16 entries are Vector128 indices.  m8 entries are fp16 indices into
    // its one-row packed stage because the PPU x2 provider window is 64 bits.
    int a_smem_read[BInnerIters]{};
    int b_smem_read[BInnerIters]{};
    int scale_smem_read[BInnerIters]{};
    bool a_copy_pred[ASharedWriteIters]{};
    bool scale_copy_pred = false;
  };

  struct SegmentState {
    marlin_ppu_detail::Vector128 const* a[ASharedWriteIters]{};
    marlin_ppu_detail::Vector128 const* b[BInnerIters]{};
    marlin_ppu_detail::Vector128 const* scale = nullptr;
    int scale_tile_phase = 0;
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
    // Preserve classic's exact operator order: `(row_stride*row + col) ^ row`.
    // It is equivalent to row_stride*row + (col^row) for TK128, but only the
    // former remains a bijection when TK64's eight-vector row lets a high row
    // bit overlap the row-stride bit.
    return (AGlobalOuter * row + index % AGlobalOuter) ^ row;
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
    int const warp_id = tid / 32;
    int const warp_n = warp_id % WarpOnN;
    int const warp_k = warp_id / WarpOnN;
    int const lane = tid % 32;
    int const a_lane_read =
        ASharedStride * (lane % 16) + lane / 16;
    #pragma unroll
    for (int i = 0; i < BInnerIters; ++i) {
      int const k_inner = i / BLoadsPerKInner;
      int const n_load = i % BLoadsPerKInner;
      int const k_block = k_inner * WarpOnK + warp_k;
      int const n_group = warp_n * BLoadsPerKInner + n_load;
      state.b_smem_read[i] =
          k_block * BSharedStride + n_group * 32 + lane;
      state.scale_smem_read[i] = 8 * n_group + lane / 4;
    }
    if constexpr (InstructionM == 8) {
      // Only row zero is resident.  A ppu x2 source lane `p` owns a 64-bit
      // window at K = 4*(p%2) + 8*(p/16) within the current 16-wide atom.
      // Output rows 1..7 are masked for M=1, so their providers deliberately
      // alias the same safe packed row rather than requiring padding storage.
      #pragma unroll
      for (int i = 0; i < BInnerIters; ++i) {
        int const k_inner = i / BLoadsPerKInner;
        int const k_block = k_inner * WarpOnK + warp_k;
        state.a_smem_read[i] =
            k_block * 16 + 4 * (lane % 2) +
            8 * (lane / 16);
      }
    } else {
      #pragma unroll
      for (int i = 0; i < BInnerIters; ++i) {
        int const k_inner = i / BLoadsPerKInner;
        int const k_block = k_inner * WarpOnK + warp_k;
        state.a_smem_read[i] =
            transform_a_index(2 * k_block + a_lane_read);
      }
    }
    // A packed M=1 stage is one 128-half row: 256 B, exactly sixteen
    // Vector128 transactions.  Threads 0..15 own those chunks once; all 256
    // threads execute the same stage cadence and concurrently own the 512
    // B-stage Vector128 transactions below.  Giving every thread an A
    // transaction would copy the same row sixteen times, not add useful
    // cooperation.
    state.scale_copy_pred = tid < ScaleSharedStride;
    auto const* a = reinterpret_cast<marlin_ppu_detail::Vector128 const*>(
        params.ptr_A);
    auto const* b = reinterpret_cast<marlin_ppu_detail::Vector128 const*>(
        params.ptr_B);
    auto const* scale =
        reinterpret_cast<marlin_ppu_detail::Vector128 const*>(params.ptr_S);
    // The A producer is a genuine vector-domain partition, not implicitly
    // "one vector per CTA thread".  TN64/TK128 m16 needs two rounds, while
    // the TN256/TK64 and packed-m8 cases use only an active prefix.  Form each
    // inactive pointer inside row zero so a false predicate never relies on
    // constructing an out-of-range C++ pointer.
    #pragma unroll
    for (int i = 0; i < ASharedWriteIters; ++i) {
      int const linear = a_producer_linear(i, tid);
      int const row = linear / AGlobalOuter;
      int const col = linear % AGlobalOuter;
      bool const active = a_producer_active(linear, problem_m);
      state.a_smem_write[i] = transform_a_index(active ? linear : 0);
      state.a_copy_pred[i] = active;
      state.a_thread_base[i] = a + a_global_stride * (active ? row : 0) + col;
    }
    state.b_thread_base =
        b + b_global_stride * (tid / BSharedStride) + tid % BSharedStride;
    // Only [0,ScaleSharedStride) owns scale copies.  Keep every inactive
    // thread's pointer within that resident vector row before the predicate is
    // evaluated; active-thread addresses are exactly unchanged.
    state.scale_thread_base = scale + tid % ScaleSharedStride;
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
    #pragma unroll
    for (int i = 0; i < ASharedWriteIters; ++i) {
      segment.a[i] =
          state.a_thread_base[i] + AGlobalOuter * k_tile_begin;
    }
    marlin_ppu_detail::Vector128 const* b =
        state.b_thread_base + BSharedStride * n_tile +
        state.b_k_delta * k_tile_begin;
    #pragma unroll
    for (int i = 0; i < BInnerIters; ++i) {
      segment.b[i] = b + state.b_inner_delta * i;
    }
    segment.scale = state.scale_thread_base + ScaleSharedStride * n_tile +
                    state.scale_k_delta * scale_group_index(k_tile_begin);
    segment.scale_tile_phase = k_tile_begin % ScaleTilesPerGroup;
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
    uint64_t const scale_groups = k / uint64_t(GroupSize);
    uint64_t const last_scale_group =
        scale_groups == 0 ? 0 : scale_groups - 1;

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
            a_global_stride, AStoredRows - 1,
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
        mul_fits_int(scale_k_delta, last_scale_group) &&
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
    bool const m_supported = InstructionM == 8 ? m == 1 : (m > 0 && m <= TileM);
    return args.ptr_A != nullptr && args.ptr_B != nullptr && args.ptr_S != nullptr &&
           aligned_16(args.ptr_A) && aligned_16(args.ptr_B) && aligned_16(args.ptr_S) &&
           args.group_size == GroupSize && m_supported && l == 1 &&
           n > 0 && n % 256 == 0 && k > 0 &&
           k % TileK == 0 && k % GroupSize == 0 &&
           address_arithmetic_supported(problem_shape);
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return {args.ptr_A, args.ptr_B, args.ptr_S};
  }

  template <int NBlock>
  CUTLASS_DEVICE static void multiply_n_block(
      marlin_ppu_detail::FragmentAFor<InstructionM> const& fragment_a,
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
    marlin_ppu_detail::mma_n16<InstructionM, NBlock>(
        fragment_a, b0, b1, accum);
  }

  CUTLASS_DEVICE static void run_segment(
      CtaState const& state, SegmentState const& segment,
      SharedBases const& shared, Accumulator& accum) {
    using FragmentA = marlin_ppu_detail::FragmentAFor<InstructionM>;
    using marlin_ppu_detail::FragmentScale;
    using marlin_ppu_detail::Vector128;

    int const tid = state.tid;
    int k_tiles_remaining = segment.k_tiles_remaining;

    Vector128 const* a_pointer[ASharedWriteIters];
    #pragma unroll
    for (int i = 0; i < ASharedWriteIters; ++i) {
      a_pointer[i] = segment.a[i];
    }
    Vector128 const* b_pointer[BInnerIters];
    #pragma unroll
    for (int i = 0; i < BInnerIters; ++i) {
      b_pointer[i] = segment.b[i];
    }
    Vector128 const* scale_group_base = segment.scale;
    int scale_tile_base = 0;

    FragmentA fragment_a[2];
    marlin_ppu_detail::Vector128 fragment_b_quant[2];
    FragmentScale fragment_scale[2][4];

    auto copy_stage = [&](int pipe, int a_offset, bool predicate) {
      if (predicate) {
        Vector128* a_stage = shared.a + ASharedStage * pipe;
        #pragma unroll
        for (int i = 0; i < ASharedWriteIters; ++i) {
          marlin_ppu_detail::cp_async_16_if(
              &a_stage[state.a_smem_write[i]],
              &a_pointer[i][AGlobalOuter * a_offset],
              state.a_copy_pred[i]);
        }
        Vector128* b_stage = shared.b + BSharedStage * pipe;
        #pragma unroll
        for (int i = 0; i < BInnerIters; ++i) {
          marlin_ppu_detail::cp_async_16(
              &b_stage[Threads * i + tid], b_pointer[i]);
          b_pointer[i] += state.b_k_delta;
        }
        // TK64 consumes the same gs128 scale group for two adjacent K tiles.
        // Each ring slot still owns its own staged copy; only the global
        // source group advances at the exact absolute-K boundary.
        int const scale_group = scale_group_offset(
            segment.scale_tile_phase, scale_tile_base, a_offset);
        Vector128 const* scale_pointer =
            scale_group_base + state.scale_k_delta * scale_group;
        if (state.scale_copy_pred) {
          marlin_ppu_detail::cp_async_16(
              &shared.scale[ScaleSharedStage * pipe + tid], scale_pointer);
        }
      }
      marlin_ppu_detail::cp_async_commit();
    };

    auto wait_stage = [&]() {
      marlin_ppu_detail::cp_async_wait<Stages - 2>();
      __syncthreads();
    };

    auto load_registers = [&](auto inner_c, int pipe) {
      constexpr int inner = decltype(inner_c)::value;
      constexpr int slot = inner % BInnerIters;
      Vector128 const* scale_stage =
          shared.scale + ScaleSharedStage * pipe;
      *reinterpret_cast<Vector128*>(&fragment_scale[inner & 1]) =
          scale_stage[state.scale_smem_read[slot]];

      Vector128 const* a_stage = shared.a + ASharedStage * pipe;
      if constexpr (InstructionM == 8) {
        auto const* a_half = reinterpret_cast<ElementA const*>(a_stage);
        marlin_ppu_detail::ldmatrix_a<InstructionM>(
            fragment_a[inner & 1],
            &a_half[state.a_smem_read[slot]]);
      } else {
        marlin_ppu_detail::ldmatrix_a<InstructionM>(
            fragment_a[inner & 1],
            &a_stage[state.a_smem_read[slot]]);
      }

      Vector128 const* b_stage = shared.b + BSharedStage * pipe;
      fragment_b_quant[inner & 1] =
          b_stage[state.b_smem_read[slot]];
    };

    auto multiply = [&](auto inner_c) {
      constexpr int inner = decltype(inner_c)::value;
      constexpr int accum_base = 4 * (inner % BLoadsPerKInner);
      // Every staged Vector128 owns four local n16 blocks.  The N-load index
      // selects which four native accumulator fragments receive them, with no
      // runtime switch or dynamic fragment index.
      multiply_n_block<0>(fragment_a[inner & 1], fragment_b_quant[inner & 1],
                          fragment_scale[inner & 1], accum.fragments[accum_base + 0]);
      multiply_n_block<1>(fragment_a[inner & 1], fragment_b_quant[inner & 1],
                          fragment_scale[inner & 1], accum.fragments[accum_base + 1]);
      multiply_n_block<2>(fragment_a[inner & 1], fragment_b_quant[inner & 1],
                          fragment_scale[inner & 1], accum.fragments[accum_base + 2]);
      multiply_n_block<3>(fragment_a[inner & 1], fragment_b_quant[inner & 1],
                          fragment_scale[inner & 1], accum.fragments[accum_base + 3]);
    };

    #pragma unroll
    for (int i = 0; i < Stages - 1; ++i) {
      copy_stage(i, i, i < k_tiles_remaining);
    }
    wait_stage();
    load_registers(cute::Int<0>{}, 0);
    #pragma unroll
    for (int n_block = 0; n_block < NBlocksPerWarp; ++n_block) {
      #pragma unroll
      for (int value = 0; value < AccumulatorValues; ++value) {
        accum.fragments[n_block].value[value] = 0.0f;
      }
    }
    #pragma unroll
    for (int i = 0; i < ASharedWriteIters; ++i) {
      a_pointer[i] += AGlobalOuter * (Stages - 1);
    }
    scale_tile_base += Stages - 1;

    while (k_tiles_remaining > 0) {
#if PPU_MARLIN_PIPE_ROLL == 0
      #pragma unroll
#else
      #pragma unroll 1
#endif
      for (int pipe = 0; pipe < Stages;) {
#if PPU_MARLIN_PIPE_ROLL == 2
        #pragma unroll 1
#else
        #pragma unroll
#endif
        cute::for_each(cute::make_int_sequence<BInnerIters>{},
                      [&](auto inner_c) {
          constexpr int inner = decltype(inner_c)::value;
          load_registers(cute::Int<inner + 1>{}, pipe % Stages);
          if constexpr (inner == BInnerIters - 2) {
            copy_stage(
                (pipe + Stages - 1) % Stages, pipe,
                k_tiles_remaining >= Stages);
            ++pipe;
            wait_stage();
          }
          multiply(inner_c);
        });
        --k_tiles_remaining;
        if (k_tiles_remaining == 0) {
          break;
        }
      }
      #pragma unroll
      for (int i = 0; i < ASharedWriteIters; ++i) {
        a_pointer[i] += AGlobalOuter * Stages;
      }
      scale_tile_base += Stages;
    }
    marlin_ppu_detail::cp_async_wait<0>();
  }
};

}  // namespace cutlass::gemm::collective
