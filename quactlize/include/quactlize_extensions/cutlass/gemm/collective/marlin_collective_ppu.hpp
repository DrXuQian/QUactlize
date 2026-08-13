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
#include <type_traits>

#include <cuda_fp16.h>

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"

namespace cutlass::gemm::collective {

struct MarlinCpAsyncLoadPolicyPPU {};
struct MarlinAiuLoadPolicyPPU {};

namespace marlin_ppu_detail {

struct alignas(16) Vector128 {
  uint32_t x, y, z, w;
};

struct FragmentA {
  __half2 value[4];
};

struct FragmentB {
  __half2 value[2];
};

struct FragmentScale {
  __half2 value[1];
};

template <int Lut>
CUTLASS_DEVICE int lop3(int a, int b, int c) {
  int result;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(result)
               : "r"(a), "r"(b), "r"(c), "n"(Lut));
  return result;
}

CUTLASS_DEVICE FragmentB dequantize_biased_int4(int q) {
  constexpr int kLo = 0x000f000f;
  constexpr int kHi = 0x00f000f0;
  constexpr int kExponent = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, kLo, kExponent);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, kHi, kExponent);
  constexpr int kSubtract = 0x64086408;
  constexpr int kMultiply = 0x2c002c00;
  constexpr int kAdd = 0xd480d480;
  FragmentB result;
  result.value[0] = __hsub2(
      *reinterpret_cast<__half2*>(&lo),
      *reinterpret_cast<__half2 const*>(&kSubtract));
  result.value[1] = __hfma2(
      *reinterpret_cast<__half2*>(&hi),
      *reinterpret_cast<__half2 const*>(&kMultiply),
      *reinterpret_cast<__half2 const*>(&kAdd));
  return result;
}

CUTLASS_DEVICE void scale(FragmentB& fragment, FragmentScale const& scales, int half) {
  __half const scalar = reinterpret_cast<__half const*>(&scales)[half];
  __half2 const pair = __half2half2(scalar);
  fragment.value[0] = __hmul2(fragment.value[0], pair);
  fragment.value[1] = __hmul2(fragment.value[1], pair);
}

CUTLASS_DEVICE void cp_async_16(void* smem_ptr, void const* global_ptr) {
  uint32_t const smem = uint32_t(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem), "l"(global_ptr)
               : "memory");
}

CUTLASS_DEVICE void cp_async_16_if(
    void* smem_ptr, void const* global_ptr, bool predicate) {
  // PPU lowers cp.async to an intrinsic.  Predicating the asm instruction itself can force a
  // compatibility expansion, so the classic PPU port uses a uniform C++ branch around it.
  if (predicate) {
    cp_async_16(smem_ptr, global_ptr);
  }
}

CUTLASS_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int Count>
CUTLASS_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(Count) : "memory");
}

CUTLASS_DEVICE void ldmatrix_a(FragmentA& fragment, void const* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&fragment);
  // PPU x4 returns v1/v2 in the opposite register order from the A operand.  Bind those
  // outputs directly to a2/a1: no move or temporary is emitted.
  asm volatile(
      "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[2]), "=r"(a[1]), "=r"(a[3])
      : "l"(smem_ptr));
}

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
    int group_size = 0;
  };

  // Classic computes these lane-invariant and CTA-invariant terms once, then
  // only rebases the three global sources when its stripe crosses an output
  // tile.  Keep that lifetime explicit: CtaState survives the cooperative,
  // while SegmentState is deliberately short-lived and must die before the
  // CTA reduction.  Device numRegs is a post-build observation, not something
  // this source-level separation pretends to prove.
  struct CtaState {
    marlin_ppu_detail::Vector128 const* ptr_A = nullptr;
    marlin_ppu_detail::Vector128 const* ptr_B = nullptr;
    marlin_ppu_detail::Vector128 const* ptr_S = nullptr;
    int problem_n = 0;
    int problem_k = 0;
    int tid = 0;
    int a_global_stride = 0;
    int a_global_inner = 0;
    int b_global_stride = 0;
    int b_global_outer = 0;
    int b_global_inner = 0;
    int scale_global_stride = 0;
    bool a_predicate[ASharedWriteIters]{};
    int a_write_transformed[ASharedWriteIters]{};
    int a_read_transformed[BInnerIters]{};
    bool valid = false;
  };

  struct SegmentState {
    int n_tile = 0;
    int k_tile_begin = 0;
    int k_tiles_remaining = 0;
    int a_global_read = 0;
    int b_global_read[BInnerIters]{};
    int scale_global_read = 0;
    bool valid = false;
  };

  static_assert(std::is_standard_layout_v<CtaState> &&
                    std::is_trivially_copyable_v<CtaState> &&
                    std::is_standard_layout_v<SegmentState> &&
                    std::is_trivially_copyable_v<SegmentState>,
                "standalone Marlin address state must stay register-local");

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
    state.ptr_A = reinterpret_cast<marlin_ppu_detail::Vector128 const*>(
        params.ptr_A);
    state.ptr_B = reinterpret_cast<marlin_ppu_detail::Vector128 const*>(
        params.ptr_B);
    state.ptr_S = reinterpret_cast<marlin_ppu_detail::Vector128 const*>(
        params.ptr_S);
    state.problem_n = problem_n;
    state.problem_k = problem_k;
    state.tid = tid;
    state.a_global_stride = problem_k / 8;
    state.a_global_inner =
        state.a_global_stride * (Threads / AGlobalOuter);
    state.b_global_stride = 16 * problem_n / 32;
    state.b_global_outer = state.b_global_stride * KBlocks;
    state.b_global_inner =
        state.b_global_stride * (Threads / BSharedStride);
    state.scale_global_stride = problem_n / 8;
    int const a_shared_write =
        ASharedStride * (tid / AGlobalOuter) + tid % AGlobalOuter;
    int const a_shared_read =
        ASharedStride * ((tid % 32) % 16) + (tid % 32) / 16 +
        2 * ((tid / 32) / (NBlocks / 4));
    #pragma unroll
    for (int i = 0; i < ASharedWriteIters; ++i) {
      int const logical = ASharedWriteDelta * i + a_shared_write;
      state.a_predicate[i] = logical < ASharedStride * problem_m;
      state.a_write_transformed[i] = transform_a_index(logical);
    }
    #pragma unroll
    for (int i = 0; i < BInnerIters; ++i) {
      state.a_read_transformed[i] =
          transform_a_index(ASharedReadOuter * i + a_shared_read);
    }
    state.valid = params.ptr_A != nullptr && params.ptr_B != nullptr &&
                  params.ptr_S != nullptr && params.group_size == GroupSize &&
                  problem_m > 0 && problem_m <= TileM && problem_n > 0 &&
                  problem_n % 256 == 0 && problem_k > 0 &&
                  problem_k % TileK == 0 && tid >= 0 && tid < Threads;
    return state;
  }

  template <class WorkTile>
  CUTLASS_HOST_DEVICE static SegmentState rebase_segment(
      CtaState const& state, WorkTile const& work) {
    SegmentState segment;
    if (!state.valid || !work.is_valid()) {
      return segment;
    }
    int const n_tile = int(work.N_idx);
    int const k_tile_begin = int(work.K_idx);
    int const k_tile_count = int(work.k_tile_count);
    if (int(work.M_idx) != 0 || int(work.L_idx) != 0 || n_tile < 0 ||
        n_tile >= state.problem_n / TileN || k_tile_begin < 0 ||
        k_tile_count <= 0 ||
        k_tile_begin + k_tile_count > state.problem_k / TileK) {
      return segment;
    }

    segment.n_tile = n_tile;
    segment.k_tile_begin = k_tile_begin;
    segment.k_tiles_remaining = k_tile_count;
    segment.a_global_read =
        state.a_global_stride * (state.tid / AGlobalOuter) +
        state.tid % AGlobalOuter + AGlobalOuter * k_tile_begin;
    int const b_global_read =
        state.b_global_stride * (state.tid / BSharedStride) +
        state.tid % BSharedStride + BSharedStride * n_tile +
        state.b_global_outer * k_tile_begin;
    #pragma unroll
    for (int i = 0; i < BInnerIters; ++i) {
      segment.b_global_read[i] = state.b_global_inner * i + b_global_read;
    }
    segment.scale_global_read =
        state.scale_global_stride * k_tile_begin +
        ScaleSharedStride * n_tile + state.tid;
    segment.valid = true;
    return segment;
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
           n > 0 && n % 256 == 0 && k > 0 && k % TileK == 0;
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return {args.ptr_A, args.ptr_B, args.ptr_S, args.group_size};
  }

  template <class Accumulator>
  CUTLASS_DEVICE static void run_segment(
      CtaState const& state, SegmentState const& segment,
      Accumulator& accum, SharedStorage& shared) {
    using marlin_ppu_detail::FragmentA;
    using marlin_ppu_detail::FragmentB;
    using marlin_ppu_detail::FragmentScale;
    using marlin_ppu_detail::Vector128;

    int const tid = state.tid;
    int k_tiles_remaining = segment.k_tiles_remaining;

    auto const* a = state.ptr_A;
    auto const* b = state.ptr_B;
    auto const* scales = state.ptr_S;
    Vector128* const smem = shared.storage;
    Vector128* const smem_a = smem;
    Vector128* const smem_b = smem_a + Stages * ASharedStage;
    Vector128* const smem_scale = smem_b + Stages * BSharedStage;

    int a_global_read = segment.a_global_read;
    Vector128 const* b_pointer[BInnerIters];
    #pragma unroll
    for (int i = 0; i < BInnerIters; ++i) {
      b_pointer[i] = b + segment.b_global_read[i];
    }
    int scale_global_read = segment.scale_global_read;

    FragmentA fragment_a[2];
    marlin_ppu_detail::Vector128 fragment_b_quant[2];
    FragmentScale fragment_scale[2][4];

    auto copy_stage = [&](int pipe, int a_offset, bool predicate) {
      if (predicate) {
        Vector128* a_stage = smem_a + ASharedStage * pipe;
        #pragma unroll
        for (int i = 0; i < ASharedWriteIters; ++i) {
          marlin_ppu_detail::cp_async_16_if(
              &a_stage[state.a_write_transformed[i]],
              &a[state.a_global_inner * i + a_global_read +
                 AGlobalOuter * a_offset],
              state.a_predicate[i]);
        }
        Vector128* b_stage = smem_b + BSharedStage * pipe;
        #pragma unroll
        for (int i = 0; i < BInnerIters; ++i) {
          marlin_ppu_detail::cp_async_16(
              &b_stage[Threads * i + tid], b_pointer[i]);
          b_pointer[i] += state.b_global_outer;
        }
        // TileK == GroupSize in the admitted target, hence every pipeline stage begins a group.
        if (tid < ScaleSharedStride) {
          marlin_ppu_detail::cp_async_16(
              &smem_scale[ScaleSharedStage * pipe + tid],
              &scales[scale_global_read]);
        }
        scale_global_read += state.scale_global_stride;
      }
      marlin_ppu_detail::cp_async_commit();
    };

    auto wait_stage = [&]() {
      marlin_ppu_detail::cp_async_wait<Stages - 2>();
      __syncthreads();
    };

    auto load_registers = [&](int inner, int pipe) {
      Vector128 const* scale_stage = smem_scale + ScaleSharedStage * pipe;
      int const warp_n = (tid / 32) % (NBlocks / 4);
      int const scale_read = 8 * warp_n + (tid % 32) / 4;
      *reinterpret_cast<Vector128*>(&fragment_scale[inner & 1]) =
          scale_stage[scale_read];

      Vector128 const* a_stage = smem_a + ASharedStage * pipe;
      marlin_ppu_detail::ldmatrix_a(
          fragment_a[inner & 1],
          &a_stage[state.a_read_transformed[inner % BInnerIters]]);

      Vector128 const* b_stage = smem_b + BSharedStage * pipe;
      fragment_b_quant[inner & 1] =
          b_stage[Threads * (inner % BInnerIters) + tid];
    };

    auto multiply = [&](int inner) {
      uint32_t const* quant =
          reinterpret_cast<uint32_t const*>(&fragment_b_quant[inner & 1]);
      #pragma unroll
      for (int n_block = 0; n_block < 4; ++n_block) {
        int const q = int(quant[n_block]);
        FragmentB b0 = marlin_ppu_detail::dequantize_biased_int4(q);
        FragmentB b1 = marlin_ppu_detail::dequantize_biased_int4(q >> 8);
        marlin_ppu_detail::scale(b0, fragment_scale[inner & 1][n_block], 0);
        marlin_ppu_detail::scale(b1, fragment_scale[inner & 1][n_block], 1);
        if (n_block == 0) {
          marlin_ppu_detail::mma_n16<0>(fragment_a[inner & 1], b0, b1, accum);
        }
        else if (n_block == 1) {
          marlin_ppu_detail::mma_n16<1>(fragment_a[inner & 1], b0, b1, accum);
        }
        else if (n_block == 2) {
          marlin_ppu_detail::mma_n16<2>(fragment_a[inner & 1], b0, b1, accum);
        }
        else {
          marlin_ppu_detail::mma_n16<3>(fragment_a[inner & 1], b0, b1, accum);
        }
      }
    };

    #pragma unroll
    for (int i = 0; i < Stages - 1; ++i) {
      copy_stage(i, i, i < k_tiles_remaining);
    }
    wait_stage();
    load_registers(0, 0);
    cute::clear(accum);
    a_global_read += AGlobalOuter * (Stages - 1);

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
      a_global_read += AGlobalOuter * Stages;
    }
    marlin_ppu_detail::cp_async_wait<0>();
  }
};

}  // namespace cutlass::gemm::collective
