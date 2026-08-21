/***************************************************************************************************
 * Copyright (c) 2026 quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fixed C64 / Dk128 / Dv128 / BV64 chunked-GDN forward collective.
 *
 * The CUDA branch is a deliberately scalar, independently runnable device
 * reference.  On ppu0010 the two products whose operands both come from
 * global memory (QK^T and KK^T) use actlize's proved BF16 AIU collective and
 * the m16n16k16 BF16/BF16->FP32 atom.  Matrices produced inside the CTA do not
 * have a proved register-to-swizzled-shared writer on ppu0010; those products
 * currently use the same scalar shared-memory implementation as the CUDA
 * reference.  This is a correctness v1, not an all-AIU performance claim.
 **************************************************************************************************/
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_aiu.cuh"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_types.hpp"

// actlize's CUTLASS_DEVICE intentionally follows hgcc's compilation macros.
// Keep the independently runnable CUDA reference a real device function when
// this header is parsed by NVIDIA nvcc instead.
#if defined(__CUDACC__) && !defined(__HGGCCC__)
#define QZ_PPU_GDN_DEVICE __device__ __forceinline__
#else
#define QZ_PPU_GDN_DEVICE CUTLASS_DEVICE
#endif

// NVIDIA nvcc can compile and execute the scalar device reference without
// parsing actlize's PPU collectives.  hgcc defines __HGGCCC__ in both passes;
// only that build owns the production AIU CollectiveBuilder specialization.
#if defined(__HGGCCC__)
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/ppu_mma_aiu_multistage.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "cutlass/layout/layout.h"
#endif

namespace cutlass::linear_attention {

namespace detail {

enum class PpuChunkedGdnGlobalDotKind : int {
  kCausalQk,
  kStrictLowerKk,
};

#if defined(__HGGCCC__)
using PpuChunkedGdnGlobalDotTile =
    cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>;
// The actlize builder uses its ClusterShape slot as the explicit warp tile.
using PpuChunkedGdnGlobalDotWarp =
    cute::Shape<cute::Int<32>, cute::Int<32>, cute::Int<64>>;

using PpuChunkedGdnGlobalDotMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010,
    cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    8,
    cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor,
    8,
    float,
    PpuChunkedGdnGlobalDotTile,
    PpuChunkedGdnGlobalDotWarp,
    cute::Int<2>,
    cutlass::gemm::KernelMultistage>::CollectiveOp;

static_assert(cute::size(typename PpuChunkedGdnGlobalDotMainloop::TiledMma{}) == 128,
              "C64 PPU global-dot collective must launch four warps");
static_assert(sizeof(typename PpuChunkedGdnGlobalDotMainloop::SharedStorage) == 32768,
              "C64xC64xK64 stage-2 BF16 AIU mainloop must use exactly 32 KiB");
#else
// Keeps the public type surface identical for the nvcc scalar-reference build.
struct PpuChunkedGdnGlobalDotMainloop {
  struct alignas(32) SharedStorage {
    std::uint8_t bytes[32768];
  };
};
#endif

}  // namespace detail

template <class Traits_, class Arguments_>
struct PpuChunkedGdnCollectiveBf16C64D128BV64 {
  using Traits = Traits_;
  using Arguments = Arguments_;
  using Params = Arguments;
  using Element = cutlass::bfloat16_t;
  using ElementOutput = typename Arguments::ElementOutput;
  using ElementState = typename Arguments::ElementState;
  using GlobalDotMainloop = detail::PpuChunkedGdnGlobalDotMainloop;

  static constexpr int kChunk = 64;
  static constexpr int kHeadK = 128;
  static constexpr int kHeadV = 128;
  static constexpr int kValueBlock = 64;
  static constexpr int kValueBlocks = kHeadV / kValueBlock;
  static constexpr int kThreadCount = 128;
  static constexpr int kGlobalDotAlignmentBytes = 16;
  static constexpr int kStateElements = kHeadK * kHeadV;
  static constexpr int kScoreElements = kChunk * kChunk;
  static constexpr int kWElements = kChunk * kHeadK;
  static constexpr int kValueTileElements = kChunk * kValueBlock;

  // Phase offsets are a checked liveness plan, not arbitrary scratch:
  //   [0,32K)  AIU mainloop OR {strict_lower,inverse}
  //   [0, 8K)  A_bf16 after the inverse no longer needs strict_lower
  //   [8,24K)  W, after inverse is no longer live
  //   [24,32K) U for the current V tile
  //   [32,48K) exp(G) Q H_start
  //   [48,56K) causal gated QK^T (computed before KK^T)
  //   [56,72K) FP32 v_new for the current V tile
  static constexpr int kPhaseBytes = 72 * 1024;
  static constexpr int kOffsetStrictLower = 0;
  static constexpr int kOffsetInverse = 16 * 1024;
  static constexpr int kOffsetA = 0;
  static constexpr int kOffsetW = 8 * 1024;
  static constexpr int kOffsetU = 24 * 1024;
  static constexpr int kOffsetOState = 32 * 1024;
  static constexpr int kOffsetP = 48 * 1024;
  static constexpr int kOffsetVNew = 56 * 1024;

  // The GDN is mathematically complete in this file.  The narrower performance
  // property is stated separately so callers/tests cannot misreport scalar
  // shared products as AIU products.
  static constexpr bool kAllStagesConnected = true;
  static constexpr bool kGlobalQkAndKkUseAiuOnPpu0010 = true;
  static constexpr bool kGeneratedOperandMmaConnected = false;
  static constexpr bool kAllMatrixProductsUseAiu = false;

  static_assert(Traits::ChunkSize == kChunk && Traits::HeadSizeK == kHeadK &&
                    Traits::HeadSizeV == kHeadV,
                "this collective is the fixed C64/D128 implementation");
  static_assert(std::is_same_v<typename Arguments::ElementQKV, Element> &&
                    std::is_same_v<ElementOutput, Element> &&
                    std::is_same_v<ElementState, float>,
                "v1 requires BF16 Q/K/V/O and FP32 recurrent state");

  struct alignas(32) SharedStorage {
    // Full state remains resident while chunks are the outer loop.  BV64 is a
    // compute tile, not a split ownership contract, so KK/QK/A/W are computed
    // only once per chunk.
    float state[kStateElements];
    float gamma[kChunk];
    float beta[kChunk];
    alignas(32) std::uint8_t phase[kPhaseBytes];
  };

  static constexpr int kSharedStorageBytes = int(sizeof(SharedStorage));
  static_assert(kSharedStorageBytes == 139776,
                "C64/D128/BV64 shared-memory liveness ledger changed");
  static_assert(kSharedStorageBytes <= 262144,
                "PPU0010 exposes at most 256 KiB shared storage per CTA");
  static_assert(sizeof(typename GlobalDotMainloop::SharedStorage) <= kPhaseBytes,
                "AIU global-dot mainloop must fit the phase arena");

  CUTLASS_HOST_DEVICE static constexpr PpuChunkedGdnStatus argument_status(
      Arguments const& args) {
    PpuChunkedGdnStatus const status = can_implement_ppu_chunked_gdn<Traits>(args);
    if (status != PpuChunkedGdnStatus::kSuccess) return status;
    if ((reinterpret_cast<std::uintptr_t>(args.q) % kGlobalDotAlignmentBytes) != 0 ||
        (reinterpret_cast<std::uintptr_t>(args.k) % kGlobalDotAlignmentBytes) != 0) {
      return PpuChunkedGdnStatus::kMisalignedPointer;
    }
    return PpuChunkedGdnStatus::kSuccess;
  }

 private:
  QZ_PPU_GDN_DEVICE static float* strict_lower(SharedStorage& s) {
    return reinterpret_cast<float*>(s.phase + kOffsetStrictLower);
  }
  QZ_PPU_GDN_DEVICE static float* inverse(SharedStorage& s) {
    return reinterpret_cast<float*>(s.phase + kOffsetInverse);
  }
  QZ_PPU_GDN_DEVICE static Element* inverse_bf16(SharedStorage& s) {
    return reinterpret_cast<Element*>(s.phase + kOffsetA);
  }
  QZ_PPU_GDN_DEVICE static Element* w_matrix(SharedStorage& s) {
    return reinterpret_cast<Element*>(s.phase + kOffsetW);
  }
  QZ_PPU_GDN_DEVICE static Element* u_tile(SharedStorage& s) {
    return reinterpret_cast<Element*>(s.phase + kOffsetU);
  }
  QZ_PPU_GDN_DEVICE static float* o_state(SharedStorage& s) {
    return reinterpret_cast<float*>(s.phase + kOffsetOState);
  }
  QZ_PPU_GDN_DEVICE static Element* causal_score(SharedStorage& s) {
    return reinterpret_cast<Element*>(s.phase + kOffsetP);
  }
  QZ_PPU_GDN_DEVICE static float* v_new(SharedStorage& s) {
    return reinterpret_cast<float*>(s.phase + kOffsetVNew);
  }

  QZ_PPU_GDN_DEVICE static float exp2_gate(float x) {
    return ::exp2f(x);
  }

  // actlize's BF16 conversion members are hgcc device functions.  NVIDIA nvcc
  // still needs a runnable scalar reference, so provide the identical RNE bit
  // conversion locally instead of calling those host-only members.
  QZ_PPU_GDN_DEVICE static float to_float(Element value) {
#if defined(__CUDA_ARCH__) && !defined(__HGGCCC__)
    std::uint32_t const bits =
        std::uint32_t(*reinterpret_cast<std::uint16_t const*>(&value)) << 16;
    return __uint_as_float(bits);
#else
    return float(value);
#endif
  }

  QZ_PPU_GDN_DEVICE static Element to_bf16(float value) {
#if defined(__CUDA_ARCH__) && !defined(__HGGCCC__)
    std::uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
      bits += 0x7fffu + ((bits >> 16) & 1u);
    } else if ((bits & 0x007fffffu) != 0) {
      bits = 0x7fffffffu;
    }
    Element result{};
    *reinterpret_cast<std::uint16_t*>(&result) = std::uint16_t(bits >> 16);
    return result;
#else
    return Element(value);
#endif
  }

  QZ_PPU_GDN_DEVICE static std::int64_t qk_offset(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int local_token, int feature) {
    std::int64_t const token = std::int64_t(work.token_begin) + local_token;
    return (token * params.problem.num_qk_heads + work.qk_head_idx) * kHeadK + feature;
  }

  QZ_PPU_GDN_DEVICE static std::int64_t vo_offset(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int local_token, int value) {
    std::int64_t const token = std::int64_t(work.token_begin) + local_token;
    return (token * params.problem.num_v_heads + work.v_head_idx) * kHeadV + value;
  }

  QZ_PPU_GDN_DEVICE static std::int64_t state_offset(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int feature, int value) {
    return ((std::int64_t(work.sequence_idx) * params.problem.num_v_heads +
             work.v_head_idx) *
                kHeadK +
            feature) *
               kHeadV +
           value;
  }

  QZ_PPU_GDN_DEVICE static void load_state(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      SharedStorage& shared, int thread_idx) {
    for (int i = thread_idx; i < kStateElements; i += kThreadCount) {
      int const feature = i / kHeadV;
      int const value = i % kHeadV;
      shared.state[i] = params.initial_state == nullptr
                            ? 0.0f
                            : params.initial_state[state_offset(
                                  params, work, feature, value)];
    }
    __syncthreads();
  }

  QZ_PPU_GDN_DEVICE static void store_final_state(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      SharedStorage const& shared, int thread_idx) {
    if (params.final_state != nullptr) {
      for (int i = thread_idx; i < kStateElements; i += kThreadCount) {
        int const feature = i / kHeadV;
        int const value = i % kHeadV;
        params.final_state[state_offset(params, work, feature, value)] = shared.state[i];
      }
    }
  }

  QZ_PPU_GDN_DEVICE static int load_chunk_scalars(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_idx, SharedStorage& shared, int thread_idx) {
    int const chunk_begin = chunk_idx * kChunk;
    int const valid = work.token_count - chunk_begin < kChunk
                          ? work.token_count - chunk_begin
                          : kChunk;
    for (int i = thread_idx; i < kChunk; i += kThreadCount) {
      if (i < valid) {
        std::int64_t const token = std::int64_t(work.token_begin) + chunk_begin + i;
        std::int64_t const gh = token * params.problem.num_v_heads + work.v_head_idx;
        shared.gamma[i] = params.gamma_log2_cumsum[gh];
        shared.beta[i] = params.beta[gh];
      } else {
        shared.gamma[i] = 0.0f;
        shared.beta[i] = 0.0f;
      }
    }
    __syncthreads();
    return valid;
  }

  QZ_PPU_GDN_DEVICE static void scalar_global_dot(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, detail::PpuChunkedGdnGlobalDotKind kind,
      SharedStorage& shared, int thread_idx) {
    float* const l = strict_lower(shared);
    Element* const p = causal_score(shared);
    for (int index = thread_idx; index < kScoreElements; index += kThreadCount) {
      int const row = index / kChunk;
      int const col = index % kChunk;
      float dot = 0.0f;
      if (row < valid && col < valid) {
        Element const* lhs = kind == detail::PpuChunkedGdnGlobalDotKind::kCausalQk
                                 ? params.q
                                 : params.k;
        for (int d = 0; d < kHeadK; ++d) {
          std::int64_t const lhs_i = qk_offset(params, work, chunk_begin + row, d);
          std::int64_t const rhs_i = qk_offset(params, work, chunk_begin + col, d);
          dot += to_float(lhs[lhs_i]) * to_float(params.k[rhs_i]);
        }
      }
      if (kind == detail::PpuChunkedGdnGlobalDotKind::kCausalQk) {
        float const value = row < valid && col < valid && row >= col
                                ? dot * exp2_gate(shared.gamma[row] - shared.gamma[col])
                                : 0.0f;
        p[index] = to_bf16(value);
      } else {
        l[index] = row < valid && col < valid && row > col
                       ? shared.beta[row] * dot *
                             exp2_gate(shared.gamma[row] - shared.gamma[col])
                       : 0.0f;
      }
    }
    __syncthreads();
  }

#if defined(__HGGCCC__)
  QZ_PPU_GDN_DEVICE static void ppu0010_global_dot(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, detail::PpuChunkedGdnGlobalDotKind kind,
      SharedStorage& shared, int thread_idx) {
    using namespace cute;
    using X = Underscore;
    using Mainloop = GlobalDotMainloop;
    using Tile = detail::PpuChunkedGdnGlobalDotTile;
    using StrideA = typename Mainloop::StrideA;
    using StrideB = typename Mainloop::StrideB;

    Element const* const lhs =
        (kind == detail::PpuChunkedGdnGlobalDotKind::kCausalQk ? params.q : params.k) +
        qk_offset(params, work, chunk_begin, 0);
    Element const* const rhs = params.k + qk_offset(params, work, chunk_begin, 0);
    std::int64_t const row_pitch = std::int64_t(params.problem.num_qk_heads) * kHeadK;
    StrideA const dA{row_pitch, Int<1>{}, Int<0>{}};
    StrideB const dB{row_pitch, Int<1>{}, Int<0>{}};
    auto const problem_shape =
        make_shape(valid, valid, Int<kHeadK>{}, Int<1>{});
    typename Mainloop::Arguments mainloop_args{lhs, dA, rhs, dB};
    auto const mainloop_params = Mainloop::to_underlying_arguments(
        problem_shape, mainloop_args, nullptr);

    // make_mix_tensor_like is required: the collective constructor initializes
    // its descriptor with a null gmem pointer, and the mix iterator is what
    // carries the real per-work-tile pointer into Copy_Traits.
    Tensor mA_mkl = make_tensor(
        make_gmem_ptr(mainloop_params.ptr_A),
        make_shape(valid, Int<kHeadK>{}, Int<1>{}),
        mainloop_params.dA);
    Tensor mB_nkl = make_tensor(
        make_gmem_ptr(mainloop_params.ptr_B),
        make_shape(valid, Int<kHeadK>{}, Int<1>{}),
        mainloop_params.dB);
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_, _, 0));
    Tensor mB_nk = make_mix_tensor_like(mB_nkl(_, _, 0));
    auto const block_coord = make_coord(0, 0, _, 0);
    Tensor gA = local_tile(
        mA_mk, Tile{}, take<0, 3>(block_coord), Step<_1, X, _1>{});
    Tensor gB = local_tile(
        mB_nk, Tile{}, take<0, 3>(block_coord), Step<X, _1, _1>{});

    typename Mainloop::TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, make_shape(Int<64>{}, Int<64>{}));
    clear(accum);
    auto k_tile_iter = make_coord_iterator(shape<2>(gA));
    int k_tile_count = size<2>(gA);
    auto const residue = make_tuple(
        valid, valid, Int<kHeadK>{} - size<1>(gA) * size<2>(gA));
    Mainloop mainloop(mainloop_params, problem_shape);
    mainloop(
        accum, gA, gB, accum, k_tile_iter, k_tile_count, residue, thread_idx,
        reinterpret_cast<char*>(shared.phase));

    auto identity = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    auto coord = thread_mma.partition_C(identity);
    float* const l = strict_lower(shared);
    Element* const p = causal_score(shared);
    for (int i = 0; i < int(size(accum)); ++i) {
      int const row = int(get<0>(coord(i)));
      int const col = int(get<1>(coord(i)));
      float const dot = accum(i);
      if (kind == detail::PpuChunkedGdnGlobalDotKind::kCausalQk) {
        float const value = row < valid && col < valid && row >= col
                                ? dot * exp2_gate(shared.gamma[row] - shared.gamma[col])
                                : 0.0f;
        p[row * kChunk + col] = to_bf16(value);
      } else {
        l[row * kChunk + col] = row < valid && col < valid && row > col
                                    ? shared.beta[row] * dot *
                                          exp2_gate(shared.gamma[row] - shared.gamma[col])
                                    : 0.0f;
      }
    }
    __syncthreads();
  }
#endif

  QZ_PPU_GDN_DEVICE static void global_dot(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, detail::PpuChunkedGdnGlobalDotKind kind,
      SharedStorage& shared, int thread_idx) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    ppu0010_global_dot(params, work, chunk_begin, valid, kind, shared, thread_idx);
#elif defined(__HGGC_ARCH__)
    (void)params;
    (void)work;
    (void)chunk_begin;
    (void)valid;
    (void)kind;
    (void)shared;
    (void)thread_idx;
    CUTE_INVALID_CONTROL_PATH("PPU chunked GDN supports only ppu0010");
#else
    scalar_global_dot(params, work, chunk_begin, valid, kind, shared, thread_idx);
#endif
  }

  QZ_PPU_GDN_DEVICE static void solve_inverse(
      SharedStorage& shared, int thread_idx) {
    float* const l = strict_lower(shared);
    float* const a = inverse(shared);
    // One thread per RHS column; rows are a true dependency chain.
    for (int row = 0; row < kChunk; ++row) {
      if (thread_idx < kChunk) {
        int const col = thread_idx;
        float value = row == col ? 1.0f : 0.0f;
        for (int k = 0; k < row; ++k) {
          value -= l[row * kChunk + k] * a[k * kChunk + col];
        }
        a[row * kChunk + col] = value;
      }
      __syncthreads();
    }
    Element* const ab = inverse_bf16(shared);
    for (int i = thread_idx; i < kScoreElements; i += kThreadCount) {
      ab[i] = to_bf16(a[i]);
    }
    __syncthreads();
  }

  QZ_PPU_GDN_DEVICE static void compute_w(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, SharedStorage& shared, int thread_idx) {
    Element const* const a = inverse_bf16(shared);
    Element* const w = w_matrix(shared);
    for (int index = thread_idx; index < kWElements; index += kThreadCount) {
      int const row = index / kHeadK;
      int const d = index % kHeadK;
      float sum = 0.0f;
      for (int j = 0; j < kChunk; ++j) {
        // Upstream forms the scaled K operand in the input dtype before the
        // A@K tensor product.  Keep that BF16 boundary explicit.
        Element const kb = to_bf16(
            j < valid
                ? shared.beta[j] * exp2_gate(shared.gamma[j]) *
                      to_float(params.k[qk_offset(
                          params, work, chunk_begin + j, d)])
                : 0.0f);
        sum += to_float(a[row * kChunk + j]) * to_float(kb);
      }
      w[index] = to_bf16(sum);
    }
    __syncthreads();
  }

  QZ_PPU_GDN_DEVICE static void compute_value_tile(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, int value_base,
      SharedStorage& shared, int thread_idx) {
    Element const* const a = inverse_bf16(shared);
    Element const* const w = w_matrix(shared);
    Element* const u = u_tile(shared);
    float* const os = o_state(shared);
    Element const* const p = causal_score(shared);
    float* const vn = v_new(shared);

    for (int index = thread_idx; index < kValueTileElements; index += kThreadCount) {
      int const row = index / kValueBlock;
      int const v = index % kValueBlock;
      float sum = 0.0f;
      for (int j = 0; j < kChunk; ++j) {
        // beta*V is likewise rounded to V's dtype before A@V.
        Element const vb = to_bf16(
            j < valid
                ? shared.beta[j] * to_float(params.v[vo_offset(
                      params, work, chunk_begin + j, value_base + v)])
                : 0.0f);
        sum += to_float(a[row * kChunk + j]) * to_float(vb);
      }
      u[index] = to_bf16(sum);
    }
    __syncthreads();

    for (int index = thread_idx; index < kValueTileElements; index += kThreadCount) {
      int const row = index / kValueBlock;
      int const v = index % kValueBlock;
      float qh = 0.0f;
      float wh = 0.0f;
      if (row < valid) {
        for (int d = 0; d < kHeadK; ++d) {
          // The reference pipeline materializes the recurrent-state boundary
          // in BF16 before QH/WH. Preserve that rounding point even though the
          // canonical in-CTA state remains FP32 across chunks.
          float const h = to_float(to_bf16(shared.state[d * kHeadV + value_base + v]));
          qh += to_float(params.q[qk_offset(params, work, chunk_begin + row, d)]) * h;
          wh += to_float(w[row * kHeadK + d]) * h;
        }
      }
      os[index] = row < valid ? exp2_gate(shared.gamma[row]) * qh : 0.0f;
      vn[index] = row < valid ? to_float(u[index]) - wh : 0.0f;
    }
    __syncthreads();

    for (int index = thread_idx; index < kValueTileElements; index += kThreadCount) {
      int const row = index / kValueBlock;
      int const v = index % kValueBlock;
      if (row < valid) {
        float causal = 0.0f;
        for (int j = 0; j <= row; ++j) {
          causal += to_float(p[row * kChunk + j]) *
                    to_float(to_bf16(vn[j * kValueBlock + v]));
        }
        params.output[vo_offset(params, work, chunk_begin + row, value_base + v)] =
            to_bf16(params.scale * os[index] + params.scale * causal);
      }
    }
    __syncthreads();

    float const gamma_last = shared.gamma[valid - 1];
    float const state_decay = exp2_gate(gamma_last);
    for (int index = thread_idx; index < kHeadK * kValueBlock; index += kThreadCount) {
      int const d = index / kValueBlock;
      int const v = index % kValueBlock;
      float update = 0.0f;
      for (int row = 0; row < valid; ++row) {
        // Match the chunk recurrence's BF16 dot boundary: gate in FP32, then
        // round the scaled delta value once before K^T V_new.
        Element const scaled_v = to_bf16(
            exp2_gate(gamma_last - shared.gamma[row]) *
            vn[row * kValueBlock + v]);
        update += to_float(params.k[qk_offset(params, work, chunk_begin + row, d)]) *
                  to_float(scaled_v);
      }
      int const h_index = d * kHeadV + value_base + v;
      shared.state[h_index] = state_decay * shared.state[h_index] + update;
    }
    __syncthreads();
  }

 public:
  QZ_PPU_GDN_DEVICE static void run(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      SharedStorage& shared) {
    int const thread_idx = int(threadIdx.x);
    load_state(params, work, shared, thread_idx);
    for (int chunk = 0; chunk < work.chunk_count; ++chunk) {
      int const chunk_begin = chunk * kChunk;
      int const valid = load_chunk_scalars(params, work, chunk, shared, thread_idx);

      // QK runs first so its compact BF16 causal score can live above the
      // 32-KiB AIU arena while KK reuses that arena.
      global_dot(
          params, work, chunk_begin, valid,
          detail::PpuChunkedGdnGlobalDotKind::kCausalQk,
          shared, thread_idx);
      global_dot(
          params, work, chunk_begin, valid,
          detail::PpuChunkedGdnGlobalDotKind::kStrictLowerKk,
          shared, thread_idx);
      solve_inverse(shared, thread_idx);
      compute_w(params, work, chunk_begin, valid, shared, thread_idx);
      for (int value_tile = 0; value_tile < kValueBlocks; ++value_tile) {
        compute_value_tile(
            params, work, chunk_begin, valid, value_tile * kValueBlock,
            shared, thread_idx);
      }
    }
    store_final_state(params, work, shared, thread_idx);
  }
};

}  // namespace cutlass::linear_attention

#undef QZ_PPU_GDN_DEVICE
