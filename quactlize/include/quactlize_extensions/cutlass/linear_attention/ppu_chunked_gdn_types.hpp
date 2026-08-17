/***************************************************************************************************
 * Copyright (c) 2026 quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Pure CUDA C++ / CUTLASS-facing ABI for the PPU chunked Gated Delta Network
 * forward kernel.  The chunk algebra follows the GDN implementation in
 * flash-linear-attention.  The fused-kernel boundary and one-CTA recurrent
 * state ownership follow inclusionAI/cuLA's Apache-2.0 SM90 KDA C++ kernel;
 * no CuTeDSL or Triton source is used by this implementation.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HGGCCC__)
#  define QZ_GDN_HOST_DEVICE __host__ __device__
#elif defined(CUTLASS_HOST_DEVICE)
#  define QZ_GDN_HOST_DEVICE CUTLASS_HOST_DEVICE
#else
#  define QZ_GDN_HOST_DEVICE
#endif

namespace cutlass::linear_attention {

enum class PpuChunkedGdnStatus : std::int32_t {
  kSuccess = 0,
  kNullPointer = 1,
  kInvalidProblem = 2,
  kUnsupportedHeadDimension = 3,
  kUnsupportedChunkSize = 4,
  kUnsupportedHeadMapping = 5,
  kInvalidSequenceLayout = 6,
  kMisalignedPointer = 7,
};

// Public inputs use the standard packed-token convention:
//   q/k    [total_tokens, num_qk_heads, head_size_k]
//   v/o    [total_tokens, num_v_heads,  head_size_v]
//   gamma  [total_tokens, num_v_heads]
//   beta   [total_tokens, num_v_heads]
//   state  [num_sequences, num_v_heads, head_size_k, head_size_v]
//
// gamma_log2_cumsum is deliberately named for what it contains.  It is the
// chunk-local cumulative log2 decay, not the raw per-token gate.  Keeping raw
// gate activation/cumsum in a separate entry prevents the same silent ABI
// ambiguity that a generic `g` pointer would create.
struct PpuChunkedGdnProblem {
  std::int32_t total_tokens = 0;
  std::int32_t num_sequences = 0;
  std::int32_t sequence_length = 0;  // fixed-length v1; 0 when cu_seqlens owns it
  std::int32_t num_qk_heads = 0;
  std::int32_t num_v_heads = 0;
  std::int32_t head_size_k = 0;
  std::int32_t head_size_v = 0;
  std::int32_t chunk_size = 0;
};

template <class ElementQKV_, class ElementOutput_ = ElementQKV_, class ElementState_ = float>
struct PpuChunkedGdnArguments {
  using ElementQKV = ElementQKV_;
  using ElementOutput = ElementOutput_;
  using ElementState = ElementState_;

  ElementQKV const* q = nullptr;
  ElementQKV const* k = nullptr;
  ElementQKV const* v = nullptr;
  float const* gamma_log2_cumsum = nullptr;
  float const* beta = nullptr;
  ElementState const* initial_state = nullptr;  // optional: zero state when null
  ElementOutput* output = nullptr;
  ElementState* final_state = nullptr;          // optional
  std::int32_t const* cu_seqlens = nullptr;     // reserved for varlen successor
  PpuChunkedGdnProblem problem{};
  float scale = 1.0f;
};

struct PpuChunkedGdnWorkTileInfo {
  std::int32_t sequence_idx = 0;
  std::int32_t v_head_idx = 0;
  std::int32_t qk_head_idx = 0;
  std::int32_t token_begin = 0;
  std::int32_t token_count = 0;
  std::int32_t chunk_count = 0;
  bool valid = false;
};

template <int ChunkSize_, int HeadSizeK_, int HeadSizeV_>
struct PpuChunkedGdnTraits {
  static constexpr int ChunkSize = ChunkSize_;
  static constexpr int HeadSizeK = HeadSizeK_;
  static constexpr int HeadSizeV = HeadSizeV_;

  static_assert(ChunkSize == 16 || ChunkSize == 32 || ChunkSize == 64,
                "PPU GDN chunk size must be one of 16, 32, or 64");
  static_assert(HeadSizeK > 0 && HeadSizeK % 16 == 0,
                "PPU GDN K head dimension must be a positive multiple of 16");
  static_assert(HeadSizeV > 0 && HeadSizeV % 16 == 0,
                "PPU GDN V head dimension must be a positive multiple of 16");
};

template <class Traits>
struct PpuChunkedGdnScheduler {
  QZ_GDN_HOST_DEVICE static constexpr std::int32_t ceil_div(std::int32_t x, std::int32_t y) {
    return x / y + (x % y != 0);
  }

  QZ_GDN_HOST_DEVICE static constexpr std::int32_t grid_size(PpuChunkedGdnProblem const& p) {
    std::int64_t const grid =
        std::int64_t(p.num_sequences) * std::int64_t(p.num_v_heads);
    return grid > 0 && grid <= std::numeric_limits<std::int32_t>::max()
               ? std::int32_t(grid)
               : 0;
  }

  // A work tile owns the complete recurrence chain for one (sequence,V-head).
  // Chunks are intentionally serial inside that owner: there is no global
  // state handoff, counter, or lock in the first fully-fused implementation.
  QZ_GDN_HOST_DEVICE static constexpr PpuChunkedGdnWorkTileInfo
  work(std::int32_t linear_block, PpuChunkedGdnProblem const& p) {
    PpuChunkedGdnWorkTileInfo w{};
    std::int32_t const grid = grid_size(p);
    if (linear_block < 0 || grid <= 0 || linear_block >= grid || p.num_v_heads <= 0 ||
        p.num_qk_heads <= 0 || p.num_v_heads % p.num_qk_heads != 0 ||
        p.sequence_length <= 0) {
      return w;
    }
    std::int32_t const value_heads_per_qk_head = p.num_v_heads / p.num_qk_heads;
    w.sequence_idx = linear_block / p.num_v_heads;
    w.v_head_idx = linear_block % p.num_v_heads;
    w.qk_head_idx = w.v_head_idx / value_heads_per_qk_head;
    std::int64_t const token_begin =
        std::int64_t(w.sequence_idx) * std::int64_t(p.sequence_length);
    if (token_begin < 0 || token_begin > std::numeric_limits<std::int32_t>::max()) {
      return PpuChunkedGdnWorkTileInfo{};
    }
    w.token_begin = std::int32_t(token_begin);
    w.token_count = p.sequence_length;
    w.chunk_count = ceil_div(w.token_count, Traits::ChunkSize);
    w.valid = true;
    return w;
  }
};

template <class Traits, class Arguments>
QZ_GDN_HOST_DEVICE constexpr PpuChunkedGdnStatus
can_implement_ppu_chunked_gdn(Arguments const& args) {
  auto const& p = args.problem;
  if (args.q == nullptr || args.k == nullptr || args.v == nullptr ||
      args.gamma_log2_cumsum == nullptr || args.beta == nullptr || args.output == nullptr) {
    return PpuChunkedGdnStatus::kNullPointer;
  }
  std::int64_t const token_count =
      std::int64_t(p.num_sequences) * std::int64_t(p.sequence_length);
  std::int64_t const grid_count =
      std::int64_t(p.num_sequences) * std::int64_t(p.num_v_heads);
  if (p.total_tokens <= 0 || p.num_sequences <= 0 || p.sequence_length <= 0 ||
      token_count != std::int64_t(p.total_tokens) || p.num_qk_heads <= 0 ||
      p.num_v_heads <= 0) {
    return PpuChunkedGdnStatus::kInvalidProblem;
  }
  if (grid_count <= 0 || grid_count > std::numeric_limits<std::int32_t>::max()) {
    return PpuChunkedGdnStatus::kInvalidProblem;
  }
  std::int64_t const qk_row_elements =
      std::int64_t(p.num_qk_heads) * std::int64_t(Traits::HeadSizeK);
  std::int64_t const vo_row_elements =
      std::int64_t(p.num_v_heads) * std::int64_t(Traits::HeadSizeV);
  if (qk_row_elements <= 0 || vo_row_elements <= 0 ||
      std::int64_t(p.total_tokens) >
          std::numeric_limits<std::int64_t>::max() / qk_row_elements ||
      std::int64_t(p.total_tokens) >
          std::numeric_limits<std::int64_t>::max() / vo_row_elements) {
    return PpuChunkedGdnStatus::kInvalidProblem;
  }
  if (p.head_size_k != Traits::HeadSizeK || p.head_size_v != Traits::HeadSizeV) {
    return PpuChunkedGdnStatus::kUnsupportedHeadDimension;
  }
  if (p.chunk_size != Traits::ChunkSize) {
    return PpuChunkedGdnStatus::kUnsupportedChunkSize;
  }
  if (p.num_v_heads % p.num_qk_heads != 0) {
    return PpuChunkedGdnStatus::kUnsupportedHeadMapping;
  }
  // Varlen is represented in the ABI but deliberately fails closed until the
  // device scheduler reads cu_seqlens.  A non-null pointer must never be
  // silently ignored by the fixed-length kernel.
  if (args.cu_seqlens != nullptr) {
    return PpuChunkedGdnStatus::kInvalidSequenceLayout;
  }
  return PpuChunkedGdnStatus::kSuccess;
}

}  // namespace cutlass::linear_attention

#undef QZ_GDN_HOST_DEVICE
