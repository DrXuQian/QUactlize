/***************************************************************************************************
 * Copyright (c) 2026 quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

// Stable C ABI for PPU linear-attention device kernels.  This header is kept
// separate from the GEMM/GEMV ABI: chunked GDN has recurrent-state and gate
// semantics that must not be inferred from matrix dimensions.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1 1u

// Status values 1..7 intentionally mirror the header-only CUTLASS admission
// checker.  kRuntimeError is reserved for a launch/runtime failure after the
// problem has passed admission.
#define QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS 0
#define QUACTLIZE_PPU_CHUNKED_GDN_NULL_POINTER 1
#define QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM 2
#define QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_HEAD_DIMENSION 3
#define QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_CHUNK_SIZE 4
#define QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_HEAD_MAPPING 5
#define QUACTLIZE_PPU_CHUNKED_GDN_INVALID_SEQUENCE_LAYOUT 6
#define QUACTLIZE_PPU_CHUNKED_GDN_MISALIGNED_POINTER 7
#define QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR 8

typedef struct quactlize_ppu_chunked_gdn_problem_v1 {
  uint32_t schema_version;
  int32_t total_tokens;
  int32_t num_sequences;
  int32_t sequence_length;
  int32_t num_qk_heads;
  int32_t num_v_heads;
  int32_t head_size_k;
  int32_t head_size_v;
  int32_t chunk_size;
} quactlize_ppu_chunked_gdn_problem_v1;

// Fixed-length BF16 forward entry for the first PPU specialization.
//
// Device tensor layouts:
//   q/k       [total_tokens, num_qk_heads, head_size_k]
//   v/output  [total_tokens, num_v_heads,  head_size_v]
//   gamma/beta[total_tokens, num_v_heads]
//   state     [num_sequences, num_v_heads, head_size_k, head_size_v]
//
// q/k/v/output are raw BF16 payloads. q and k bases must be aligned to 16
// bytes, as required by the PPU AIU global-dot collective. output may alias v
// exactly (in-place V -> O), but otherwise output must not overlap q, k, v,
// gamma, beta, initial_state, or final_state. initial_state and final_state may
// alias exactly. Partially overlapping tensors are unsupported.
//
// gamma_log2_cumsum is the chunk-local
// cumulative log2 decay already produced by the GDN gate preprocessing step;
// it is not a raw gate. beta is post-sigmoid. initial_state may be null (zero
// initial state), and final_state may be null when the caller does not request
// it. The v1 entry is asynchronous and owns no global workspace.
// `problem` is a host-resident launch descriptor read synchronously by this
// function; it need only remain alive until the function returns. Device tensor
// storage and `stream` must obey the usual asynchronous-launch lifetime rules.
int quactlize_ppu_chunked_gdn_fwd_bf16_v1(
    uint16_t const* q,
    uint16_t const* k,
    uint16_t const* v,
    float const* gamma_log2_cumsum,
    float const* beta,
    float const* initial_state,
    uint16_t* output,
    float* final_state,
    quactlize_ppu_chunked_gdn_problem_v1 const* problem,
    float scale,
    void* stream);

#ifdef __cplusplus
}
#endif
