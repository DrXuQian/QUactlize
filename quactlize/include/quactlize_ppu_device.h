#pragma once
// Asynchronous device-pointer consumers exported by libquactlize_ppu.so.

#include <stdint.h>

#include "quactlize_ppu_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// All data pointers name device memory. stream is the caller's native hggc/CUDA stream handle cast to void*;
// nullptr selects the runtime default stream. A zero return means the kernel was enqueued, not completed.
// The caller retains every allocation and must enforce stream lifetime and dependency ordering.
int quactlize_ppu_vecdot_dense_dev_v1(uint8_t const* blocks, int64_t block_bytes,
                                      uint16_t const* x, float* out,
                                      int rows, int blocks_per_row, int qtype, void* stream);

// Scale-first CUDA-core tactic over the same fp16 affine planes as dense_lowbit. This entry owns no workspace and
// only enqueues; config_name must come from the dense inventory's enable_cuda_kernel record (or be null for it).
int quactlize_ppu_gemv_lowbit_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, void* stream, char const* config_name);

// The placed low/high planes and packed units are the same artifact consumed by quactlize_ppu_bc_gemv.
// experts==0 selects dense and requires total_rows==1. Grouped offsets are cumulative int[experts+1].
int quactlize_ppu_bc_gemv_dev_v1(uint16_t const* x,
                                 uint8_t const* low, uint8_t const* high, uint8_t const* units,
                                 int const* offsets, float* out,
                                 int total_rows, int n, int k, int experts, int max_rows, int qtype,
                                 void* stream);
// Arrangement-aware successor.  The artifact supplies the descriptor; a null/unknown/mismatched descriptor is an
// error and never falls back to the default reader map.
int quactlize_ppu_bc_gemv_for_arrangement_dev_v1(
    uint16_t const* x, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, float* out, int total_rows, int n, int k, int experts, int max_rows, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement, void* stream);

// Fully-quantized tensor-core GEMM uses caller-owned device workspace. The size queries return -1 when the
// dimensions or qtype do not match this format-selected library. A successful device entry only enqueues work.
int64_t quactlize_ppu_dense_fully_quantized_workspace_bytes_v1(
    int m, int n, int k, int qtype);
int quactlize_ppu_dense_fully_quantized_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream);
// Config-selecting legacy reader-default successor. The v1 entry remains ABI-compatible and delegates with a
// null/default name; both keep the historical registry fully_quantized_tile_k tactic and artifact arrangement.
int quactlize_ppu_dense_fully_quantized_dev_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream,
    char const* config_name);
// Arrangement-aware successor.  Unlike v1/v2, this entry never infers the resident fold from qtype or tactic.
// A null/unknown/mismatched descriptor fails before launch; the old entries keep their registry-default behavior.
int quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream,
    char const* config_name, quactlize_ppu_placed_arrangement_v1 const* arrangement);

// Grouped activations/output are concatenated in expert order. offsets is a cumulative device int[experts+1]
// with offsets[0]=0 and offsets[experts]=total_rows; max_rows is an upper bound on every expert row count.
// grouped_lowbit consumes the scale-first artifact selected offline: low/high code planes plus fp16 scales and no
// zero plane (FinegrainedScaleOnly). Its workspace holds the same ptr/stride arrays and m-tile prefix as the packed
// grouped route; neither device entry allocates or synchronizes.
int64_t quactlize_ppu_grouped_lowbit_workspace_bytes_v1(
    int max_rows, int n, int k, int group_size, int experts, int qtype);
int quactlize_ppu_grouped_lowbit_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint16_t const* scale,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream);
int quactlize_ppu_grouped_lowbit_dev_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint16_t const* scale,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream, char const* config_name);

int64_t quactlize_ppu_grouped_fully_quantized_workspace_bytes_v1(
    int max_rows, int n, int k, int experts, int qtype);
int quactlize_ppu_grouped_fully_quantized_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream);
// Config-selecting successor. The v1 entry remains ABI-compatible and delegates with a null/default name.
int quactlize_ppu_grouped_fully_quantized_dev_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream, char const* config_name);

#ifdef __cplusplus
}
#endif
