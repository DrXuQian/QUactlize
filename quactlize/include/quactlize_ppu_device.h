#pragma once
// Asynchronous device-pointer consumers exported by libquactlize_ppu.so.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// All data pointers name device memory. stream is the caller's native hggc/CUDA stream handle cast to void*;
// nullptr selects the runtime default stream. A zero return means the kernel was enqueued, not completed.
// The caller retains every allocation and must enforce stream lifetime and dependency ordering.
int quactlize_ppu_vecdot_dense_dev_v1(uint8_t const* blocks, int64_t block_bytes,
                                      uint16_t const* x, float* out,
                                      int rows, int blocks_per_row, int qtype, void* stream);

// The placed low/high planes and packed units are the same artifact consumed by quactlize_ppu_bc_gemv.
// experts==0 selects dense and requires total_rows==1. Grouped offsets are cumulative int[experts+1].
int quactlize_ppu_bc_gemv_dev_v1(uint16_t const* x,
                                 uint8_t const* low, uint8_t const* high, uint8_t const* units,
                                 int const* offsets, float* out,
                                 int total_rows, int n, int k, int experts, int max_rows, int qtype,
                                 void* stream);

// Fully-quantized tensor-core GEMM uses caller-owned device workspace. The size queries return -1 when the
// dimensions or qtype do not match this format-selected library. A successful device entry only enqueues work.
int64_t quactlize_ppu_dense_fully_quantized_workspace_bytes_v1(
    int m, int n, int k, int qtype);
int quactlize_ppu_dense_fully_quantized_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream);

// Grouped activations/output are concatenated in expert order. offsets is a cumulative device int[experts+1]
// with offsets[0]=0 and offsets[experts]=total_rows; max_rows is an upper bound on every expert row count.
int64_t quactlize_ppu_grouped_fully_quantized_workspace_bytes_v1(
    int max_rows, int n, int k, int experts, int qtype);
int quactlize_ppu_grouped_fully_quantized_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream);

#ifdef __cplusplus
}
#endif
