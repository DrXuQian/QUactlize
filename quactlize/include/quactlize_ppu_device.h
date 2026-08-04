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

#ifdef __cplusplus
}
#endif
