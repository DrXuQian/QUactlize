#pragma once
// Host-pointer producer half of libquactlize_ppu's fully-quantized artifact ABI.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bytes written by quactlize_ppu_prepare_units for one dense tensor. Returns -1 for an invalid shape or qtype.
// Grouped output requires experts * this many bytes.
int64_t quactlize_ppu_units_bytes(int n, int k, int qtype);

// blocks are official GGUF [N,K/256,raw-byte] bytes; units are [K-unit,N,unit-byte].
int quactlize_ppu_prepare_units(uint8_t const* blocks, uint8_t* units, int n, int k, int qtype);

// blocks are [E,N,K/256,raw-byte]; units are [E,K-unit,N,unit-byte].
int quactlize_ppu_prepare_units_grouped(uint8_t const* blocks, uint8_t* units,
                                        int n, int k, int experts, int qtype);

// Complete host-only artifact seam used by online loaders. blocks/recovered are official GGUF
// [experts,N,K/256,raw-byte] records; low/high/units are exactly the resident artifact consumed by
// quactlize_ppu_bc_gemv_dev_v1. experts must be positive (use 1 for a dense tensor).
//
// The inverse is deliberately separate from dequantization: a loader can first require a byte-exact round trip of
// the shuffled representation, then pass the recovered official blocks to quactlize_ppu_dequantize. The `_v1`
// suffix prevents an older library with a different pointer contract from accepting either call.
int quactlize_ppu_prepare_fully_quantized_v1(uint8_t const* blocks,
                                             uint8_t* low, uint8_t* high, uint8_t* units,
                                             int n, int k, int experts, int qtype);
int quactlize_ppu_recover_fully_quantized_v1(uint8_t const* low, uint8_t const* high,
                                             uint8_t const* units, uint8_t* recovered,
                                             int n, int k, int experts, int qtype);

#ifdef __cplusplus
}
#endif
