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

#ifdef __cplusplus
}
#endif
