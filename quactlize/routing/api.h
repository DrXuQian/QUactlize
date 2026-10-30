#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { QZR_OK=0, QZR_INVALID=20, QZR_OVERFLOW=26, QZR_CAPACITY=37, QZR_RUNTIME=41 };
enum { QZR_AUTO=0, QZR_WARP=1, QZR_TILED=2, QZR_SCALAR=3 };

// No weight dependency. IDs are caller-validated: 0 <= id < experts and
// distinct slots within each token (the same contract as top-k routing).
// All strides count int32 elements. Maps are stable in [token,slot] order
// within each expert. dst_rows[compact] = token*topk+slot; forward src_rows
// is token*source_stride+slot%channels. inverse=1 instead writes
// src_rows[token*topk+slot] = compact. offsets has experts+1 elements.
// Any token count >= 0 fitting the int32 output-index ABI is supported by
// TILED; no shared-memory allocation grows with total token count. Experts
// 1..1024, topk 1..32, channels 1..topk. Zero tokens writes zero offsets.
// Inputs/outputs/workspace must not overlap. Caller owns per-stream scratch.
typedef struct {
    uint32_t version, size;
    int32_t tokens, experts, topk, channels, inverse, reserved;
    int64_t ids_stride, source_stride;
    int32_t const* ids;
    int32_t *src_rows, *dst_rows, *offsets;
    void* workspace;
    uint64_t workspace_bytes;
    void* stream;
} qzr_call_v1;

typedef struct {
    uint32_t version, size;
    int32_t algorithm, tile_rows;
} qzr_config_v1;

typedef struct {
    uint64_t workspace_bytes;
    int32_t algorithm, tile_rows, tiles, stages;
} qzr_sizes_v1;

// Host-only: no data pointers are read. tile_rows=0 chooses 512; explicit
// TILED candidates are 256/512/1024. WARP uses at most 8192 tokens / 32 KiB
// of shared memory, retaining a single-kernel candidate across prefill M.
// AUTO uses WARP up to 1024 routed rows, otherwise TILED. This new entry is
// not wired into existing production selection; device timing admission is
// required before replacing its incumbent helper.
// SCALAR explicitly selects the register/ballot reader for <=512 routed
// rows, with no shared memory or workspace. AUTO keeps its original policy.
int quactlize_ppu_moe_route_query_v1(qzr_call_v1 const*,qzr_config_v1 const*,qzr_sizes_v1*);
// Enqueue only: no allocation, synchronization, readback or first-use JIT.
// Rebuilds maps each invocation, including replay with changed IDs.
int quactlize_ppu_moe_route_run_v1(qzr_call_v1 const*,qzr_config_v1 const*);

#ifdef __cplusplus
}
#endif
