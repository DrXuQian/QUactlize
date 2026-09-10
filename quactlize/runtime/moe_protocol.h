#pragma once
#include "abi.h"
#include "../integrations/llama/indexed.h"

// Private module-to-dispatch protocol, never a tensor layout for callers.
// Version and the actual CuTe member offsets are checked across JIT images.
typedef struct {
  uint32_t version, size;
  uint32_t shape_size, stride_size, shape_offsets[3], stride_offset;
  int32_t m, n, k, experts, tile_m, splits, device, reserved;
  void *a, *output, *partials, *shapes, *outputs, *strides;
  int32_t *offsets, *rows;
  void *directory_header, *directory_entries;
  uint64_t directory_capacity;
  void *workspace;
  uint64_t workspace_bytes;
  qk_llama_indexed_v1 io;
} qk_moe_projection_v1;

typedef struct {
  // merged=1: gate describes [M,2N], up is absent. Otherwise separate N.
  uint32_t version, size, merged, reserved;
  qk_moe_projection_v1 gate, up, down;
  qk_llama_router_v1 router;
} qk_moe_plan_v1;

enum { QK_MOE_PREPARE=0, QK_MOE_PRODUCER=1, QK_MOE_ACTIVATE=2, QK_MOE_FINISH=3 };
#ifdef __cplusplus
extern "C" {
#endif
int quactlize_kpack_moe_projection_v1(void*, qk_moe_projection_v1*);
int quactlize_kpack_moe_stage_v1(void*,qk_moe_plan_v1 const*,int,void*);
#ifdef __cplusplus
}
#endif
