#pragma once
#include "api.h"
#include "simt.h"
#include "../runtime/moe_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif
// Host-only descriptor binding. Scratch includes private F32 intermediates
// and a route directory; no weight data or device IDs are read here.
int quactlize_kpack_moe_simt_query_v1(qkg_call_v1 const*,uint64_t*);
int quactlize_kpack_moe_simt_bind_v1(qkg_call_v1 const*,int device,void*,uint64_t,qk_moe_projection_v1*);
// Shared preparation and SwiGLU only. Bits 0/1/2 name SIMT gate/up/down.
// TC producers and final TC reduction retain their existing module entries.
int quactlize_kpack_moe_mixed_stage_v1(qk_moe_plan_v1 const*,uint32_t,int,void*);
int quactlize_kpack_moe_weighted_finish_v1(qk_moe_plan_v1 const*,uint32_t,
    qk_llama_moe_finish_v1 const*,void*);

// All TC endpoints in this chain must use the named compute type. BF16
// projection values are rounded into BF16 before SwiGLU; SwiGLU arithmetic
// stays F32 and writes BF16 for TC down, or F32 for a BF16-rounding SIMT
// reader. Weighted finish returns F32 in slot order. No clipping is used.
typedef struct {
    uint32_t version,size;
    qk_moe_plan_v1 plan;
    uint32_t simt_mask;
    int32_t compute_type;
} qkg_moe_compute_v2;
int quactlize_kpack_moe_mixed_stage_v2(qkg_moe_compute_v2 const*,int,void*);
int quactlize_kpack_moe_weighted_finish_v2(qkg_moe_compute_v2 const*,
    qk_llama_moe_finish_v1 const*,void*);
#ifdef __cplusplus
}
#endif
