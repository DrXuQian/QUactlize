#pragma once
#include "api.h"
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
#ifdef __cplusplus
}
#endif
