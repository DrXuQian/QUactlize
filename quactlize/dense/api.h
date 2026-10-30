#pragma once
#include "../packing/api.h"
#include <stddef.h>

// Complete K-pack dense consumer. The caller never selects a kernel, expands
// metadata, loads an internal DSO, or composes a reducer. All device pointers
// belong to the context device. The weight planes remain immutable/alive
// until the context and every queued/captured use have been retired.
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t version, size;
    int32_t qtype, n, k, m;
    int64_t input_stride, output_stride;
    quactlize_ppu_placed_arrangement_v2 arrangement;
    quactlize_ppu_kpack_sizes_v1 sizes;
} quactlize_ppu_dense_params_v1;

enum { QUACTLIZE_DENSE_FQ_TC=0, QUACTLIZE_DENSE_SF_TC=1,
       QUACTLIZE_DENSE_SIMT=2, QUACTLIZE_DENSE_FULL=3,
       QUACTLIZE_DENSE_FULL_BF16=QUACTLIZE_DENSE_FULL }; // legacy name/ABI value
enum { QUACTLIZE_DENSE_F16=0, QUACTLIZE_DENSE_BF16=1 };
typedef struct {
    uint32_t version, size;
    uint64_t workspace_bytes;
    int32_t route, policy, split, compute_type;
    char config[192];
} quactlize_ppu_dense_plan_info_v1;

// Optional precision policy, fixed for a context, never inferred from a
// pointer. F16 preserves the PoC arithmetic/selection: values at each F16
// compute boundary must be representable. BF16 avoids the FP16 range limit;
// it is NOT numerically identical or automatically as fast as F16. F32 input
// and output storage and FP32 accumulation are used by both. No int8
// activation quantization and no clipping. Reviewed F16 dense prefill uses
// Full-F16 when selected; older/out-of-scope Full routes retain BF16. Route 3
// denotes the Full provider; plan_info.compute_type reports actual precision.
typedef struct {
    uint32_t version, size;
    int32_t compute_type, reserved;
} quactlize_ppu_dense_options_v1;

// Pure host check, no device/runtime/JIT access. 1=supported, 0=unsupported,
// -1=invalid ABI. Q8_0,Q2_K..Q6_K, N%256==0, K aligned to the format.
// Positive M must satisfy M*N<=INT32_MAX and checked byte extents. Strides
// are F32 elements, at least K/N. M=0 is weight admission: every positive M
// in this integer domain has a route (not merely a handful of sampled M).
// Resource availability and installed SDK/JIT/provider failures remain
// errors, not permission to reinterpret packed bytes or change a winner.
int quactlize_ppu_dense_supported_v1(quactlize_ppu_dense_params_v1 const*);

// Outside graph capture; current device must equal device. v1 retains the
// PoC's F16 dense compute. Wider-range callers must explicitly choose BF16
// with create_with_options_v1; finite F32 values need not fit FP16.
// Internal resources are resolved relative to libncp_quactlize.so. Optional
// environment: PPU_SDK, QUACTLIZE_PPU_CACHE_DIR, QUACTLIZE_PPU_PYTHON.
// PPU1.0 ZW610 uses ZW810 exact-table latencies/configs and bucket rules as
// selection estimates (not ZW610 measurements), with its actual resources.
int quactlize_ppu_dense_create_v1(int device,void** context);
int quactlize_ppu_dense_create_with_options_v1(int device,
    quactlize_ppu_dense_options_v1 const*,void** context);

// Outside capture. Context owns immutable plans and their modules. JIT is
// selected-parent compilation, never online tuning. Output info starts with
// {1,sizeof(info)}. Returns 0 on success; *plan stays NULL on failure.
int quactlize_ppu_dense_prepare_v1(void* context,quactlize_ppu_dense_params_v1 const*,
    void const* low,void const* high,void const* units,void** plan,
    quactlize_ppu_dense_plan_info_v1* info);

// Explicit initialization candidates; default prepare_v1 is unchanged. This
// is not an instruction to change production routing or a measured-win claim.
// kind: 0=TC, 1=generic SIMT, 2=existing selected Q4 reader, 3=full BF16.
// route: 0=FQ, 1=SF, 3=full (same public route values as above).
// TC fields describe the exact compiled parent and runtime recipe; SIMT uses
// variant/columns/warps/values/split. Unused fields must be zero, except
// dequant_config=-1 means the existing default SF prepass. native_f16 retains
// the original adapters/rounding; it is forbidden for BF16 contexts. Q4 uses
// its already selected reader (not an unchecked arbitrary specialization).
typedef struct {
    uint32_t version,size;
    int32_t kind,route,native_f16,legacy_adapters,dequant_config;
    int32_t tm,tn,tk,wm,wn,stages,ap,delivery_n,split,grid_mode,grid_b;
    int32_t variant,columns,warps,values;
    char parent[192];
} quactlize_ppu_dense_candidate_v1;
int quactlize_ppu_dense_prepare_candidate_v1(void* context,quactlize_ppu_dense_params_v1 const*,
    void const* low,void const* high,void const* units,quactlize_ppu_dense_candidate_v1 const*,
    void** plan,quactlize_ppu_dense_plan_info_v1* info);

// Enqueue only. Each invocation accepts new 16-byte-aligned A/C pointers,
// stream and private 256-byte-aligned workspace. Disjoint output/workspace
// is required for concurrent streams. Includes per-call SF/full expansion,
// GEMM and its actual reducer. Matched compact decode has no gather/scatter.
// Native F16 prefill and legacy fallback retain their original adapters;
// changing that contract is a separate numerical/performance admission. Strided decode
// requests may use F32 compact copies, included in workspace and timing.
// No allocation, JIT, D2H, host/device wait or pointer rebinding of a shared
// GEMM handle in run. Eager warmup is required before capture/timing for the
// installed cuBLAS provider. Captured graphs retain captured addresses.
int quactlize_ppu_dense_run_v1(void* plan,float const* input,float* output,
    void* workspace,uint64_t workspace_bytes,void* stream);
void quactlize_ppu_dense_destroy_v1(void* context);
char const* quactlize_ppu_dense_error_v1(void);

#ifdef __cplusplus
}
static_assert(sizeof(quactlize_ppu_dense_params_v1)==112,"dense ABI drift");
static_assert(offsetof(quactlize_ppu_dense_params_v1,arrangement)==40,"dense ABI drift");
static_assert(offsetof(quactlize_ppu_dense_params_v1,sizes)==80,"dense ABI drift");
static_assert(sizeof(quactlize_ppu_dense_plan_info_v1)==224,"dense ABI drift");
static_assert(offsetof(quactlize_ppu_dense_plan_info_v1,config)==32,"dense ABI drift");
static_assert(sizeof(quactlize_ppu_dense_candidate_v1)==280,"candidate ABI drift");
static_assert(offsetof(quactlize_ppu_dense_candidate_v1,parent)==88,"candidate ABI drift");
#endif
