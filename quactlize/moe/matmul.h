#pragma once
#include "../packing/api.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Complete independent MUL_MAT_ID projection, not fused FFN or weighted sum.
// Canonical weights [experts,N,K]; F32 input/output and FP32 accumulation,
// BF16 operands. IDs [tokens,topk] must be valid and distinct within a token.
// A[token*input_token_stride+(slot%channels)*input_row_stride+K],
// C[(token*topk+slot)*output_stride+N]. All strides are elements, not bytes.
typedef struct {
    uint32_t version,size;
    int32_t qtype,n,k,experts,tokens,topk,channels,reserved;
    int64_t input_row_stride,input_token_stride,ids_stride,output_stride;
    quactlize_ppu_placed_arrangement_v2 arrangement;
    quactlize_ppu_kpack_sizes_v1 sizes;
} quactlize_ppu_moe_matmul_params_v1;

enum { QUACTLIZE_MOE_FQ_TC=0, QUACTLIZE_MOE_SF_TC=1, QUACTLIZE_MOE_SIMT=2 };
typedef struct {
    uint32_t version,size;
    uint64_t workspace_bytes;
    int32_t route,policy,split,compute_type,source_tokens,prepare_algorithm,prepare_helper,reserved;
    char config[192];
} quactlize_ppu_moe_matmul_plan_info_v1;

// Host only: 1 supported, 0 unsupported, -1 invalid ABI. Q8_0,Q2_K..Q6_K;
// N%256=0, N<=65535*16, K%256=0 (Q3/Q6: K%512=0), 1..1024 experts, 1..32 topk.
// tokens=0 admits a weight independently of inference M. Every positive M
// with M*topk*max(N,K)<=INT32_MAX and the checked stride extents has a path.
// Input tokens must be an integral number of input_row_stride apart.
// Exact model tables, then same-family M buckets; unknown large-M families
// use an explicitly unmeasured SF proposal, NEVER a legacy FQ fallback.
int quactlize_ppu_moe_matmul_supported_v1(quactlize_ppu_moe_matmul_params_v1 const*);

// Create/prepare outside capture on the context device. Prepare selects and
// materializes a parent, not an online sweep. Context owns immutable plans;
// weights stay immutable/alive through every queued/captured use. ZW610 uses
// ZW810 selection estimates but its actual device resources.
int quactlize_ppu_moe_matmul_create_v1(int32_t device,void** context);
int quactlize_ppu_moe_matmul_prepare_v1(void* context,quactlize_ppu_moe_matmul_params_v1 const*,
    void const* low,void const* high,void const* units,void** plan,
    quactlize_ppu_moe_matmul_plan_info_v1*);

// Complete selected projection, including GPU routing/gather, SF expansion,
// actual Split-K reducer and output scatter. Direct indexed SIMT needs none
// of the TC routing helpers. New A/IDs/C/stream on EVERY call; no retained IDs,
// JIT, allocation, D2H, synchronization or shared handle pointer rebinding.
// A/C are 16-byte aligned, IDs 4-byte aligned; scratch 256-byte aligned and
// private to each concurrently live stream/graph. Graph replay retains its
// captured addresses. Caller completes work before destroying the context.
int quactlize_ppu_moe_matmul_run_v1(void* plan,float const* input,int32_t const* ids,float* output,
    void* workspace,uint64_t workspace_bytes,void* stream);
void quactlize_ppu_moe_matmul_destroy_v1(void* context);
char const* quactlize_ppu_moe_matmul_error_v1(void);

#ifdef __cplusplus
}
static_assert(sizeof(quactlize_ppu_moe_matmul_params_v1)==144,"MoE params ABI drift");
static_assert(offsetof(quactlize_ppu_moe_matmul_params_v1,arrangement)==72,"MoE arrangement ABI drift");
static_assert(sizeof(quactlize_ppu_moe_matmul_plan_info_v1)==240,"MoE plan ABI drift");
#endif
