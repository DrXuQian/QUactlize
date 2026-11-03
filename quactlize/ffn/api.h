#pragma once
#include "../packing/api.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { QUACTLIZE_FFN_DENSE=0, QUACTLIZE_FFN_MOE=1 };
enum { QUACTLIZE_FFN_GATE=0, QUACTLIZE_FFN_UP=1, QUACTLIZE_FFN_DOWN=2 };

// Canonical K-pack only, not G4/U4 paired storage. Plan preparation never
// repacks/copies weights. The caller retains immutable planes until all
// queued work and graph replays have completed and the context is destroyed.
typedef struct {
    int32_t qtype,n,k,experts;
    quactlize_ppu_placed_arrangement_v2 arrangement;
    quactlize_ppu_kpack_sizes_v1 sizes;
    void const *low,*high,*units;
} quactlize_ppu_ffn_weight_v1;

// Computes down(silu(gate(A))*up(A)). MoE additionally computes the weighted
// sum in original slot order, using changing IDs and weights on every call.
// All endpoints are F32; all accumulators F32. Dense/shared uses the existing
// F16 dense policy, MoE BF16. Actual producer rounding is preserved, with no
// clipping or INT8 activations. Bias, residual, router and TP collectives are
// outside this ABI. Do not fuse across a required collective/nonlinear input.
//
// merged=1: gate holds canonical [E,2*hidden,K] (gate rows followed by up),
// and up is entirely zero. Otherwise gate/up may have different qtypes but
// identical N/K/E. Down.K=hidden; Down.N need not equal Gate.K.
// SwiGLU runs in a separate transition, not in a projection or its reducer.
// Strides count elements. Input is A[token*input_token_stride +
// (slot%channels)*input_row_stride + K], output [token*output_stride+Down.N].
// Dense requires topk=channels=E=1, ids_stride=weights_stride=0; run receives
// NULL IDs/weights. MoE requires valid, distinct-per-token IDs, topk<=E<=1024.
typedef struct {
    uint32_t version,size;
    int32_t kind,tokens,topk,channels,merged,reserved;
    int64_t input_row_stride,input_token_stride,ids_stride,weights_stride,output_stride;
    quactlize_ppu_ffn_weight_v1 gate,up,down;
} quactlize_ppu_ffn_params_v1;

typedef struct {
    int32_t route,policy,split,compute_type;
    char config[192];
} quactlize_ppu_ffn_projection_info_v1;
typedef struct {
    uint32_t version,size;
    uint64_t workspace_bytes;
    // -1 means no TC routing preparation. helper=0 means not used.
    int32_t prepare_stage,prepare_helper,finish_helper,reserved;
    // Same route/precision values as dense and moe/matmul. Absent up has
    // route=-1. Each projection keeps its independent selector's configuration.
    quactlize_ppu_ffn_projection_info_v1 projections[3];
} quactlize_ppu_ffn_plan_info_v1;

// Pure host shape/layout admission, ignoring plane addresses. 1=supported,
// 0=unsupported, -1=invalid ABI. tokens=0 is weight admission; prepare needs
// tokens>0. No M<=8 global limit: each projection uses its existing selector.
int quactlize_ppu_ffn_supported_v1(quactlize_ppu_ffn_params_v1 const*);
// Outside capture. Context owns immutable plans; run does not tune, JIT,
// allocate, synchronize, read back IDs or mutate shared handle bindings.
int quactlize_ppu_ffn_create_v1(int32_t device,void** context);
int quactlize_ppu_ffn_prepare_v1(void* context,quactlize_ppu_ffn_params_v1 const*,
    void** plan,quactlize_ppu_ffn_plan_info_v1*);
// A/C 16-byte aligned; IDs/weights 4-byte; scratch 256-byte and disjoint from
// every input/weight/output. Concurrent streams need private C and scratch.
// Eager warmup before capture remains necessary for dense cuBLAS providers.
int quactlize_ppu_ffn_run_v1(void* plan,float const* input,int32_t const* ids,float const* weights,
    float* output,void* workspace,uint64_t workspace_bytes,void* stream);
// Same plan/shape ABI. Final output may overlap A, IDs or routing weights:
// A/IDs are consumed before output; transition retains overlapping routing
// weights without another launch. Scratch/packed planes must remain disjoint.
// v1 keeps its original no-overlap contract.
int quactlize_ppu_ffn_run_v2(void* plan,float const* input,int32_t const* ids,float const* weights,
    float* output,void* workspace,uint64_t workspace_bytes,void* stream);
void quactlize_ppu_ffn_destroy_v1(void* context);
char const* quactlize_ppu_ffn_error_v1(void);

#ifdef __cplusplus
}
static_assert(sizeof(quactlize_ppu_ffn_weight_v1)==112,"FFN weight ABI drift");
static_assert(offsetof(quactlize_ppu_ffn_params_v1,gate)==72,"FFN params ABI drift");
static_assert(sizeof(quactlize_ppu_ffn_params_v1)==408,"FFN params ABI drift");
static_assert(sizeof(quactlize_ppu_ffn_plan_info_v1)==656,"FFN plan ABI drift");
#endif
