#pragma once
#include "api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable helper IDs. Retired experiment IDs are deliberately not renumbered
// or silently mapped onto another implementation.
enum {
    QKP_DENSE_DEQUANT_DIRECT = 0,
    QKP_DENSE_DEQUANT_PAIR = 8,
    QKP_DENSE_DEQUANT_COOPERATIVE_K128 = 20,
    QKP_DENSE_DEQUANT_COOPERATIVE_K256 = 21,
    // N64 paired-word Q2_K/Q3_K/Q8_0 helpers.
    QKP_DENSE_DEQUANT_PLANE_PAIR_K128 = 22,
    QKP_DENSE_DEQUANT_PLANE_PAIR_K256 = 23,
    // Explicit Q8 shared-load exchange experiment; never selected by policy.
    QKP_DENSE_DEQUANT_Q8_EXCHANGE_K128 = 24,
    QKP_DENSE_DEQUANT_Q8_EXCHANGE_K256 = 25,
#define QKP_Q8_TUNE(ID, FAST, THREADS, VECTOR) QKP_DENSE_DEQUANT_Q8_TUNE_##ID = ID,
#include "dense_q8_tuning.inc"
#undef QKP_Q8_TUNE
#define QKP_Q8_HALF2(ID, FAST, THREADS) QKP_DENSE_DEQUANT_Q8_HALF2_##ID = ID,
#include "dense_q8_half2.inc"
#undef QKP_Q8_HALF2
};

// Private helper inventory; no change to caller structs or default policy.
static inline int qkp_dense_q8_tuning(int variant, int* fast, int* threads, int* vector_scale) {
    if (!fast || !threads || !vector_scale) return 0;
    switch (variant) {
#define QKP_Q8_TUNE(ID, FAST, THREADS, VECTOR) case ID: *fast=FAST; *threads=THREADS; *vector_scale=VECTOR; return 1;
#include "dense_q8_tuning.inc"
#undef QKP_Q8_TUNE
        default: return 0;
    }
}

static inline int qkp_dense_dequant_variant_valid(int variant) {
    int fast,threads,vector_scale;
    if (qkp_dense_q8_tuning(variant,&fast,&threads,&vector_scale)) return 1;
    if (variant>=QKP_DENSE_DEQUANT_Q8_HALF2_35 && variant<=QKP_DENSE_DEQUANT_Q8_HALF2_38) return 1;
    return variant == QKP_DENSE_DEQUANT_DIRECT || variant == QKP_DENSE_DEQUANT_PAIR ||
           variant == QKP_DENSE_DEQUANT_COOPERATIVE_K128 ||
           variant == QKP_DENSE_DEQUANT_COOPERATIVE_K256 ||
           variant == QKP_DENSE_DEQUANT_PLANE_PAIR_K128 ||
           variant == QKP_DENSE_DEQUANT_PLANE_PAIR_K256 ||
           variant == QKP_DENSE_DEQUANT_Q8_EXCHANGE_K128 ||
           variant == QKP_DENSE_DEQUANT_Q8_EXCHANGE_K256;
}

static inline int qkp_dense_dequant_variant_supports(int qtype, int variant) {
    int fast,threads,vector_scale;
    if (variant>=QKP_DENSE_DEQUANT_Q8_HALF2_35 && variant<=QKP_DENSE_DEQUANT_Q8_HALF2_38) return qtype==8;
    if (qkp_dense_q8_tuning(variant,&fast,&threads,&vector_scale)) return qtype==8;
    if (!qkp_dense_dequant_variant_valid(variant)) return 0;
    if (variant == QKP_DENSE_DEQUANT_Q8_EXCHANGE_K128 || variant == QKP_DENSE_DEQUANT_Q8_EXCHANGE_K256)
        return qtype == 8;
    if (variant == QKP_DENSE_DEQUANT_DIRECT) return qtype == 8 || (qtype >= 10 && qtype <= 14);
    if (variant == QKP_DENSE_DEQUANT_PLANE_PAIR_K128 || variant == QKP_DENSE_DEQUANT_PLANE_PAIR_K256)
        return qtype == 8 || qtype == 10 || qtype == 11;
    return qtype >= 12 && qtype <= 14;
}

// Dense-only helpers. Existing BF16/MoE entries are unchanged.
// Storage 0=FP16, 1=BF16. FP32 external endpoints and accumulation.
// F16 conversion has IEEE RNE overflow, never clipping or BF16 intermediates.
typedef struct {
    uint32_t version, size;
    char const* sdk;
    int32_t storage, vector_io;
    // -1: existing full BF16 dequant (call.weight.config).
    // Otherwise use the stable helper IDs above: direct Q2..Q6 and Q8_0; pair and
    // cooperative Q4/Q5/Q6; plane-pair Q2/Q3/Q8. Not the weight.config namespace.
    int32_t dequant_variant;
    int32_t math_mode, algorithm; // explicit cuBLAS controls; compute is always 32F
} qkp_dense_options_v1;

// Host-only helper lookup for the measured device/shape/dtype. Returns
// QKG_SHAPE on a miss with both outputs -1; caller retains its old helpers.
// Does not select a GEMM, change precision, compile or launch any work.
int quactlize_kpack_dense_helper_policy_v1(qkp_call_v1 const*, int storage,
    int* dequant_variant, int* vector_io);
// Separately measured Full-provider helpers. The old query above retains
// its TC/SF adapter semantics; a full-only measurement cannot change them.
int quactlize_kpack_dense_full_helper_policy_v1(qkp_call_v1 const*, int storage,
    int* dequant_variant, int* vector_io);

// Typed dense workspace query; Q8_0 does not enable the legacy BF16 provider.
int quactlize_kpack_dense_full_workspace_v1(qkp_call_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, uint64_t* bytes);

// Same workspace/layout contract as query_dynamic_v1 (including private
// cuBLAS workspace). Prepare is address-free and must occur outside capture.
int quactlize_kpack_dense_full_prepare_v1(qkp_call_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*,qkp_dense_options_v1 const*,void**);
// stage=-1: complete call; 0: input; 1: weights; 2: GEMM; 3: output; 4: reset.
// Individual stages are diagnostic entries: caller must prepare all inputs
// first. Streams need disjoint workspaces; no allocation/JIT/wait in run.
int quactlize_kpack_dense_full_run_v1(void*,qkp_call_v1 const*,int stage);
void quactlize_kpack_dense_full_destroy_v1(void*);
char const* quactlize_kpack_dense_full_error_v1(void);

// Dense contiguous/strided cast. No row-map semantics. vector_io=0 retains
// row-CTA scalar control; =1 enables aligned 8-element chunks and safe tails;
// =2 uses two warp-contiguous vec4 chunks, same eight elements/thread.
// =3 uses four such chunks for aligned contiguous whole CTAs, otherwise =2.
// error is optional; output propagates nonzero status as NaNs, input ignores it.
int quactlize_dense_io_v1(int storage,int input,int vector_io,void const* src,void* dst,
    int rows,int width,int64_t stride,int const* error,void* stream);

// Dense full-dequant helpers. qzd_call_v1.operation must be 1;
// config is not reinterpreted: variant has its own explicit namespace above.
int quactlize_kpack_dequant_dense_v1(qzd_call_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*,int storage,int variant);
#ifdef __cplusplus
}
#endif
