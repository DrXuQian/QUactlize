#pragma once
#include "api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Additive dense-only candidates. No existing BF16/MoE entry or selector is
// changed. Storage 0=FP16, 1=BF16. FP32 external endpoints and accumulation.
// F16 conversion has IEEE RNE overflow, never clipping or BF16 intermediates.
typedef struct {
    uint32_t version, size;
    char const* sdk;
    int32_t storage, vector_io;
    // -1: existing full BF16 dequant (call.weight.config). Otherwise:
    // 0: direct typed reference, 1/2: packed exchange K128/K256,
    // 3: K256 with K-fast CTA order. 1..3 support Q4/Q5/Q6; 0 Q2..Q6.
    int32_t dequant_variant;
    int32_t math_mode, algorithm; // explicit cuBLAS controls; compute is always 32F
} qkp_dense_options_v1;

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
// row-CTA scalar control; =1 enables aligned 8-element chunks and safe tails.
// error is optional; output propagates nonzero status as NaNs, input ignores it.
int quactlize_dense_io_v1(int storage,int input,int vector_io,void const* src,void* dst,
    int rows,int width,int64_t stride,int const* error,void* stream);

// Independent full dequant candidate. qzd_call_v1.operation must be 1;
// config is not reinterpreted: variant has its own explicit namespace above.
int quactlize_kpack_dequant_dense_v1(qzd_call_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*,int storage,int variant);
#ifdef __cplusplus
}
#endif
