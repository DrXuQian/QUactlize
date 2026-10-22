#pragma once
#include "api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dense-only helpers. Existing BF16/MoE entries are unchanged.
// Storage 0=FP16, 1=BF16. FP32 external endpoints and accumulation.
// F16 conversion has IEEE RNE overflow, never clipping or BF16 intermediates.
typedef struct {
    uint32_t version, size;
    char const* sdk;
    int32_t storage, vector_io;
    // -1: existing full BF16 dequant (call.weight.config). Otherwise:
    // 0: direct typed reference, 1/2: packed exchange K128/K256,
    // 3: K256 with K-fast CTA order. 4/5: unsigned K128/K256 N32/T128;
    // 6: unsigned K256 N16/T128; 7: unsigned K256 N32/T256.
    // 8/9: paired N32/K128, 10/11: paired N32/K256, 12/13: paired N16/K256.
    // Even paired variants use scalar code conversion; odd use exact FP32
    // mantissa construction. All retain FP32 multiply/subtract rounding.
    // 14/15/16: paired bank layout N32/K128, N32/K256, N16/K256.
    // 17/18/19: same with cooperative unit metadata; 20/21: N64/K128/K256.
    // 1..21 support Q4/Q5/Q6; 0 Q2..Q6. Not the legacy weight.config namespace.
    // 22/23: experimental Q5/Q6 N64 K128/K256, unique physical high loads.
    // Metadata, consumer, output store and rounding match 20/21 respectively.
    int32_t dequant_variant;
    int32_t math_mode, algorithm; // explicit cuBLAS controls; compute is always 32F
} qkp_dense_options_v1;

// Host-only helper lookup for the measured device/shape/dtype. Returns
// QKG_SHAPE on a miss with both outputs -1; caller retains its old helpers.
// Does not select a GEMM, change precision, compile or launch any work.
int quactlize_kpack_dense_helper_policy_v1(qkp_call_v1 const*, int storage,
    int* dequant_variant, int* vector_io);

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

// Independent full dequant candidate. qzd_call_v1.operation must be 1;
// config is not reinterpreted: variant has its own explicit namespace above.
int quactlize_kpack_dequant_dense_v1(qzd_call_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*,int storage,int variant);
#ifdef __cplusplus
}
#endif
