#pragma once
#include "../routing/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opt-in scaffolding, independent of weight format and GEMM selection. Existing
// decode chains are unchanged. M means tokens; R=M*topk means routed rows.
// All entries enqueue on caller-owned streams/storage: no allocation, JIT,
// synchronization, device readback, or persistent caching of changing IDs.
enum { QZM_F32=0, QZM_F16=1, QZM_BF16=2 };
enum { QZM_GROUPED=0, QZM_TOKEN_SLOT=1 };
enum { QZM_PROJECTED=0, QZM_PARTIALS=1, QZM_ACTIVATED=2 };
enum {
    QZM_REDUCE_GATE=1, QZM_REDUCE_UP=2, QZM_SWIGLU=4,
    QZM_COPY_ACTIVATED=8, QZM_NOOP=16
};

// Final producer output, NOT the producer's original Split-K configuration.
// PARTIALS is F32 with partial_count planes; PROJECTED/ACTIVATED have one.
// partial_stride and row_stride count elements, not bytes. round_type is the
// projection boundary (F32=no extra rounding), applied AFTER reduction and
// BEFORE activation/weighting. ACTIVATED has already crossed that boundary;
// it requires round_type=F32 and must never be activated a second time.
typedef struct {
    void const* data;
    int32_t dtype, order, state, partial_count;
    int64_t row_stride, partial_stride;
    int32_t round_type, reserved;
} qzm_value_v1;

typedef struct {
    uint32_t version, size;
    qzr_call_v1 route;
    qzr_config_v1 config;
    int32_t columns, dtype;
    int64_t input_row_stride;
    float const* input;
    void* output;
    int32_t *inverse_rows, *expert_rows, *active_experts, *active_count;
} qzm_prepare_v1;

// route.inverse must be 0. Besides its forward maps/offsets, prepare writes
// inverse_rows[token*topk+slot]=grouped_row, expert_rows[E], active_experts[E]
// (only its active_count prefix is defined), and compact output[R,columns].
// Input is F32 at route.src_rows[grouped_row]*input_row_stride. IDs retain
// the route API's valid/distinct-per-token contract. Every output is disjoint.
// TILED: histogram/rank -> prefix+expert metadata -> maps+gather/convert.
// WARP: existing map producer -> expert metadata -> gather/convert+inverse.
// Zero tokens clears offsets/counts; it launches no input/output kernel.
// Select WARP/TILED explicitly. This entry does not publish an unmeasured
// AUTO crossover; the existing production small-M prepare remains separate.
int quactlize_ppu_moe_prepare_query_v1(qzm_prepare_v1 const*,qzr_sizes_v1*);
int quactlize_ppu_moe_prepare_run_v1(qzm_prepare_v1 const*);

typedef struct {
    uint32_t version, size;
    int32_t tokens, topk, hidden, merged;
    qzm_value_v1 gate, up;
    int32_t output_dtype, output_order;
    int64_t output_stride;
    void* output;
    int32_t const *row_ids, *inverse_rows;
    void* stream;
} qzm_transition_v1;

// PROJECTED/PARTIALS gate is [R,2*hidden] when merged, otherwise gate and up
// are [R,hidden]. ACTIVATED gate is always [R,hidden]; up is absent.
// Maps are only required when orders differ. Arithmetic is F32 SwiGLU with
// explicit projection and destination rounding; no clipping or INT8 A.
// An already-activated, identical view is a true no-op (zero kernel launches).
int quactlize_ppu_moe_transition_query_v1(qzm_transition_v1 const*,uint32_t* operations);
int quactlize_ppu_moe_transition_run_v1(qzm_transition_v1 const*);

typedef struct {
    uint32_t version, size;
    int32_t tokens, topk, columns, reserved;
    qzm_value_v1 down;
    int32_t const* inverse_rows;
    float const* weights;
    float* output;
    int64_t weights_stride, output_stride;
    void* stream;
} qzm_finish_v1;

// Optional Down partial reduction + projection rounding + routing weights +
// top-k sum. F32 output [tokens,columns], deterministic original slot order,
// not expert order or floating atomics. No fixed token/route-row capacity.
int quactlize_ppu_moe_finish_run_v1(qzm_finish_v1 const*);

#ifdef __cplusplus
}
#endif
