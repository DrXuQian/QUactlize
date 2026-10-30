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

// Selected helpers are additive: the original v1 calls above keep their
// fixed implementations. AUTO ranks only the measured model-shaped domain;
// outside it the original implementation is retained. Explicit alternatives
// remain available for any shape allowed by their query, not just measured M.
enum {
    QZM_AUTO=0,
    QZM_PREPARE_FUSED=1, QZM_PREPARE_HYBRID=2, QZM_PREPARE_SEPARATE=3,
    QZM_FINISH_SERIAL=16, QZM_FINISH_SLOTS4=17, QZM_FINISH_SLOTS8=18,
    QZM_FINISH_SEPARATE=19
};
typedef struct {
    uint32_t version, size;
    int32_t implementation, stages;
    uint64_t workspace_bytes;
} qzm_plan_v1;

// Host-only query; no pointers, IDs or route distribution are read. Prepare
// retains the caller's WARP/TILED and tile_rows choice. Its workspace is the
// existing route.workspace. The measured TILED512 choice is NOT a claim of
// optimality over every mm_ids implementation. Transition already selects its
// required operations from the actual producer state, including zero-op.
int quactlize_ppu_moe_prepare_select_v1(qzm_prepare_v1 const*,int32_t implementation,qzm_plan_v1*);
int quactlize_ppu_moe_prepare_execute_v1(qzm_prepare_v1 const*,qzm_plan_v1 const*);

// Additive combined route/helper selection. v1 keeps the caller's explicit
// route. In v2, config={1,sizeof(config),QZR_AUTO,0} requests measured host-key
// selection; an explicit route keeps v1 selection. Unmeasured AUTO geometry
// uses the previous general TILED512 preparation, not a small-M-only kernel.
// The returned plan owns no addresses/stream. Its helper.workspace_bytes is
// the required route.workspace capacity; allocate it before execute/capture.
typedef struct {
    uint32_t version, size;
    qzr_config_v1 config;
    qzm_plan_v1 helper;
} qzm_prepare_plan_v2;
int quactlize_ppu_moe_prepare_select_v2(qzm_prepare_v1 const*,qzm_prepare_plan_v2*);
int quactlize_ppu_moe_prepare_execute_v2(qzm_prepare_v1 const*,qzm_prepare_plan_v2 const*);
int quactlize_ppu_moe_finish_select_v1(qzm_finish_v1 const*,int32_t implementation,qzm_plan_v1*);

// A plan contains no addresses/stream. Pass current call bindings every time;
// query again if its shape/state changes. Separate finish needs two compact
// F32 [tokens*topk,columns] scratch planes, each aligned to 16 bytes. Caller
// owns per-stream scratch and keeps it alive through graph replay. Insufficient
// or overlapping scratch is rejected before any launch, never silently fused.
int quactlize_ppu_moe_finish_execute_v1(qzm_finish_v1 const*,qzm_plan_v1 const*,void* workspace,uint64_t bytes);

#ifdef __cplusplus
}
#endif
