#pragma once
#include "api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Additive serving helpers. Same ABI in separately built PPU/CUDA libraries.
// v1 matmul/router/SwiGLU/merge layouts and numerical semantics are unchanged.
enum { QPM_F32 = 2, QPM_SCORE_SOFTMAX = 0 };
enum { QPM_BAD_COUNT = 4 }; // sticky OR, alongside BAD_ID=1 and BAD_LOGIT=2

// Call outside capture, once per device/context, before any serving helper.
// Loads and warms all helper variants on the EXPLICIT non-null stream and
// synchronizes it. No device allocation, hidden scratch, handle or host routing.
// Repeat preparation after device/context reset. Keep the DSO loaded until all
// captured graphs and pending launches have been destroyed/drained.
int quactlize_serving_prepare_v1(int32_t device, void *stream);
char const *quactlize_serving_error_v1(void);

// BF16/F32 [capacity,E], contiguous E, element strides. E<=1024, topk<=32.
// Top-k ORIGINAL widened logits, descending, exact ties favor LARGER expert ID.
// Stable softmax of selected logits only, then multiplier (normally 1).
// Only score_func=SOFTMAX, renormalize=1 and no correction bias are supported.
// Live count is optional device I32 (NULL means capacity). Never read inactive
// logits. Inactive output = selected 0, IDs -1, weights 0. Nonfinite ACTIVE
// logits clear that row and OR BAD_LOGIT; invalid count clears ALL rows and ORs
// BAD_COUNT. No helper clears device_error. All outputs/weights remain F32.
typedef struct {
    uint32_t version, size;
    int32_t capacity, experts, topk, dtype, score_func, renormalize;
    float multiplier;
    int32_t reserved;
    int64_t logits_stride, selected_stride, ids_stride, weights_stride;
    uint64_t logits_bytes, selected_bytes, ids_bytes, weights_bytes;
    void const *logits;
    float *selected_logits;
    int32_t *ids;
    float *weights;
    int32_t const *num_valid_tokens;
    int32_t *device_error;
    void *stream;
} qpm_router_v2;
int quactlize_serving_router_v2(qpm_router_v2 const *);

// Native F16/BF16. Explicit round_silu=1: T(T(silu(F32(gate)))*F32(up));
// round_silu=0: one final rounding. No clipping. merged means [gate;up] in
// EACH activation row, NOT paired/interleaved weights; up must then be NULL.
// All strides count elements. Output must be disjoint from gate/up.
typedef struct {
    uint32_t version, size;
    int32_t rows, hidden, dtype, merged, round_silu, reserved;
    int64_t gate_stride, up_stride, output_stride;
    uint64_t gate_bytes, up_bytes, output_bytes;
    void const *gate, *up;
    void *output, *stream;
} qpm_swiglu_v2;
int quactlize_serving_swiglu_v2(qpm_swiglu_v2 const *);

// RAW GGUF ONLY: Q8_0=8 (32 values/34 bytes), IQ4_NL=20 (32/18).
// Contiguous table[table_rows,columns], columns%32=0, IDs I64[capacity].
// Output contiguous native F16/BF16[capacity,columns], F32 decode then T round.
// Duplicate IDs are independent. -1 -> zero without error; other invalid IDs
// -> zero + BAD_ID. Inactive rows -> zero, read neither IDs nor raw. Invalid
// device num_rows -> clear all outputs + BAD_COUNT. NULL count = capacity.
// 64-bit offsets and capacity checks also apply to tables larger than 4 GiB.
// IQ4_NL support here does NOT extend the matmul format inventory.
typedef struct {
    uint32_t version, size;
    int32_t qtype, dtype, capacity, columns;
    int64_t table_rows;
    uint64_t raw_bytes, ids_bytes, output_bytes;
    void const *raw;
    int64_t const *ids;
    void *output;
    int32_t const *num_rows;
    int32_t *device_error;
    void *stream;
} qpm_rows_v1;
int quactlize_serving_gather_rows_v1(qpm_rows_v1 const *);

// Shared merge contract, also exported by the standalone CUDA helper DSO.
// Native slot values x F32 weights, fixed slot order FP32 sum -> F32 output.
int quactlize_serving_merge_v1(qpm_merge_v1 const *);

// Separate NVIDIA compatibility binary, never substituted for the PPU route.
// Reuses qpm v1 geometry/planes/rounding. Query reports bounded column-tiled
// expansion: <=2*512*K weight bytes, 2*R*K input + 2*R*min(ceil8(N),512)
// output scratch + alignment + 32 MiB cuBLAS workspace. No per-slot weights.
// One prepared cuBLAS handle per concurrent chain/stream; prepare warms every
// tile outside capture. CUDA timings do not qualify the PPU implementation.
int quactlize_cuda_matmul_query_v1(qpm_problem_v1 const *, quactlize_ppu_placed_arrangement_v2 const *,
                                   qpm_sizes_v1 *);
int quactlize_cuda_matmul_prepare_v1(qpm_problem_v1 const *, quactlize_ppu_placed_arrangement_v2 const *,
                                     qpm_buffers_v1 const *, void **);
int quactlize_cuda_matmul_run_v1(void *, qpm_buffers_v1 const *);
void quactlize_cuda_matmul_destroy_v1(void *);
char const *quactlize_cuda_matmul_error_v1(void);

// All run calls: enqueue only, caller's current device/non-null stream. No
// malloc/free, JIT/tuning, synchronization, implicit default stream, or D2H.
// Pointers must be on that device; scalar count/status pointers have 4 bytes.
// Minimum alignment: typed16=2, F32/IDs32/count/status=4, IDs64=8, raw=1.
// Caller zeroes status ONCE at chain/replay entry. Outputs/status must not
// overlap ANY live input/output/count. Read-only inputs may alias each other.
// Buffers must outlive queued work and graph use. Same fixed addresses can
// hold new counts/logits/IDs at replay. Concurrent chains need disjoint writes;
// sequential chains may reuse scratch with normal stream/event dependencies.
#ifdef __cplusplus
}
static_assert(sizeof(qpm_router_v2) == 160, "router v2 ABI drift");
static_assert(sizeof(qpm_swiglu_v2) == 112, "SwiGLU v2 ABI drift");
static_assert(sizeof(qpm_rows_v1) == 104, "row lookup ABI drift");
#endif
