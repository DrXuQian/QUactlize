#pragma once
#include "../packing/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Additive native-16-bit compatibility API. Old K-pack APIs and their tuned
// selectors are unchanged. RAW_GGUF is block storage, not canonical K-pack.
// This is a separate compatibility route, NOT an admission of the old tuned
// table's latency on raw/special shapes. PPU hardware qualification is required.
enum { QPM_RAW_GGUF = 0, QPM_CANONICAL_KPACK = 1 };
enum { QPM_DENSE = 0, QPM_INDEXED = 1 };
enum { QPM_F16 = 0, QPM_BF16 = 1 };
enum { QPM_SIMT = 0, QPM_GROUPED_TC = 1 };
// Explicit direct-packed candidates, not automatic changes to the v1 route.
enum { QPM_PACKED_AUTO = 0, QPM_PACKED_SIMT = 2, QPM_PACKED_TC = 3 };
enum { QPM_OK = 0, QPM_INVALID = 20, QPM_RUNTIME = 41 };
enum { QPM_BAD_ID = 1, QPM_BAD_LOGIT = 2 };

// Logical weights [experts,N,K]. Strides count ELEMENTS, not bytes; the last
// dimension is contiguous. A[token*input_token_stride+channel*input_row_stride+k],
// channel=0 for channels=1, else slot. C[(token*topk+slot)*output_stride+n].
// Dense: experts=topk=channels=1, ids absent. Indexed: IDs[token*ids_stride+slot].
// Native F16/BF16 A and C; weights decode in FP32 then round ONCE to that type;
// GEMM accumulates FP32. No activation quantization, clipping or requantization.
typedef struct {
    uint32_t version, size;
    int32_t qtype, layout, mode, dtype, m, n, k, experts, topk, channels, device, reserved;
    int64_t input_row_stride, input_token_stride, ids_stride, output_stride;
} qpm_problem_v1;

typedef struct {
    uint32_t version, size;
    uint64_t raw_bytes, low_bytes, high_bytes, units_bytes;
    uint64_t input_bytes, output_bytes, ids_bytes, workspace_bytes;
    uint64_t expanded_weight_bytes, grouped_input_bytes, grouped_output_bytes, routing_bytes;
    int32_t route, expert_tile, column_tile, reserved;
} qpm_sizes_v1;

typedef struct {
    uint32_t version, size;
    void const *input, *raw, *low, *high, *units;
    int32_t const *ids;
    void *output, *workspace;
    uint64_t input_bytes, raw_bytes, low_bytes, high_bytes, units_bytes;
    uint64_t ids_bytes, output_bytes, workspace_bytes;
    int32_t *device_error;
    void *stream;
} qpm_buffers_v1;

// Host-only. Q5_1(7), Q8_0(8), Q2_K..Q6_K(10..14), Q2_0(42). RAW requires
// K%32==0 for 7/8, K%256==0 for K-quants, K%64==0 for Q2_0, with N>0. Canonical keeps
// producer's EXACT descriptor/alignment checks: Q5_1 N%16/K%128; Q8_0
// N%256/K%32; Q2_0 N>0/K%64; K-quants retain their existing constraints. M>=0; E<=1024,
// topk<=32, channels=1 or topk. Canonical storage does not by itself select
// a tuned packed kernel: this v1 remains the native16 compatibility route.
// Null arrangement for RAW, canonical producer descriptor otherwise.
// Query is also the workspace authority. Let R=M*topk, Np=ceil(N/8)*8.
// For M<=8, direct indexed SIMT uses no scratch. Otherwise scratch contains:
//   2*(R+32)*K bytes gathered input, 2*R*Np bytes grouped output;
//   2*min(E,4)*min(Np,512)*K bytes expanded weights;
//   4*(E+1)+4*E+8*R bytes route arrays and 4096 bytes descriptors,
// with EACH allocation rounded to 256 bytes. Weight scratch is independent
// of M; experts/slots are never expanded into an [M,topk,N,K] tensor.
// Sequential layers may share max(query.workspace_bytes); live projection
// outputs, activation intermediates and weights are NOT part of that scratch.
int quactlize_ppu_matmul_query_v1(qpm_problem_v1 const *, quactlize_ppu_placed_arrangement_v2 const *,
                                  qpm_sizes_v1 *);

// Outside capture, on problem.device. Runs and synchronizes the initial call
// using caller-owned buffers to warm the actual selected kernels; output and
// scratch are overwritten. No hidden weight allocation. No online tuning/JIT.
// Handle stores geometry/resources only, not buffer addresses. On failure
// *handle is NULL. Query/prepare diagnostics use error_v1.
int quactlize_ppu_matmul_prepare_v1(qpm_problem_v1 const *, quactlize_ppu_placed_arrangement_v2 const *,
                                    qpm_buffers_v1 const *, void **handle);

// PPU direct-packed Q5_1 candidates. Same arithmetic/strides/graph lifetime as
// v1; canonical descriptor required. No global expanded weights or grouped
// A/C. AUTO uses SIMT for M<=8, TC otherwise; either can be forced for tests.
// Query reports the actual PACKED_* route, N tile (32/64), and exact workspace.
// TC uses M16 tiles; indexed scratch is three aligned 4*(E+1) route arrays
// and one aligned 4*M*topk slot map. Dense TC and SIMT use no workspace.
// Performance admission is separate: old v1/tuned selectors are unchanged.
// Returned handles use the existing run/dequant/destroy functions below.
int quactlize_ppu_matmul_packed_query_v1(qpm_problem_v1 const *,
    quactlize_ppu_placed_arrangement_v2 const *, int route, qpm_sizes_v1 *);
int quactlize_ppu_matmul_packed_prepare_v1(qpm_problem_v1 const *,
    quactlize_ppu_placed_arrangement_v2 const *, int route, qpm_buffers_v1 const *, void **handle);

// Enqueue only: no allocation, JIT, synchronization, D2H or host routing reads.
// Duplicates are independent slots; -1 produces zero; other invalid IDs
// produce zero and atomicOr(QPM_BAD_ID) into caller-owned device_error.
// Error bits are STICKY: caller must zero status before a call/chain/replay.
// Run accepts new addresses with the prepared geometry/capacity/strides.
// Disjoint scratch/output/status for overlapping streams. Graphs retain
// captured addresses; keep ALL buffers and the handle alive through graph use.
// All pointers belong to problem.device. Weight buffers are immutable. Minimum
// alignment: A/C=2, raw=1, canonical=16, IDs/status=4, workspace=256 bytes.
int quactlize_ppu_matmul_run_v1(void *handle, qpm_buffers_v1 const *);
void quactlize_ppu_matmul_destroy_v1(void *handle);
char const *quactlize_ppu_matmul_error_v1(void);

// Decode one expert's contiguous N-row range into native [rows,K], for format
// qualification or bounded consumers. No GEMM or route expansion. The output
// must be disjoint from all input planes. expert/first/rows are HOST indices.
int quactlize_ppu_matmul_dequant_v1(void *handle, qpm_buffers_v1 const *, int32_t expert, int32_t first,
                                    int32_t rows, void *output, uint64_t bytes);

// Router: F32 logits[M,E] -> IDs[M,topk] + F32 weights[M,topk]. Full softmax,
// descending probability (tie: lower expert ID), then selected-weight
// normalization and multiplier. Nonfinite logits set BAD_LOGIT and write
// all -1/zero for that token. Strides count elements. No workspace.
typedef struct {
    uint32_t version, size;
    int32_t m, experts, topk, reserved;
    int64_t logits_stride, ids_stride, weights_stride;
    float multiplier;
    float const *logits;
    int32_t *ids;
    float *weights;
    int32_t *device_error;
    void *stream;
} qpm_router_v1;
int quactlize_ppu_router_run_v1(qpm_router_v1 const *);

// Native F16/BF16 SwiGLU: separate [rows,H] gate/up, or merged [rows,2H]
// in canonical [gate;up] row order (NOT paired-N4). All strides in elements.
// F32 activation arithmetic followed by one output rounding; no workspace.
typedef struct {
    uint32_t version, size;
    int32_t rows, hidden, dtype, merged;
    int64_t gate_stride, up_stride, output_stride;
    void const *gate, *up;
    void *output;
    void *stream;
} qpm_swiglu_v1;
int quactlize_ppu_swiglu_run_v1(qpm_swiglu_v1 const *);

// Typed slot output[M,topk,N] times F32 route weights -> F32 [M,N]. Sum in
// original slot order using FP32, no floating atomics. Native-16-bit output
// of matmul has already crossed its projection rounding boundary.
typedef struct {
    uint32_t version, size;
    int32_t m, topk, n, dtype;
    int64_t input_stride, weights_stride, output_stride;
    void const *input;
    float const *weights;
    float *output;
    void *stream;
} qpm_merge_v1;
int quactlize_ppu_merge_run_v1(qpm_merge_v1 const *);

#ifdef __cplusplus
}
static_assert(sizeof(qpm_problem_v1) == 88, "matmul problem ABI drift");
static_assert(sizeof(qpm_sizes_v1) == 120, "matmul sizes ABI drift");
static_assert(sizeof(qpm_buffers_v1) == 152, "matmul buffers ABI drift");
#endif
