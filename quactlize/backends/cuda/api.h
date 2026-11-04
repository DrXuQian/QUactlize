#pragma once
#include <stdint.h>
#include "../../include/quactlize_ppu_config.h"
#include "../../compat/api.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { QCU_RAW_GGUF = 0, QCU_CANONICAL_KPACK = 1 };
enum { QCU_DENSE = 0, QCU_INDEXED = 1 };
enum { QCU_F16 = 0, QCU_BF16 = 1 };

// Native 16-bit input/output, FP32 accumulation, no activation quantization.
// m is the token count. INDEXED uses ids[m,topk] and input[m,channels,k],
// where channels=1 broadcasts the token and channels=topk selects each slot.
// Its output is [m,topk,n]. IDs may repeat; -1 denotes an inactive slot.
typedef struct {
    uint32_t version, size;
    int32_t qtype, layout, mode, compute_type, m, n, k, experts, topk, channels, device;
} qcu_problem_v1;

typedef struct {
    uint64_t raw_bytes, low_bytes, high_bytes, units_bytes;
    uint64_t input_bytes, output_bytes, ids_bytes;
    uint64_t expanded_weight_bytes, expanded_input_bytes, workspace_bytes;
} qcu_sizes_v1;

typedef struct {
    uint32_t version, size;
    void const *input, *raw, *low, *high, *units, *ids;
    void *output, *workspace;
    uint64_t raw_bytes, low_bytes, high_bytes, units_bytes;
    uint64_t input_bytes, output_bytes, ids_bytes, workspace_bytes;
    int32_t *device_error;
    void *stream;
} qcu_buffers_v1;

// Query is CPU-only. Preparation creates the cuBLAS handle outside capture.
// Call run once before capture to warm library kernels. Run enqueues complete
// weight expansion and 16-bit GEMM, with caller-owned output/workspace. It
// performs no host readback, allocation, JIT, or stream synchronization.
// Raw GGUF and canonical K-pack remain distinct, explicit storage contracts.
int quactlize_cuda_query_v1(qcu_problem_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, qcu_sizes_v1*);
int quactlize_cuda_prepare_v1(qcu_problem_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, void* stream, void** handle);
int quactlize_cuda_run_v1(void* handle, qcu_buffers_v1 const*);
// Expose expansion alone for format qualification, using the same workspace.
int quactlize_cuda_dequant_v1(void* handle, qcu_buffers_v1 const*);

// Direct GGUF row lookup for vocabulary and staged embedding tables. All
// buffers are caller-owned; IDs and optional live-row count stay on device.
// No handle, scratch, dummy GEMM operands or persistent expanded table.
typedef struct {
    uint32_t version, size;
    int32_t qtype, compute_type, columns, capacity;
    int64_t table_rows;
    uint64_t raw_bytes, ids_bytes, output_bytes;
    void const *raw;
    int64_t const *ids;
    int32_t const *num_rows;
    void *output;
    int32_t *device_error;
    void *stream;
} qcu_rows_v1;
// -1 and inactive suffix rows produce zero. Other invalid IDs set error=1;
// a live-row count outside [0,capacity] zeros the output and sets error=2.
int quactlize_cuda_gather_rows_v1(qcu_rows_v1 const*);
void quactlize_cuda_destroy_v1(void* handle);
char const* quactlize_cuda_error_v1(void);

// Explicit NVIDIA implementation of the new PPU matmul record layout. The
// symbols keep their CUDA identity; no PPU kernel or dispatch is substituted.
// Unlike the legacy entry points, these admit padded element strides and keep
// device_error sticky. Prepare executes and synchronizes one warmup; run only
// enqueues dequantization and 16-bit GEMM. Query reports NVIDIA scratch, which
// expands selected slots and is NOT the PPU grouped route's bounded workspace.
int quactlize_cuda_matmul_query_v1(qpm_problem_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, qpm_sizes_v1*);
int quactlize_cuda_matmul_prepare_v1(qpm_problem_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, qpm_buffers_v1 const*, void**);
int quactlize_cuda_matmul_run_v1(void*, qpm_buffers_v1 const*);
void quactlize_cuda_matmul_destroy_v1(void*);
char const* quactlize_cuda_matmul_error_v1(void);

// A handle is serialized during preparation/capture. Independent concurrent
// owners need separate handles and scratch. Retain the handle and all buffers
// until GPU work drains. device_error=1 reports an invalid routing ID; checking
// this device value belongs outside the captured computation.
#ifdef __cplusplus
}
#endif
