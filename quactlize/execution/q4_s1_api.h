#pragma once
#include "api.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { QKG_Q4_META = 0, QKG_Q4_MEDIUM = 1, QKG_Q4_REUSE = 2 };

// Additive, explicit S1 reader API. Not a reinterpretation of qkg_config_v1.
// META: variant=header (0/1), values=1, eight output columns/CTA.
// MEDIUM: variant=fold | (unsigned_index<<1), values=2/4, four N workers.
// REUSE: variant=6/7 (cooperative metadata, optional cooperative A),
//        values=4/8, four N workers. All variants retain canonical K-pack4.
// Warps is the number of 32-thread warps. Each CTA owns ONE output row.
// Query succeeds only for an explicitly compiled recipe and N/K shape.
typedef struct {
    uint32_t version, size;
    int32_t reader, variant, warps, values;
} qkg_q4_s1_config_v1;

// Same INDEXED/GROUPED/DENSE addressing and F32 output as qkg_call_v1.
// F32 A rounds to F16 in registers. META reconstructs each weight in F16;
// MEDIUM/REUSE use FP32 group-affine arithmetic. All accumulate in FP32.
// low/units/A must be 16-byte aligned; A row/token strides must preserve
// 16-byte alignment. Output and device IDs/offsets must be 4-byte aligned.
// No workspace, prepass, inter-CTA reducer, allocation or host wait.
// Invalid device expert IDs produce NaN, never an out-of-range weight read.
// Device offsets must satisfy the existing nondecreasing/endpoints contract.
int quactlize_q4_s1_query_v1(qkg_call_v1 const*, qkg_q4_s1_config_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*, qkg_sizes_v1*);
int quactlize_q4_s1_run_v1(qkg_call_v1 const*, qkg_q4_s1_config_v1 const*,
    quactlize_ppu_placed_arrangement_v2 const*);

#ifdef __cplusplus
}
#endif
