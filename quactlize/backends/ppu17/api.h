#pragma once

#include <stddef.h>
#include <stdint.h>
#include "quactlize_ppu_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independent ABI: no actlize/CuTe types, legacy tactic IDs, or device-0 state.
enum q17_dtype { Q17_F16 = 0, Q17_BF16 = 1, Q17_F32 = 2 };
enum q17_route { Q17_FQ = 0, Q17_SF = 1 };
enum q17_operator { Q17_DENSE = 0, Q17_GROUPED = 1 };
enum q17_engine { Q17_SIMT = 0, Q17_WGMMA = 1, Q17_TMA_WGMMA = 2 };
enum q17_status {
  Q17_SUCCESS = 0, Q17_INVALID = 1, Q17_UNSUPPORTED = 2,
  Q17_WRONG_DEVICE = 3, Q17_LAUNCH_ERROR = 4
};

typedef struct q17_problem_v1 {
  uint32_t version, struct_bytes;
  int32_t qtype, operation, route, compute_type, input_type, output_type;
  int32_t m, n, k, experts, max_rows;
  quactlize_ppu_placed_arrangement_v2 arrangement;
} q17_problem_v1;

typedef struct q17_config_v1 {
  uint32_t version, struct_bytes;
  int32_t engine, tile_m, tile_n, tile_k, threads, lanes_n;
} q17_config_v1;

typedef struct q17_sizes_v1 {
  uint64_t low_bytes, high_bytes, units_bytes, scale_bytes, zero_bytes;
} q17_sizes_v1;

// A and C are row-major compact [m,k] and [m,n]; strides count elements.
// Grouped offsets are device int32[E+1], monotone, offsets[0]=0,
// offsets[E]=m; every difference <= max_rows. Empty experts are legal.
// Typed pointers must satisfy their element alignment. Output/expanded planes
// must be disjoint from inputs and from each other; allocations must cover the
// queried plane sizes and strided A/C extents.
// Each weight plane concatenates E equal-size expert slices. NULL high is
// required when high_bits=0. FQ units retain original GGUF FP16 headers;
// SF scale/zero are compute_type [E,K/group_size,N], never FP16 as a BF16
// intermediate. Offsets and all device storage must outlive stream execution.
// No device allocation, synchronization, host routing readback, or implicit
// prepass occurs inside launch. Prepared execution also has no host allocation.
// SF expansion is a separate, explicitly timed launch.
typedef struct q17_call_v1 {
  uint32_t version, struct_bytes;
  q17_problem_v1 problem;
  q17_config_v1 config;
  void const *a, *low, *high, *units, *scale, *zero;
  int32_t const *offsets;
  void *output, *stream;
  int64_t a_stride, output_stride;
  int32_t device, reserved;
} q17_call_v1;

// Host-only calls: do not initialize a GPU context. Inventory is structural,
// not a performance ranking. Callers must explicitly choose a config.
int q17_canonical_arrangement_v1(int qtype, quactlize_ppu_placed_arrangement_v2 *out);
int q17_sizes_query_v1(q17_problem_v1 const *, q17_sizes_v1 *);
int q17_config_count_v1(q17_problem_v1 const *);
int q17_config_get_v1(q17_problem_v1 const *, int index, q17_config_v1 *);
int q17_can_implement_v1(q17_problem_v1 const *, q17_config_v1 const *);

int q17_launch_v1(q17_call_v1 const *);
// Prepare once outside graph capture to encode TMA descriptors. The immutable
// plan binds all pointers, shapes, strides, config and device from call. Execute
// enqueues exactly one kernel; it does not encode descriptors or allocate.
// A plan can be destroyed after enqueue (no owned GPU storage). Recreate it
// when any binding changes. Storage remains the caller's responsibility.
typedef struct q17_plan_v1 q17_plan_v1;
int q17_prepare_v1(q17_call_v1 const *, q17_plan_v1 **);
int q17_execute_v1(q17_plan_v1 const *);
void q17_destroy_v1(q17_plan_v1 *);
// Writes scale/zero from units. low/high/a/output/offsets are not accessed.
int q17_expand_scales_v1(q17_call_v1 const *);
char const *q17_last_error_v1(void);
char const *q17_build_target_v1(void);

#ifdef __cplusplus
}
#endif
