#pragma once
#include <stdint.h>

// Versioned, device-pointer-only interface for one compiled K-pack parent.
// No GGUF repacking, allocation, profiling or implicit configuration choice is
// performed by prepare/run. Callers own all buffers and the execution stream.
#ifdef __cplusplus
extern "C" {
#endif

enum { QK_OK = 0, QK_UNSUPPORTED = 1, QK_INVALID = 2,
       QK_RUNTIME_ERROR = 3, QK_INITIALIZE_ERROR = 4 };
enum { QK_DENSE_FQ = 0, QK_DENSE_SF = 1, QK_GROUPED_FQ = 2, QK_GROUPED_SF = 3 };
enum { QK_ORDINARY = 0, QK_PERSISTENT = 1 };
enum { QK_COMPUTE_F16=0, QK_COMPUTE_BF16=1 };
enum { QK_METADATA_F16=0, QK_METADATA_BF16=1 };

typedef struct {
  uint32_t version, size;
  int32_t qtype, route, tm, tn, tk, wm, wn, stages, ap, delivery_n;
  uint64_t mapping_id;
  char const* parent;
  char const* build_key;
} qk_identity_v1;

typedef struct {
  uint32_t version, size;
  int32_t m, n, k, experts, group_size, device, compute_units;
  uint64_t mapping_id;
  void const* a;
  void const* low;
  void const* high;
  // FQ: packed units, zero=NULL. SF: resident FP16 scale/zero planes.
  void const* metadata;
  void const* zero;
  void* output;
  // Grouped rows_host[experts] and matching device rows/offsets. m is the
  // concatenated row count. Empty experts are legal. No hidden D2H occurs.
  int32_t const* rows_host;
  int32_t const* rows_device;
  int32_t const* offsets_device;
  void* workspace;
  uint64_t workspace_bytes;
  void* stream;
} qk_call_v1;

// Additive device-only grouped successor. The embedded call keeps its v1
// layout/version; rows_host and rows_device must be NULL. offsets_device is
// a nondecreasing device array [experts+1], from 0 to call.m. Every expert's
// row count must be <= max_rows. Host query sizes from that bound only.
// prepare owns no asynchronous host copies. run rebuilds device metadata
// from the current offsets, so graph replay can change expert routing.
typedef struct {
  uint32_t version, size;
  qk_call_v1 call;
  int32_t max_rows, reserved;
} qk_device_call_v2;

typedef struct {
  uint32_t version,size;
  qk_device_call_v2 device_call;
  int32_t compute_type;
} qk_compute_device_call_v3;
typedef struct {
  uint32_t version,size;
  qk_identity_v1 const* parent;
  int32_t compute_type;
} qk_compute_identity_v3;

// Explicit metadata precision. For K-quants, SF metadata/zero point to
// planes of this type; FQ still takes packed units and unfolds to this type.
// Q8_0 has no expanded metadata: its original FP16 d plane stays FP16.
typedef struct {
  uint32_t version,size;
  qk_device_call_v2 device_call;
  int32_t compute_type, metadata_type;
} qk_compute_device_call_v4;
typedef struct {
  uint32_t version,size;
  qk_identity_v1 const* parent;
  int32_t compute_type, metadata_type;
} qk_compute_identity_v4;

typedef struct {
  uint32_t version, size;
  int32_t algorithm, split, grid;
} qk_recipe_v1;

typedef struct {
  uint32_t version, size;
  uint64_t workspace_bytes, shared_bytes;
  int32_t occupancy, runtime_status, cutlass_status;
} qk_resources_v1;

qk_identity_v1 const* quactlize_kpack_identity_v1(void);
int quactlize_kpack_device_v1(char* name, int capacity, int32_t* ordinal, int32_t* compute_units);
// Query never dereferences device data. Exact occupancy can use the runtime.
int quactlize_kpack_query_v1(qk_call_v1 const*, qk_recipe_v1 const*, qk_resources_v1*);
// Prepare may enqueue small grouped metadata copies. The handle owns their
// host backing until destroy. All workspace/device data remain caller-owned.
int quactlize_kpack_prepare_v1(qk_call_v1 const*, qk_recipe_v1 const*, void**);
// Includes the grouped directory build or dense Split-K reducer when needed.
int quactlize_kpack_run_v1(void*, void* stream);
// Dense-only additive lifecycle. Prepare stores geometry/recipe, not buffers;
// all pointer fields in shape must be NULL (stream is used only to check that
// preparation is outside capture). Run accepts a complete current call, with
// the SAME geometry/device/layout. FQ, SF and Q8 retain their original FP16
// storage and arithmetic; SF expansion remains the caller's responsibility.
// Run builds private launch parameters on the stack: no allocation, JIT,
// occupancy query, handle mutation or synchronization. Concurrent streams
// require disjoint output/workspace; retain them through graph/device use.
// Dynamic handles use destroy_v1; run_v1 rejects them. Grouped is unsupported.
int quactlize_kpack_prepare_dynamic_v1(qk_call_v1 const* shape, qk_recipe_v1 const*, void**);
int quactlize_kpack_run_dynamic_v1(void*, qk_call_v1 const*);
// Caller must complete work using the handle before destroy/reusing workspace.
void quactlize_kpack_destroy_v1(void*);
int quactlize_kpack_measure_v1(void*, void* stream, int repeats, double* microseconds);

int quactlize_kpack_grouped_query_v2(qk_device_call_v2 const*, qk_recipe_v1 const*, qk_resources_v1*);
int quactlize_kpack_grouped_prepare_v2(qk_device_call_v2 const*, qk_recipe_v1 const*, void**);
// v2 handles use the existing run_v1/destroy_v1 lifecycle. This does not
// grant unmeasured any-M or performance admission to a particular parent.

// Explicit compute successor. A/output are contiguous BF16 for BF16 compute
// or FP16 for FP16 compute; metadata is ALWAYS the original FP16/packed-unit
// representation. F32 llama endpoints use the fused indexed/MoE adapters.
// BF16 modules reject v1/v2 preparation, so old callers cannot reinterpret
// their pointers through a changed compute contract.
qk_compute_identity_v3 const* quactlize_kpack_compute_identity_v3(void);
int quactlize_kpack_grouped_query_v3(qk_compute_device_call_v3 const*,qk_recipe_v1 const*,qk_resources_v1*);
int quactlize_kpack_grouped_prepare_v3(qk_compute_device_call_v3 const*,qk_recipe_v1 const*,void**);

// Typed metadata successor. A BF16-metadata module rejects the older v3
// entry before dereferencing any buffer. F16 modules continue to accept v3.
qk_compute_identity_v4 const* quactlize_kpack_compute_identity_v4(void);
int quactlize_kpack_grouped_query_v4(qk_compute_device_call_v4 const*,qk_recipe_v1 const*,qk_resources_v1*);
int quactlize_kpack_grouped_prepare_v4(qk_compute_device_call_v4 const*,qk_recipe_v1 const*,void**);

// Additive general grouped recipe, not the small-M indexed/chain endpoint.
// Prepare takes shape only (all data/workspace pointers NULL), outside capture.
// Each run supplies current buffers/offsets/stream with unchanged geometry and
// precision. The handle stores no data pointers and is shareable across streams
// with disjoint live scratch/output. Caller keeps buffers through graph use.
// COMPLETE retains v4's ordered reducer. PRODUCER leaves F32 partial planes
// for a fused transition/finish; split1 is the same typed final in both modes.
// No allocation, resource query, JIT, D2H or synchronization occurs in run.
enum { QK_GROUPED_COMPLETE=0, QK_GROUPED_PRODUCER=1 };
enum { QK_RESULT_F16=0, QK_RESULT_BF16=1, QK_RESULT_F32=2 };
typedef struct {
  uint32_t version, size;
  void const* data;
  int32_t dtype, partial_count, compute_type, reserved;
  int64_t row_stride, partial_stride;
} qk_grouped_result_v1;
int quactlize_kpack_grouped_prepare_dynamic_v1(qk_compute_device_call_v4 const* shape,
    qk_recipe_v1 const*,void** handle);
int quactlize_kpack_grouped_run_dynamic_v1(void* handle,qk_compute_device_call_v4 const*,int32_t mode);
// Host-only output description, also validates current bindings without launch.
// partial_count>1 means ordered F32 planes, not a finalized projection. Apply
// compute_type rounding AFTER reducing, before activation/weighting. No implied
// SwiGLU here. Descriptions are in grouped row order. Destroy with destroy_v1;
// run_v1 and old indexed/MoE binds reject these dynamic handles.
int quactlize_kpack_grouped_output_dynamic_v1(void* handle,qk_compute_device_call_v4 const*,
    int32_t mode,qk_grouped_result_v1*);

#ifdef __cplusplus
}
#endif
