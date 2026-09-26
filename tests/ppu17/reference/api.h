#pragma once
#include <stddef.h>
#include <stdint.h>

// Diagnostic only. All inputs are already resident and representation-specific.
// Active contract: existing GPTQ symmetric INT4, group128, desc_act=false.
// dtype=0: FP16 A/scales/C; FP32 accumulation. zeros MUST be NULL.
// Native packing differs, but every code and original scale bit is retained.
struct q17_ref_call {
  int32_t dtype, config, m, n, k;
  void const *a, *b, *scales, *zeros;
  void *output, *stream;
};

#define Q17_REF_EXPORT extern "C" __attribute__((visibility("default")))
Q17_REF_EXPORT char const *q17_ref_error();
Q17_REF_EXPORT char const *q17_ref_contract();
Q17_REF_EXPORT int q17_ref_config_count();
Q17_REF_EXPORT char const *q17_ref_config_name(int config);
Q17_REF_EXPORT int q17_ref_prepare(q17_ref_call const *, void **plan);
Q17_REF_EXPORT int q17_ref_execute(void *plan);
Q17_REF_EXPORT void q17_ref_destroy(void *plan);
Q17_REF_EXPORT size_t q17_ref_workspace(void *plan);
Q17_REF_EXPORT int q17_ref_device(int *l2_bytes, int *sm_count);
// Sequential plans may share one scratch allocation (never concurrent calls).
// owner must outlive all users and have the same configuration/geometry.
Q17_REF_EXPORT int q17_ref_share_workspace(void *plan, void *owner);
// Host-only native representation packing, input is logical uint4 [K,N].
Q17_REF_EXPORT int q17_ref_pack(int dtype, int n, int k,
                              uint8_t const *logical, uint8_t *packed);
