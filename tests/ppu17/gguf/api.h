#pragma once
#include <stddef.h>
#include <stdint.h>

// Diagnostic-only private packing; NOT the shipping q17 or K-pack ABI.
// Resident Q4_K SF: original GGUF group32, signed converter q-8, scale/zero
// already expanded with the existing typed GGUF affine correction. dtype0
// is FP16 A/S/Z/C; dtype1 BF16 A/S/Z/C. Multiply/add round separately;
// accumulator and Stream-K fixup remain FP32. S/Z are [K/32,N].
struct q17_gguf_call {
  int32_t dtype, config, m, n, k;
  void const *a, *b, *scales, *zeros;
  void *output, *stream;
};
#define Q17_GGUF_EXPORT extern "C" __attribute__((visibility("default")))
Q17_GGUF_EXPORT char const* q17_gguf_error();
Q17_GGUF_EXPORT char const* q17_gguf_contract();
Q17_GGUF_EXPORT int q17_gguf_prepare(q17_gguf_call const*, void**);
Q17_GGUF_EXPORT int q17_gguf_execute(void*);
Q17_GGUF_EXPORT void q17_gguf_destroy(void*);
Q17_GGUF_EXPORT size_t q17_gguf_workspace(void*);
Q17_GGUF_EXPORT int q17_gguf_pack(int dtype, int n, int k,
                               uint8_t const* logical, uint8_t* packed);
