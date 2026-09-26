#pragma once
#include "format.cuh"

namespace quactlize::ppu17 {

CUTLASS_HOST_DEVICE float load_element(void const *p, int type, int64_t index) {
  if (type == Q17_F32) return static_cast<float const *>(p)[index];
  if (type == Q17_BF16) return float(static_cast<cutlass::bfloat16_t const *>(p)[index]);
  return float(static_cast<cutlass::half_t const *>(p)[index]);
}
CUTLASS_HOST_DEVICE void store_element(void *p, int type, int64_t index, float value) {
  if (type == Q17_F32) static_cast<float *>(p)[index] = value;
  else if (type == Q17_BF16) static_cast<cutlass::bfloat16_t *>(p)[index] = cutlass::bfloat16_t(value);
  else static_cast<cutlass::half_t *>(p)[index] = cutlass::half_t(value);
}

struct RowRange { int begin, size; };
CUTLASS_HOST_DEVICE RowRange row_range(q17_call_v1 const &c, int expert) {
  if (c.problem.operation == Q17_DENSE) return {0, c.problem.m};
  int begin = c.offsets[expert], end = c.offsets[expert + 1];
  // Offset validity is the caller's contract. Never access outside A/C even
  // if a bad directory is supplied; a device numerical gate must still fail.
  if (begin < 0 || end < begin || end > c.problem.m || end - begin > c.problem.max_rows)
    return {0, 0};
  return {begin, end - begin};
}
} // namespace quactlize::ppu17
