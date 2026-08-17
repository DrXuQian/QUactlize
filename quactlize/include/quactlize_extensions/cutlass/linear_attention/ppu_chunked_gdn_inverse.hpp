/***************************************************************************************************
 * Copyright (c) 2026 quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Scalar authority for the unit-lower-triangular inverse used by PPU GDN.
 * The production collective replaces the block products with PPU AIU MMA but
 * is required to preserve this 16 -> 32 -> 64 decomposition exactly.
 **************************************************************************************************/
#pragma once

#include <cstddef>

namespace cutlass::linear_attention::detail {

inline void invert_unit_lower_direct(
    float const* strict_lower, float* inverse, int leading_dim, int begin, int size) {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      inverse[(begin + i) * leading_dim + begin + j] = i == j ? 1.0f : 0.0f;
    }
  }
  for (int i = 1; i < size; ++i) {
    for (int j = 0; j < i; ++j) {
      float sum = 0.0f;
      for (int k = j; k < i; ++k) {
        sum += strict_lower[(begin + i) * leading_dim + begin + k] *
               inverse[(begin + k) * leading_dim + begin + j];
      }
      inverse[(begin + i) * leading_dim + begin + j] = -sum;
    }
  }
}

// For L = [[A,0],[C,D]], inv(L) =
// [[inv(A),0],[-inv(D) C inv(A),inv(D)]].  Recursing at 16 rows is
// inclusionAI/cuLA's chunk-64 inverse structure; this scalar form is the
// independent authority used by the host gate.
inline void invert_unit_lower_blocked_impl(
    float const* strict_lower, float* inverse, int leading_dim, int begin, int size, int base) {
  if (size <= base) {
    invert_unit_lower_direct(strict_lower, inverse, leading_dim, begin, size);
    return;
  }
  int const half = size / 2;
  invert_unit_lower_blocked_impl(strict_lower, inverse, leading_dim, begin, half, base);
  invert_unit_lower_blocked_impl(strict_lower, inverse, leading_dim, begin + half, half, base);

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      float value = 0.0f;
      for (int r = 0; r < half; ++r) {
        float right = 0.0f;
        for (int s = 0; s < half; ++s) {
          right += strict_lower[(begin + half + r) * leading_dim + begin + s] *
                   inverse[(begin + s) * leading_dim + begin + j];
        }
        value += inverse[(begin + half + i) * leading_dim + begin + half + r] * right;
      }
      inverse[(begin + half + i) * leading_dim + begin + j] = -value;
    }
  }
}

inline bool invert_unit_lower_blocked(
    float const* strict_lower, float* inverse, int size, int base = 16) {
  if (strict_lower == nullptr || inverse == nullptr || size <= 0 || base <= 0 ||
      size % base != 0 || (size & (size - 1)) != 0) {
    return false;
  }
  for (int i = 0; i < size * size; ++i) inverse[i] = 0.0f;
  invert_unit_lower_blocked_impl(strict_lower, inverse, size, 0, size, base);
  return true;
}

inline float unit_lower_inverse_residual(
    float const* strict_lower, float const* inverse, int size) {
  float max_abs = 0.0f;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      float sum = inverse[i * size + j];
      for (int k = 0; k < i; ++k) {
        sum += strict_lower[i * size + k] * inverse[k * size + j];
      }
      float const want = i == j ? 1.0f : 0.0f;
      float const err = sum > want ? sum - want : want - sum;
      if (err > max_abs) max_abs = err;
    }
  }
  return max_abs;
}

}  // namespace cutlass::linear_attention::detail
