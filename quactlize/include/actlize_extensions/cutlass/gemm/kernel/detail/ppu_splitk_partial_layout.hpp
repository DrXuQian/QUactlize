/***************************************************************************************************
 * Copyright (c) 2026 Quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * One CuTe layout authority for the fixed Split-K FP32 workspace.
 *
 * The physical ABI is [M,N,S] with row-major N and split-major planes:
 *
 *   stride = (N, 1, M*N)
 *   offset(m,n,s) = (s*M + m)*N + n
 *
 * Do not derive this stride with make_cute_packed_stride(..., S==1): CuTe is
 * allowed to canonicalize an unused singleton mode to stride zero.  Split-K's
 * L/S mode names a physical plane even for the diagnostic S==1 control, so its
 * pitch is part of the ABI rather than an optimization of a logical tensor.
 **************************************************************************************************/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel::detail {

CUTLASS_HOST_DEVICE constexpr int64_t fp32_partial_linear_offset(
    int split, int row, int column, int64_t rows, int64_t columns) {
  return (int64_t(split) * rows + int64_t(row)) * columns +
      int64_t(column);
}

template <class Stride>
CUTLASS_HOST_DEVICE constexpr Stride make_compact_fp32_partial_stride(
    int64_t rows, int64_t columns) {
  static_assert(cute::rank(Stride{}) == 3,
                "fixed Split-K partial stride must be rank-3 [M,N,S]");
  static_assert(cute::is_static<decltype(cute::get<1>(Stride{}))>::value &&
                    int(cute::get<1>(Stride{})) == 1,
                "fixed Split-K partial N mode must be statically contiguous");
  Stride stride{};
  cute::get<0>(stride) = columns;
  cute::get<2>(stride) = rows * columns;
  return stride;
}

template <class Stride>
CUTLASS_HOST_DEVICE constexpr bool is_compact_fp32_partial_stride(
    Stride const& stride, int64_t rows, int64_t columns) {
  return rows > 0 && columns > 0 &&
      int64_t(cute::get<0>(stride)) == columns &&
      int64_t(cute::get<1>(stride)) == 1 &&
      int64_t(cute::get<2>(stride)) == rows * columns;
}

template <class Stride>
CUTLASS_HOST_DEVICE constexpr int64_t fp32_partial_cute_offset(
    Stride const& stride, int64_t rows, int64_t columns, int splits,
    int row, int column, int split) {
  auto layout = cute::make_layout(
      cute::make_shape(rows, columns, splits), stride);
  return int64_t(layout(cute::make_coord(row, column, split)));
}

}  // namespace cutlass::gemm::kernel::detail
