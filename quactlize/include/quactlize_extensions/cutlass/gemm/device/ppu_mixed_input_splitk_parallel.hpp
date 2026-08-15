/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Dense v1 reduction/device seam for a mixed-input parallel Split-K GEMM.
 *
 * Contract:
 *   - the main kernel writes FP32 partials in [split_k][M][N] row-major order;
 *   - this kernel adds partitions in increasing split_k order;
 *   - alpha is one, beta is zero, and conversion to FP16 happens exactly once;
 *   - the main launch and reduction launch are enqueued on the same stream.
 *
 * This header deliberately does not lower a mixed-input mainloop.  The mainloop owns B/S/Z/B2 and
 * artifact semantics, while this file owns only the partial-workspace ABI and final reduction.  The
 * device reduction primitive is kept independent of the launch kernel so a later last-arriver path
 * can use the identical fixed-order arithmetic without copying it.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/ppu_host_adapter.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_fixed_splitk_partition.hpp"

namespace cutlass::gemm::device::splitk_parallel {

// The producer and consumer share this indexing function.  Keeping the unit (FP32 elements, not
// bytes and not sub-byte logical codes) in the type makes the expert-pitch class of bugs harder to
// recreate at this seam.
CUTLASS_HOST_DEVICE
constexpr int64_t fp32_partial_offset(
    int split_k, int row, int column, int64_t rows, int64_t columns) {
  return (int64_t(split_k) * rows + int64_t(row)) * columns + int64_t(column);
}

inline bool fp32_workspace_size(
    int64_t rows, int64_t columns, int split_k_slices, size_t& bytes) {
  bytes = 0;
  if (rows <= 0 || columns <= 0 ||
      !cutlass::gemm::kernel::fixed_splitk::supported_split_count(
          uint32_t(split_k_slices))) {
    return false;
  }

  size_t const max = (std::numeric_limits<size_t>::max)();
  size_t const m = size_t(rows);
  size_t const n = size_t(columns);
  size_t const s = size_t(split_k_slices);
  if (m > max / n) {
    return false;
  }
  size_t const mn = m * n;
  if (s > max / mn) {
    return false;
  }
  size_t const elements = s * mn;
  if (elements > max / sizeof(float)) {
    return false;
  }
  bytes = elements * sizeof(float);
  return true;
}

// Shared arithmetic for both the standalone reduction kernel and a future fused last-arriver.
// Each valid lane observes precisely s=0,1,...,S-1.  Invalid lanes are zeroed so callers may use
// one vector conversion without reading or writing beyond N.
template <int ElementsPerAccess>
CUTLASS_DEVICE Array<float, ElementsPerAccess> reduce_fp32_fixed_s_order(
    float const* workspace, int partitions, int64_t partition_stride,
    int64_t element_offset, int valid_elements = ElementsPerAccess) {
  static_assert(ElementsPerAccess > 0, "the reduction vector must be non-empty");

  Array<float, ElementsPerAccess> accumulator;
  accumulator.clear();

  CUTLASS_PRAGMA_NO_UNROLL
  for (int split_k = 0; split_k < partitions; ++split_k) {
    float const* partial = workspace + int64_t(split_k) * partition_stride + element_offset;
    CUTLASS_PRAGMA_UNROLL
    for (int lane = 0; lane < ElementsPerAccess; ++lane) {
      if (lane < valid_elements) {
        accumulator[lane] += partial[lane];
      }
    }
  }
  return accumulator;
}

template <int ElementsPerAccess>
class PpuMixedInputSplitKParallelReductionKernel {
 public:
  static_assert(ElementsPerAccess > 0 && ElementsPerAccess <= 16,
                "v1 supports a compact 1..16-element output vector");

  static constexpr int kRowsPerCta = 4;
  static constexpr int kThreadsPerRow = 32;
  static constexpr int kColumnsPerCta = kThreadsPerRow * ElementsPerAccess;

  struct Params {
    float const* workspace = nullptr;
    half_t* destination = nullptr;
    int64_t rows = 0;
    int64_t columns = 0;
    int64_t destination_stride = 0;
    int partitions = 0;
    int64_t partition_stride = 0;  // FP32 elements; v1 requires rows * columns.
  };

  struct SharedStorage {};

  CUTLASS_HOST_DEVICE
  static dim3 block_shape() {
    return dim3(kThreadsPerRow, kRowsPerCta, 1);
  }

  CUTLASS_HOST_DEVICE
  static dim3 grid_shape(int64_t rows, int64_t columns) {
    return dim3(
        unsigned((rows + kRowsPerCta - 1) / kRowsPerCta),
        unsigned((columns + kColumnsPerCta - 1) / kColumnsPerCta), 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage&) {
    int64_t const row =
        int64_t(blockIdx.x) * kRowsPerCta + int64_t(threadIdx.y);
    int64_t const column =
        int64_t(blockIdx.y) * kColumnsPerCta +
        int64_t(threadIdx.x) * ElementsPerAccess;
    if (row >= params.rows || column >= params.columns) {
      return;
    }

    int const valid = int(
        params.columns - column < ElementsPerAccess
            ? params.columns - column
            : ElementsPerAccess);
    int64_t const element_offset = row * params.columns + column;
    Array<float, ElementsPerAccess> const accumulator =
        reduce_fp32_fixed_s_order<ElementsPerAccess>(
            params.workspace, params.partitions, params.partition_stride,
            element_offset, valid);

    NumericArrayConverter<half_t, float, ElementsPerAccess,
                          FloatRoundStyle::round_to_nearest>
        convert;
    Array<half_t, ElementsPerAccess> const output = convert(accumulator);
    half_t* destination =
        params.destination + row * params.destination_stride + column;
    CUTLASS_PRAGMA_UNROLL
    for (int lane = 0; lane < ElementsPerAccess; ++lane) {
      if (lane < valid) {
        destination[lane] = output[lane];
      }
    }
  }
};

template <int ElementsPerAccess = 8>
class PpuMixedInputSplitKParallelReduction {
 public:
  using Kernel = PpuMixedInputSplitKParallelReductionKernel<ElementsPerAccess>;
  using Params = typename Kernel::Params;

  struct Arguments {
    int64_t rows = 0;
    int64_t columns = 0;
    int split_k_slices = 0;
    float const* workspace = nullptr;
    size_t workspace_bytes = 0;
    half_t* destination = nullptr;
    int64_t destination_stride = 0;
  };

 private:
  Params params_{};
  bool installed_ = false;

  static bool half_output_bytes(Arguments const& args, size_t& bytes) {
    bytes = 0;
    size_t const max = (std::numeric_limits<size_t>::max)();
    size_t const rows = size_t(args.rows);
    size_t const columns = size_t(args.columns);
    size_t const stride = size_t(args.destination_stride);
    if (rows == 0 || columns == 0 || stride < columns ||
        rows - 1 > max / stride) {
      return false;
    }
    size_t const last_row = (rows - 1) * stride;
    if (columns > max - last_row) {
      return false;
    }
    size_t const elements = last_row + columns;
    if (elements > max / sizeof(half_t)) {
      return false;
    }
    bytes = elements * sizeof(half_t);
    return true;
  }

  static bool ranges_do_not_overlap(
      void const* lhs, size_t lhs_bytes, void const* rhs, size_t rhs_bytes) {
    uintptr_t const max = (std::numeric_limits<uintptr_t>::max)();
    uintptr_t const lhs_begin = reinterpret_cast<uintptr_t>(lhs);
    uintptr_t const rhs_begin = reinterpret_cast<uintptr_t>(rhs);
    if (lhs_bytes > max - lhs_begin || rhs_bytes > max - rhs_begin) {
      return false;
    }
    uintptr_t const lhs_end = lhs_begin + lhs_bytes;
    uintptr_t const rhs_end = rhs_begin + rhs_bytes;
    return lhs_end <= rhs_begin || rhs_end <= lhs_begin;
  }

 public:
  static size_t get_workspace_size(Arguments const& args) {
    size_t bytes = 0;
    return fp32_workspace_size(
               args.rows, args.columns, args.split_k_slices, bytes)
        ? bytes
        : 0;
  }

  static Status can_implement(Arguments const& args) {
    size_t required = 0;
    if (!fp32_workspace_size(
            args.rows, args.columns, args.split_k_slices, required)) {
      return Status::kErrorInvalidProblem;
    }
    if (args.workspace == nullptr) {
      return Status::kErrorWorkspaceNull;
    }
    if (args.destination == nullptr) {
      return Status::kErrorInvalidProblem;
    }
    if (args.destination_stride < args.columns ||
        args.workspace_bytes < required) {
      return Status::kErrorInvalidProblem;
    }
    if (args.rows > (std::numeric_limits<int64_t>::max)() / args.columns) {
      return Status::kErrorInvalidProblem;
    }
    uint64_t const grid_x = (uint64_t(args.rows) - 1) / Kernel::kRowsPerCta + 1;
    uint64_t const grid_y =
        (uint64_t(args.columns) - 1) / Kernel::kColumnsPerCta + 1;
    if (grid_x > uint64_t((std::numeric_limits<unsigned>::max)()) ||
        grid_y > uint64_t((std::numeric_limits<unsigned>::max)())) {
      return Status::kErrorInvalidProblem;
    }
    if ((reinterpret_cast<uintptr_t>(args.workspace) % 16) != 0 ||
        (reinterpret_cast<uintptr_t>(args.destination) % alignof(half_t)) != 0) {
      return Status::kErrorMisalignedOperand;
    }
    size_t destination_bytes = 0;
    if (!half_output_bytes(args, destination_bytes) ||
        !ranges_do_not_overlap(
            args.workspace, required, args.destination, destination_bytes)) {
      return Status::kErrorInvalidProblem;
    }
    return Status::kSuccess;
  }

  Status initialize(Arguments const& args) {
    installed_ = false;
    Status const status = can_implement(args);
    if (status != Status::kSuccess) {
      return status;
    }
    params_ = Params{
        args.workspace,
        args.destination,
        args.rows,
        args.columns,
        args.destination_stride,
        args.split_k_slices,
        args.rows * args.columns};
    installed_ = true;
    return Status::kSuccess;
  }

  Status run(
      hggcStream_t stream = nullptr, HostAdapter* host_adapter = nullptr,
      int32_t kernel_index = 0) {
    if (!installed_) {
      return Status::kErrorInvalidProblem;
    }

    dim3 const grid = Kernel::grid_shape(params_.rows, params_.columns);
    dim3 const block = Kernel::block_shape();
    if constexpr (CUTLASS_ENABLE_HOST_ADAPTER) {
      if (host_adapter == nullptr) {
        return Status::kErrorInvalidProblem;
      }
      void* kernel_params[] = {&params_};
      Status const launch_status = host_adapter->launch(
          grid, dim3(1, 1, 1), block, 0, stream, kernel_params,
          kernel_index);
      if (launch_status != Status::kSuccess) {
        return launch_status;
      }
    } else {
      if (host_adapter != nullptr) {
        return Status::kErrorInvalidProblem;
      }
      cutlass::arch::synclog_setup();
      cutlass::Kernel<Kernel><<<grid, block, 0, stream>>>(params_);
    }

    hggcError_t const result = hggcGetLastError();
    return result == hggcSuccess ? Status::kSuccess
                                 : Status::kErrorInternal;
  }

  Status run(
      Arguments const& args, hggcStream_t stream = nullptr,
      HostAdapter* host_adapter = nullptr, int32_t kernel_index = 0) {
    Status const status = initialize(args);
    return status == Status::kSuccess
        ? run(stream, host_adapter, kernel_index)
        : status;
  }

  Params const& params() const = delete;
};

// Explicit same-stream two-launch seam.  MainLaunch must enqueue the mixed-input GEMM on the stream
// it receives and return cutlass::Status.  No event or host synchronization is inserted here: stream
// order is the producer/consumer edge for the FP32 workspace.
template <class MainLaunch, class ReductionHandle>
Status launch_main_then_reduce_same_stream(
    MainLaunch&& launch_main, ReductionHandle& reduction,
    hggcStream_t stream = nullptr, HostAdapter* host_adapter = nullptr,
    int32_t reduction_kernel_index = 0) {
  Status const main_status = std::forward<MainLaunch>(launch_main)(stream);
  if (main_status != Status::kSuccess) {
    return main_status;
  }
  return reduction.run(
      stream, host_adapter, reduction_kernel_index);
}

}  // namespace cutlass::gemm::device::splitk_parallel
