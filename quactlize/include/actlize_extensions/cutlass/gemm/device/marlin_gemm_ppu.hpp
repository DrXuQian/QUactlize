/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Owned host handle for the standalone Marlin PPU kernel.
 *
 * The generic GemmUniversalAdapter deliberately exposes update(Arguments)
 * and run(Params).  Both bypass the checked Arguments -> Params installation
 * on which MarlinCollectivePPU's assume-valid device path relies.  Keep that
 * raw adapter private and expose only a full initialize followed by launches
 * of the installed Params object.
 **************************************************************************************************/
#pragma once

#include "cutlass/gemm/device/gemm_universal_adapter.h"

namespace cutlass::gemm::device {

namespace detail {

template <class RawGemm_>
class MarlinCheckedHandlePPU {
 private:
  using RawGemm = RawGemm_;

  RawGemm raw_{};
  bool installed_ = false;

 public:
  using GemmKernel = typename RawGemm::GemmKernel;
  using TileShape = typename RawGemm::TileShape;
  using ElementA = typename RawGemm::ElementA;
  using ElementB = typename RawGemm::ElementB;
  using ElementC = typename RawGemm::ElementC;
  using ElementD = typename RawGemm::ElementD;
  using ElementAccumulator = typename RawGemm::ElementAccumulator;
  using CollectiveMainloop = typename RawGemm::CollectiveMainloop;
  using CollectiveEpilogue = typename RawGemm::CollectiveEpilogue;
  using Arguments = typename RawGemm::Arguments;
  using Params = typename RawGemm::Params;

  static Status can_implement(Arguments const& args) {
    return RawGemm::can_implement(args);
  }

  static size_t get_workspace_size(Arguments const& args) {
    return can_implement(args) == Status::kSuccess
        ? RawGemm::get_workspace_size(args)
        : 0;
  }

  static int maximum_active_blocks(int smem_capacity = -1) {
    return RawGemm::maximum_active_blocks(smem_capacity);
  }

  // The raw adapter lowers Arguments even when they are unsupported.  A
  // diagnostic grid query must not manufacture apparently launchable Params.
  static dim3 get_grid_shape(Arguments const& args, void* workspace = nullptr) {
    return can_implement(args) == Status::kSuccess
        ? RawGemm::get_grid_shape(args, workspace)
        : dim3(0, 0, 0);
  }

  static dim3 get_grid_shape(Params const&) = delete;

  Status initialize(
      Arguments const& args, void* workspace = nullptr,
      hggcStream_t stream = nullptr, HostAdapter* host_adapter = nullptr) {
    installed_ = false;
    Status const status = raw_.initialize(
        args, workspace, stream, host_adapter);
    installed_ = status == Status::kSuccess;
    return status;
  }

  Status run(
      hggcStream_t stream = nullptr, HostAdapter* host_adapter = nullptr,
      bool launch_with_pdl = false) {
    if (!installed_) {
      return Status::kErrorInvalidProblem;
    }
    return raw_.run(stream, host_adapter, launch_with_pdl);
  }

  Status run(
      Arguments const& args, void* workspace = nullptr,
      hggcStream_t stream = nullptr, HostAdapter* host_adapter = nullptr,
      bool launch_with_pdl = false) {
    Status const status = initialize(args, workspace, stream, host_adapter);
    return status == Status::kSuccess
        ? run(stream, host_adapter, launch_with_pdl)
        : status;
  }

  Status operator()(
      Arguments const& args, void* workspace = nullptr,
      hggcStream_t stream = nullptr, HostAdapter* host_adapter = nullptr,
      bool launch_with_pdl = false) {
    return run(args, workspace, stream, host_adapter, launch_with_pdl);
  }

  Status operator()(
      hggcStream_t stream = nullptr, HostAdapter* host_adapter = nullptr,
      bool launch_with_pdl = false) {
    return run(stream, host_adapter, launch_with_pdl);
  }

  // These are intentionally declared, rather than merely omitted: accidental
  // calls fail by naming the unsafe seam instead of falling through another
  // overload.  Private composition also prevents an upcast to RawGemm.
  Status update(Arguments const&, void* = nullptr) = delete;
  static Status run(
      Params&, hggcStream_t = nullptr, HostAdapter* = nullptr,
      bool = false) = delete;
  Status operator()(
      Params&, hggcStream_t = nullptr, HostAdapter* = nullptr,
      bool = false) = delete;
  Params const& params() const = delete;
};

}  // namespace detail

template <class GemmKernel_>
using MarlinGemmPPU = detail::MarlinCheckedHandlePPU<
    GemmUniversalAdapter<GemmKernel_>>;

}  // namespace cutlass::gemm::device
