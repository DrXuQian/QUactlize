// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Isolated host/type builder for the grouped Stream-K mechanism gate.  It is
// intentionally not included by the production grouped route: production has
// only device-resident row counts and cannot form the exact tactic-dependent
// q prefix without synchronising its asynchronous ABI.
#pragma once

#include <cstdint>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "fold_traits.hpp"
#include "ppu_group_schedule.hpp"
#include "ppu_mixed_policy.hpp"
#include "ppu_tactic_space.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_streamk.hpp"

namespace moe_grouped_streamk_ppu {

using QuantMode = ppu_mixed_policy::QuantMode;
using GroupShape = cute::Shape<int, int, int>;
using GroupProblemShape = cutlass::gemm::GroupProblemShape<GroupShape>;
using DStride = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

struct Plan {
  int q = 0;
  int kt = 0;
  int real_cu = 0;
  int ctas_per_cu = 0;
  int workers = 0;
  uint32_t sk_tiles = 0;
  uint64_t sk_units = 0;
  uint64_t units_per_problem = 0;
  int splits = 0;
  uint32_t separate_reduction_units = 0;
  size_t workspace_bytes = 0;
  size_t scheduler_workspace_bytes = 0;
  size_t scheduler_barrier_bytes = 0;
};

template <QuantMode QuantOp, class BaseSchedule, class TileShape,
          class ScaleTileShape, class WarpShape, int Stages,
          bool AiuInterleaved, class ElementB = cutlass::int4b_t,
          class PlaneB2 = void, int ArtifactTileK = 0,
          uint32_t MinSkIters = 8u>
struct Operation {
  using MainloopPolicy = ppu_mixed_policy::MainloopPolicy<
      QuantOp, BaseSchedule, TileShape, ScaleTileShape, WarpShape, Stages,
      AiuInterleaved, ElementB, PlaneB2, ArtifactTileK>;
  using CollectiveMainloop = typename MainloopPolicy::CollectiveOp;
  using ElementA = typename MainloopPolicy::ElementA;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ClusterShape = WarpShape;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using EpilogueSchedule = cutlass::epilogue::EpiloguePtrArraySimtVectorized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, OperatorClass, TileShape, ClusterShape,
      EpilogueTileType, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC, ElementD, LayoutD*, AlignmentD,
      EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<
          ElementC, ElementAccumulator>>::CollectiveOp;
  static_assert(
      cute::size<0>(typename CollectiveEpilogue::SmemLayout{}) ==
          cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}) *
              cute::size<1>(typename CollectiveMainloop::TiledMma::ThrLayoutVMNK{}),
      "grouped Stream-K epilogue M layout must match MMA atom and M warps");
  using Kernel = cutlass::gemm::kernel::GroupStreamKMixedInputKernel<
      GroupProblemShape, CollectiveMainloop, CollectiveEpilogue, MinSkIters>;
 private:
  using RawGemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

 public:
  // This isolated handle deliberately exposes no update(): the generic adapter
  // relowers Params without reinstalling the ragged prefix/shape mirror. A
  // private base prevents callers from upcasting around the deleted seam.
  class Gemm : private RawGemm {
   public:
    using Arguments = typename RawGemm::Arguments;
    using Params = typename RawGemm::Params;
    using RawGemm::can_implement;
    using RawGemm::get_grid_shape;
    using RawGemm::get_workspace_size;
    using RawGemm::initialize;
    using RawGemm::maximum_active_blocks;
    using RawGemm::params;

    cutlass::Status run(hggcStream_t stream = nullptr,
                        cutlass::HostAdapter* host_adapter = nullptr,
                        bool launch_with_pdl = false) {
      return RawGemm::run(stream, host_adapter, launch_with_pdl);
    }

    cutlass::Status update(Arguments const&, void* = nullptr) = delete;
  };
  using Arguments = typename Gemm::Arguments;
  using Params = typename Gemm::Params;
  using Census = typename Kernel::Census;
  using TileSchedulerParams = typename Kernel::TileSchedulerParams;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  using StrideS = typename CollectiveMainloop::StrideScale;

  static constexpr int LowBits = MainloopPolicy::LowBits;
  static constexpr int LowFold = MainloopPolicy::ArtifactLowFold;
  static constexpr int ExpectedGroupSize =
      CollectiveMainloop::DispatchPolicy::StaticGroupSize;
  static_assert(ExpectedGroupSize > 0,
                "the isolated grouped Stream-K gate requires static gs");
  static_assert(std::is_same_v<DStride, cute::remove_pointer_t<StrideC>> &&
                    std::is_same_v<DStride, cute::remove_pointer_t<StrideD>>,
                "grouped Stream-K C/D stride ABI drifted");
  static_assert(ppu_mixed_policy::kernel_policy_valid_v<
                    ppu_tactics::GroupedSpace, MainloopPolicy>);

  static Arguments make_arguments(
      ElementA const* A, ElementB const* B,
      cutlass::half_t const* scales, cutlass::half_t const* zeros,
      ElementC const** ptr_C, DStride* stride_C,
      ElementD** ptr_D, DStride* stride_D,
      int m_max, int n, int k, int groups,
      GroupShape* group_shapes_device, GroupShape const* group_shapes_host,
      int const* group_row_offsets, float alpha, float beta,
      PlaneB2 const* B2 = nullptr) {
    static_assert(ArtifactTileK > 0,
                  "the isolated Stream-K gate requires an explicit artifact TileK");
    int const scale_k = (k + ExpectedGroupSize - 1) / ExpectedGroupSize;
    StrideA sA = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(m_max, k, groups));
    StrideB sB = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(n / LowFold, k * LowFold, groups));
    StrideS sS = cutlass::make_cute_packed_stride(
        StrideS{}, cute::make_shape(n, scale_k, groups));
    GroupProblemShape ps;
    ps.num_groups = groups;
    ps.problem_shapes = group_shapes_device;
    ps.host_problem_shapes = group_shapes_host;

    Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        ps,
        {A, sA, B, sB, scales, sS, ExpectedGroupSize, zeros,
         group_row_offsets},
        {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D},
        cutlass::KernelHardwareInfo{},
    };
    args.representative_m = m_max;
    args.representative_n = n;
    args.representative_k = k;
    args.domain_valid =
        !AiuInterleaved || (n % 256 == 0 && k % 256 == 0);
    if constexpr (!std::is_void_v<PlaneB2>) {
      args.mainloop.ptr_B2 = B2;
      constexpr int HighFold = MainloopPolicy::ArtifactHighFold;
      if constexpr (HighFold != LowFold) {
        args.mainloop.dB2 = cutlass::make_cute_packed_stride(
            StrideB{}, cute::make_shape(n / HighFold, k * HighFold, groups));
        args.mainloop.dB2_valid = true;
      }
    } else {
      (void)B2;
    }
    return args;
  }

  static void configure_runtime(Arguments& args, int device_id, int real_cu,
                                int ctas_per_cu, Census census = {}) {
    args.hw_info = cutlass::KernelHardwareInfo{device_id, real_cu};
    args.ctas_per_cu = ctas_per_cu;
    args.scheduler.splits = 1;
    args.scheduler.max_swizzle_size = 1;
    args.scheduler.raster_order = Kernel::TileScheduler::RasterOrderOptions::AlongN;
    args.scheduler.reduction_mode =
        TileSchedulerParams::ReductionMode::Deterministic;
    args.scheduler.decomposition_mode =
        TileSchedulerParams::DecompositionMode::StreamK;
    args.census = census;
  }

  static Plan inspect(Arguments const& args, void* workspace) {
    if (Gemm::can_implement(args) != cutlass::Status::kSuccess) {
      return {};
    }
    Params params = Kernel::to_underlying_arguments(args, workspace);
    Plan out;
    out.q = params.output_tiles;
    out.kt = params.k_tiles_per_output;
    out.real_cu = params.real_hw_info.cu_count;
    out.ctas_per_cu = params.ctas_per_cu;
    out.workers = params.scheduler_hw_info.cu_count;
    out.sk_tiles = params.scheduler.sk_tiles_;
    out.sk_units = params.scheduler.sk_units_;
    out.units_per_problem = params.scheduler.units_per_problem_;
    out.splits = params.scheduler.divmod_splits_.divisor;
    out.separate_reduction_units = params.scheduler.separate_reduction_units_;
    out.workspace_bytes = Gemm::get_workspace_size(args);
    out.scheduler_workspace_bytes = params.scheduler_workspace_bytes;
    out.scheduler_barrier_bytes = params.scheduler_barrier_bytes;
    return out;
  }
};

}  // namespace moe_grouped_streamk_ppu
