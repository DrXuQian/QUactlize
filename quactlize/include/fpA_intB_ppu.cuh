// Canonical K-pack dense kernel types shared by S1 and Split-K launchers.
#pragma once

#include <cstdio>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "fold_traits.hpp"
#include "ppu_group_schedule.hpp"
#include "ppu_tactic_space.hpp"

#include "quactlize_actlize.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "ppu_mixed_policy.hpp"
#include "cutlass/epilogue/collective/builders/ppu_builder.inl"

namespace fpa_intb_ppu {
using namespace cute;
using TacticSpace = ppu_tactics::DenseSpace;
using QuantMode = ppu_mixed_policy::QuantMode;

// Exact type authority for the canonical Q4_K K-pack4-transposed weight
// layout.  It intentionally has no ArtifactTileK template argument: every
// TK64/128/256 tactic consumes the same physical bytes.
template <QuantMode QuantOp, class BaseSchedule,
          class TileShape, class ScaleTileShape, class WarpShape,
          int Stages, bool AiuInterleaved, int APackRows = 0,
          int KPack4DeliveryN = 0,
          class MetadataPublication = cutlass::gemm::InterleavedHalf2,
          class Compute = cutlass::half_t>
struct DenseQ4KPack4KernelTypes {
  using MainloopPolicy = ppu_mixed_policy::Q4KPack4MainloopPolicy<
      QuantOp, BaseSchedule, TileShape, ScaleTileShape, WarpShape,
      Stages, AiuInterleaved, APackRows, KPack4DeliveryN,
      MetadataPublication, Compute>;
  using ElementA = typename MainloopPolicy::ElementA;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::half_t;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ClusterShape = WarpShape;
  using EpilogueSchedule = cutlass::epilogue::EpilogueSimtVectorizedWithoutEvt;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, OperatorClass, TileShape, ClusterShape,
      EpilogueTileType, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop = typename MainloopPolicy::CollectiveOp;
  static_assert(
      cute::size<0>(typename CollectiveEpilogue::SmemLayout{}) ==
          cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}) *
              cute::size<1>(typename CollectiveMainloop::TiledMma::ThrLayoutVMNK{}),
      "K-pack4 dense epilogue M ownership must match the mainloop");
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop,
      CollectiveEpilogue, cutlass::gemm::SplitKSerialScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static constexpr size_t SharedStorageSize = GemmKernel::SharedStorageSize;
};

// Exact type authority for the Q2/Q3/Q5/Q6 per-plane b16 K-pack layout.
// Q4 retains DenseQ4KPack4KernelTypes so its historical type identity remains
// unchanged; the kernel/epilogue construction is otherwise intentionally the
// same.
template <QuantMode QuantOp, class BaseSchedule,
          class TileShape, class ScaleTileShape, class WarpShape,
          int Stages, bool AiuInterleaved, class ElementB,
          class PlaneB2 = void, int APackRows = 0,
          int KPackDeliveryN = 0, class Compute = cutlass::half_t>
struct DenseKPackKernelTypes {
  using MainloopPolicy = ppu_mixed_policy::KPackMainloopPolicy<
      QuantOp, BaseSchedule, TileShape, ScaleTileShape, WarpShape,
      Stages, AiuInterleaved, ElementB, PlaneB2, APackRows,
      KPackDeliveryN, Compute>;
  using ElementA = typename MainloopPolicy::ElementA;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::half_t;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ClusterShape = WarpShape;
  using EpilogueSchedule = cutlass::epilogue::EpilogueSimtVectorizedWithoutEvt;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, OperatorClass, TileShape, ClusterShape,
      EpilogueTileType, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop = typename MainloopPolicy::CollectiveOp;
  static_assert(
      cute::size<0>(typename CollectiveEpilogue::SmemLayout{}) ==
          cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}) *
              cute::size<1>(typename CollectiveMainloop::TiledMma::ThrLayoutVMNK{}),
      "K-pack dense epilogue M ownership must match the mainloop");
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop,
      CollectiveEpilogue, cutlass::gemm::SplitKSerialScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static constexpr size_t SharedStorageSize = GemmKernel::SharedStorageSize;
};

} // namespace fpa_intb_ppu
