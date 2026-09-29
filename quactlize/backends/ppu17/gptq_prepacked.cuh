#pragma once

#include "target.cuh"
#include "marlin_scheduler.cuh"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "libtorch_stable/cutlass_extensions/vllm_numeric_conversion.cuh"
#include "libtorch_stable/quantization/machete/machete_prepacked_layout.cuh"

namespace quactlize::ppu17 {

// Experimental reuse of the pinned Machete reader/collective on PPU
// CUTLASS4.3, not on the reference's CUTLASS. No torch host API is used.
// The private MMA-aligned layout preserves every original GPTQ code and
// FP16 scale bit; it is NOT a new GGUF format or canonical K-pack replacement.
// Keep the conventional reader as an independent baseline.
template<int N, int M, int ClusterN = 1,
         class Scheduler = cutlass::gemm::StreamKScheduler, int Stages = 0>
struct PrepackedGptq {
  using T = cutlass::half_t;
  using Code = cutlass::vllm_uint4b8_t;
  using Schedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using Tile = cute::Shape<cute::Int<N>, cute::Int<M>, cute::_64>;
  using Cluster = cute::Shape<cute::Int<ClusterN>, cute::_1, cute::_1>;
  using Packing = machete::PrepackedLayoutBTemplate<T, Code, T, float,
      cutlass::layout::ColumnMajor, Schedule>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, cutlass::layout::ColumnMajor, 0, T, cutlass::layout::ColumnMajor, 8,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>>::CollectiveOp;
  using Mainloop = machete::MacheteCollectiveMma<cute::tuple<Code,T>, Packing, 32,
      T, cutlass::layout::ColumnMajor, 8, float, Tile, Cluster,
      std::conditional_t<Stages == 0,
          cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
          cutlass::gemm::collective::StageCount<Stages>>, Schedule>;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>, Mainloop, Epilogue, Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
} // namespace quactlize::ppu17
