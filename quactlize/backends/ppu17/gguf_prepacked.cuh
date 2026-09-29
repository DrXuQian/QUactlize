#pragma once

#include "target.cuh"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "libtorch_stable/cutlass_extensions/vllm_numeric_conversion.cuh"
#include "libtorch_stable/quantization/machete/machete_prepacked_layout.cuh"
#include "gguf_prepacked_mainloop.cuh"

namespace quactlize::ppu17 {

// Experimental, resident ScaleFirst Q4_K. The private lossless code layout
// is not the canonical K-pack ABI. Metadata retains eight independent affine
// groups per original GGUF superblock; BF16 never passes through FP16 compute.
template<class T, int NTile, int MTile, int ClusterN, int Stages, bool PackedUnits>
struct GgufQ4Prepacked {
  static_assert(std::is_same_v<T, cutlass::half_t> ||
                std::is_same_v<T, cutlass::bfloat16_t>);
  using Code = cutlass::vllm_uint4b8_t;
  using Schedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using Tile = cute::Shape<cute::Int<NTile>, cute::Int<MTile>, cute::_64>;
  using Cluster = cute::Shape<cute::Int<ClusterN>, cute::_1, cute::_1>;
  using Packing = machete::PrepackedLayoutBTemplate<T, Code, T, float,
      cutlass::layout::ColumnMajor, Schedule>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, cutlass::layout::ColumnMajor, 0, T, cutlass::layout::ColumnMajor, 8,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>>::CollectiveOp;
  using Mainloop = detail::GgufPrepackedCollectiveMma<32,
      cute::tuple<Code,T,T>, Packing, 32,
      T, cutlass::layout::ColumnMajor, 8, float, Tile, Cluster,
      cutlass::gemm::collective::StageCount<Stages>, Schedule, PackedUnits>;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>, Mainloop, Epilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
template<class T, int NTile, int MTile, int ClusterN = 1, int Stages = 4>
using GgufQ4Sf = GgufQ4Prepacked<T,NTile,MTile,ClusterN,Stages,false>;
template<class T, int NTile, int MTile, int Stages = 4>
using GgufQ4Fq = GgufQ4Prepacked<T,NTile,MTile,1,Stages,true>;
} // namespace quactlize::ppu17
