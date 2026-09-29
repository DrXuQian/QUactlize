#pragma once

#include "target.cuh"
#include "marlin_scheduler.cuh"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

namespace quactlize::ppu17 {

// Hopper port of the existing FinegrainedScaleOnly GPTQ contract, NOT a
// GGUF adapter. Logical B[k,n] = half((q_u[k,n]-8) * scales[k/128,n]).
// A/C and scales are FP16, the accumulator is FP32. Native B is signed int4,
// K-contiguous within each N column; the offline pack is lossless. There is
// no additive-zero plane or metadata expansion. This header does not change
// the actlize path, canonical K-pack bytes, or the GGUF-only q17 v1 ABI.
template<int TileN, int TileM, int TileK = 64,
         class Scheduler = cutlass::gemm::StreamKScheduler, int Stages = 0> struct GptqScaleOnly {
  using T = cutlass::half_t;
  using Tile = cute::Shape<cute::Int<TileN>, cute::Int<TileM>, cute::Int<TileK>>;
  using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
  static_assert(TileK <= 128 && 128 % TileK == 0);
  // Explicit B^T * A^T lets the narrow operand use the register-fed RS
  // collective while retaining a TMA epilogue. Output bytes remain [M,N].
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, cutlass::layout::ColumnMajor, 0,
      T, cutlass::layout::ColumnMajor, 8,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cute::tuple<cutlass::int4b_t, T>, cutlass::layout::RowMajor, 32,
      T, cutlass::layout::ColumnMajor, 8, float, Tile, Cluster,
      cute::conditional_t<Stages == 0,
          cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
          cutlass::gemm::collective::StageCount<Stages>>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>, Mainloop, Epilogue, Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

} // namespace quactlize::ppu17
