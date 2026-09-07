#pragma once

#include "fpA_intB_ppu.cuh"
#include "dense_splitk_multiformat_ppu.cuh"
#include "moe_grouped_ppu.cuh"
#include "ppu_format_config.hpp"
#include "ppu_group_schedule.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_persistent.hpp"

namespace quactlize::runtime {
using Half = cutlass::half_t;
inline constexpr auto mode = ppu_mixed_policy::QuantMode::FinegrainedScaleZero;

template<int Q> struct Format {
  static constexpr auto spec = ppu_formats::for_qtype(Q);
  static_assert(Q >= 10 && Q <= 14);
  using Low = std::conditional_t<spec.low_bits == 2, cutlass::uint2b_t, cutlass::int4b_t>;
  using High = std::conditional_t<spec.high_bits == 0, void,
      std::conditional_t<spec.high_bits == 1, cutlass::uint1b_t, cutlass::uint2b_t>>;
};

template<int Q, int TM, int TN, int TK, int WM, int WN, int ST, int AP, int DN>
struct DenseTypes {
  using F = Format<Q>;
  using Low = typename F::Low;
  using High = typename F::High;
  using Schedule = ppu_group_schedule::FinegrainedSchedule<F::spec.group_size>;
  using Tile = cute::Shape<cute::C<TM>, cute::C<TN>, cute::C<TK>>;
  using ScaleTile = cute::Shape<cute::C<TN>, cute::C<
      ppu_group_schedule::scale_groups_v<TK, F::spec.group_size>>>;
  using Warp = cute::Shape<cute::C<WM>, cute::C<WN>, cute::C<TK>>;
  using Shipping = std::conditional_t<Q == 12,
      fpa_intb_ppu::DenseQ4KPack4KernelTypes<mode, Schedule, Tile, ScaleTile, Warp, ST, true, AP, DN>,
      fpa_intb_ppu::DenseKPackKernelTypes<mode, Schedule, Tile, ScaleTile, Warp, ST, true, Low, High, AP, DN>>;
  using Mainloop = typename Shipping::CollectiveMainloop;
  using PersistentKernel = cutlass::gemm::kernel::PersistentMixedInputKernel<
      cute::Shape<int,int,int,int>, Mainloop, typename Shipping::CollectiveEpilogue>;
  using PersistentGemm = cutlass::gemm::device::GemmUniversalAdapter<PersistentKernel>;
  using Prepared = dense_splitk_parallel_ppu::PreparedMultiformatLauncher<
      Shipping, Tile, Warp, High, (PPU_PACKED_SCALE != 0)>;
};

template<int Q, int TM, int TN, int TK, int WM, int WN, int ST, int DN, bool Persistent>
struct GroupedTypes {
  using F = Format<Q>;
  using Low = typename F::Low;
  using High = typename F::High;
  using Schedule = ppu_group_schedule::FinegrainedSchedule<F::spec.group_size>;
  using Tile = cute::Shape<cute::C<TM>, cute::C<TN>, cute::C<TK>>;
  using ScaleTile = cute::Shape<cute::C<TN>, cute::C<
      ppu_group_schedule::scale_groups_v<TK, F::spec.group_size>>>;
  using Warp = cute::Shape<cute::C<WM>, cute::C<WN>, cute::C<TK>>;
  // Both measured grouped routes use separate-half publication. Dense has
  // its own factory default; do not infer this choice from PackedScale.
  using Publication = cutlass::gemm::SeparateHalfPlanes;
  using Policy = std::conditional_t<Q == 12,
      ppu_mixed_policy::Q4KPack4MainloopPolicy<mode, Schedule, Tile, ScaleTile, Warp, ST, true, 0, DN, Publication>,
      ppu_mixed_policy::KPackMainloopPolicy<mode, Schedule, Tile, ScaleTile, Warp, ST, true, Low, High, 0, DN>>;
  using Mainloop = typename Policy::CollectiveOp;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp, Tile, Warp,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      Half, cutlass::layout::RowMajor*, 8, Half, cutlass::layout::RowMajor*, 8,
      cutlass::epilogue::EpiloguePtrArraySimtVectorized,
      cutlass::epilogue::fusion::LinearCombination<Half, float>>::CollectiveOp;
  using Kernel = std::conditional_t<Persistent,
      cutlass::gemm::kernel::GroupPersistentMixedInputKernel<moe_grouped_ppu::GroupProblemShape, Mainloop, Epilogue>,
      cutlass::gemm::kernel::GemmUniversal<moe_grouped_ppu::GroupProblemShape, Mainloop, Epilogue>>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static_assert(cute::size<0>(typename Epilogue::SmemLayout{}) ==
      cute::size<0>(typename Mainloop::TiledMma::AtomShape_MNK{}) *
      cute::size<1>(typename Mainloop::TiledMma::ThrLayoutVMNK{}));
};
}  // namespace quactlize::runtime
