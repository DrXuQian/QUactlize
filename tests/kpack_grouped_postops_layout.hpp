#pragma once
#include "cute/tensor.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cutlass/ppu_host_adapter.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/detail/ppu_grouped_splitk_direct_epilogue.hpp"

// A CUDA-compilable projection of the production PPU C-coordinate types.
// kpack_grouped_postops_types.cu checks this projection against the actual
// five-format runtime types with hgcc. No mainloop or PPU opcode is emulated.
template<int TM> struct PostopsLayout {
  using Atom = std::conditional_t<TM==8,cute::PPU0010_8x16x16_F32F16F16F32_TN,
      cute::PPU0010_16x16x16_F32F16F16F32_TN>;
  using Mma = cute::TiledMMA<cute::MMA_Atom<Atom>,cute::Layout<cute::Shape<cute::_1,cute::_4,cute::_1>>,
      cute::Tile<cute::Int<TM>,cute::_64,cute::_64>>;
  struct DestinationLayout {
    using ElementD = float;
    using StrideC = cute::Stride<int64_t,cute::_1,cute::_0>*;
    using StrideD = StrideC;
    using SmemLayout = cute::Layout<cute::Shape<cute::Int<TM>,cute::_64>>;
    using ThreadEpilogueOp = void;
    struct SharedStorage {};
  };
  using Epilogue = cutlass::gemm::kernel::detail::GroupedSplitKDirectEpilogue<DestinationLayout>;
  using Tile = cute::Shape<cute::Int<TM>,cute::_64,cute::_256>;
  struct Mainloop { using TiledMma = Mma; };
};
