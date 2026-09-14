#pragma once
#include "../runtime/kernel_types.cuh"

namespace quactlize::decode {
template<class Mainloop,class Source> struct InputCollective;
template<class Arch,class Dispatch,class Tile,class A,class SA,class B,class SB,
    class Mma,class GA,class SLA,class SCA,class TA,class GB,class SLB,class SCB,class TB,class Source>
struct InputCollective<cutlass::gemm::collective::CollectiveMma<Arch,Dispatch,Tile,A,SA,B,SB,
    Mma,GA,SLA,SCA,TA,GB,SLB,SCB,TB>,Source> {
  using type=cutlass::gemm::collective::CollectiveMma<Arch,Dispatch,Tile,A,SA,B,SB,
      Mma,GA,SLA,SCA,cutlass::gemm::collective::detail::DecodeInput<Source>,GB,SLB,SCB,TB>;
};

template<class Core,class Source,class Output> struct DenseTypes {
  using Base=typename Core::Shipping;
  using Tile=typename Core::Tile;
  using Warp=typename Core::Warp;
  struct Shipping:Base {
    using CollectiveMainloop=typename InputCollective<typename Base::CollectiveMainloop,Source>::type;
    using ElementC=Output;
    using ElementD=Output;
    using CollectiveEpilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::PPU0010,cutlass::arch::OpClassTensorOp,Tile,Warp,
        cutlass::epilogue::collective::EpilogueTileAuto,float,float,
        Output,cutlass::layout::RowMajor,16/sizeof(Output),
        Output,cutlass::layout::RowMajor,16/sizeof(Output),
        cutlass::epilogue::EpilogueSimtVectorizedWithoutEvt>::CollectiveOp;
    using GemmKernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,
        CollectiveMainloop,CollectiveEpilogue,cutlass::gemm::SplitKSerialScheduler>;
    using Gemm=cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  };
  using Mainloop=typename Shipping::CollectiveMainloop;
  using Split=dense_splitk_parallel_ppu::KernelTypes<Shipping,Tile,Warp>;
  using PersistentKernel=cutlass::gemm::kernel::PersistentMixedInputKernel<
      cute::Shape<int,int,int,int>,Mainloop,typename Shipping::CollectiveEpilogue>;
  using PersistentGemm=cutlass::gemm::device::GemmUniversalAdapter<PersistentKernel>;
};
} // namespace quactlize::decode
