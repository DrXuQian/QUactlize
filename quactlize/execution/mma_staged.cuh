#pragma once
// Runtime adapter only; the optional collective owns staged delivery and arithmetic.
#include "mma_reader.cuh"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_exact_code_staged.hpp"

namespace quactlize::execution::mma_reader {
template<int Q,int CtaN,int WK,int S>
constexpr size_t staged_shared_bytes() {
    return size_t(S)*ec::exact_code_ppu_detail::StageLayout<
        Q,CtaN,ec::exact_code_ppu_detail::kStageSb<Q,WK>>::Bytes;
}

template<int Q,int Compute,int T,int WN,int WK,int Tokens,int S>
using StagedCollective=ec::ExactCodeCollectivePPU<Q,
    std::conditional_t<Compute==0,cutlass::half_t,cutlass::bfloat16_t>,
    T,WN,WK,Tokens,ec::ExactCodeCpAsyncPPU<S>>;

template<int Q,int Mode,int Compute,int T,int WN,int WK,int Tokens,int S>
__global__ void __launch_bounds__(WN*WK*32) staged_kernel(qkg_call_v1 call,int split) {
    extern __shared__ __align__(128) uint8_t storage[];
    run_tile<Mode,StagedCollective<Q,Compute,T,WN,WK,Tokens,S>>(call,split,storage);
}

template<int Q,int Mode,int Compute,int T,int WN,int WK,int Tokens,int S>
int launch_staged(qkg_call_v1 const& c,int split) {
    static_assert(Compute==0 || Compute==1);
    using Collective=StagedCollective<Q,Compute,T,WN,WK,Tokens,S>;
    return launch<Mode,Collective>(c,split,staged_kernel<Q,Mode,Compute,T,WN,WK,Tokens,S>);
}
} // namespace quactlize::execution::mma_reader
