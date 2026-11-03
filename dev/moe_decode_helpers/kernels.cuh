#pragma once
#include "contract.hpp"
// Experimental helpers only. Canonical bytes, rounding and slot order stay
// unchanged; these entrypoints do not participate in the production selector.
namespace quactlize::moe::decode_helpers {

template<int Threads,int Round>
__global__ void finish_top8(qzm_finish_v1 c) {
    int64_t col=int64_t(blockIdx.x)*Threads+threadIdx.x;
    if(col>=c.columns) return;
    for(int64_t token=blockIdx.y;token<c.tokens;token+=gridDim.y) {
        auto base=static_cast<float const*>(c.down.data)+token*8*c.down.row_stride+col;
        auto weights=c.weights+token*c.weights_stride;
        float sum=0.f;
        #pragma unroll
        for(int slot=0;slot<8;++slot) {
            float v=rounded(base[int64_t(slot)*c.down.row_stride],Round);
            float product=__fmul_rn(v,weights[slot]);
            sum=slot?__fadd_rn(sum,product):product;
        }
        c.output[token*c.output_stride+col]=sum;
    }
}

template<int Threads,bool Snapshot>
__global__ void transition_direct(qzm_transition_v1 c,int nr,float const* weights,
                                  int64_t weights_stride,float* retained) {
    int64_t col=int64_t(blockIdx.x)*Threads+threadIdx.x;
    for(int64_t row=blockIdx.y;row<nr;row+=gridDim.y) {
        if constexpr(Snapshot) {
            // Exactly one writer per token-slot. This launch completes before
            // Down/finish starts; it does not write the caller's final output.
            if(blockIdx.x==0 && threadIdx.x==0)
                retained[row]=weights[(row/c.topk)*weights_stride+row%c.topk];
        }
        if(col>=c.hidden) continue;
        auto input=static_cast<float const*>(c.gate.data)+row*c.gate.row_stride+col;
        float g=input[0],u=input[c.hidden];
        float v=(g/(1.f+expf(-g)))*u;
        static_cast<float*>(c.output)[row*c.output_stride+col]=v;
    }
}
template<int Threads>
int finish_launch(qzm_finish_v1 const& c) {
    dim3 grid((int64_t(c.columns)+Threads-1)/Threads,c.tokens>65535?65535:c.tokens);
    auto stream=static_cast<hggcStream_t>(c.stream);
    if(c.down.round_type==QZM_F16) finish_top8<Threads,QZM_F16><<<grid,Threads,0,stream>>>(c);
    else if(c.down.round_type==QZM_BF16) finish_top8<Threads,QZM_BF16><<<grid,Threads,0,stream>>>(c);
    else finish_top8<Threads,QZM_F32><<<grid,Threads,0,stream>>>(c);
    return launch_status();
}
template<int Threads,bool Snapshot>
int transition_launch(qzm_transition_v1 const& c,float const* weights,int64_t stride,float* retained) {
    int nr=c.tokens*c.topk;
    dim3 grid((int64_t(c.hidden)+Threads-1)/Threads,nr>65535?65535:nr);
    transition_direct<Threads,Snapshot><<<grid,Threads,0,static_cast<hggcStream_t>(c.stream)>>>(c,nr,weights,stride,retained);
    return launch_status();
}
} // namespace quactlize::moe::decode_helpers
