#pragma once
#include "prefill_helpers.hpp"

namespace quactlize::moe::prefill_helpers {
struct alignas(8) Bf4 {__ppu_bfloat162 lo,hi;};
static_assert(sizeof(Bf4)==8 && alignof(Bf4)==8,"four contiguous BF16 values");
__device__ Bf4 pack(float4 v) {return {__floats2bfloat162_rn(v.x,v.y),__floats2bfloat162_rn(v.z,v.w)};}
__device__ float4 unpack(Bf4 v) {
    float2 a=__bfloat1622float2(v.lo),b=__bfloat1622float2(v.hi);
    return make_float4(a.x,a.y,b.x,b.y);
}
__device__ float4 projected4(void const* data,int64_t at) {
    auto x=*reinterpret_cast<Bf4 const*>(static_cast<__ppu_bfloat16 const*>(data)+at);
    // Retain the explicit projection BF16 RNE boundary, including NaNs.
    return unpack(pack(unpack(x)));
}
__device__ void store4(void* data,int64_t at,Bf4 v) {
    *reinterpret_cast<Bf4*>(static_cast<__ppu_bfloat16*>(data)+at)=v;
}
__device__ int destination(qzm_prepare_v1 const& c,int token,int slot,int tile,int r,
                           int const* prefix,int const* ranks) {
    int id=c.route.ids[int64_t(token)*c.route.ids_stride+slot];
    return c.route.offsets[id]+prefix[int64_t(tile)*c.route.experts+id]+ranks[r];
}
__device__ void maps(qzm_prepare_v1 const& c,int r,int to,int source) {
    c.route.src_rows[to]=source;c.route.dst_rows[to]=r;c.inverse_rows[r]=to;
}
__global__ void prepare_token_vec4(qzm_prepare_v1 c,int nr,int const* prefix,int const* ranks) {
    (void)nr;
    int64_t col=int64_t(blockIdx.x)*1024+int(threadIdx.x)*4;
    if(col>=c.columns) return;
    for(int64_t row=blockIdx.y;row<c.route.tokens;row+=gridDim.y) {
        int token=int(row),tile=int(row/64);
        int source=int(int64_t(token)*c.route.source_stride);
        Bf4 value=pack(*reinterpret_cast<float4 const*>(c.input+int64_t(source)*c.input_row_stride+col));
        #pragma unroll
        for(int slot=0;slot<8;++slot) {
            int r=int(row*8)+slot;
            int to=destination(c,token,slot,tile,r,prefix,ranks);
            if(blockIdx.x==0 && threadIdx.x==0) maps(c,r,to,source);
            store4(c.output,int64_t(to)*c.columns+col,value);
        }
    }
}
__global__ void finish_vec4(qzm_finish_v1 c) {
    constexpr int Threads=128;
    int64_t col=int64_t(blockIdx.x)*Threads*4+int(threadIdx.x)*4;
    if(col>=c.columns) return;
    for(int64_t token=blockIdx.y;token<c.tokens;token+=gridDim.y) {
        float4 sum=make_float4(0,0,0,0);
        #pragma unroll
        for(int slot=0;slot<8;++slot) {
            int row=c.inverse_rows[int64_t(token)*8+slot];
            float4 v=projected4(c.down.data,int64_t(row)*c.down.row_stride+col);
            float weight=c.weights[int64_t(token)*c.weights_stride+slot];
            float4 w=make_float4(__fmul_rn(v.x,weight),__fmul_rn(v.y,weight),
                                __fmul_rn(v.z,weight),__fmul_rn(v.w,weight));
            sum=slot?make_float4(__fadd_rn(sum.x,w.x),__fadd_rn(sum.y,w.y),
                                __fadd_rn(sum.z,w.z),__fadd_rn(sum.w,w.w)):w;
        }
        *reinterpret_cast<float4*>(c.output+int64_t(token)*c.output_stride+col)=sum;
    }
}
__device__ float swiglu(float g,float u) {return (g/(1.f+expf(-g)))*u;}
template<bool Snapshot>
__global__ void transition_vec4(qzm_transition_v1 c,int nr,float const* weights,int64_t stride,float* retained) {
    constexpr int Threads=128;
    int64_t col=int64_t(blockIdx.x)*Threads*4+int(threadIdx.x)*4;
    if(col>=c.hidden) return;
    for(int64_t row=blockIdx.y;row<nr;row+=gridDim.y) {
        if constexpr(Snapshot) {
            // One writer per original token-slot, independent of grouped order.
            if(blockIdx.x==0 && threadIdx.x==0) {
                int original=int(row); // validated row count fits int32
                retained[row]=weights[int64_t(original/c.topk)*stride+original%c.topk];
            }
        }
        int64_t at=int64_t(row)*c.gate.row_stride+col;
        float4 g=projected4(c.gate.data,at),u=projected4(c.gate.data,at+c.hidden);
        float4 v=make_float4(swiglu(g.x,u.x),swiglu(g.y,u.y),swiglu(g.z,u.z),swiglu(g.w,u.w));
        store4(c.output,int64_t(row)*c.output_stride+col,pack(v));
    }
}
} // namespace quactlize::moe::prefill_helpers
