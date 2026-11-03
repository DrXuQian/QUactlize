#include "dense_swiglu.hpp"
#include "../routing/api.h"
#include <hggc_runtime.h>

namespace quactlize::ffn {
__device__ float dense_silu_mul(float gate,float up) {
    // Preserve the generic transition's F32 expression, not an approximate
    // sigmoid, half activation, reassociation or fast-math substitution.
    return (gate/(1.f+expf(-gate)))*up;
}
__global__ void dense_swiglu_vec4(float const* gate,float const* up,float* output,int rows,int hidden,
                                int64_t gate_stride,int64_t up_stride,int64_t output_stride) {
    int col=(int(blockIdx.x)*128+int(threadIdx.x))*4;
    if(col>=hidden) return;
    for(int64_t row=blockIdx.y;row<rows;row+=gridDim.y) {
        float4 g=*reinterpret_cast<float4 const*>(gate+row*gate_stride+col);
        float4 u=*reinterpret_cast<float4 const*>(up+row*up_stride+col);
        float4 v=make_float4(dense_silu_mul(g.x,u.x),dense_silu_mul(g.y,u.y),
                            dense_silu_mul(g.z,u.z),dense_silu_mul(g.w,u.w));
        *reinterpret_cast<float4*>(output+row*output_stride+col)=v;
    }
}
__global__ void dense_swiglu_scalar(float const* gate,float const* up,float* output,int rows,int hidden,
                                  int64_t gate_stride,int64_t up_stride,int64_t output_stride) {
    int col=int(blockIdx.x)*256+int(threadIdx.x);
    if(col>=hidden) return;
    for(int64_t row=blockIdx.y;row<rows;row+=gridDim.y)
        output[row*output_stride+col]=dense_silu_mul(gate[row*gate_stride+col],up[row*up_stride+col]);
}
int dense_swiglu(float const* gate,float const* up,float* output,int rows,int hidden,
                int64_t gate_stride,int64_t up_stride,int64_t output_stride,void* stream) {
    if(!rows) return QZR_OK;
    auto s=static_cast<hggcStream_t>(stream);unsigned gy=rows>65535?65535:rows;
    if(dense_swiglu_vectorized(gate,up,output,hidden,gate_stride,up_stride,output_stride)) {
        dim3 grid((int64_t(hidden)+511)/512,gy);
        dense_swiglu_vec4<<<grid,128,0,s>>>(gate,up,output,rows,hidden,gate_stride,up_stride,output_stride);
    } else {
        dim3 grid((int64_t(hidden)+255)/256,gy);
        dense_swiglu_scalar<<<grid,256,0,s>>>(gate,up,output,rows,hidden,gate_stride,up_stride,output_stride);
    }
    return hggcGetLastError()==hggcSuccess?QZR_OK:QZR_RUNTIME;
}
}
