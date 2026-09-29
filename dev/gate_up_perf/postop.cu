// Minimal standalone post-op: deliberately no application routing overhead.
// This is a conservative unfused control, not llama.cpp's whole-chain timing.
#include <hggc_runtime.h>
#include "cutlass/numeric_types.h"
#include <cstdint>

__global__ void standalone_swiglu(float const* gate,float const* up,float* out,
    int rows,int n,int projection_stride,int output_stride,int bf16) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=int64_t(rows)*n)return;
    int r=int(i/n),c=int(i%n);
    float g=gate[int64_t(r)*projection_stride+c],u=up[int64_t(r)*projection_stride+c];
    if(bf16) {g=float(cutlass::bfloat16_t(g));u=float(cutlass::bfloat16_t(u));}
    out[int64_t(r)*output_stride+c]=(g/(1.f+expf(-g)))*u;
}
extern "C" int gate_up_perf_postop(float const* gate,float const* up,float* out,
    int rows,int n,int projection_stride,int output_stride,int bf16,void* stream) {
    if(!gate||!up||!out||rows<=0||n<=0||projection_stride<n||output_stride<n||
       (bf16!=0&&bf16!=1))return 1;
    standalone_swiglu<<<(int64_t(rows)*n+127)/128,128,0,static_cast<hggcStream_t>(stream)>>>(
        gate,up,out,rows,n,projection_stride,output_stride,bf16);
    return hggcGetLastError()==hggcSuccess?0:1;
}
