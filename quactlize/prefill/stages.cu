#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include "layout.hpp"

namespace quactlize::prefill {
__global__ void gather_bf16(float const* a,__ppu_bfloat16* out,int const* ids,int k,int64_t stride,
                          int a_rows,int* error) {
    int64_t row=blockIdx.x,from=ids ? ids[row] : row;
    bool valid=from>=0 && from<a_rows;
    if (!valid && threadIdx.x==0) atomicExch(error,4);
    for (int col=threadIdx.x;col<k;col+=blockDim.x)
        out[row*k+col]=__float2bfloat16_rn(valid ? a[from*stride+col] : 0.f);
}
__global__ void finish_bf16(__ppu_bfloat16 const* a,float* out,int const* ids,int n,int64_t stride,
                          int m,int* error) {
    int64_t row=blockIdx.x,to=ids ? ids[row] : row;
    if (to<0 || to>=m) {
        if (threadIdx.x==0) atomicExch(error,5);
        return;
    }
    for (int col=threadIdx.x;col<n;col+=blockDim.x)
        out[to*stride+col]=*error ? __int_as_float(0x7fc00000) : __bfloat162float(a[row*n+col]);
}
__global__ void active_experts(int const* offsets,int* rows,int* ids,int* count,int* error,int m) {
    __shared__ int prefix[256];
    __shared__ int invalid;
    int e=threadIdx.x;
    if (e==0) invalid=offsets[0]!=0 || offsets[256]!=m;
    __syncthreads();
    int begin=offsets[e],end=offsets[e+1];
    bool bad=begin<0 || begin>end || end>m;
    if (bad) atomicExch(&invalid,1);
    __syncthreads();
    // One malformed interval invalidates the whole directory. Keeping the
    // positive intervals would let their sum exceed the allocated M rows.
    if (invalid) {
        rows[e]=0;
        if (e==0) { *count=0;*error=3; }
        return;
    }
    int length=end-begin;
    rows[e]=length;prefix[e]=length>0;
    __syncthreads();
    for (int d=1;d<256;d*=2) {
        int v=e>=d ? prefix[e-d] : 0;
        __syncthreads();prefix[e]+=v;__syncthreads();
    }
    if (length>0) ids[prefix[e]-1]=e;
    if (e==255) *count=prefix[e];
}
}

extern "C" int qkp_stage_prepare(qkp_call_v1 const* c,quactlize::prefill::Layout const* l,void* stream) {
    using namespace quactlize::prefill;
    auto base=static_cast<uint8_t*>(c->workspace);auto s=static_cast<hggcStream_t>(stream);
    if (hggcMemsetAsync(base+l->error,0,4,s)!=hggcSuccess) return QKG_RUNTIME;
    if (c->weight.experts>1) {
        active_experts<<<1,256,0,s>>>(c->offsets,reinterpret_cast<int*>(base+l->rows),
            reinterpret_cast<int*>(base+l->ids),reinterpret_cast<int*>(base+l->count),
            reinterpret_cast<int*>(base+l->error),c->m);
        if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    }
    gather_bf16<<<c->m,256,0,s>>>(c->a,reinterpret_cast<__ppu_bfloat16*>(base+l->a),c->src_rows,
        c->weight.k,c->a_stride,c->a_rows,reinterpret_cast<int*>(base+l->error));
    return hggcGetLastError()==hggcSuccess ? 0 : QKG_RUNTIME;
}
extern "C" int qkp_stage_finish(qkp_call_v1 const* c,quactlize::prefill::Layout const* l,void* stream) {
    auto base=static_cast<uint8_t*>(c->workspace);
    quactlize::prefill::finish_bf16<<<c->m,256,0,static_cast<hggcStream_t>(stream)>>>(
        reinterpret_cast<__ppu_bfloat16 const*>(base+l->out),c->output,c->dst_rows,c->weight.n,c->output_stride,
        c->m,reinterpret_cast<int*>(base+l->error));
    return hggcGetLastError()==hggcSuccess ? 0 : QKG_RUNTIME;
}
