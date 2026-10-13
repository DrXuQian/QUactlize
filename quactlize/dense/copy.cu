#include "hggc_runtime.h"
#include <cstdint>
#include <climits>
#include "cutlass/numeric_types.h"

namespace {
// Only strided TC endpoints need this copy. Compact F32 -> TC loads and
// F32 epilogues operate directly; no FP16/BF16 round trip is inserted here.
__global__ void copy_rows(float const* src,float* dst,int columns,int64_t src_stride,
    int64_t dst_stride,int64_t count) {
    for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        int64_t row=i/columns,col=i%columns;
        dst[row*dst_stride+col]=src[row*src_stride+col];
    }
}
// Same one-CTA-per-row mapping and roundings as the PoC's dense gather/
// scatter. These prefill adapters are not substituted for typed decode.
template<bool Input> __global__ void convert_rows(void const* src,void* dst,int width,int64_t stride) {
    int64_t row=blockIdx.x;
    for (int i=threadIdx.x;i<width;i+=blockDim.x) {
        if constexpr(Input) static_cast<cutlass::half_t*>(dst)[row*width+i]=cutlass::half_t(static_cast<float const*>(src)[row*stride+i]);
        else static_cast<float*>(dst)[row*stride+i]=float(static_cast<cutlass::half_t const*>(src)[row*width+i]);
    }
}
// The older FQ fallback used a flat 256-thread cast, including M<=8.
// Keep that launch geometry distinct from the native prefill row adapters.
template<bool Input> __global__ void convert_flat(void const* src,void* dst,int64_t count) {
    for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        if constexpr(Input) static_cast<cutlass::half_t*>(dst)[i]=cutlass::half_t(static_cast<float const*>(src)[i]);
        else static_cast<float*>(dst)[i]=float(static_cast<cutlass::half_t const*>(src)[i]);
    }
}
}
extern "C" int quactlize_dense_convert_f16(int input,void const* src,void* dst,int rows,int columns,
    int64_t stride,int linear,void* stream) {
    if (linear && stride==columns) {
        int64_t count=int64_t(rows)*columns,grid=(count+255)/256;
        int blocks=int(grid>INT_MAX?INT_MAX:grid);
        if (input) convert_flat<true><<<blocks,256,0,static_cast<hggcStream_t>(stream)>>>(src,dst,count);
        else convert_flat<false><<<blocks,256,0,static_cast<hggcStream_t>(stream)>>>(src,dst,count);
    }
    else if (input) convert_rows<true><<<rows,256,0,static_cast<hggcStream_t>(stream)>>>(src,dst,columns,stride);
    else convert_rows<false><<<rows,256,0,static_cast<hggcStream_t>(stream)>>>(src,dst,columns,stride);
    return hggcGetLastError()==hggcSuccess ? 0 : 1;
}
extern "C" int quactlize_dense_copy_f32(float const* src,float* dst,int rows,int columns,
    int64_t src_stride,int64_t dst_stride,void* stream) {
    int64_t count=int64_t(rows)*columns;
    int blocks=int((count+255)/256);if (blocks>65535) blocks=65535;
    copy_rows<<<blocks,256,0,static_cast<hggcStream_t>(stream)>>>(src,dst,columns,src_stride,dst_stride,count);
    return hggcGetLastError()==hggcSuccess ? 0 : 1;
}
