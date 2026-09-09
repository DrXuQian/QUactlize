// Diagnostic common F32 endpoints; no CPU router work is inside timing.
// order maps sorted-expert row -> original indexed row. Both order and
// offsets are pre-existing GPU routing inputs, just like GEMV's ids array.
#include <hggc_runtime.h>
#include "api.h"
#include "cutlass/numeric_types.h"
#include <cstdint>

namespace kpack_comparison {
using Half = cutlass::half_t;
__global__ void gather_f32_to_f16(float const* input, Half* output,
        int const* order, int m, int k, int channels) {
    int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i < int64_t(m)*k) {
        int const row=int(i/k), col=int(i%k);
        int const slot=order[row];
        output[i]=Half(input[int64_t(slot%channels)*k+col]);
    }
}
__global__ void scatter_f16_to_f32(Half const* input, float* output,
        int const* order, int m, int n) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<int64_t(m)*n) {
        int const row=int(i/n), col=int(i%n);
        output[int64_t(order[row])*n+col]=float(input[i]);
    }
}
}

extern "C" int qkg_comparison_adapter_v1(int direction, void const* input,
        void* output, int const* order, int m, int extent, int channels, void* stream_ptr) {
    if ((direction!=0 && direction!=1) || !input || !output || !order ||
        m<=0 || m>8 || extent<=0 || extent>262144 || channels<=0 || channels>m || m%channels)
        return QKG_INVALID;
    auto stream=static_cast<hggcStream_t>(stream_ptr);
    unsigned blocks=unsigned((int64_t(m)*extent+255)/256);
    if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    if (direction==0)
        kpack_comparison::gather_f32_to_f16<<<blocks,256,0,stream>>>(
            static_cast<float const*>(input),static_cast<cutlass::half_t*>(output),order,m,extent,channels);
    else
        kpack_comparison::scatter_f16_to_f32<<<blocks,256,0,stream>>>(
            static_cast<cutlass::half_t const*>(input),static_cast<float*>(output),order,m,extent);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
