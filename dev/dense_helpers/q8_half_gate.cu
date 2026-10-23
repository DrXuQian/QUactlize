#include <hggc_runtime.h>
#include "quactlize/dequant/dense_q8_half2.hpp"

// No timed work: exhaustive finite F16 scales x all unsigned8 canonical codes.
// Compare the real candidate helper to the original FP32 multiply, bit for bit.
__global__ void q8_half2_arithmetic_gate(uint32_t* bad,unsigned fault) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=65536u*256u) return;
    unsigned d=i>>8,raw=i&255u;
    if((d&0x7c00u)==0x7c00u) return;
    float scale=__half2float(__ushort_as_half(uint16_t(d)));
    uint32_t want=__half_as_ushort(__float2half_rn(__fmul_rn(float(int(raw)-128),scale)))|
        (uint32_t(__half_as_ushort(__float2half_rn(__fmul_rn(float(127-int(raw)),scale))))<<16);
    uint32_t codes=raw|((255u-raw)<<16),scales=d|(d<<16);
    uint32_t a=quactlize::dequant::DenseQ8Half2::weight<false>(codes,scales);
    uint32_t b=quactlize::dequant::DenseQ8Half2::weight<true>(codes,scales);
    if(fault && i==0) b^=1;
    if(a!=want)atomicAdd(bad,1u);
    if(b!=want)atomicAdd(bad+1,1u);
    if(a!=want || b!=want)atomicMin(bad+2,i);
    if(threadIdx.x==0)atomicAdd(bad+3,256u); // prove every finite-scale CTA ran
}
extern "C" int quactlize_q8_half2_gate(uint32_t* bad,int fault,void* stream) {
    if(!bad || (fault!=0 && fault!=1)) return 1;
    q8_half2_arithmetic_gate<<<65536,256,0,static_cast<hggcStream_t>(stream)>>>(bad,unsigned(fault));
    return int(hggcGetLastError());
}
