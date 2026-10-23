#include <hggc_runtime.h>
#include "quactlize/dequant/dense_q8_half2.hpp"
#include "quactlize/dequant/dense_q8_paired.hpp"

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

__global__ void q8_paired_arithmetic_gate(uint32_t* bad,unsigned fault) {
    using P=quactlize::dequant::DenseQ8Paired;
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,d=i>>8,raw=i&255u;
    if((d&0x7c00u)==0x7c00u) return;
    // A bijection over finite scales, distinct for the two N lanes. This
    // catches accidentally broadcasting one expert/column's scale to both.
    unsigned ordinal=(d>>15)*31744+(d&32767),other=(ordinal+12345)%63488;
    other+=other>=31744?1024:0;
    float s0=__half2float(__ushort_as_half(uint16_t(d)));
    float s1=__half2float(__ushort_as_half(uint16_t(other)));
    uint32_t scales=d|(other<<16),word=raw|((raw^90)<<8)|((255-raw)<<16)|((255-(raw^90))<<24);
    uint32_t want[2],signed_value[2],fast[2];
    #pragma unroll
    for(unsigned phase=0;phase<2;++phase) {
        unsigned r=raw^(phase?90:0);
        want[phase]=__half_as_ushort(__float2half_rn(__fmul_rn(float(int(r)-128),s0)))|
            (uint32_t(__half_as_ushort(__float2half_rn(__fmul_rn(float(127-int(r)),s1))))<<16);
        signed_value[phase]=P::weight<true>(P::stage<true>(word),phase*8,scales);
        fast[phase]=P::weight<false>(word,phase*8,scales);
        if(fault && i==0 && phase==0)fast[phase]^=1;
        if(signed_value[phase]!=want[phase])atomicAdd(bad,1u);
        if(fast[phase]!=want[phase])atomicAdd(bad+1,1u);
        if(signed_value[phase]!=want[phase] || fast[phase]!=want[phase])atomicMin(bad+2,i*2+phase);
    }
    // Gate the real byte-permute instructions, not just host bit operations.
    if(!fault && (P::output0(fast[0],fast[1])!=((want[0]&65535u)|(want[1]<<16)) ||
                  P::output1(fast[0],fast[1])!=((want[0]>>16)|(want[1]&0xffff0000u)))) {
        atomicAdd(bad+1,1u);atomicMin(bad+2,i*2);
    }
    if(threadIdx.x==0)atomicAdd(bad+3,512u);
}
extern "C" int quactlize_q8_paired_gate(uint32_t* bad,int fault,void* stream) {
    if(!bad || (fault!=0 && fault!=1)) return 1;
    q8_paired_arithmetic_gate<<<65536,256,0,static_cast<hggcStream_t>(stream)>>>(bad,unsigned(fault));
    return int(hggcGetLastError());
}
