#pragma once
#include "dense_q8_half2.hpp"

namespace quactlize::dequant {
template<bool Fast,unsigned Threads>
__global__ __launch_bounds__(Threads) void dense_dequant_q8_half2(
        uint16_t const* __restrict__ low,uint16_t const* __restrict__ units,
        uint16_t* __restrict__ output,int n,int k) {
    static_assert(Threads==128 || Threads==256);
    using L=DensePlanePair<8,128>;using Lo=typename L::Low;using A=typename L::Affine;
    __shared__ __align__(16) uint32_t tile[Lo::Cells];
    // Same 4-byte slot and address map as the FP32 control; no metadata shuffle.
    __shared__ uint32_t scale[64*L::Groups];
    unsigned tid=threadIdx.x,n0=blockIdx.x*64,k0=blockIdx.y*128,extent=unsigned(n);
    #pragma unroll
    for(unsigned i=tid;i<64*L::Groups;i+=Threads) {
        uint32_t d=units[DenseQ8TunedLayout::scale(extent,n0,k0,i,false)];
        scale[A::offset(i%64,i/64)]=d|(d<<16);
    }
    #pragma unroll
    for(unsigned i=tid;i<Lo::Rows*8;i+=Threads) {
        unsigned row=i/8,col=i%8*8;
        uint4 v=*reinterpret_cast<uint4 const*>(low+DenseQ8TunedLayout::low(extent,n0,k0,i));
        #pragma unroll
        for(unsigned p=0;p<4;++p)tile[Lo::shared(col/2+p,row)]=p==0?v.x:p==1?v.y:p==2?v.z:v.w;
    }
    __syncthreads();
    #pragma unroll
    for(unsigned i=tid;i<32*(128/8);i+=Threads) {
        unsigned pair=i/16,kk=i%16*8;
        uint4 a=*reinterpret_cast<uint4 const*>(tile+Lo::shared(pair,Lo::row(kk)));
        uint4 b=*reinterpret_cast<uint4 const*>(tile+Lo::shared(pair,Lo::row(kk)+4));
        uint32_t words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
        uint32_t s0=scale[A::offset(2*pair,kk/32)],s1=scale[A::offset(2*pair+1,kk/32)];
        uint32_t out0[4],out1[4];
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            uint32_t x=L::codes(words[2*p],0,kk),y=L::codes(words[2*p+1],0,kk);
            out0[p]=DenseQ8Half2::weight<Fast>((x&255u)|((y&255u)<<16),s0);
            out1[p]=DenseQ8Half2::weight<Fast>((x>>16)|(y&0x00ff0000u),s1);
        }
        auto dst=output+DenseQ8TunedLayout::output(unsigned(k),n0,k0,i);
        *reinterpret_cast<uint4*>(dst)={out0[0],out0[1],out0[2],out0[3]};
        *reinterpret_cast<uint4*>(dst+unsigned(k))={out1[0],out1[1],out1[2],out1[3]};
    }
}
} // namespace quactlize::dequant
