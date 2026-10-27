#pragma once
#include "dense_q8_paired.hpp"

namespace quactlize::dequant {
template<bool Signed,unsigned Threads,bool BytePerm=false>
__global__ __launch_bounds__(Threads) void dense_dequant_q8_paired(
        uint16_t const* __restrict__ low,uint16_t const* __restrict__ units,
        uint16_t* __restrict__ output,int n,int k) {
    static_assert(Threads==128 || Threads==256);
    using P=DenseQ8Paired;using L=typename P::Low;
    __shared__ __align__(16) uint32_t tile[L::Cells];
    __shared__ uint32_t scales[128];
    unsigned t=threadIdx.x,n0=blockIdx.x*64,k0=blockIdx.y*128;
    if(t<128) {
        unsigned pair=t%32,g=t/32;
        auto src=units+uint64_t(k0/32+g)*unsigned(n)+n0+2*pair;
        scales[P::scale(pair,g)]=*reinterpret_cast<uint32_t const*>(src);
    }
    unsigned row=t/8,col=t%8*8;
    auto src=reinterpret_cast<uint4 const*>(low)+uint64_t(k0/2+row)*(unsigned(n)/8)+n0/8+t%8;
    uint64_t step=uint64_t(Threads/8)*(unsigned(n)/8);
    #pragma unroll
    for(unsigned part=0;part<512/Threads;++part) {
        uint4 v=src[part*step];
        unsigned r=row+part*(Threads/8);
        tile[L::shared(col/2+0,r)]=P::stage<Signed>(v.x);
        tile[L::shared(col/2+1,r)]=P::stage<Signed>(v.y);
        tile[L::shared(col/2+2,r)]=P::stage<Signed>(v.z);
        tile[L::shared(col/2+3,r)]=P::stage<Signed>(v.w);
    }
    __syncthreads();
    unsigned pair0=t/16,kk=t%16*8,shift=L::shift(kk),r=L::row(kk),g=kk/32;
    auto dst=reinterpret_cast<uint4*>(output)+uint64_t(n0+2*pair0)*(unsigned(k)/8)+k0/8+t%16;
    uint64_t out_step=uint64_t(2*(Threads/16))*(unsigned(k)/8);
    #pragma unroll
    for(unsigned part=0;part<512/Threads;++part) {
        unsigned pair=pair0+part*(Threads/16);
        uint4 a=*reinterpret_cast<uint4 const*>(tile+L::shared(pair,r));
        uint4 b=*reinterpret_cast<uint4 const*>(tile+L::shared(pair,r+4));
        uint32_t words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
        uint32_t scale=scales[P::scale(pair,g)],out0[4],out1[4];
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            uint32_t x=P::weight<Signed,BytePerm>(words[2*p],shift,scale);
            uint32_t y=P::weight<Signed,BytePerm>(words[2*p+1],shift,scale);
            out0[p]=P::output0(x,y);out1[p]=P::output1(x,y);
        }
        dst[part*out_step]={out0[0],out0[1],out0[2],out0[3]};
        dst[part*out_step+unsigned(k)/8]={out1[0],out1[1],out1[2],out1[3]};
    }
}
} // namespace quactlize::dequant
