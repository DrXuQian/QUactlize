#pragma once
#include "dense_q8_exchange.hpp"

namespace quactlize::dequant {
// Explicit experiment only. Defaults remain the measured DQ22/DQ23 bodies.
template<bool BF16,unsigned TileK>
__global__ __launch_bounds__(128) void dense_dequant_q8_exchange(
        uint16_t const* __restrict__ low,uint16_t const* __restrict__ units,
        uint16_t* __restrict__ output,int n,int k) {
    using L=DenseQ8Exchange<TileK>;using Lo=typename L::Low;using A=typename L::Affine;
    __shared__ __align__(16) uint32_t tile[Lo::Cells];
    __shared__ float scale[64*L::Groups];
    unsigned tid=threadIdx.x,n0=blockIdx.x*64,k0=blockIdx.y*TileK,extent=unsigned(n);
    #pragma unroll
    for(unsigned i=tid;i<64*L::Groups;i+=128) {
        unsigned col=i%64,g=i/64;
        scale[A::offset(col,g)]=float(cutlass::half_t::bitcast(units[uint64_t(k0/32+g)*extent+n0+col]));
    }
    #pragma unroll
    for(unsigned i=tid;i<Lo::Rows*8;i+=128) {
        unsigned row=i/8,col=i%8*8;
        uint4 v=*reinterpret_cast<uint4 const*>(low+uint64_t(k0/2+row)*extent+n0+col);
        #pragma unroll
        for(unsigned p=0;p<4;++p)tile[Lo::shared(col/2+p,row)]=p==0?v.x:p==1?v.y:p==2?v.z:v.w;
    }
    __syncthreads();
    #pragma unroll
    for(unsigned i=tid;i<32*(TileK/8);i+=128) {
        unsigned pair=L::pair(i),kk=L::k(i);
        uint4 v=*reinterpret_cast<uint4 const*>(tile+L::source(i));
        uint32_t own[4]={v.x,v.y,v.z,v.w},other[4];
        #pragma unroll
        for(unsigned r=0;r<4;++r)other[r]=__shfl_xor_sync(0xffffffffu,own[r],1);
        float s0=scale[A::offset(2*pair,kk/32)],s1=scale[A::offset(2*pair+1,kk/32)];
        uint32_t out0[4],out1[4];
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            unsigned r=2*p;
            uint32_t x=L::codes(L::word(i,r,own[r%4],other[r%4]),0,kk);
            uint32_t y=L::codes(L::word(i,r+1,own[(r+1)%4],other[(r+1)%4]),0,kk);
            out0[p]=uint32_t(L::template weight<BF16>(x&0xffffu,s0))|
                (uint32_t(L::template weight<BF16>(y&0xffffu,s0))<<16);
            out1[p]=uint32_t(L::template weight<BF16>(x>>16,s1))|
                (uint32_t(L::template weight<BF16>(y>>16,s1))<<16);
        }
        auto dst=output+uint64_t(n0+2*pair)*unsigned(k)+k0+kk;
        *reinterpret_cast<uint4*>(dst)={out0[0],out0[1],out0[2],out0[3]};
        *reinterpret_cast<uint4*>(dst+unsigned(k))={out1[0],out1[1],out1[2],out1[3]};
    }
}
} // namespace quactlize::dequant
