#pragma once
#include "dense_q8_tuned.hpp"

namespace quactlize::dequant {
template<bool BF16,bool Fast,unsigned Threads,bool VectorScale>
__global__ __launch_bounds__(Threads) void dense_dequant_q8_tuned(
        uint16_t const* __restrict__ low,uint16_t const* __restrict__ units,
        uint16_t* __restrict__ output,int n,int k) {
    static_assert(Threads==128 || Threads==256 || Threads==512);
    using L=DensePlanePair<8,128>;using Lo=typename L::Low;using A=typename L::Affine;
    __shared__ __align__(16) uint32_t tile[Lo::Cells];
    __shared__ float scale[64*L::Groups];
    unsigned tid=threadIdx.x,n0=blockIdx.x*64,k0=blockIdx.y*128,extent=unsigned(n);
    if constexpr(VectorScale) {
        // Four adjacent N scales per owner: each vector instruction's warp
        // spans two full128B segments rather than64B of one128B segment.
        if(tid<64) {
            unsigned col=(tid%16)*4,g=tid/16;
            uint2 words=*reinterpret_cast<uint2 const*>(units+DenseQ8TunedLayout::scale(extent,n0,k0,tid,true));
            #pragma unroll
            for(unsigned v=0;v<4;++v) {
                uint32_t word=v<2?words.x:words.y;
                scale[A::offset(col+v,g)]=float(cutlass::half_t::bitcast(uint16_t(word>>(16*(v%2)))));
            }
        }
    } else {
        #pragma unroll
        for(unsigned i=tid;i<64*L::Groups;i+=Threads) {
            unsigned col=i%64,g=i/64;
            scale[A::offset(col,g)]=float(cutlass::half_t::bitcast(units[DenseQ8TunedLayout::scale(extent,n0,k0,i,false)]));
        }
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
        float s0=scale[A::offset(2*pair,kk/32)],s1=scale[A::offset(2*pair+1,kk/32)];
        uint32_t out0[4],out1[4];
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            uint32_t x=L::codes(words[2*p],0,kk),y=L::codes(words[2*p+1],0,kk);
            if constexpr(Fast) {
                out0[p]=uint32_t(DenseQ8Fast::weight<BF16>(x&0xffffu,s0))|
                    (uint32_t(DenseQ8Fast::weight<BF16>(y&0xffffu,s0))<<16);
                out1[p]=uint32_t(DenseQ8Fast::weight<BF16>(x>>16,s1))|
                    (uint32_t(DenseQ8Fast::weight<BF16>(y>>16,s1))<<16);
            } else {
                out0[p]=uint32_t(L::template weight<BF16>(x&0xffffu,s0))|
                    (uint32_t(L::template weight<BF16>(y&0xffffu,s0))<<16);
                out1[p]=uint32_t(L::template weight<BF16>(x>>16,s1))|
                    (uint32_t(L::template weight<BF16>(y>>16,s1))<<16);
            }
        }
        auto dst=output+DenseQ8TunedLayout::output(unsigned(k),n0,k0,i);
        *reinterpret_cast<uint4*>(dst)={out0[0],out0[1],out0[2],out0[3]};
        *reinterpret_cast<uint4*>(dst+unsigned(k))={out1[0],out1[1],out1[2],out1[3]};
    }
}
} // namespace quactlize::dequant
