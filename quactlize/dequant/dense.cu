#include <hggc_runtime.h>
#include "../prefill/dense_api.h"
#include "dense_reader.hpp"
#include "dense_unsigned.hpp"
#include "dense_pair.hpp"
#include "dense_cooperative.hpp"
#include "dense_high_unique.hpp"
#include "packed_exchange.hpp"
#include "../execution/validation.hpp"

namespace quactlize::dequant {
template<KType T,bool BF16>
__global__ void dense_dequant_direct(uint16_t const* low,uint16_t const* high,uint8_t const* units,
                                      uint16_t* output,int n,int k) {
    using R=DenseReader<T,BF16>;
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=int64_t(n)*k) return;
    int col=i/k,kk=i%k;
    int raw=R::code(low,high,col,kk,n)+(R::lo_bits==4?8:0);
    output[i]=R::weight(raw,R::affine(units,col,kk/R::group,n));
}

// Four K+8 codes travel in ONE shared word, not four expanded FP32 cells.
// Q6's two high bits are selected before exchange; the global layout is not
// modified. Producers load contiguous N8 uint4s, consumers store contiguous
// K8 uint4s. Q6 metadata is read once per (N,K16), with no warp broadcasts.
template<KType T,bool BF16,int TileK,bool KMajor>
__global__ __launch_bounds__(128) void dense_dequant_packed(
        uint16_t const* __restrict__ low,uint16_t const* __restrict__ high,
        uint8_t const* __restrict__ units,uint16_t* __restrict__ output,int n,int k) {
    static_assert(T==KType::Q4_K || T==KType::Q5_K || T==KType::Q6_K);
    using R=DenseReader<T,BF16>;using L=PackedExchangeLayout<TileK>;
    constexpr int Groups=TileK/R::group;
    using A=DenseAffineLayout<Groups>;
    __shared__ __align__(16) uint32_t tile[L::kCells];
    __shared__ typename R::Affine affine[32*Groups];
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    int n0=(KMajor?blockIdx.y:blockIdx.x)*32,k0=(KMajor?blockIdx.x:blockIdx.y)*TileK;
    int c0=(lane&3)*8,residue=lane>>2;
    // Metadata ownership is independent of the code producer. Consecutive
    // lanes cover consecutive N; each affine is computed only once per CTA.
    for (int i=tid;i<32*Groups;i+=128) {
        int col=i%32,g=i/32;
        affine[A::offset(col,g)]=R::affine(units,n0+col,k0/R::group+g,n);
    }
    #pragma unroll
    for (int part=0;part<TileK/128;++part) {
        int kb=k0+part*128+warp*32;
        uint4 lv=*reinterpret_cast<uint4 const*>(low+R::LowMap::word_index(n0+c0,kb+residue,n));
        uint4 hv{};
        if constexpr(R::hi_bits) hv=*reinterpret_cast<uint4 const*>(high+R::HighMap::word_index(n0+c0,kb+residue,n));
        #pragma unroll
        for (int v=0;v<8;++v) {
            uint32_t lw=v/2==0?lv.x:v/2==1?lv.y:v/2==2?lv.z:lv.w;
            uint32_t hw=v/2==0?hv.x:v/2==1?hv.y:v/2==2?hv.z:hv.w;
            uint16_t l=uint16_t(lw>>(16*(v%2))),h=uint16_t(hw>>(16*(v%2)));
            uint32_t codes=DenseCodes<T>::combine(l,h,n0+c0+v,kb+residue);
            tile[L::offset(c0+v,(part*4+warp)*8+residue)]=codes;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int iter=0;iter<TileK/32;++iter) {
        int i=tid+iter*128,row=i/(TileK/8),kk=(i%(TileK/8))*8;
        int g=kk/32,slot=(kk/8)%4;
        uint4 a=*reinterpret_cast<uint4 const*>(tile+L::offset(row,g*8));
        uint4 b=*reinterpret_cast<uint4 const*>(tile+L::offset(row,g*8+4));
        auto f=affine[A::offset(row,kk/R::group)];
        uint32_t words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};uint32_t pairs[4];
        #pragma unroll
        for (int p=0;p<4;++p)
            pairs[p]=uint32_t(R::weight(DenseCodes<T>::code(words[2*p],slot),f)) |
                (uint32_t(R::weight(DenseCodes<T>::code(words[2*p+1],slot),f))<<16);
        *reinterpret_cast<uint4*>(output+int64_t(n0+row)*k+k0+kk)={pairs[0],pairs[1],pairs[2],pairs[3]};
    }
}
// Coordinate-only K128/K256 controls, then independent N/thread geometries.
template<KType T,bool BF16,int TileN,int TileK,int Threads>
__global__ __launch_bounds__(Threads) void dense_dequant_unsigned(
        uint16_t const* __restrict__ low,uint16_t const* __restrict__ high,
        uint8_t const* __restrict__ units,uint16_t* __restrict__ output,int n,int k) {
    using R=DenseUnsignedReader<T,BF16>;using L=PackedExchangeLayout<TileK>;
    constexpr unsigned Groups=TileK/R::group,VectorsN=TileN/8;
    using A=DenseAffineLayout<Groups>;
    __shared__ __align__(16) uint32_t tile[TileN*L::kWords];
    __shared__ typename R::Affine affine[TileN*Groups];
    unsigned tid=threadIdx.x,n0=blockIdx.x*TileN,k0=blockIdx.y*TileK,extent=unsigned(n);
    #pragma unroll
    for(unsigned i=tid;i<TileN*Groups;i+=Threads){
        unsigned col=i%TileN,g=i/TileN;
        affine[A::offset(col,g)]=R::affine(units,n0+col,k0/R::group+g,extent);
    }
    #pragma unroll
    for(unsigned i=tid;i<(TileK/32)*8*VectorsN;i+=Threads){
        unsigned c0=(i%VectorsN)*8,residue=(i/VectorsN)%8,g=i/(VectorsN*8),kb=k0+g*32;
        uint4 lv=*reinterpret_cast<uint4 const*>(low+R::LowMap::word_index(n0+c0,kb+residue,extent));
        uint4 hv{};
        if constexpr(R::hi_bits)hv=*reinterpret_cast<uint4 const*>(high+R::HighMap::word_index(n0+c0,kb+residue,extent));
        #pragma unroll
        for(unsigned v=0;v<8;++v){
            uint32_t lw=v/2==0?lv.x:v/2==1?lv.y:v/2==2?lv.z:lv.w;
            uint32_t hw=v/2==0?hv.x:v/2==1?hv.y:v/2==2?hv.z:hv.w;
            tile[L::offset(c0+v,g*8+residue)]=R::combine(uint16_t(lw>>(16*(v%2))),uint16_t(hw>>(16*(v%2))),n0+c0+v,kb+residue);
        }
    }
    __syncthreads();
    #pragma unroll
    for(unsigned i=tid;i<TileN*(TileK/8);i+=Threads){
        unsigned row=i/(TileK/8),kk=(i%(TileK/8))*8,g=kk/32,slot=(kk/8)%4;
        uint4 a=*reinterpret_cast<uint4 const*>(tile+L::offset(row,g*8));
        uint4 b=*reinterpret_cast<uint4 const*>(tile+L::offset(row,g*8+4));
        auto f=affine[A::offset(row,kk/R::group)];
        uint32_t words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w},pairs[4];
        #pragma unroll
        for(unsigned p=0;p<4;++p)pairs[p]=uint32_t(R::weight(int((words[2*p]>>(slot*8))&255),f))|
            (uint32_t(R::weight(int((words[2*p+1]>>(slot*8))&255),f))<<16);
        *reinterpret_cast<uint4*>(output+uint64_t(n0+row)*unsigned(k)+k0+kk)={pairs[0],pairs[1],pairs[2],pairs[3]};
    }
}
// Two N columns per consumer; each column still publishes a contiguous K8
// vector. The native packed low/high words are read once for both columns.
template<KType T,bool BF16,int TileN,int TileK,bool Fast>
__global__ __launch_bounds__(128) void dense_dequant_pair(
        uint16_t const* __restrict__ low,uint16_t const* __restrict__ high,
        uint8_t const* __restrict__ units,uint16_t* __restrict__ output,int n,int k) {
    using R=DensePairReader<T,BF16,Fast>;using Codes=DensePairCodes<T>;
    using L=PackedExchangeLayout<TileK>;
    constexpr unsigned Groups=TileK/R::group,VectorsN=TileN/8,Cells=TileN/2*L::kWords;
    using A=DenseAffineLayout<Groups>;
    __shared__ __align__(16) uint32_t tile[Cells];
    __shared__ __align__(16) uint16_t upper[R::hi_bits ? Cells : 1];
    __shared__ typename R::Affine affine[TileN*Groups];
    unsigned tid=threadIdx.x,n0=blockIdx.x*TileN,k0=blockIdx.y*TileK,extent=unsigned(n);
    #pragma unroll
    for(unsigned i=tid;i<TileN*Groups;i+=128) {
        unsigned col=i%TileN,g=i/TileN;
        affine[A::offset(col,g)]=R::affine(units,n0+col,k0/R::group+g,extent);
    }
    #pragma unroll
    for(unsigned i=tid;i<(TileK/32)*8*VectorsN;i+=128) {
        unsigned c0=(i%VectorsN)*8,residue=(i/VectorsN)%8,g=i/(VectorsN*8),kb=k0+g*32;
        uint4 lv=*reinterpret_cast<uint4 const*>(low+R::LowMap::word_index(n0+c0,kb+residue,extent));
        uint4 hv{};
        if constexpr(R::hi_bits) hv=*reinterpret_cast<uint4 const*>(high+R::HighMap::word_index(n0+c0,kb+residue,extent));
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            unsigned at=L::offset(c0/2+p,g*8+residue);
            tile[at]=p==0?lv.x:p==1?lv.y:p==2?lv.z:lv.w;
            if constexpr(R::hi_bits) {
                uint32_t h=p==0?hv.x:p==1?hv.y:p==2?hv.z:hv.w;
                upper[at]=Codes::high(h,n0+c0+2*p,kb);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for(unsigned i=tid;i<(TileN/2)*(TileK/8);i+=128) {
        unsigned pair=i/(TileK/8),kk=(i%(TileK/8))*8,g=kk/32,slot=(kk/8)%4;
        unsigned at0=L::offset(pair,g*8),at1=L::offset(pair,g*8+4);
        uint4 a=*reinterpret_cast<uint4 const*>(tile+at0),b=*reinterpret_cast<uint4 const*>(tile+at1);
        uint2 ha{},hb{};
        if constexpr(R::hi_bits) {
            ha=*reinterpret_cast<uint2 const*>(upper+at0);
            hb=*reinterpret_cast<uint2 const*>(upper+at1);
        }
        uint32_t words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w},high_words[4]={ha.x,ha.y,hb.x,hb.y};
        auto f0=affine[A::offset(2*pair,kk/R::group)],f1=affine[A::offset(2*pair+1,kk/R::group)];
        uint32_t out0[4],out1[4];
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            unsigned x=Codes::codes(words[2*p],uint16_t(high_words[p]),slot);
            unsigned y=Codes::codes(words[2*p+1],uint16_t(high_words[p]>>16),slot);
            out0[p]=uint32_t(R::weight(x&0xffffu,f0))|(uint32_t(R::weight(y&0xffffu,f0))<<16);
            out1[p]=uint32_t(R::weight(x>>16,f1))|(uint32_t(R::weight(y>>16,f1))<<16);
        }
        auto dst=output+uint64_t(n0+2*pair)*unsigned(k)+k0+kk;
        *reinterpret_cast<uint4*>(dst)={out0[0],out0[1],out0[2],out0[3]};
        *reinterpret_cast<uint4*>(dst+unsigned(k))={out1[0],out1[1],out1[2],out1[3]};
    }
}
template<KType T,bool BF16,unsigned TileN,unsigned TileK,bool Metadata>
__global__ __launch_bounds__(128) void dense_dequant_cooperative(
        uint16_t const* __restrict__ low,uint16_t const* __restrict__ high,
        uint8_t const* __restrict__ units,uint16_t* __restrict__ output,int n,int k) {
    using R=DensePairReader<T,BF16,false>;using Codes=DensePairCodes<T>;using U=typename R::U;
    using L=DenseCooperativeLayout<TileN,TileK>;
    constexpr unsigned Groups=TileK/R::group,MetaWords=TileN*U::kUnitTotal/4;
    using A=DenseCooperativeAffine<Groups>;
    __shared__ __align__(16) uint32_t tile[L::Cells];
    __shared__ __align__(16) uint16_t upper[R::hi_bits?L::Cells:1];
    __shared__ __align__(16) uint32_t metadata[Metadata?MetaWords:1];
    __shared__ typename R::Affine affine[TileN*Groups];
    unsigned tid=threadIdx.x,n0=blockIdx.x*TileN,k0=blockIdx.y*TileK,extent=unsigned(n);
    if constexpr(Metadata) {
        auto src=units+(uint64_t(k0/(256*U::kSbPerUnit))*extent+n0)*U::kUnitTotal;
        #pragma unroll
        for(unsigned i=tid;i<MetaWords/4;i+=128)
            reinterpret_cast<uint4*>(metadata)[i]=reinterpret_cast<uint4 const*>(src)[i];
    } else {
        #pragma unroll
        for(unsigned i=tid;i<TileN*Groups;i+=128) {
            unsigned col=i%TileN,g=i/TileN;
            affine[A::offset(col,g)]=R::affine(units,n0+col,k0/R::group+g,extent);
        }
    }
    #pragma unroll
    for(unsigned i=tid;i<(TileK/32)*8*L::V;i+=128) {
        auto lp=L::producer(i,false),hp=L::producer(i,true);
        uint4 lv=*reinterpret_cast<uint4 const*>(low+R::LowMap::word_index(n0+lp.col,k0+lp.group*32+lp.residue,extent));
        uint4 hv{};
        if constexpr(R::hi_bits)hv=*reinterpret_cast<uint4 const*>(high+R::HighMap::word_index(n0+hp.col,k0+hp.group*32+hp.residue,extent));
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            tile[L::low(lp.col/2+p,lp.group*8+lp.residue)]=p==0?lv.x:p==1?lv.y:p==2?lv.z:lv.w;
            if constexpr(R::hi_bits) {
                uint32_t h=p==0?hv.x:p==1?hv.y:p==2?hv.z:hv.w;
                upper[L::high(hp.col/2+p,hp.group*8+hp.residue)]=Codes::high(h,n0+hp.col+2*p,k0+hp.group*32);
            }
        }
    }
    __syncthreads();
    if constexpr(Metadata) {
        if(tid<TileN) {
            auto values=DenseRegisterAffine<T,TileK>::load(metadata+tid*(U::kUnitTotal/4),k0);
            #pragma unroll
            for(unsigned g=0;g<Groups;++g)affine[A::offset(tid,g)]=values.group(g);
        }
        // Raw metadata publication and affine publication are separate
        // lifetimes. Price this extra barrier instead of omitting it.
        __syncthreads();
    }
    #pragma unroll
    for(unsigned i=tid;i<(TileN/2)*(TileK/8);i+=128) {
        unsigned pair=i/(TileK/8),kk=(i%(TileK/8))*8,g=kk/32,slot=(kk/8)%4;
        uint4 a=*reinterpret_cast<uint4 const*>(tile+L::low(pair,g*8));
        uint4 b=*reinterpret_cast<uint4 const*>(tile+L::low(pair,g*8+4));
        uint2 ha{},hb{};
        if constexpr(R::hi_bits) {
            ha=*reinterpret_cast<uint2 const*>(upper+L::high(pair,g*8));
            hb=*reinterpret_cast<uint2 const*>(upper+L::high(pair,g*8+4));
        }
        uint32_t words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w},high_words[4]={ha.x,ha.y,hb.x,hb.y};
        auto f0=affine[A::offset(2*pair,kk/R::group)],f1=affine[A::offset(2*pair+1,kk/R::group)];
        uint32_t out0[4],out1[4];
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            unsigned x=Codes::codes(words[2*p],uint16_t(high_words[p]),slot);
            unsigned y=Codes::codes(words[2*p+1],uint16_t(high_words[p]>>16),slot);
            out0[p]=uint32_t(R::weight(x&0xffffu,f0))|(uint32_t(R::weight(y&0xffffu,f0))<<16);
            out1[p]=uint32_t(R::weight(x>>16,f1))|(uint32_t(R::weight(y>>16,f1))<<16);
        }
        auto dst=output+uint64_t(n0+2*pair)*unsigned(k)+k0+kk;
        *reinterpret_cast<uint4*>(dst)={out0[0],out0[1],out0[2],out0[3]};
        *reinterpret_cast<uint4*>(dst+unsigned(k))={out1[0],out1[1],out1[2],out1[3]};
    }
}
// Keep the admitted cooperative body above unchanged for paired codegen and
// timing controls. Only the high producer changes in this bounded candidate.
template<KType T,bool BF16,unsigned TileK>
__global__ __launch_bounds__(128) void dense_dequant_unique_high(
        uint16_t const* __restrict__ low,uint16_t const* __restrict__ high,
        uint8_t const* __restrict__ units,uint16_t* __restrict__ output,int n,int k) {
    constexpr unsigned TileN=64;
    constexpr bool Metadata=true;
    using R=DensePairReader<T,BF16,false>;using Codes=DensePairCodes<T>;using U=typename R::U;
    using L=DenseCooperativeLayout<TileN,TileK>;
    constexpr unsigned Groups=TileK/R::group,MetaWords=TileN*U::kUnitTotal/4;
    using A=DenseCooperativeAffine<Groups>;
    __shared__ __align__(16) uint32_t tile[L::Cells];
    __shared__ __align__(16) uint16_t upper[R::hi_bits?L::Cells:1];
    __shared__ __align__(16) uint32_t metadata[Metadata?MetaWords:1];
    __shared__ typename R::Affine affine[TileN*Groups];
    unsigned tid=threadIdx.x,n0=blockIdx.x*TileN,k0=blockIdx.y*TileK,extent=unsigned(n);
    if constexpr(Metadata) {
        auto src=units+(uint64_t(k0/(256*U::kSbPerUnit))*extent+n0)*U::kUnitTotal;
        #pragma unroll
        for(unsigned i=tid;i<MetaWords/4;i+=128)
            reinterpret_cast<uint4*>(metadata)[i]=reinterpret_cast<uint4 const*>(src)[i];
    }
    #pragma unroll
    for(unsigned i=tid;i<(TileK/32)*8*L::V;i+=128) {
        auto lp=L::producer(i,false);
        uint4 lv=*reinterpret_cast<uint4 const*>(low+R::LowMap::word_index(n0+lp.col,k0+lp.group*32+lp.residue,extent));
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            tile[L::low(lp.col/2+p,lp.group*8+lp.residue)]=p==0?lv.x:p==1?lv.y:p==2?lv.z:lv.w;
        }
    }
    {
        // One physical vector, all of its logical high-bit destinations.
        using H=DenseUniqueHigh<T,TileK>;
        #pragma unroll
        for(unsigned i=tid;i<H::Loads;i+=128) {
            auto owner=H::owner(i);
            uint4 hv=*reinterpret_cast<uint4 const*>(high+R::HighMap::word_index(
                n0+owner.col,k0+owner.group64*64+owner.residue,extent));
            #pragma unroll
            for(unsigned fan=0;fan<H::Fans;++fan) {
                auto dst=H::target(owner,fan);
                #pragma unroll
                for(unsigned p=0;p<4;++p) {
                    uint32_t h=p==0?hv.x:p==1?hv.y:p==2?hv.z:hv.w;
                    upper[L::high(dst.col/2+p,dst.group32*8+owner.residue)]=
                        Codes::high(h,n0+dst.col+2*p,k0+dst.group32*32);
                }
            }
        }
    }
    __syncthreads();
    if constexpr(Metadata) {
        if(tid<TileN) {
            auto values=DenseRegisterAffine<T,TileK>::load(metadata+tid*(U::kUnitTotal/4),k0);
            #pragma unroll
            for(unsigned g=0;g<Groups;++g)affine[A::offset(tid,g)]=values.group(g);
        }
        // Raw metadata publication and affine publication are separate
        // lifetimes. Price this extra barrier instead of omitting it.
        __syncthreads();
    }
    #pragma unroll
    for(unsigned i=tid;i<(TileN/2)*(TileK/8);i+=128) {
        unsigned pair=i/(TileK/8),kk=(i%(TileK/8))*8,g=kk/32,slot=(kk/8)%4;
        uint4 a=*reinterpret_cast<uint4 const*>(tile+L::low(pair,g*8));
        uint4 b=*reinterpret_cast<uint4 const*>(tile+L::low(pair,g*8+4));
        uint2 ha{},hb{};
        if constexpr(R::hi_bits) {
            ha=*reinterpret_cast<uint2 const*>(upper+L::high(pair,g*8));
            hb=*reinterpret_cast<uint2 const*>(upper+L::high(pair,g*8+4));
        }
        uint32_t words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w},high_words[4]={ha.x,ha.y,hb.x,hb.y};
        auto f0=affine[A::offset(2*pair,kk/R::group)],f1=affine[A::offset(2*pair+1,kk/R::group)];
        uint32_t out0[4],out1[4];
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            unsigned x=Codes::codes(words[2*p],uint16_t(high_words[p]),slot);
            unsigned y=Codes::codes(words[2*p+1],uint16_t(high_words[p]>>16),slot);
            out0[p]=uint32_t(R::weight(x&0xffffu,f0))|(uint32_t(R::weight(y&0xffffu,f0))<<16);
            out1[p]=uint32_t(R::weight(x>>16,f1))|(uint32_t(R::weight(y>>16,f1))<<16);
        }
        auto dst=output+uint64_t(n0+2*pair)*unsigned(k)+k0+kk;
        *reinterpret_cast<uint4*>(dst)={out0[0],out0[1],out0[2],out0[3]};
        *reinterpret_cast<uint4*>(dst+unsigned(k))={out1[0],out1[1],out1[2],out1[3]};
    }
}
template<KType T,bool BF16>
int dense_launch(qzd_call_v1 const& c,int variant) {
    auto l=static_cast<uint16_t const*>(c.low),h=static_cast<uint16_t const*>(c.high);
    auto u=static_cast<uint8_t const*>(c.units);auto out=static_cast<uint16_t*>(c.output);
    auto s=static_cast<hggcStream_t>(c.stream);
    if (variant==22 || variant==23) {
        if constexpr(T==KType::Q5_K || T==KType::Q6_K) {
            if(variant==22)dense_dequant_unique_high<T,BF16,128><<<dim3(c.n/64,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
            else dense_dequant_unique_high<T,BF16,256><<<dim3(c.n/64,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
            return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
        } else return QKG_FORMAT;
    }
    if (!variant) dense_dequant_direct<T,BF16><<<(int64_t(c.n)*c.k+255)/256,256,0,s>>>(l,h,u,out,c.n,c.k);
    else if constexpr(T==KType::Q4_K || T==KType::Q5_K || T==KType::Q6_K) {
        if (variant==14) dense_dequant_cooperative<T,BF16,32,128,false><<<dim3(c.n/32,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==15) dense_dequant_cooperative<T,BF16,32,256,false><<<dim3(c.n/32,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==16) dense_dequant_cooperative<T,BF16,16,256,false><<<dim3(c.n/16,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==17) dense_dequant_cooperative<T,BF16,32,128,true><<<dim3(c.n/32,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==18) dense_dequant_cooperative<T,BF16,32,256,true><<<dim3(c.n/32,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==19) dense_dequant_cooperative<T,BF16,16,256,true><<<dim3(c.n/16,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==20) dense_dequant_cooperative<T,BF16,64,128,true><<<dim3(c.n/64,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==21) dense_dequant_cooperative<T,BF16,64,256,true><<<dim3(c.n/64,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==8) dense_dequant_pair<T,BF16,32,128,false><<<dim3(c.n/32,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==9) dense_dequant_pair<T,BF16,32,128,true><<<dim3(c.n/32,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==10) dense_dequant_pair<T,BF16,32,256,false><<<dim3(c.n/32,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==11) dense_dequant_pair<T,BF16,32,256,true><<<dim3(c.n/32,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==12) dense_dequant_pair<T,BF16,16,256,false><<<dim3(c.n/16,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==13) dense_dequant_pair<T,BF16,16,256,true><<<dim3(c.n/16,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==4) dense_dequant_unsigned<T,BF16,32,128,128><<<dim3(c.n/32,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==5) dense_dequant_unsigned<T,BF16,32,256,128><<<dim3(c.n/32,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==6) dense_dequant_unsigned<T,BF16,16,256,128><<<dim3(c.n/16,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==7) dense_dequant_unsigned<T,BF16,32,256,256><<<dim3(c.n/32,c.k/256),256,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==1) dense_dequant_packed<T,BF16,128,false><<<dim3(c.n/32,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
        else if (variant==2) dense_dequant_packed<T,BF16,256,false><<<dim3(c.n/32,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
        else dense_dequant_packed<T,BF16,256,true><<<dim3(c.k/256,c.n/32),128,0,s>>>(l,h,u,out,c.n,c.k);
    } else return QKG_FORMAT;
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
template<KType T> int dense_type(qzd_call_v1 const& c,int storage,int variant) {
    return storage?dense_launch<T,true>(c,variant):dense_launch<T,false>(c,variant);
}
}
extern "C" int quactlize_kpack_dequant_dense_v1(qzd_call_v1 const* c,
        quactlize_ppu_placed_arrangement_v2 const* arr,int storage,int variant) {
    using namespace quactlize::execution;using namespace quactlize::dequant;
    if (!c || c->version!=1 || c->size!=sizeof(*c) || c->operation!=1 || c->experts!=1 ||
        c->zero || (storage!=0 && storage!=1) || variant<0 || variant>23) return QKG_INVALID;
    qkg_sizes_v1 s{};int rc=sizes(c->qtype,c->n,c->k,c->experts,arr,s);if(rc)return rc;
    if (c->qtype<10 || c->qtype>14 || (variant && c->qtype<12)) return QKG_FORMAT;
    if (variant>=22 && c->qtype<13) return QKG_FORMAT;
    if (c->k/128>65535 || c->n/32>65535 || (int64_t(c->n)*c->k+255)/256>INT32_MAX) return QKG_SHAPE;
    uint64_t bytes=uint64_t(c->n)*c->k*2;
    if (c->low_bytes<s.low_bytes || c->high_bytes<s.high_bytes || c->unit_bytes<s.units_bytes || c->output_bytes<bytes) return QKG_CAPACITY;
    uintptr_t p[]={uintptr_t(c->low),uintptr_t(c->high),uintptr_t(c->units),uintptr_t(c->output)};
    uint64_t len[]={s.low_bytes,s.high_bytes,s.units_bytes,bytes};
    for (int i=0;i<4;++i) if (len[i]) {
        if (!p[i] || p[i]%16 || !span(p[i],len[i])) return QKG_INVALID;
        for (int j=0;j<i;++j) if(overlap(p[i],len[i],p[j],len[j])) return QKG_INVALID;
    }
    switch(c->qtype) {
        case 10:return dense_type<KType::Q2_K>(*c,storage,variant);
        case 11:return dense_type<KType::Q3_K>(*c,storage,variant);
        case 12:return dense_type<KType::Q4_K>(*c,storage,variant);
        case 13:return dense_type<KType::Q5_K>(*c,storage,variant);
        case 14:return dense_type<KType::Q6_K>(*c,storage,variant);
        default:return QKG_FORMAT;
    }
}
