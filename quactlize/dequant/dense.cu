#include <hggc_runtime.h>
#include "../prefill/dense_api.h"
#include "dense_reader.hpp"
#include "dense_cooperative.hpp"
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
template<KType T,bool BF16>
int dense_launch(qzd_call_v1 const& c,int variant) {
    auto l=static_cast<uint16_t const*>(c.low),h=static_cast<uint16_t const*>(c.high);
    auto u=static_cast<uint8_t const*>(c.units);auto out=static_cast<uint16_t*>(c.output);
    auto s=static_cast<hggcStream_t>(c.stream);
    if (variant==QKP_DENSE_DEQUANT_DIRECT) {
        dense_dequant_direct<T,BF16><<<(int64_t(c.n)*c.k+255)/256,256,0,s>>>(l,h,u,out,c.n,c.k);
    } else if constexpr(T==KType::Q4_K || T==KType::Q5_K || T==KType::Q6_K) {
        switch(variant) {
            case QKP_DENSE_DEQUANT_PAIR:
                dense_dequant_pair<T,BF16,32,128,false><<<dim3(c.n/32,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
                break;
            case QKP_DENSE_DEQUANT_COOPERATIVE_K128:
                dense_dequant_cooperative<T,BF16,64,128,true><<<dim3(c.n/64,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
                break;
            case QKP_DENSE_DEQUANT_COOPERATIVE_K256:
                dense_dequant_cooperative<T,BF16,64,256,true><<<dim3(c.n/64,c.k/256),128,0,s>>>(l,h,u,out,c.n,c.k);
                break;
            default: return QKG_INVALID;
        }
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
        c->zero || (storage!=0 && storage!=1) || !qkp_dense_dequant_variant_valid(variant)) return QKG_INVALID;
    qkg_sizes_v1 s{};int rc=sizes(c->qtype,c->n,c->k,c->experts,arr,s);if(rc)return rc;
    if (c->qtype<10 || c->qtype>14 || (variant && c->qtype<12)) return QKG_FORMAT;
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
