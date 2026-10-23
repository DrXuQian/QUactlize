#include <hggc_runtime.h>
#include "../prefill/dense_api.h"
#include "dense_reader.hpp"
#include "dense_q8.hpp"
#include "dense_cooperative.hpp"
#include "dense_plane_pair.hpp"
#include "packed_exchange.hpp"
#include "../execution/validation.hpp"

namespace quactlize::dequant {
// N64 paired-word exchange for the formats not covered by the Q4/Q5/Q6
// cooperative helper. A CTA reads each low/high physical word exactly once.
template<int Q,bool BF16,unsigned TileK>
__global__ __launch_bounds__(128) void dense_dequant_plane_pair(
        uint16_t const* __restrict__ low,uint16_t const* __restrict__ high,
        uint8_t const* __restrict__ units,uint16_t* __restrict__ output,int n,int k) {
    using L=DensePlanePair<Q,TileK>;using Lo=typename L::Low;using Hi=typename L::High;
    using A=typename L::Affine;
    constexpr unsigned MetaWords=Q==11?64*28/4:1;
    __shared__ __align__(16) uint32_t tile[Lo::Cells];
    __shared__ __align__(16) uint32_t upper[Q==11?Hi::Cells:1];
    __shared__ __align__(16) uint32_t metadata[MetaWords];
    __shared__ float scale[64*L::Groups];
    unsigned tid=threadIdx.x,n0=blockIdx.x*64,k0=blockIdx.y*TileK,extent=unsigned(n);
    if constexpr(Q==11) {
        auto src=units+(uint64_t(k0/512)*extent+n0)*28;
        #pragma unroll
        for(unsigned i=tid;i<MetaWords/4;i+=128)
            reinterpret_cast<uint4*>(metadata)[i]=reinterpret_cast<uint4 const*>(src)[i];
    } else {
        #pragma unroll
        for(unsigned i=tid;i<64*L::Groups;i+=128) {
            unsigned col=i%64,g=i/64;
            uint16_t d=reinterpret_cast<uint16_t const*>(units)[uint64_t(k0/32+g)*extent+n0+col];
            scale[A::offset(col,g)]=float(cutlass::half_t::bitcast(d));
        }
    }
    #pragma unroll
    for(unsigned i=tid;i<Lo::Rows*8;i+=128) {
        unsigned row=i/8,col=i%8*8;
        uint4 v=*reinterpret_cast<uint4 const*>(low+uint64_t(k0*Lo::Rows/TileK+row)*extent+n0+col);
        #pragma unroll
        for(unsigned p=0;p<4;++p)tile[Lo::shared(col/2+p,row)]=p==0?v.x:p==1?v.y:p==2?v.z:v.w;
    }
    if constexpr(Q==11) {
        #pragma unroll
        for(unsigned i=tid;i<Hi::Rows*8;i+=128) {
            unsigned row=i/8,col=i%8*8;
            uint4 v=*reinterpret_cast<uint4 const*>(high+uint64_t(k0/16+row)*extent+n0+col);
            #pragma unroll
            for(unsigned p=0;p<4;++p)upper[Hi::shared(col/2+p,row)]=p==0?v.x:p==1?v.y:p==2?v.z:v.w;
        }
    }
    __syncthreads();
    if constexpr(Q==11) {
        if(tid<64) {
            auto values=DenseRegisterAffine<KType::Q3_K,TileK>::load(metadata+tid*7,k0);
            #pragma unroll
            for(unsigned g=0;g<L::Groups;++g)scale[A::offset(tid,g)]=values.group(g).scale;
        }
        __syncthreads();
    }
    #pragma unroll
    for(unsigned i=tid;i<32*(TileK/8);i+=128) {
        unsigned pair=i/(TileK/8),kk=(i%(TileK/8))*8;
        uint4 a=*reinterpret_cast<uint4 const*>(tile+Lo::shared(pair,Lo::row(kk)));
        uint4 b=*reinterpret_cast<uint4 const*>(tile+Lo::shared(pair,Lo::row(kk)+4));
        uint4 ha{},hb{};
        if constexpr(Q==11) {
            ha=*reinterpret_cast<uint4 const*>(upper+Hi::shared(pair,Hi::row(kk)));
            hb=*reinterpret_cast<uint4 const*>(upper+Hi::shared(pair,Hi::row(kk)+4));
        }
        uint32_t words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
        uint32_t highs[8]={ha.x,ha.y,ha.z,ha.w,hb.x,hb.y,hb.z,hb.w};
        float s0=scale[A::offset(2*pair,kk/L::Group)],s1=scale[A::offset(2*pair+1,kk/L::Group)];
        uint32_t out0[4],out1[4];
        #pragma unroll
        for(unsigned p=0;p<4;++p) {
            uint32_t x=L::codes(words[2*p],highs[2*p],kk),y=L::codes(words[2*p+1],highs[2*p+1],kk);
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
template<int Q,bool BF16> void dense_plane_pair_launch(qzd_call_v1 const& c,int variant) {
    auto l=static_cast<uint16_t const*>(c.low),h=static_cast<uint16_t const*>(c.high);
    auto u=static_cast<uint8_t const*>(c.units);auto o=static_cast<uint16_t*>(c.output);
    auto s=static_cast<hggcStream_t>(c.stream);
    if(variant==QKP_DENSE_DEQUANT_PLANE_PAIR_K128)
        dense_dequant_plane_pair<Q,BF16,128><<<dim3(c.n/64,c.k/128),128,0,s>>>(l,h,u,o,c.n,c.k);
    else dense_dequant_plane_pair<Q,BF16,256><<<dim3(c.n/64,c.k/256),128,0,s>>>(l,h,u,o,c.n,c.k);
}
// One read per packed word; shared exchange preserves contiguous K8 stores.
template<bool BF16>
__global__ __launch_bounds__(128) void dense_dequant_q8(
        uint16_t const* low, uint16_t const* units, uint16_t* output, int n, int k) {
    using L = DenseQ8;
    __shared__ uint32_t tile[L::Words];
    __shared__ uint16_t scales[L::N];
    unsigned tid = threadIdx.x, n0 = blockIdx.x * L::N, k0 = blockIdx.y * L::K;
    unsigned row = tid / 8, col = tid % 8 * 8;
    auto source = low + uint64_t(k0 / 2 + row) * unsigned(n) + n0 + col;
    uint4 v = *reinterpret_cast<uint4 const*>(source);
    unsigned at = L::shared(row, col);
    tile[at] = v.x; tile[at + 1] = v.y; tile[at + 2] = v.z; tile[at + 3] = v.w;
    if (tid < L::N) scales[tid] = units[uint64_t(k0 / 32) * unsigned(n) + n0 + tid];
    __syncthreads();
    #pragma unroll
    for (unsigned part = 0; part < 2; ++part) {
        unsigned column = tid / 4 + part * 32, kk = tid % 4 * 8;
        uint32_t out[4];
        #pragma unroll
        for (unsigned p = 0; p < 4; ++p) {
            auto a = tile[L::shared(L::packed_row(kk + 2*p), column)];
            auto b = tile[L::shared(L::packed_row(kk + 2*p + 1), column)];
            out[p] = uint32_t(L::weight<BF16>(L::code(a, column, kk), scales[column])) |
                     (uint32_t(L::weight<BF16>(L::code(b, column, kk), scales[column])) << 16);
        }
        auto dst = output + uint64_t(n0 + column) * unsigned(k) + k0 + kk;
        *reinterpret_cast<uint4*>(dst) = {out[0], out[1], out[2], out[3]};
    }
}
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
    if (!qkp_dense_dequant_variant_supports(c->qtype,variant)) return QKG_FORMAT;
    if (c->k/128>65535 || c->n/32>65535 || (int64_t(c->n)*c->k+255)/256>INT32_MAX) return QKG_SHAPE;
    uint64_t bytes=uint64_t(c->n)*c->k*2;
    if (c->low_bytes<s.low_bytes || c->high_bytes<s.high_bytes || c->unit_bytes<s.units_bytes || c->output_bytes<bytes) return QKG_CAPACITY;
    uintptr_t p[]={uintptr_t(c->low),uintptr_t(c->high),uintptr_t(c->units),uintptr_t(c->output)};
    uint64_t len[]={s.low_bytes,s.high_bytes,s.units_bytes,bytes};
    if (c->qtype==8 && c->high) return QKG_INVALID;
    for (int i=0;i<4;++i) if (len[i]) {
        if (!p[i] || p[i]%16 || !span(p[i],len[i])) return QKG_INVALID;
        for (int j=0;j<i;++j) if(overlap(p[i],len[i],p[j],len[j])) return QKG_INVALID;
    }
    if(variant==QKP_DENSE_DEQUANT_PLANE_PAIR_K128 || variant==QKP_DENSE_DEQUANT_PLANE_PAIR_K256) {
        if(c->qtype==8) {
            if(storage)dense_plane_pair_launch<8,true>(*c,variant);
            else dense_plane_pair_launch<8,false>(*c,variant);
        } else {
            if(storage)dense_plane_pair_launch<11,true>(*c,variant);
            else dense_plane_pair_launch<11,false>(*c,variant);
        }
        return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
    }
    if (c->qtype==8) {
        if (c->k/32>65535) return QKG_SHAPE;
        auto stream=static_cast<hggcStream_t>(c->stream);
        auto low=static_cast<uint16_t const*>(c->low), units=static_cast<uint16_t const*>(c->units);
        auto output=static_cast<uint16_t*>(c->output);
        if (storage) dense_dequant_q8<true><<<dim3(c->n/64,c->k/32),128,0,stream>>>(low,units,output,c->n,c->k);
        else dense_dequant_q8<false><<<dim3(c->n/64,c->k/32),128,0,stream>>>(low,units,output,c->n,c->k);
        return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
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
