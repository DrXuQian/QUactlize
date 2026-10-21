#include <hggc_runtime.h>
#include "../prefill/dense_api.h"
#include "dense_reader.hpp"
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
template<KType T,bool BF16>
int dense_launch(qzd_call_v1 const& c,int variant) {
    auto l=static_cast<uint16_t const*>(c.low),h=static_cast<uint16_t const*>(c.high);
    auto u=static_cast<uint8_t const*>(c.units);auto out=static_cast<uint16_t*>(c.output);
    auto s=static_cast<hggcStream_t>(c.stream);
    if (!variant) dense_dequant_direct<T,BF16><<<(int64_t(c.n)*c.k+255)/256,256,0,s>>>(l,h,u,out,c.n,c.k);
    else if constexpr(T==KType::Q4_K || T==KType::Q5_K || T==KType::Q6_K) {
        if (variant==1) dense_dequant_packed<T,BF16,128,false><<<dim3(c.n/32,c.k/128),128,0,s>>>(l,h,u,out,c.n,c.k);
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
        c->zero || (storage!=0 && storage!=1) || variant<0 || variant>3) return QKG_INVALID;
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
