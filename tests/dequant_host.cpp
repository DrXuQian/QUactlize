#include <hggc_runtime.h>
#include "quactlize/dequant/reader.hpp"
#include "quactlize/dequant/api.h"

template<gguf_scale::KType T>
void decode(uint16_t const* low, uint16_t const* high, uint8_t const* units,
            uint16_t* out, int n, int k) {
    using R=quactlize::dequant::Reader<T>;
    for(int col=0;col<n;++col) for(int kk=0;kk<k;++kk) {
        int const raw=R::code(low,high,col,kk,n)+(R::lo_bits==4?8:0);
        out[int64_t(col)*k+kk]=R::weight(raw,R::affine(units,col,kk/R::group,n));
    }
}
extern "C" int dequant_host(int q,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,int n,int k) {
    using gguf_scale::KType;
    switch(q) {
        case 10:decode<KType::Q2_K>(l,h,u,o,n,k);break;
        case 11:decode<KType::Q3_K>(l,h,u,o,n,k);break;
        case 12:decode<KType::Q4_K>(l,h,u,o,n,k);break;
        case 13:decode<KType::Q5_K>(l,h,u,o,n,k);break;
        case 14:decode<KType::Q6_K>(l,h,u,o,n,k);break;
        default:return 1;
    }
    return 0;
}
extern "C" int dequant_call_size() { return sizeof(qzd_call_v1); }

template<gguf_scale::KType T>
void tiled(uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* out,int n,int k) {
    using R=quactlize::dequant::Reader<T>;
    for(int col=0;col<n;++col) for(int k0=0;k0<k;k0+=32) {
        auto first=R::affine(u,col,k0/R::group,n),second=first;
        if constexpr(R::group==16) second=R::affine(u,col,k0/R::group+1,n);
        for(int residue=0;residue<8;++residue) {
            uint16_t low=l[R::LowMap::word_index(col,k0+residue,n)],high=0;
            if constexpr(R::hi_bits) high=h[R::HighMap::word_index(col,k0+residue,n)];
            for(int slot=0;slot<4;++slot) {
                int const kk=k0+residue+8*slot;
                out[int64_t(col)*k+kk]=R::weight(R::raw_from_words(low,high,col,kk),slot<2?first:second);
            }
        }
    }
}
extern "C" int dequant_tiled_host(int q,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,int n,int k) {
    using gguf_scale::KType;
    switch(q) {
        case 10:tiled<KType::Q2_K>(l,h,u,o,n,k);break;
        case 11:tiled<KType::Q3_K>(l,h,u,o,n,k);break;
        case 12:tiled<KType::Q4_K>(l,h,u,o,n,k);break;
        case 13:tiled<KType::Q5_K>(l,h,u,o,n,k);break;
        case 14:tiled<KType::Q6_K>(l,h,u,o,n,k);break;
        default:return 1;
    }
    return 0;
}
