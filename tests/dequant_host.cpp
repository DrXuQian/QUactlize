#include <hggc_runtime.h>
#include "quactlize/dequant/reader.hpp"
#include "quactlize/dequant/api.h"
#include "quactlize/dequant/unit16.hpp"
#include "quactlize/dequant/transpose_layout.hpp"
#include <vector>

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
int cached16(uint8_t const* units, uint16_t* scale, uint16_t* zero, int n, int k) {
    using U=quactlize::dequant::Unit16<T>;
    using R=quactlize::dequant::Reader<T>;
    for(int sb=0;sb<k/256;++sb) for(int col=0;col<n;++col) {
        auto const* p=units+(int64_t(sb)*n+col)*16;
        auto v=U::load(p);
        for(int g=0;g<8;++g) {
            auto sz=v.scale(g);int64_t o=(int64_t(sb)*8+g)*n+col;
            scale[o]=sz.scale.raw();zero[o]=sz.zero.raw();
            auto a=v.affine(g),b=R::affine(units,col,sb*8+g,n);
            if(a.scale!=b.scale || a.minimum!=b.minimum) return 2;
        }
    }
    return 0;
}
extern "C" int dequant_unit16_host(int q,uint8_t const* u,uint16_t* s,uint16_t* z,int n,int k) {
    if(q==12)return cached16<gguf_scale::KType::Q4_K>(u,s,z,n,k);
    if(q==13)return cached16<gguf_scale::KType::Q5_K>(u,s,z,n,k);
    return 1;
}

extern "C" int dequant_shared_offset(int stage_k, int row, int k) {
    using namespace quactlize::dequant;
    if (stage_k == 128) return FullTransposeLayout<128>::offset(row,k);
    if (stage_k == 256) return FullTransposeLayout<256>::offset(row,k);
    return -1;
}

template<gguf_scale::KType T, int TileK, int StageK, bool CacheMetadata>
int shared_decode(uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* out,int n,int k,int fault) {
    using namespace quactlize::dequant;
    using R=Reader<T>;
    using Layout=FullTransposeLayout<StageK>;
    std::vector<uint32_t> tile(Layout::kCells);
    std::vector<int> owners(Layout::kCells);
    Unit16<T> metadata[32];
    for(int n0=0;n0<n;n0+=32) for(int k0=0;k0<k;k0+=TileK) {
        auto const* base=u+int64_t(k0/256)*n*16;
        if constexpr(CacheMetadata) for(int col=0;col<32;++col) metadata[col]=Unit16<T>::load(base+(n0+col)*16);
        for(int stage=0;stage<TileK/StageK;++stage) {
            std::fill(owners.begin(),owners.end(),0);
            for(int t=0;t<128;++t) {
                int lane=t%32,warp=t/32,col0=(lane%4)*8,residue=lane/4;
                for(int v=0;v<8;++v) {
                    int col=col0+v;
                    auto meta=CacheMetadata?metadata[col]:Unit16<T>::load(base+(n0+col)*16);
                    for(int part=0;part<StageK/128;++part) {
                        int kb=k0+stage*StageK+part*128+warp*32;
                        auto affine=meta.affine((kb/32)%8);
                        int64_t lp=R::LowMap::word_index(n0+col0,kb+residue,n);
                        uint16_t lw=l[lp+v],hw=0;
                        if constexpr(R::hi_bits) hw=h[R::HighMap::word_index(n0+col0,kb+residue,n)+v];
                        for(int s=0;s<4;++s) {
                            int kk=kb+residue+8*s, local_k=part*128+warp*32+residue+8*s;
                            int index=Layout::offset(col,local_k);
                            if(++owners[index]!=1) return 2;
                            tile[index]=R::weight(R::raw_from_words(lw,hw,n0+col,kk),affine);
                        }
                    }
                }
            }
            for(int count:owners) if(count!=1) return 3;
            for(int i=0;i<Layout::kCells/8;++i) {
                int row=i/(StageK/8),kk=(i%(StageK/8))*8;
                // The consumer performs two real K4 contiguous-vector reads.
                for(int half=0;half<2;++half) {
                    int pos=Layout::offset(row,kk+4*half);
                    if(fault) pos^=8;
                    for(int v=0;v<4;++v)
                        out[int64_t(n0+row)*k+k0+stage*StageK+kk+4*half+v]=tile[pos+v];
                }
            }
        }
    }
    return 0;
}
template<gguf_scale::KType T>
int shared_config(int c,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,int n,int k,int fault) {
    if(c==6)return shared_decode<T,128,128,false>(l,h,u,o,n,k,fault);
    if(c==7)return shared_decode<T,128,128,true>(l,h,u,o,n,k,fault);
    if(c==8)return shared_decode<T,256,256,true>(l,h,u,o,n,k,fault);
    if(c==9)return shared_decode<T,256,128,true>(l,h,u,o,n,k,fault);
    return 1;
}
extern "C" int dequant_shared_host(int q,int c,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,int n,int k,int fault) {
    if(q==12)return shared_config<gguf_scale::KType::Q4_K>(c,l,h,u,o,n,k,fault);
    if(q==13)return shared_config<gguf_scale::KType::Q5_K>(c,l,h,u,o,n,k,fault);
    return 1;
}

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
