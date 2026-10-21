#include <hggc_runtime.h>
#include "quactlize/dequant/dense_reader.hpp"
#include "quactlize/dequant/packed_exchange.hpp"
#include <vector>

template<gguf_scale::KType T,bool BF16,int TileK>
int decode(uint16_t const* low,uint16_t const* high,uint8_t const* units,uint16_t* output,int n,int k,int fault) {
    using namespace quactlize::dequant;using R=DenseReader<T,BF16>;using L=PackedExchangeLayout<TileK>;
    constexpr int G=TileK/R::group;
    using A=DenseAffineLayout<G>;
    std::vector<uint32_t> tile(L::kCells);std::vector<typename R::Affine> aff(32*G);
    std::vector<int> owned(L::kCells);
    for(int n0=0;n0<n;n0+=32)for(int k0=0;k0<k;k0+=TileK) {
        std::fill(owned.begin(),owned.end(),0);
        for(int i=0;i<32*G;++i)aff[A::offset(i%32,i/32)]=R::affine(units,n0+i%32,k0/R::group+i/32,n);
        for(int tid=0;tid<128;++tid)for(int part=0;part<TileK/128;++part) {
            int lane=tid%32,warp=tid/32,c0=(lane%4)*8,r=lane/4,kb=k0+part*128+warp*32;
            auto li=R::LowMap::word_index(n0+c0,kb+r,n);
            if((li*2)%16)return 1;
            int64_t hi=0;
            if constexpr(R::hi_bits){hi=R::HighMap::word_index(n0+c0,kb+r,n);if((hi*2)%16)return 2;}
            for(int v=0;v<8;++v) {
                uint16_t h=R::hi_bits?high[hi+v]:0;
                int idx=L::offset(c0+v,(part*4+warp)*8+r);
                if(++owned[idx]!=1)return 3;
                tile[idx]=DenseCodes<T>::combine(low[li+v],h,n0+c0+v,kb+r);
            }
        }
        for(int x:owned)if(x!=1)return 4;
        for(int i=0;i<32*TileK/8;++i) {
            int row=i/(TileK/8),kk=(i%(TileK/8))*8,g=kk/32,s=(kk/8)%4;
            auto a=aff[A::offset(row,kk/R::group)];
            if(fault)s^=1;
            for(int half=0;half<2;++half) {
                int offset=L::offset(row,g*8+half*4);if(offset%4)return 5;
                for(int v=0;v<4;++v)output[int64_t(n0+row)*k+k0+kk+half*4+v]=
                    R::weight(DenseCodes<T>::code(tile[offset+v],s),a);
            }
        }
    }
    return 0;
}
template<gguf_scale::KType T> int typed(int bf,int tk,uint16_t const* l,uint16_t const* h,
    uint8_t const* u,uint16_t* o,int n,int k,int fault) {
    if(bf)return tk==128?decode<T,true,128>(l,h,u,o,n,k,fault):decode<T,true,256>(l,h,u,o,n,k,fault);
    return tk==128?decode<T,false,128>(l,h,u,o,n,k,fault):decode<T,false,256>(l,h,u,o,n,k,fault);
}
extern "C" int dense_host(int q,int bf,int tk,uint16_t const* l,uint16_t const* h,uint8_t const* u,
    uint16_t* o,int n,int k,int fault) {
    using gguf_scale::KType;
    if(q==12)return typed<KType::Q4_K>(bf,tk,l,h,u,o,n,k,fault);
    if(q==13)return typed<KType::Q5_K>(bf,tk,l,h,u,o,n,k,fault);
    if(q==14)return typed<KType::Q6_K>(bf,tk,l,h,u,o,n,k,fault);
    return 1;
}
