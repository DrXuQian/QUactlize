#include <hggc_runtime.h>
#include "quactlize/dequant/dense_pair.hpp"
#include "quactlize/dequant/packed_exchange.hpp"
#include <vector>

// Enumerate the actual shared producer/consumer mappings, independently of
// the GPU launcher. Python supplies canonical bytes and raw-GGUF oracle bits.
template<gguf_scale::KType T,bool BF16,unsigned TN,unsigned TK,bool Fast>
int decode(uint16_t const* low,uint16_t const* high,uint8_t const* units,uint16_t* output,
           unsigned n,unsigned k,int fault) {
    using namespace quactlize::dequant;
    using R=DensePairReader<T,BF16,Fast>;using Codes=DensePairCodes<T>;using L=PackedExchangeLayout<TK>;
    constexpr unsigned G=TK/R::group,V=TN/8;using A=DenseAffineLayout<G>;
    std::vector<uint32_t> tile(TN/2*L::kWords);
    std::vector<uint16_t> upper(tile.size());std::vector<typename R::Affine> affine(TN*G);
    std::vector<unsigned> writes(tile.size()),metadata(affine.size()),reads(TN*TK);
    for(unsigned n0=0;n0<n;n0+=TN)for(unsigned k0=0;k0<k;k0+=TK) {
        std::fill(writes.begin(),writes.end(),0);std::fill(metadata.begin(),metadata.end(),0);
        std::fill(reads.begin(),reads.end(),0);
        for(unsigned tid=0;tid<128;++tid) {
            for(unsigned i=tid;i<TN*G;i+=128) {
                unsigned col=i%TN,g=i/TN,index=A::offset(col,g);
                if(index>=metadata.size() || ++metadata[index]!=1)return 1;
                affine[index]=R::affine(units,n0+col,k0/R::group+g,n);
            }
            for(unsigned i=tid;i<(TK/32)*8*V;i+=(fault==2?256:128)) {
                unsigned c0=(i%V)*8,r=(i/V)%8,g=i/(V*8),kb=k0+g*32;
                auto li=R::LowMap::word_index(n0+c0,kb+r,n);
                if(li%8)return 2;
                uint64_t hi=0;
                if constexpr(R::hi_bits) {hi=R::HighMap::word_index(n0+c0,kb+r,n);if(hi%8)return 3;}
                for(unsigned p=0;p<4;++p) {
                    unsigned at=L::offset(c0/2+p,g*8+r);
                    if(at>=writes.size() || ++writes[at]!=1)return 4;
                    tile[at]=uint32_t(low[li+2*p])|(uint32_t(low[li+2*p+1])<<16);
                    if constexpr(R::hi_bits) {
                        uint32_t h=uint32_t(high[hi+2*p])|(uint32_t(high[hi+2*p+1])<<16);
                        upper[at]=Codes::high(h,n0+c0+2*p,kb);
                    }
                }
            }
        }
        for(unsigned x:writes)if(x!=1)return 5;
        for(unsigned x:metadata)if(x!=1)return 6;
        for(unsigned tid=0;tid<128;++tid)for(unsigned i=tid;i<TN/2*(TK/8);i+=128) {
            unsigned pair=i/(TK/8),kk=(i%(TK/8))*8,g=kk/32,slot=(kk/8)%4;
            if(fault==1)slot^=1;
            for(unsigned part=0;part<2;++part) {
                unsigned offset=L::offset(pair,g*8+part*4);
                if(offset%4 || offset+4>tile.size())return 7;
                for(unsigned v=0;v<4;++v) {
                    unsigned raw=Codes::codes(tile[offset+v],upper[offset+v],slot);
                    for(unsigned side=0;side<2;++side) {
                        unsigned row=2*pair+side,index=row*TK+kk+part*4+v;
                        if(++reads[index]!=1)return 8;
                        if((uint64_t(n0+row)*k+k0+kk)%8)return 9;
                        auto f=affine[A::offset(row,kk/R::group)];
                        unsigned code=(raw>>(16*(side^(fault==3))))&0xffffu;
                        output[uint64_t(n0+row)*k+k0+kk+part*4+v]=R::weight(code,f);
                    }
                }
            }
        }
        for(unsigned x:reads)if(x!=1)return 10;
    }
    return 0;
}
template<gguf_scale::KType T,bool B> int geometry(int variant,uint16_t const* l,uint16_t const* h,
        uint8_t const* u,uint16_t* out,int n,int k,int fault) {
    switch(variant) {
        case 8:return decode<T,B,32,128,false>(l,h,u,out,n,k,fault);
        case 9:return decode<T,B,32,128,true>(l,h,u,out,n,k,fault);
        case 10:return decode<T,B,32,256,false>(l,h,u,out,n,k,fault);
        case 11:return decode<T,B,32,256,true>(l,h,u,out,n,k,fault);
        case 12:return decode<T,B,16,256,false>(l,h,u,out,n,k,fault);
        case 13:return decode<T,B,16,256,true>(l,h,u,out,n,k,fault);
        default:return 99;
    }
}
template<gguf_scale::KType T> int typed(int bf,int variant,uint16_t const* l,uint16_t const* h,
        uint8_t const* u,uint16_t* out,int n,int k,int fault) {
    return bf?geometry<T,true>(variant,l,h,u,out,n,k,fault):geometry<T,false>(variant,l,h,u,out,n,k,fault);
}
extern "C" int dense_r3_host(int q,int bf,int variant,uint16_t const* l,uint16_t const* h,uint8_t const* u,
        uint16_t* out,int n,int k,int fault) {
    using gguf_scale::KType;
    if(q==12)return typed<KType::Q4_K>(bf,variant,l,h,u,out,n,k,fault);
    if(q==13)return typed<KType::Q5_K>(bf,variant,l,h,u,out,n,k,fault);
    if(q==14)return typed<KType::Q6_K>(bf,variant,l,h,u,out,n,k,fault);
    return 99;
}
extern "C" float dense_r3_fp32(unsigned q,unsigned code) {
    using namespace quactlize::dequant;
    return q==14?DensePairCodes<KType::Q6_K>::value<true>(code):DensePairCodes<KType::Q4_K>::value<true>(code);
}
