#include <hggc_runtime.h>
#include "quactlize/dequant/dense_cooperative.hpp"
#include "quactlize/dequant/dense_high_unique.hpp"
#include "quactlize/prefill/dense_io_layout.hpp"
#include <set>
#include <vector>

template<gguf_scale::KType T,bool B,unsigned TN,unsigned TK,bool Metadata,bool UniqueHigh=false>
int decode(uint16_t const* low,uint16_t const* high,uint8_t const* units,uint16_t* output,
           unsigned n,unsigned k,int fault) {
    using namespace quactlize::dequant;
    using R=DensePairReader<T,B,false>;using C=DensePairCodes<T>;using U=typename R::U;
    using L=DenseCooperativeLayout<TN,TK>;constexpr unsigned G=TK/R::group,MW=TN*U::kUnitTotal/4;
    using A=DenseCooperativeAffine<G>;
    std::vector<uint32_t> tile(L::Cells),raw(MW);std::vector<uint16_t> upper(L::Cells);
    std::vector<typename R::Affine> affine(TN*G);
    std::vector<unsigned> wl(L::Cells),wh(L::Cells),wm(MW),wa(TN*G),wo(TN*TK);
    for(unsigned n0=0;n0<n;n0+=TN)for(unsigned k0=0;k0<k;k0+=TK) {
        for(auto* v:{&wl,&wh,&wm,&wa,&wo})std::fill(v->begin(),v->end(),0);
        if constexpr(Metadata) {
            uint64_t start=(uint64_t(k0/(256*U::kSbPerUnit))*n+n0)*U::kUnitTotal;
            if(start%16 || start+MW*4>uint64_t(n)*k/256*U::kSbBytes)return 1;
            for(unsigned tid=0;tid<128;++tid)for(unsigned i=tid;i<MW/4;i+=128) {
                if(fault==4 && i==MW/4-1)continue;
                std::memcpy(raw.data()+i*4,units+start+i*16,16);
                for(unsigned v=0;v<4;++v)if(++wm[i*4+v]!=1)return 2;
            }
            for(unsigned x:wm)if(x!=1)return 3;
        }
        for(unsigned tid=0;tid<128;++tid)for(unsigned i=tid;i<(TK/32)*8*L::V;i+=(fault==2?256:128)) {
            auto lp=L::producer(i,false),hp=L::producer(i,true);
            auto li=R::LowMap::word_index(n0+lp.col,k0+lp.group*32+lp.residue,n);
            if(li%8 || li+8>uint64_t(n)*k/4)return 4;
            uint64_t hi=0;
            if constexpr(R::hi_bits && !UniqueHigh) {
                hi=R::HighMap::word_index(n0+hp.col,k0+hp.group*32+hp.residue,n);
                if(hi%8 || hi+8>uint64_t(n)*k*R::hi_bits/16)return 5;
            }
            for(unsigned p=0;p<4;++p) {
                unsigned lo=L::low(lp.col/2+p,lp.group*8+lp.residue);
                if(lo>=wl.size() || ++wl[lo]!=1)return 6;
                tile[lo]=uint32_t(low[li+2*p])|(uint32_t(low[li+2*p+1])<<16);
                if constexpr(R::hi_bits && !UniqueHigh) {
                    unsigned at=L::high(hp.col/2+p,hp.group*8+hp.residue);
                    if(at>=wh.size() || ++wh[at]!=1)return 7;
                    uint32_t h=uint32_t(high[hi+2*p])|(uint32_t(high[hi+2*p+1])<<16);
                    upper[at]=C::high(h,n0+hp.col+2*p,k0+hp.group*32);
                }
            }
        }
        if constexpr(UniqueHigh) {
            using H=DenseUniqueHigh<T,TK>;
            std::set<uint64_t> reads;
            for(unsigned tid=0;tid<128;++tid)for(unsigned i=tid;i<H::Loads;i+=128) {
                if(fault==6 && i==H::Loads-1)continue;
                auto owner=H::owner(i);
                auto at=R::HighMap::word_index(n0+owner.col,k0+owner.group64*64+owner.residue,n);
                if(at%8 || at+8>uint64_t(n)*k*R::hi_bits/16 || !reads.insert(at).second)return 16;
                for(unsigned fan=0;fan<H::Fans;++fan) {
                    auto dst=H::target(owner,fan);
                    for(unsigned p=0;p<4;++p) {
                        unsigned index=L::high(dst.col/2+p,dst.group32*8+owner.residue);
                        if(index>=wh.size() || ++wh[index]!=1)return 17;
                        uint32_t h=uint32_t(high[at+2*p])|(uint32_t(high[at+2*p+1])<<16);
                        upper[index]=C::high(h,n0+dst.col+2*p,k0+(dst.group32^(fault==3?1:0))*32);
                    }
                }
            }
            if(reads.size()!=H::Loads)return 18;
        }
        for(unsigned x:wl)if(x!=1)return 8;
        if constexpr(R::hi_bits)for(unsigned x:wh)if(x!=1)return 9;
        for(unsigned col=0;col<TN;++col) {
            auto values=DenseRegisterAffine<T,TK>::load(raw.data()+col*(U::kUnitTotal/4),k0^(fault==5?128:0));
            for(unsigned g=0;g<G;++g) {
                unsigned at=A::offset(col,g);
                if(at>=wa.size() || ++wa[at]!=1)return 10;
                if constexpr(Metadata)affine[at]=values.group(g);
                else affine[at]=R::affine(units,n0+col,k0/R::group+g,n);
            }
        }
        for(unsigned x:wa)if(x!=1)return 11;
        for(unsigned tid=0;tid<128;++tid)for(unsigned i=tid;i<TN/2*(TK/8);i+=128) {
            unsigned pair=i/(TK/8),kk=(i%(TK/8))*8,g=kk/32,slot=(kk/8)%4;
            if(fault==1)slot^=1;
            for(unsigned part=0;part<2;++part) {
                unsigned lo=L::low(pair,g*8+part*4),hi=L::high(pair,g*8+part*4);
                if(lo%4 || hi%4 || lo+4>tile.size() || hi+4>upper.size())return 12;
                for(unsigned v=0;v<4;++v) {
                    if(L::low(pair,g*8+part*4+v)!=lo+v || L::high(pair,g*8+part*4+v)!=hi+v)return 13;
                    uint32_t codes=C::codes(tile[lo+v],upper[hi+v],slot);
                    for(unsigned side=0;side<2;++side) {
                        unsigned row=2*pair+side,at=row*TK+kk+part*4+v;
                        if(++wo[at]!=1)return 14;
                        output[uint64_t(n0+row)*k+k0+kk+part*4+v]=R::weight((codes>>(side*16))&0xffffu,affine[A::offset(row,kk/R::group)]);
                    }
                }
            }
        }
        for(unsigned x:wo)if(x!=1)return 15;
    }
    return 0;
}
template<gguf_scale::KType T,bool B> int geometry(int v,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,int n,int k,int f) {
    if constexpr(T==gguf_scale::KType::Q5_K || T==gguf_scale::KType::Q6_K) {
        if(v==22)return decode<T,B,64,128,true,true>(l,h,u,o,n,k,f);
        if(v==23)return decode<T,B,64,256,true,true>(l,h,u,o,n,k,f);
    }
    switch(v) {
        case 14:return decode<T,B,32,128,false>(l,h,u,o,n,k,f);
        case 15:return decode<T,B,32,256,false>(l,h,u,o,n,k,f);
        case 16:return decode<T,B,16,256,false>(l,h,u,o,n,k,f);
        case 17:return decode<T,B,32,128,true>(l,h,u,o,n,k,f);
        case 18:return decode<T,B,32,256,true>(l,h,u,o,n,k,f);
        case 19:return decode<T,B,16,256,true>(l,h,u,o,n,k,f);
        case 20:return decode<T,B,64,128,true>(l,h,u,o,n,k,f);
        case 21:return decode<T,B,64,256,true>(l,h,u,o,n,k,f);
        default:return 99;
    }
}
template<gguf_scale::KType T> int typed(int b,int v,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,int n,int k,int f) {
    return b?geometry<T,true>(v,l,h,u,o,n,k,f):geometry<T,false>(v,l,h,u,o,n,k,f);
}
extern "C" int dense_r4_host(int q,int b,int v,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,int n,int k,int f) {
    using gguf_scale::KType;
    if(q==12)return typed<KType::Q4_K>(b,v,l,h,u,o,n,k,f);
    if(q==13)return typed<KType::Q5_K>(b,v,l,h,u,o,n,k,f);
    if(q==14)return typed<KType::Q6_K>(b,v,l,h,u,o,n,k,f);
    return 99;
}
template<unsigned N,unsigned K> unsigned layout(unsigned plane,unsigned a,unsigned b) {
    using L=quactlize::dequant::DenseCooperativeLayout<N,K>;
    if(plane==0)return L::low(a,b);
    if(plane==1)return L::high(a,b);
    auto p=L::producer(a,b);return p.col|(p.residue<<8)|(p.group<<16);
}
extern "C" unsigned dense_r4_layout(unsigned n,unsigned k,unsigned plane,unsigned a,unsigned b) {
    if(n==16)return k==128?layout<16,128>(plane,a,b):layout<16,256>(plane,a,b);
    if(n==32)return k==128?layout<32,128>(plane,a,b):layout<32,256>(plane,a,b);
    return k==128?layout<64,128>(plane,a,b):layout<64,256>(plane,a,b);
}
extern "C" uint64_t dense_full4(uint64_t block,unsigned thread,unsigned part) {
    return quactlize::prefill::DenseFull4<256>::element(block,thread,part);
}
template<gguf_scale::KType T,unsigned K> unsigned high_unique(unsigned i,unsigned fan) {
    using H=quactlize::dequant::DenseUniqueHigh<T,K>;
    auto p=H::owner(i);auto dst=H::target(p,fan);
    return dst.col|(p.residue<<8)|(dst.group32<<16);
}
extern "C" unsigned dense_r5_high(unsigned q,unsigned k,unsigned i,unsigned fan) {
    using gguf_scale::KType;
    if(q==13)return k==128?high_unique<KType::Q5_K,128>(i,fan):high_unique<KType::Q5_K,256>(i,fan);
    return k==128?high_unique<KType::Q6_K,128>(i,fan):high_unique<KType::Q6_K,256>(i,fan);
}
