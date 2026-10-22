#include <hggc_runtime.h>
#include "quactlize/dequant/dense_unsigned.hpp"
#include "quactlize/dequant/packed_exchange.hpp"
#include "quactlize/prefill/dense_io_layout.hpp"
#include <vector>

// Execute every producer/consumer lane, using the actual device mapping and
// reader, against independent Python GGUF bits. Faults must fail this oracle.
template<gguf_scale::KType T,bool BF16,unsigned TN,unsigned TK,unsigned Threads>
int decode(uint16_t const* low,uint16_t const* high,uint8_t const* units,uint16_t* output,
           unsigned n,unsigned k,int fault) {
    using namespace quactlize::dequant;using R=DenseUnsignedReader<T,BF16>;using L=PackedExchangeLayout<TK>;
    constexpr unsigned G=TK/R::group,V=TN/8;using A=DenseAffineLayout<G>;
    std::vector<uint32_t> tile(TN*L::kWords);std::vector<typename R::Affine> affine(TN*G);
    std::vector<unsigned> writes(tile.size()),metadata(affine.size()),reads(TN*TK);
    for(unsigned n0=0;n0<n;n0+=TN)for(unsigned k0=0;k0<k;k0+=TK){
        std::fill(writes.begin(),writes.end(),0);std::fill(metadata.begin(),metadata.end(),0);
        std::fill(reads.begin(),reads.end(),0);
        for(unsigned tid=0;tid<Threads;++tid){
            for(unsigned i=tid;i<TN*G;i+=Threads){
                unsigned col=i%TN,g=i/TN,index=A::offset(col,g);
                if(index>=metadata.size() || ++metadata[index]!=1)return 1;
                affine[index]=R::affine(units,n0+col,k0/R::group+g,n);
            }
            for(unsigned i=tid;i<(TK/32)*8*V;i+=(fault==2?128:Threads)){
                unsigned c0=(i%V)*8,r=(i/V)%8,g=i/(V*8),kb=k0+g*32;
                auto li=R::LowMap::word_index(n0+c0,kb+r,n);
                if(li%8)return 2;
                uint64_t hi=0;
                if constexpr(R::hi_bits){hi=R::HighMap::word_index(n0+c0,kb+r,n);if(hi%8)return 3;}
                for(unsigned v=0;v<8;++v){
                    unsigned index=L::offset(c0+v,g*8+r);
                    if(index>=writes.size() || ++writes[index]!=1)return 4;
                    tile[index]=R::combine(low[li+v],R::hi_bits?high[hi+v]:0,n0+c0+v,kb+r);
                }
            }
        }
        for(unsigned x:writes)if(x!=1)return 5;
        for(unsigned x:metadata)if(x!=1)return 6;
        for(unsigned tid=0;tid<Threads;++tid)for(unsigned i=tid;i<TN*(TK/8);i+=Threads){
            unsigned row=i/(TK/8),kk=(i%(TK/8))*8,g=kk/32,slot=(kk/8)%4;
            if((uint64_t(n0+row)*k+k0+kk)%8)return 7;
            auto f=affine[A::offset(row,kk/R::group)];if(fault==1)slot^=1;
            for(unsigned half=0;half<2;++half){
                unsigned offset=L::offset(row,g*8+half*4);if(offset%4 || offset+4>tile.size())return 8;
                for(unsigned v=0;v<4;++v){
                    if(++reads[row*TK+kk+half*4+v]!=1)return 9;
                    output[uint64_t(n0+row)*k+k0+kk+half*4+v]=R::weight((tile[offset+v]>>(slot*8))&255,f);
                }
            }
        }
        for(unsigned x:reads)if(x!=1)return 10;
    }
    return 0;
}
template<gguf_scale::KType T,bool B> int geometry(int variant,uint16_t const* l,uint16_t const* h,
        uint8_t const* u,uint16_t* out,int n,int k,int fault){
    if(variant==4)return decode<T,B,32,128,128>(l,h,u,out,n,k,fault);
    if(variant==5)return decode<T,B,32,256,128>(l,h,u,out,n,k,fault);
    if(variant==6)return decode<T,B,16,256,128>(l,h,u,out,n,k,fault);
    if(variant==7)return decode<T,B,32,256,256>(l,h,u,out,n,k,fault);
    return 99;
}
template<gguf_scale::KType T> int typed(int bf,int variant,uint16_t const* l,uint16_t const* h,
        uint8_t const* u,uint16_t* out,int n,int k,int fault){
    return bf?geometry<T,true>(variant,l,h,u,out,n,k,fault):geometry<T,false>(variant,l,h,u,out,n,k,fault);
}
extern "C" int dense_r2_host(int q,int bf,int variant,uint16_t const* l,uint16_t const* h,uint8_t const* u,
        uint16_t* out,int n,int k,int fault){
    using gguf_scale::KType;
    if(q==12)return typed<KType::Q4_K>(bf,variant,l,h,u,out,n,k,fault);
    if(q==13)return typed<KType::Q5_K>(bf,variant,l,h,u,out,n,k,fault);
    if(q==14)return typed<KType::Q6_K>(bf,variant,l,h,u,out,n,k,fault);
    return 99;
}
extern "C" uint64_t dense_stripe4(uint64_t block,unsigned thread,unsigned part){
    return quactlize::prefill::DenseStripe4<256>::element(block,thread,part);
}
extern "C" uint64_t dense_unsigned_index(unsigned bits,unsigned n,unsigned k,unsigned extent){
    using namespace quactlize::dequant;
    if(bits==4)return DenseUnsignedPlane<4>::word_index(n,k,extent);
    if(bits==2)return DenseUnsignedPlane<2>::word_index(n,k,extent);
    return DenseUnsignedQ5High::word_index(n,k,extent);
}
