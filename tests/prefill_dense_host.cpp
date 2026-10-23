// Host execution of production address/metadata readers. Python compares
// against independent raw GGUF dequantization; fault arms must fail.
#include <hggc_runtime.h>
#include "quactlize/dequant/dense_reader.hpp"
#include "quactlize/dequant/dense_q8.hpp"
#include "quactlize/dequant/dense_cooperative.hpp"
#include "quactlize/dequant/dense_plane_pair.hpp"
#include "quactlize/dequant/dense_q8_exchange.hpp"
#include "quactlize/dequant/dense_q8_tuned.hpp"
#include "quactlize/prefill/dense_api.h"
#include "quactlize/dequant/packed_exchange.hpp"
#include "quactlize/prefill/dense_io_layout.hpp"
#include <cstring>
#include <vector>

namespace plane_pair_oracle {
template<unsigned Bits,unsigned TK,unsigned Threads=128>
int stage(uint16_t const* src,std::vector<uint32_t>& dst,unsigned n,unsigned k,unsigned n0,unsigned k0,int fault) {
    using L=quactlize::dequant::DensePlanePairLayout<Bits,TK>;
    std::vector<unsigned> writes(dst.size());
    for(unsigned tid=0;tid<Threads;++tid)for(unsigned i=tid;i<L::Rows*8;i+=Threads) {
        if(fault==2 && i==0)continue;
        unsigned row=i/8,col=i%8*8;
        uint64_t at=uint64_t(k0*L::Rows/TK+row)*n+n0+col;
        if(at%8 || at+8>uint64_t(n)*k*Bits/16)return 1;
        for(unsigned p=0;p<4;++p) {
            unsigned out=L::shared(col/2+p,row);
            if(out>=writes.size() || ++writes[out]!=1)return 2;
            dst[out]=uint32_t(src[at+2*p])|(uint32_t(src[at+2*p+1])<<16);
        }
    }
    for(unsigned w:writes)if(w!=1)return 3;
    return 0;
}
template<int Q,bool BF16,unsigned TK,bool Exchange=false,bool Fast=false,unsigned Threads=128,bool VectorScale=false>
int decode(uint16_t const* low,uint16_t const* high,uint8_t const* units,uint16_t* output,unsigned n,unsigned k,int fault) {
    using namespace quactlize::dequant;
    using L=DensePlanePair<Q,TK>;using Lo=typename L::Low;using Hi=typename L::High;using A=typename L::Affine;
    constexpr unsigned UnitBytes=Q==11?28:20, UnitK=Q==11?512:256;
    std::vector<uint32_t> tile(Lo::Cells),upper(Hi::Cells),metadata(64*UnitBytes/4);
    std::vector<float> scale(64*L::Groups),minimum(64*L::Groups);
    std::vector<unsigned> wa(scale.size()),wo(64*TK);
    for(unsigned n0=0;n0<n;n0+=64)for(unsigned k0=0;k0<k;k0+=TK) {
        if(stage<Q==8?8:2,TK,Threads>(low,tile,n,k,n0,k0,fault))return 4;
        if constexpr(Q==11)if(stage<1,TK>(high,upper,n,k,n0,k0,fault==5?2:0))return 5;
        std::fill(wa.begin(),wa.end(),0);std::fill(wo.begin(),wo.end(),0);
        if constexpr(Q!=8) {
            uint64_t start=(uint64_t(k0/UnitK)*n+n0)*UnitBytes;
            if(start%16 || start+metadata.size()*4>uint64_t(n)*k/UnitK*UnitBytes)return 6;
            std::memcpy(metadata.data(),units+start,metadata.size()*4);
            for(unsigned col=0;col<64;++col) {
                auto values=DenseRegisterAffine<Q==10?gguf_scale::KType::Q2_K:gguf_scale::KType::Q3_K,TK>::load(
                    metadata.data()+col*(UnitBytes/4),k0^(fault==4?(Q==11?256:128):0));
                for(unsigned g=0;g<L::Groups;++g) {
                    unsigned at=A::offset(col,g);
                    if(at>=wa.size() || ++wa[at]!=1)return 7;
                    scale[at]=values.group(g).scale;
                    if constexpr(Q==10)minimum[at]=values.group(g).minimum;
                }
            }
        } else if constexpr(VectorScale) {
            for(unsigned tid=0;tid<64;++tid) {
                unsigned col=tid%16*4,g=tid/16;
                uint16_t words[4];
                std::memcpy(words,reinterpret_cast<uint16_t const*>(units)+uint64_t(k0/32+g)*n+n0+col,8);
                for(unsigned v=0;v<4;++v) {
                    unsigned at=A::offset(col+v,g);
                    if(at>=wa.size() || ++wa[at]!=1)return 8;
                    scale[at]=float(cutlass::half_t::bitcast(words[v]));
                }
            }
        } else {
            for(unsigned i=0;i<64*L::Groups;++i) {
                unsigned col=i%64,g=i/64,at=A::offset(col,g);
                uint16_t d=reinterpret_cast<uint16_t const*>(units)[uint64_t(k0/32+g)*n+n0+col];
                if(at>=wa.size() || ++wa[at]!=1)return 8;
                scale[at]=float(cutlass::half_t::bitcast(d));
            }
        }
        for(unsigned w:wa)if(w!=1)return 9;
        for(unsigned tid=0;tid<Threads;++tid)for(unsigned i=tid;i<32*(TK/8);i+=Threads) {
            unsigned pair=i/(TK/8),kk=i%(TK/8)*8;
            for(unsigned v=0;v<8;++v) {
                unsigned la=Lo::shared(pair,Lo::row(kk)+v),ha=Hi::shared(pair,Hi::row(kk)+v);
                if(la>=tile.size() || ha>=upper.size() ||
                   la!=Lo::shared(pair,Lo::row(kk)+(v&~3u))+v%4 ||
                   ha!=Hi::shared(pair,Hi::row(kk)+(v&~3u))+v%4 ||
                   (v%4==0 && (la%4 || ha%4)))return 10;
                uint32_t word=tile[la];
                if constexpr(Exchange) {
                    using X=DenseQ8Exchange<TK>;
                    unsigned partner=X::partner(i)^(fault==6?2:0);
                    if((partner/32)!=(i/32))return 14;
                    word=X::word(i,v,tile[X::source(i)+v%4],tile[X::source(partner)+v%4]);
                }
                uint32_t raw=L::codes(word,Q==11?upper[ha]:0,kk^(fault==1?8:0));
                for(unsigned side=0;side<2;++side) {
                    unsigned row=2*pair+side,out=row*TK+kk+v;
                    if(++wo[out]!=1)return 11;
                    auto destination=uint64_t(n0+row)*k+k0+kk+v;
                    if(v==0 && destination%8)return 12;
                    auto code=(raw>>(16*(side^(fault==3))))&0xffffu;
                    auto s=scale[A::offset(row,kk/L::Group)];
                    output[destination]=Fast?DenseQ8Fast::weight<BF16>(code,s):
                        L::template weight<BF16>(code,s,minimum[A::offset(row,kk/L::Group)]);
                }
            }
        }
        for(unsigned w:wo)if(w!=1)return 13;
    }
    return 0;
}
template<int Q,bool BF16> int geometry(int v,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,unsigned n,unsigned k,int f) {
    if constexpr(Q==8) {
#define QKP_Q8_TUNE(ID, FAST, THREADS, VECTOR) if(v==ID)return decode<Q,BF16,128,false,FAST,THREADS,VECTOR>(l,h,u,o,n,k,f);
#include "quactlize/prefill/dense_q8_tuning.inc"
#undef QKP_Q8_TUNE
        if(v==24)return decode<Q,BF16,128,true>(l,h,u,o,n,k,f);
        if(v==25)return decode<Q,BF16,256,true>(l,h,u,o,n,k,f);
    }
    if(v==22)return decode<Q,BF16,128>(l,h,u,o,n,k,f);
    if(v==23)return decode<Q,BF16,256>(l,h,u,o,n,k,f);
    return 99;
}
template<int Q> int typed(int b,int v,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,unsigned n,unsigned k,int f) {
    return b?geometry<Q,true>(v,l,h,u,o,n,k,f):geometry<Q,false>(v,l,h,u,o,n,k,f);
}
extern "C" int dense_plane_pair_host(int q,int b,int v,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,unsigned n,unsigned k,int f) {
    if(q==8)return typed<8>(b,v,l,h,u,o,n,k,f);
    if(q==10)return typed<10>(b,v,l,h,u,o,n,k,f);
    if(q==11)return typed<11>(b,v,l,h,u,o,n,k,f);
    return 99;
}
} // namespace plane_pair_oracle

extern "C" int dense_q8_fast_exhaustive(uint64_t* compared) {
    using namespace quactlize::dequant;
    *compared=0;
    for(unsigned raw=0;raw<256;++raw) {
        if(DenseQ8Fast::code(raw)!=float(int(raw)-128))return 1;
        for(unsigned bits=0;bits<65536;++bits) {
            if((bits&0x7c00u)==0x7c00u)continue;
            float s=float(cutlass::half_t::bitcast(uint16_t(bits)));
            if(DenseQ8Fast::weight<false>(raw,s)!=DensePlanePair<8,128>::weight<false>(raw,s))return 2;
            if(DenseQ8Fast::weight<true>(raw,s)!=DensePlanePair<8,128>::weight<true>(raw,s))return 3;
            *compared+=2;
        }
    }
    return 0;
}

extern "C" uint64_t dense_q8_tuned_address(unsigned kind,unsigned n,unsigned k,unsigned n0,unsigned k0,unsigned task,unsigned side) {
    using L=quactlize::dequant::DenseQ8TunedLayout;
    if(kind==0)return L::low(n,n0,k0,task);
    if(kind==1 || kind==2)return L::scale(n,n0,k0,task,kind==2);
    return L::output(k,n0,k0,task,side);
}

extern "C" int dense_q8_tuned_config(int variant,int* fast,int* threads,int* vector_scale) {
    return qkp_dense_q8_tuning(variant,fast,threads,vector_scale);
}

extern "C" unsigned dense_q8_exchange_address(unsigned tk,unsigned task,unsigned which) {
    using A=quactlize::dequant::DenseQ8Exchange<128>;
    using B=quactlize::dequant::DenseQ8Exchange<256>;
    if(which==0)return tk==128?A::source(task):B::source(task);
    if(which==1)return tk==128?A::partner(task):B::partner(task);
    if(which==2)return tk==128?A::pair(task):B::pair(task);
    return tk==128?A::k(task):B::k(task);
}

extern "C" int dense_q8_host(int bf,uint16_t const* low,uint16_t const* units,
                            uint16_t* output,unsigned n,unsigned k,int fault) {
    using L=quactlize::dequant::DenseQ8;
    uint32_t tile[L::Words];
    unsigned writes[L::Words];
    for(unsigned n0=0;n0<n;n0+=64)for(unsigned k0=0;k0<k;k0+=32) {
        std::fill(writes,writes+L::Words,0);
        for(unsigned tid=0;tid<128;++tid) {
            unsigned row=tid/8,col=tid%8*8,at=L::shared(row,col);
            uint64_t src=uint64_t(k0/2+row)*n+n0+col;
            if(src%8)return 1;
            for(unsigned p=0;p<4;++p) {
                if(++writes[at+p]!=1)return 2;
                tile[at+p]=uint32_t(low[src+2*p])|(uint32_t(low[src+2*p+1])<<16);
            }
        }
        for(unsigned tid=0;tid<128;++tid)for(unsigned part=0;part<2;++part) {
            unsigned col=tid/4+part*32,kk=tid%4*8;
            if((uint64_t(n0+col)*k+k0+kk)%8)return 3;
            uint16_t d=units[uint64_t(k0/32)*n+n0+col];
            for(unsigned v=0;v<8;++v) {
                unsigned at=L::shared(L::packed_row(kk+v),col);
                if(writes[at]!=1)return 4;
                int code=L::code(tile[at],col,kk^(fault?8:0));
                output[uint64_t(n0+col)*k+k0+kk+v]=bf?L::weight<true>(code,d):L::weight<false>(code,d);
            }
        }
    }
    return 0;
}

namespace pair_oracle {
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
            for(unsigned i=tid;i<(TK/32)*8*V;i+=128) {
                if(fault==2 && i==0)continue;
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
        default:return 99;
    }
}
template<gguf_scale::KType T> int typed(int bf,int variant,uint16_t const* l,uint16_t const* h,
        uint8_t const* u,uint16_t* out,int n,int k,int fault) {
    return bf?geometry<T,true>(variant,l,h,u,out,n,k,fault):geometry<T,false>(variant,l,h,u,out,n,k,fault);
}
extern "C" int dense_pair_host(int q,int bf,int variant,uint16_t const* l,uint16_t const* h,uint8_t const* u,
        uint16_t* out,int n,int k,int fault) {
    using gguf_scale::KType;
    if(q==12)return typed<KType::Q4_K>(bf,variant,l,h,u,out,n,k,fault);
    if(q==13)return typed<KType::Q5_K>(bf,variant,l,h,u,out,n,k,fault);
    if(q==14)return typed<KType::Q6_K>(bf,variant,l,h,u,out,n,k,fault);
    return 99;
}

} // namespace pair_oracle

namespace cooperative_oracle {
template<gguf_scale::KType T,bool B,unsigned TN,unsigned TK,bool Metadata>
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
            if constexpr(R::hi_bits) {
                hi=R::HighMap::word_index(n0+hp.col,k0+hp.group*32+hp.residue,n);
                if(hi%8 || hi+8>uint64_t(n)*k*R::hi_bits/16)return 5;
            }
            for(unsigned p=0;p<4;++p) {
                unsigned lo=L::low(lp.col/2+p,lp.group*8+lp.residue);
                if(lo>=wl.size() || ++wl[lo]!=1)return 6;
                tile[lo]=uint32_t(low[li+2*p])|(uint32_t(low[li+2*p+1])<<16);
                if constexpr(R::hi_bits) {
                    unsigned at=L::high(hp.col/2+p,hp.group*8+hp.residue);
                    if(at>=wh.size() || ++wh[at]!=1)return 7;
                    uint32_t h=uint32_t(high[hi+2*p])|(uint32_t(high[hi+2*p+1])<<16);
                    upper[at]=C::high(h,n0+hp.col+2*p,k0+hp.group*32);
                }
            }
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
    switch(v) {
        case 20:return decode<T,B,64,128,true>(l,h,u,o,n,k,f);
        case 21:return decode<T,B,64,256,true>(l,h,u,o,n,k,f);
        default:return 99;
    }
}
template<gguf_scale::KType T> int typed(int b,int v,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,int n,int k,int f) {
    return b?geometry<T,true>(v,l,h,u,o,n,k,f):geometry<T,false>(v,l,h,u,o,n,k,f);
}
extern "C" int dense_cooperative_host(int q,int b,int v,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* o,int n,int k,int f) {
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
extern "C" unsigned dense_cooperative_layout(unsigned n,unsigned k,unsigned plane,unsigned a,unsigned b) {
    if(n==16)return k==128?layout<16,128>(plane,a,b):layout<16,256>(plane,a,b);
    if(n==32)return k==128?layout<32,128>(plane,a,b):layout<32,256>(plane,a,b);
    return k==128?layout<64,128>(plane,a,b):layout<64,256>(plane,a,b);
}
extern "C" uint64_t dense_full4(uint64_t block,unsigned thread,unsigned part) {
    return quactlize::prefill::DenseFull4<256>::element(block,thread,part);
}

} // namespace cooperative_oracle

extern "C" uint64_t dense_stripe4(uint64_t block,unsigned thread,unsigned part){
    return quactlize::prefill::DenseStripe4<256>::element(block,thread,part);
}
extern "C" uint64_t dense_unsigned_index(unsigned bits,unsigned n,unsigned k,unsigned extent){
    using namespace quactlize::dequant;
    if(bits==4)return DenseUnsignedPlane<4>::word_index(n,k,extent);
    if(bits==2)return DenseUnsignedPlane<2>::word_index(n,k,extent);
    return DenseUnsignedQ5High::word_index(n,k,extent);
}


template<gguf_scale::KType T,bool B>
void direct(uint16_t const* low,uint16_t const* high,uint8_t const* units,uint16_t* out,int n,int k) {
    using R=quactlize::dequant::DenseReader<T,B>;
    for(int col=0;col<n;++col)for(int kk=0;kk<k;++kk) {
        int raw=R::code(low,high,col,kk,n)+(R::lo_bits==4?8:0);
        out[uint64_t(col)*k+kk]=R::weight(raw,R::affine(units,col,kk/R::group,n));
    }
}
template<gguf_scale::KType T>
void direct_typed(int bf,uint16_t const* l,uint16_t const* h,uint8_t const* u,uint16_t* out,int n,int k) {
    if(bf)direct<T,true>(l,h,u,out,n,k);else direct<T,false>(l,h,u,out,n,k);
}
extern "C" int dense_direct_host(int q,int bf,uint16_t const* l,uint16_t const* h,
                                 uint8_t const* u,uint16_t* out,int n,int k) {
    using gguf_scale::KType;
    switch(q) {
        case 10:direct_typed<KType::Q2_K>(bf,l,h,u,out,n,k);break;
        case 11:direct_typed<KType::Q3_K>(bf,l,h,u,out,n,k);break;
        case 12:direct_typed<KType::Q4_K>(bf,l,h,u,out,n,k);break;
        case 13:direct_typed<KType::Q5_K>(bf,l,h,u,out,n,k);break;
        case 14:direct_typed<KType::Q6_K>(bf,l,h,u,out,n,k);break;
        default:return 99;
    }
    return 0;
}
