#include "quactlize/dequant/dense_q8_paired.hpp"
#include <array>
#include <vector>

using P=quactlize::dequant::DenseQ8Paired;
using L=P::Low;

extern "C" unsigned dense_q8_paired_map(unsigned kind,unsigned x,unsigned y) {
    return kind==0?P::scale(x,y):kind==1?L::shared(x,y):kind==2?L::row(x):L::shift(x);
}

extern "C" int dense_q8_paired_exhaustive(uint64_t* count) {
    *count=0;
    for(unsigned d=0;d<65536;++d) {
        if((d&0x7c00u)==0x7c00u)continue;
        unsigned other=(((d>>15)*31744+(d&32767))+12345)%63488;
        other+=other>=31744?1024:0;
        float s0=float(cutlass::half_t::bitcast(uint16_t(d)));
        float s1=float(cutlass::half_t::bitcast(uint16_t(other)));
        uint32_t scales=d|(other<<16);
        for(unsigned raw=0;raw<256;++raw) {
            uint32_t word=raw|((raw^90)<<8)|((255-raw)<<16)|((255-(raw^90))<<24);
            uint32_t got[2],want[2];
            for(unsigned phase=0;phase<2;++phase) {
                unsigned r=raw^(phase?90:0);
                want[phase]=quactlize::dequant::DensePlanePair<8,128>::weight<false>(r,s0)|
                    (uint32_t(quactlize::dequant::DensePlanePair<8,128>::weight<false>(255-r,s1))<<16);
                if(P::weight<true>(P::stage<true>(word),phase*8,scales)!=want[phase])return 1;
                got[phase]=P::weight<false>(word,phase*8,scales);
                if(got[phase]!=want[phase])return 2;
                if(P::weight<false,true>(word,phase*8,scales)!=want[phase])return 4;
                ++*count;
            }
            if(P::output0(got[0],got[1])!=((want[0]&65535u)|(want[1]<<16)) ||
               P::output1(got[0],got[1])!=((want[0]>>16)|(want[1]&0xffff0000u)))return 3;
        }
    }
    return 0;
}

extern "C" int dense_q8_byteperm_encoded(uint64_t* count,int fault) {
    *count=0;
    for(unsigned phase=0;phase<2;++phase)
    for(unsigned x=0;x<256;++x)for(unsigned y=0;y<256;++y)
    for(unsigned discarded : {0u,0xffffu,0x7f80u,0x80ffu}) {
        unsigned shift=phase*8;
        uint32_t other=((discarded&255u)|((discarded>>8)<<16))<<(8-shift);
        uint32_t word=((x|(y<<16))<<shift)|other;
        uint32_t expected=((word>>shift)&0x00ff00ffu)|0x64006400u;
        if(P::byte_encoded(word,fault?8-shift:shift)!=expected)return 1;
        ++*count;
    }
    return 0;
}

template<bool Signed,unsigned Threads,bool BytePerm=false> int layout(uint16_t const* low,uint16_t const* units,
        uint16_t* output,unsigned n,unsigned k,int fault) {
    std::vector<unsigned char> out_writes(uint64_t(n)*k);
    for(unsigned n0=0;n0<n;n0+=64)for(unsigned k0=0;k0<k;k0+=128) {
        std::array<uint32_t,L::Cells> tile{};
        std::array<uint32_t,128> scales{};
        std::array<unsigned char,L::Cells> low_writes{};
        std::array<unsigned char,128> scale_writes{};
        for(unsigned t=0;t<128;++t) {
            unsigned pair=t%32,g=t/32,at=P::scale(pair,g);
            uint64_t src=uint64_t(k0/32+g)*n+n0+2*pair;
            if(src%2 || at>=scales.size() || ++scale_writes[at]!=1)return 1;
            scales[at]=uint32_t(units[src])|(uint32_t(units[src+1])<<16);
            if(fault==1)scales[at]=(scales[at]>>16)|(scales[at]<<16);
        }
        for(unsigned t=0;t<Threads;++t)for(unsigned part=0;part<512/Threads;++part) {
            unsigned row=t/8+part*(Threads/8),col=t%8*8;
            uint64_t src=(uint64_t(k0/2+t/8)*(n/8)+n0/8+t%8+uint64_t(part)*(Threads/8)*(n/8))*8;
            if(src%8 || src+7>=uint64_t(n)*k/2)return 2;
            for(unsigned p=0;p<4;++p) {
                unsigned at=L::shared(col/2+p,row);
                if(at>=tile.size() || ++low_writes[at]!=1)return 3;
                tile[at]=P::stage<Signed>(uint32_t(low[src+2*p])|(uint32_t(low[src+2*p+1])<<16));
            }
        }
        for(auto w:low_writes)if(w!=1)return 4;
        for(auto w:scale_writes)if(w!=1)return 5;
        for(unsigned t=0;t<Threads;++t)for(unsigned part=0;part<512/Threads;++part) {
            unsigned pair=t/16+part*(Threads/16),kk=t%16*8,r=L::row(kk);
            unsigned a=L::shared(pair,r),b=L::shared(pair,r+4),g=kk/32;
            if(a%4 || b%4 || a+3>=tile.size() || b+3>=tile.size())return 6;
            uint32_t words[8]={tile[a],tile[a+1],tile[a+2],tile[a+3],tile[b],tile[b+1],tile[b+2],tile[b+3]};
            uint64_t dst=(uint64_t(n0+2*(t/16))*(k/8)+k0/8+t%16+uint64_t(part)*2*(Threads/16)*(k/8))*8;
            uint32_t scale=scales[P::scale(pair,g)];
            for(unsigned p=0;p<4;++p) {
                uint32_t x=P::weight<Signed,BytePerm>(words[2*p],L::shift(kk),scale);
                uint32_t y=P::weight<Signed,BytePerm>(words[2*p+1],L::shift(kk),scale);
                uint32_t values[2]={P::output0(x,y),P::output1(x,y)};
                for(unsigned side=0;side<2;++side)for(unsigned v=0;v<2;++v) {
                    uint64_t at=dst+side*k+2*p+v;
                    if(dst%8 || at>=out_writes.size() || ++out_writes[at]!=1)return 7;
                    output[at]=uint16_t(values[side]>>(16*v));
                }
            }
        }
    }
    for(auto w:out_writes)if(w!=1)return 8;
    return 0;
}

extern "C" int dense_q8_paired_host(int sign,unsigned threads,uint16_t const* low,uint16_t const* units,
                                    uint16_t* output,unsigned n,unsigned k,int fault) {
    if(n%64 || k%128 || (fault!=0 && fault!=1))return 99;
    if(sign==2) {
        if(threads==128)return layout<false,128,true>(low,units,output,n,k,fault);
        if(threads==256)return layout<false,256,true>(low,units,output,n,k,fault);
        return 99;
    }
    if(threads==128)return sign?layout<true,128>(low,units,output,n,k,fault):layout<false,128>(low,units,output,n,k,fault);
    if(threads==256)return sign?layout<true,256>(low,units,output,n,k,fault):layout<false,256>(low,units,output,n,k,fault);
    return 99;
}
