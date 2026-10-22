#pragma once
#include <type_traits>
#include "dense_reader.hpp"

namespace quactlize::dequant {
// Same canonical map as execution::common_map (the measured SIMT reader),
// without importing its dot/reducer/device-kernel dependencies into dequant.
// Public validation and signed Q6 scale values are intentionally unchanged.
template<unsigned Bits> struct DenseUnsignedPlane {
    static constexpr unsigned Delivery=8*(16/Bits);
    CUTLASS_HOST_DEVICE static constexpr unsigned word_slot(unsigned k){return (k%Delivery)/8;}
    CUTLASS_HOST_DEVICE static constexpr uint64_t word_index(unsigned n,unsigned k,unsigned extent){
        return uint64_t((k/Delivery)*8+k%8)*extent+n;
    }
};
struct DenseUnsignedQ5High {
    CUTLASS_HOST_DEVICE static constexpr unsigned word_slot(unsigned n,unsigned k){return (((n>>3)&1)<<3)|((k>>3)&7);}
    CUTLASS_HOST_DEVICE static constexpr uint64_t word_index(unsigned n,unsigned k,unsigned extent){
        unsigned pn=(n&~15u)|(n&7u)|(((k>>7)&1u)<<3);
        unsigned kg=(k/256)*16|(((k>>6)&1)<<3)|(k&7);
        return uint64_t(kg)*extent+pn;
    }
};
template<KType T,bool BF16> struct DenseUnsignedReader : DenseReader<T,BF16> {
    using R=DenseReader<T,BF16>;using U=typename R::U;using Affine=typename R::Affine;
    using LowMap=DenseUnsignedPlane<R::lo_bits>;
    using HighMap=std::conditional_t<T==KType::Q5_K,DenseUnsignedQ5High,DenseUnsignedPlane<(R::hi_bits?R::hi_bits:1)>>;
    CUTLASS_HOST_DEVICE static unsigned raw(uint16_t low,uint16_t high,unsigned col,unsigned k){
        unsigned q=(low>>(R::lo_bits*LowMap::word_slot(k)))&((1u<<R::lo_bits)-1);
        if constexpr(R::hi_bits) {
            unsigned s;
            if constexpr(T==KType::Q5_K)s=HighMap::word_slot(col,k);else s=HighMap::word_slot(k);
            q|=((high>>(R::hi_bits*s))&((1u<<R::hi_bits)-1))<<R::lo_bits;
        }
        return q;
    }
    CUTLASS_HOST_DEVICE static uint32_t combine(uint16_t low,uint16_t high,unsigned col,unsigned k){
        uint32_t result=0;
        CUTLASS_PRAGMA_UNROLL
        for(unsigned s=0;s<4;++s)result|=raw(low,high,col,k+s*8)<<(s*8);
        return result;
    }
    CUTLASS_HOST_DEVICE static Affine affine(uint8_t const* units,unsigned col,unsigned g,unsigned n){
        unsigned sb=g/U::kGroups;
        uint64_t offset=(uint64_t(sb/U::kSbPerUnit)*n+col)*U::kUnitTotal+(sb%U::kSbPerUnit)*U::kSbBytes;
        uint8_t const* p=units+offset;
        float d=float(cutlass::half_t::bitcast(uint16_t(p[0])|uint16_t(p[1])<<8));
        int sc=gguf_scale::packed_unit::code_of<T>(p,g%U::kGroups,0);
        if constexpr(U::kSigned)if(sc&(1<<(U::kScaleBits-1)))sc-=1<<U::kScaleBits;
        sc-=gguf_scale::Traits<T>::kScaleBias;
        float minimum=0.f;
        if constexpr(U::kHasMin){
            float dm=float(cutlass::half_t::bitcast(uint16_t(p[2])|uint16_t(p[3])<<8));
            minimum=dm*gguf_scale::packed_unit::code_of<T>(p,g%U::kGroups,1);
        }
        return {d*sc,minimum};
    }
};
}
