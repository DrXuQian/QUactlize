#pragma once
#include "reader.hpp"
#include <cstring>
#include <type_traits>

namespace quactlize::dequant {
template<int Groups> struct DenseAffineLayout {
    static_assert(Groups==4 || Groups==8 || Groups==16);
    // Preserve row-local groups, xor high row bits into group bits so N-fast
    // metadata producers do not all store to only one/two shared banks.
    CUTLASS_HOST_DEVICE static constexpr int offset(int row,int group) {
        return row*Groups+(group^(row/(32/Groups)));
    }
};
template<KType T,bool BF16> struct DenseReader : Reader<T> {
    using R=Reader<T>;using Affine=typename R::Affine;
    CUTLASS_HOST_DEVICE static uint16_t weight(int raw,Affine a) {
        constexpr int bias=T==KType::Q3_K?4:T==KType::Q6_K?32:0;
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        float value=__fsub_rn(__fmul_rn(float(raw-bias),a.scale),a.minimum);
#else
        volatile float product=float(raw-bias)*a.scale;
        float value=product-a.minimum;
#endif
        if constexpr(BF16) return cutlass::bfloat16_t(value).raw();
        else return cutlass::half_t(value).raw();
    }
};
} // namespace quactlize::dequant

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

namespace quactlize::dequant {

// A low word already contains two neighbouring N columns. Keep that word
// intact through the private shared exchange instead of expanding its four
// slots into byte codes. The high sideband has just the bits for this K32.
// Neither the canonical global artifact nor the output [N,K] layout changes.
template<KType T> struct DensePairCodes {
    static_assert(T==KType::Q4_K || T==KType::Q5_K || T==KType::Q6_K);
    static constexpr unsigned bias=T==KType::Q6_K ? 32 : 0;
    CUTLASS_HOST_DEVICE static uint16_t high(uint32_t word,unsigned col,unsigned k32) {
        if constexpr(T==KType::Q4_K) return 0;
        else if constexpr(T==KType::Q5_K) {
            unsigned shift=DenseUnsignedQ5High::word_slot(col,k32);
            word=(word>>shift)&0x000f000fu;
            return uint16_t(word|(word>>12));
        } else {
            word=(word>>((k32&32u)>>2))&0x00ff00ffu;
            return uint16_t(word|(word>>8));
        }
    }
    CUTLASS_HOST_DEVICE static uint32_t codes(uint32_t low,uint16_t high,unsigned slot) {
        uint32_t raw=(low>>(slot*4))&0x000f000fu;
        if constexpr(T==KType::Q5_K) {
            unsigned h=(unsigned(high)>>slot)&0x11u;
            raw|=((h&1u)<<4)|((h&16u)<<16);
        } else if constexpr(T==KType::Q6_K) {
            unsigned h=(unsigned(high)>>(slot*2))&0x0303u;
            raw|=((h&3u)<<4)|((h&0x300u)<<12);
        }
        return raw;
    }
    template<bool Fast> CUTLASS_HOST_DEVICE static float value(unsigned raw) {
        if constexpr(!Fast) return float(int(raw)-int(bias));
        else {
            // Every raw code is <=63. The bit construction and subtraction
            // are exact in FP32, including Q6's signed centre; no FP16 hop,
            // scaled bias reassociation or changed affine rounding point.
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
            return __fsub_rn(__uint_as_float(0x4b000000u|raw),8388608.f+bias);
#else
            uint32_t bits=0x4b000000u|raw;float result;
            std::memcpy(&result,&bits,sizeof(result));
            return result-(8388608.f+bias);
#endif
        }
    }
};

template<KType T,bool BF16,bool Fast> struct DensePairReader : DenseUnsignedReader<T,BF16> {
    using R=DenseUnsignedReader<T,BF16>;
    CUTLASS_HOST_DEVICE static uint16_t weight(unsigned raw,typename R::Affine a) {
        float q=DensePairCodes<T>::template value<Fast>(raw);
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        float value=__fsub_rn(__fmul_rn(q,a.scale),a.minimum);
#else
        volatile float product=q*a.scale;
        float value=product-a.minimum;
#endif
        if constexpr(BF16) return cutlass::bfloat16_t(value).raw();
        else return cutlass::half_t(value).raw();
    }
};
} // namespace quactlize::dequant
