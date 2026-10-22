#pragma once
#include <cstring>
#include "dense_unsigned.hpp"

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
