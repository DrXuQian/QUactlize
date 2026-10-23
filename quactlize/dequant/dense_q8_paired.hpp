#pragma once
#include "dense_q8_half2.hpp"

namespace quactlize::dequant {
// Private exchange only: canonical low/units bytes and final [N,K] are fixed.
struct DenseQ8Paired {
    using Low=DensePlanePairLayout<8,128>;
    CUTLASS_HOST_DEVICE static constexpr unsigned scale(unsigned pair,unsigned group) {
        return group*32+(pair^(group*8));
    }
    template<bool Signed> CUTLASS_HOST_DEVICE static constexpr uint32_t stage(uint32_t word) {
        return Signed?word^0x80808080u:word;
    }
    // Two neighboring N values at one K, with distinct F16 scales.
    template<bool Signed> CUTLASS_HOST_DEVICE static uint32_t weight(
            uint32_t word,unsigned shift,uint32_t scales) {
        if constexpr(!Signed) return DenseQ8Half2::weight<true>((word>>shift)&0x00ff00ffu,scales);
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        // Native-like signed consumer, compared separately with fast bits.
        // Do not assume the compiler folds sign extension into conversion.
        static_assert(char(-1)<0,"signed char contract required");
        char x=char(word>>shift),y=char(word>>(16+shift));
        __half2 q=make_half2(x,y),s;
        __builtin_memcpy(&s,&scales,4);
        __half2 value=__hmul2_rn(q,s);
        uint32_t bits;__builtin_memcpy(&bits,&value,4);return bits;
#else
        uint32_t result=0;
        for(unsigned side=0;side<2;++side) {
            unsigned raw=(word>>(shift+16*side))&255u;
            int code=raw<128?int(raw):int(raw)-256;
            volatile float product=float(code)*float(cutlass::half_t::bitcast(uint16_t(scales>>(16*side))));
            result|=uint32_t(cutlass::half_t(product).raw())<<(16*side);
        }
        return result;
#endif
    }
    CUTLASS_HOST_DEVICE static uint32_t output0(uint32_t x,uint32_t y) {
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        return __byte_perm(x,y,0x5410);
#else
        return (x&0xffffu)|(y<<16);
#endif
    }
    CUTLASS_HOST_DEVICE static uint32_t output1(uint32_t x,uint32_t y) {
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        return __byte_perm(x,y,0x7632);
#else
        return (x>>16)|(y&0xffff0000u);
#endif
    }
};
} // namespace quactlize::dequant
