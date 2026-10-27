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
    // Select bytes 0/2 or 1/3 and insert the FP16 exponent in one native
    // permutation. The two halfwords are exactly 1024+u, for unsigned u8.
    CUTLASS_HOST_DEVICE static uint32_t byte_encoded(uint32_t word,unsigned shift) {
        unsigned selector=0x4240u|((shift/8u)*0x0101u);
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        return __byte_perm(word,0x64646464u,selector);
#else
        uint64_t bytes=uint64_t(word)|(uint64_t(0x64646464u)<<32);
        uint32_t result=0;
        for(unsigned i=0;i<4;++i)
            result|=uint32_t((bytes>>(8*((selector>>(4*i))&7u)))&255u)<<(8*i);
        return result;
#endif
    }
    // Two neighboring N values at one K, with distinct F16 scales.
    template<bool Signed,bool BytePerm=false> CUTLASS_HOST_DEVICE static uint32_t weight(
            uint32_t word,unsigned shift,uint32_t scales) {
        static_assert(!Signed || !BytePerm,"byte permutation consumes canonical biased bytes");
        if constexpr(BytePerm) {
            uint32_t encoded=byte_encoded(word,shift);
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
            uint32_t bias=0x64806480u;
            __half2 q,offset,s;
            __builtin_memcpy(&q,&encoded,4);
            __builtin_memcpy(&offset,&bias,4);
            __builtin_memcpy(&s,&scales,4);
            q=__hsub2_rn(q,offset); // exact u-128; keep scale multiply separate
            __half2 value=__hmul2_rn(q,s);
            uint32_t bits;__builtin_memcpy(&bits,&value,4);return bits;
#else
            uint32_t result=0;
            for(unsigned side=0;side<2;++side) {
                float code=float(cutlass::half_t::bitcast(uint16_t(encoded>>(16*side))))-1152.f;
                volatile float product=code*float(cutlass::half_t::bitcast(uint16_t(scales>>(16*side))));
                result|=uint32_t(cutlass::half_t(product).raw())<<(16*side);
            }
            return result;
#endif
        }
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
