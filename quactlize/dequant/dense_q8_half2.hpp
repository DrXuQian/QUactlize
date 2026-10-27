#pragma once
#include "dense_plane_pair.hpp"

namespace quactlize::dequant {
struct DenseQ8Half2Layout {
    CUTLASS_HOST_DEVICE static uint64_t low(unsigned n,unsigned n0,unsigned k0,unsigned task) {
        return uint64_t(k0/2+task/8)*n+n0+task%8*8;
    }
    CUTLASS_HOST_DEVICE static uint64_t scale(unsigned n,unsigned n0,unsigned k0,unsigned task) {
        return uint64_t(k0/32+task/64)*n+n0+task%64;
    }
    CUTLASS_HOST_DEVICE static uint64_t output(unsigned k,unsigned n0,unsigned k0,unsigned task,unsigned side=0) {
        return uint64_t(n0+2*(task/16)+side)*k+k0+task%16*8;
    }
};
// Q8 only: every signed code and original F16 scale is exactly representable.
// Their product is exact in FP32, so one RNE F16 multiply has the same final
// rounding as the old FP32 product -> F16 path. Device FTZ is separately gated.
struct DenseQ8Half2 {
    // Each halfword contains 1024+u8. Keep reconstruction separate from the
    // scale multiply so all fast Q8 readers share exactly the same rounding.
    CUTLASS_HOST_DEVICE static uint32_t weight_encoded(uint32_t encoded,uint32_t scales) {
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        uint32_t bias=0x64806480u;
        __half2 q,offset,s;
        __builtin_memcpy(&q,&encoded,4);
        __builtin_memcpy(&offset,&bias,4);
        __builtin_memcpy(&s,&scales,4);
        q=__hsub2_rn(q,offset); // exact signed int8; never fuse into scale
        __half2 value=__hmul2_rn(q,s);
        uint32_t bits;__builtin_memcpy(&bits,&value,4);return bits;
#else
        uint32_t result=0;
        for(unsigned side=0;side<2;++side) {
            auto code=cutlass::half_t(float(cutlass::half_t::bitcast(uint16_t(encoded>>(16*side))))-1152.f);
            volatile float product=float(code)*float(cutlass::half_t::bitcast(uint16_t(scales>>(16*side))));
            result|=uint32_t(cutlass::half_t(product).raw())<<(16*side);
        }
        return result;
#endif
    }
    template<bool Fast> CUTLASS_HOST_DEVICE static uint32_t weight(uint32_t codes,uint32_t scales) {
        if constexpr(Fast) return weight_encoded(codes|0x64006400u,scales);
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        __half2 q=__halves2half2(__int2half_rn(int(codes&255u)-128),
                               __int2half_rn(int((codes>>16)&255u)-128)),s;
        __builtin_memcpy(&s,&scales,4);
        __half2 value=__hmul2_rn(q,s);
        uint32_t bits;__builtin_memcpy(&bits,&value,4);return bits;
#else
        uint32_t result=0;
        for(unsigned side=0;side<2;++side) {
            unsigned raw=(codes>>(16*side))&255u;
            auto code=cutlass::half_t(int(raw)-128);
            volatile float product=float(code)*float(cutlass::half_t::bitcast(uint16_t(scales>>(16*side))));
            result|=uint32_t(cutlass::half_t(product).raw())<<(16*side);
        }
        return result;
#endif
    }
};
} // namespace quactlize::dequant
