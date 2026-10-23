#pragma once
#include "dense_plane_pair.hpp"
#include <cstring>

namespace quactlize::dequant {
struct DenseQ8TunedLayout {
    CUTLASS_HOST_DEVICE static uint64_t low(unsigned n,unsigned n0,unsigned k0,unsigned task) {
        return uint64_t(k0/2+task/8)*n+n0+task%8*8;
    }
    CUTLASS_HOST_DEVICE static uint64_t scale(unsigned n,unsigned n0,unsigned k0,unsigned task,bool vector) {
        return vector?uint64_t(k0/32+task/16)*n+n0+task%16*4:
            uint64_t(k0/32+task/64)*n+n0+task%64;
    }
    CUTLASS_HOST_DEVICE static uint64_t output(unsigned k,unsigned n0,unsigned k0,unsigned task,unsigned side=0) {
        return uint64_t(n0+2*(task/16)+side)*k+k0+task%16*8;
    }
};
struct DenseQ8Fast {
    // All unsigned8 codes are exact in the FP32 mantissa at exponent23.
    // Subtract the integer bias BEFORE scaling. Fusing encoded*scale-bias
    // would suffer cancellation/rounding and is deliberately forbidden.
    CUTLASS_HOST_DEVICE static float code(unsigned raw) {
        uint32_t bits=0x4b000000u|raw;
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        return __fsub_rn(__int_as_float(int(bits)),8388736.0f);
#else
        float encoded;std::memcpy(&encoded,&bits,sizeof(encoded));
        return encoded-8388736.0f;
#endif
    }
    template<bool BF16> CUTLASS_HOST_DEVICE static uint16_t weight(unsigned raw,float scale) {
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        float value=__fmul_rn(code(raw),scale);
#else
        volatile float value=code(raw)*scale;
#endif
        // Direct final conversion; BF16 never reconstructs a rounded F16.
        if constexpr(BF16)return cutlass::bfloat16_t(value).raw();
        else return cutlass::half_t(value).raw();
    }
};
} // namespace quactlize::dequant
