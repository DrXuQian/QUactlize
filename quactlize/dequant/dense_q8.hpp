#pragma once
#include "../include/q8_kpack2.hpp"
#include "cutlass/half.h"
#include "cutlass/bfloat16.h"

namespace quactlize::dequant {
struct DenseQ8 {
    static constexpr unsigned N = 64, K = 32, Pitch = 33, Words = 16 * Pitch;
    CUTLASS_HOST_DEVICE static constexpr unsigned shared(unsigned row, unsigned column) {
        return row * Pitch + column / 2;
    }
    CUTLASS_HOST_DEVICE static constexpr unsigned packed_row(unsigned k) {
        return (k / 16) * 8 + k % 8;
    }
    CUTLASS_HOST_DEVICE static int code(uint32_t word, unsigned n, unsigned k) {
        return int((word >> ((n % 2) * 16 + (k % 16 / 8) * 8)) & 255u) - 128;
    }
    template<bool BF16> CUTLASS_HOST_DEVICE static uint16_t weight(int q, uint16_t d) {
        float scale = float(cutlass::half_t::bitcast(d));
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        float value = __fmul_rn(float(q), scale);
#else
        float value = float(q) * scale;
#endif
        if constexpr (BF16) return cutlass::bfloat16_t(value).raw();
        else return cutlass::half_t(value).raw();
    }
};
} // namespace quactlize::dequant
