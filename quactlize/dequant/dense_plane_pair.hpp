#pragma once
#include "dense_cooperative.hpp"

namespace quactlize::dequant {

// Keep both N halves of each canonical word packed through the exchange.
// A producer owns a physical word, not a logical K8 task: Q3's high plane
// must not be fetched once for every low-plane or output task.
template<unsigned Bits, unsigned TileK> struct DensePlanePairLayout {
    static_assert(Bits == 1 || Bits == 2 || Bits == 8);
    static_assert(TileK == 128 || TileK == 256);
    static constexpr unsigned N = 64, Rows = TileK * Bits / 16, Cells = N / 2 * Rows;
    CUTLASS_HOST_DEVICE static constexpr unsigned shared(unsigned pair, unsigned row) {
        // Preserve aligned four-residue consumer loads. This is a bijection,
        // not a claim that every shared instruction is conflict-free.
        return (pair * Rows + row) ^ (pair & 28u & (Rows - 1));
    }
    CUTLASS_HOST_DEVICE static constexpr unsigned row(unsigned k) {
        return (k / (128 / Bits)) * 8 + k % 8;
    }
    CUTLASS_HOST_DEVICE static constexpr unsigned shift(unsigned k) {
        return (k % (128 / Bits) / 8) * Bits;
    }
    CUTLASS_HOST_DEVICE static constexpr uint32_t codes(uint32_t word, unsigned k) {
        constexpr unsigned mask = (1u << Bits) - 1;
        return (word >> shift(k)) & (mask | (mask << 16));
    }
};

template<int Q, unsigned TileK> struct DensePlanePair {
    static_assert(Q == 8 || Q == 11);
    static constexpr unsigned N = 64, K = TileK, Group = Q == 8 ? 32 : 16;
    static constexpr unsigned Groups = K / Group;
    using Low = DensePlanePairLayout<Q == 8 ? 8 : 2, K>;
    using High = DensePlanePairLayout<1, K>;
    using Affine = DenseCooperativeAffine<Groups>;
    CUTLASS_HOST_DEVICE static uint32_t codes(uint32_t low, uint32_t high, unsigned k) {
        uint32_t value = Low::codes(low, k);
        if constexpr(Q == 11) value |= High::codes(high, k) << 2;
        return value;
    }
    template<bool BF16> CUTLASS_HOST_DEVICE static uint16_t weight(unsigned raw, float scale) {
        constexpr int bias = Q == 8 ? 128 : 4;
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        float value = __fmul_rn(float(int(raw) - bias), scale);
        if constexpr(Q == 11) value = __fsub_rn(value, 0.f);
#else
        volatile float product = float(int(raw) - bias) * scale;
        float value = Q == 11 ? product - 0.f : product;
#endif
        if constexpr(BF16) return cutlass::bfloat16_t(value).raw();
        else return cutlass::half_t(value).raw();
    }
};
} // namespace quactlize::dequant
