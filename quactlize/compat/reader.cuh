#pragma once
#include "../dequant/reader.hpp"
#include "../include/gguf_unit_pack.hpp"
#include "../include/q5_1_kpack.hpp"
#include "../include/q2_0_kpack.hpp"

namespace quactlize::compat {
CUTLASS_HOST_DEVICE float half_value(uint8_t const *p) {
    return float(cutlass::half_t::bitcast(uint16_t(p[0]) | (uint16_t(p[1]) << 8)));
}
CUTLASS_HOST_DEVICE float product(float a, float b) {
#if defined(__HGGC_ARCH__) || defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    volatile float x = a * b;
    return x;
#endif
}
CUTLASS_HOST_DEVICE float difference(float a, float b) {
#if defined(__HGGC_ARCH__) || defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}
template <int Q> CUTLASS_HOST_DEVICE float raw_value(uint8_t const *b, int i) {
    if constexpr (Q == 7) {
        uint32_t high =
            uint32_t(b[4]) | (uint32_t(b[5]) << 8) | (uint32_t(b[6]) << 16) | (uint32_t(b[7]) << 24);
        int code = ((b[8 + i % 16] >> (4 * (i / 16))) & 15) | (((high >> i) & 1) << 4);
        // Q5_1 is unsigned 5-bit code, d*q + m (NOT K-quant d*q - dmin).
        return difference(product(half_value(b), float(code)), -half_value(b + 2));
    } else if constexpr (Q == 8) {
        return product(half_value(b), float(int8_t(b[2 + i])));
    } else if constexpr (Q == 42) {
        int code = (b[2 + i / 4] >> (2 * (i % 4))) & 3;
        return product(float(code - 1), half_value(b));
    } else {
        constexpr auto T = gguf_scale::KType(Q - 10);
        using C = gguf_scale::CodeTraits<T>;
        using S = gguf_scale::Traits<T>;
        using H = gguf_scale::unit_pack::Raw<T>;
        int g = i / S::kGroupSize, j = i % S::kGroupSize, word = C::word_coord(g, j / 4);
        int code = C::Lo::extract_byte(b + C::kLoOffset, word, j % 4);
        if constexpr (Q == 11)
            code -= ((C::Hi::extract_byte(b + C::kHiOffset, word, j % 4) ^ 1) << 2);
        if constexpr (Q == 13)
            code |= C::Hi::extract_byte(b + C::kHiOffset, word, j % 4) << 4;
        if constexpr (Q == 14)
            code = (code | (C::Hi::extract_byte(b + C::kHiOffset, word, j % 4) << 4)) - 32;
        float scale = product(half_value(b + H::kDOffset),
                              float(gguf_scale::scale_of<T>(b + H::kScaleOffset, g) - S::kScaleBias));
        float minimum = 0.f;
        if constexpr (S::kHasMin)
            minimum =
                product(half_value(b + H::kDminOffset), float(gguf_scale::min_of<T>(b + H::kScaleOffset, g)));
        return difference(product(float(code), scale), minimum);
    }
}
template <int Q>
CUTLASS_HOST_DEVICE float canonical_value(uint8_t const *low, uint8_t const *high, uint8_t const *units,
                                          int n, int col, int k) {
    if constexpr (Q == 7) {
        auto word = [](uint8_t const *p, size_t index) {
            return uint16_t(p[2*index]) | (uint16_t(p[2*index+1]) << 8);
        };
        int lo = (word(low,q5_1_kpack::Low::word_index(col,k,n)) >>
                  (4*q5_1_kpack::Low::word_slot(k))) & 15;
        int hi = (word(high,q5_1_kpack::High::word_index(col,k,n)) >>
                  q5_1_kpack::High::word_slot(k)) & 1;
        auto dm = units + 4*(int64_t(k/32)*n+col);
        return difference(product(half_value(dm),float(lo | (hi << 4))),-half_value(dm+2));
    } else if constexpr (Q == 42) {
        auto index = q2_0_kpack::Low::word_index(col, k, n);
        auto word = uint16_t(low[2 * index]) | (uint16_t(low[2 * index + 1]) << 8);
        int code = (word >> (2 * q2_0_kpack::Low::word_slot(k))) & 3;
        return product(float(code - 1), half_value(units + 2 * (int64_t(k / 64) * n + col)));
    } else if constexpr (Q == 8) {
        int q = execution::Q8Reader::code(reinterpret_cast<uint16_t const *>(low), nullptr, col, k, n);
        return product(float(q), half_value(units + 2 * (int64_t(k / 32) * n + col)));
    } else if constexpr (Q >= 10 && Q <= 14) {
        using R = dequant::Reader<gguf_scale::KType(Q - 10)>;
        int code = R::code(reinterpret_cast<uint16_t const *>(low), reinterpret_cast<uint16_t const *>(high),
                           col, k, n) +
                   (R::lo_bits == 4 ? 8 : 0) -
                   (Q == 11   ? 4
                    : Q == 14 ? 32
                              : 0);
        auto scale = R::affine(units, col, k / R::group, n);
        return difference(product(float(code), scale.scale), scale.minimum);
    } else
        return 0.f; // Canonical admission rejects unsupported qtypes.
}
struct Weight {
    uint8_t const *raw, *low, *high, *units;
    uint64_t raw_stride, low_stride, high_stride, units_stride;
    int n, k;
};
template <int Q, bool Packed> CUTLASS_HOST_DEVICE float value(Weight const &w, int e, int n, int k) {
    if constexpr (Packed)
        return canonical_value<Q>(w.low + e * w.low_stride, w.high ? w.high + e * w.high_stride : nullptr,
                                  w.units + e * w.units_stride, w.n, n, k);
    else {
        constexpr int block = Q == 42 ? 64 : Q < 10 ? 32 : 256;
        constexpr int bytes = Q == 7    ? 24
                              : Q == 8  ? 34
                              : Q == 10 ? 84
                              : Q == 11 ? 110
                              : Q == 12 ? 144
                              : Q == 13 ? 176
                              : Q == 42 ? 18
                                        : 210;
        return raw_value<Q>(w.raw + e * w.raw_stride + (int64_t(n) * w.k + k) / block * bytes, k % block);
    }
}
} // namespace quactlize::compat
