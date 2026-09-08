#pragma once
#include "gguf_packed_unit.hpp"
#include "gguf_scale_layout.hpp"
#include "kquant_kpack_offline.hpp"

namespace quactlize::execution {

using gguf_scale::KType;
using Half = cutlass::half_t;

template<KType T> struct Reader {
    using C = gguf_scale::CodeTraits<T>;
    using U = gguf_scale::packed_unit::Unit<T>;
    static constexpr int lo_bits = C::Lo::kWidth;
    static constexpr int hi_bits = C::kHiBytes / 32;
    static constexpr int group = gguf_scale::Traits<T>::kGroupSize;
    using LowMap = kquant_kpack::PlaneMap<lo_bits, group>;

    template<bool High>
    CUTLASS_HOST_DEVICE static int plane(uint16_t const* data, int col, int kk, int n) {
        using Map = std::conditional_t<High,
            kquant_kpack::HighPlaneMap<lo_bits, hi_bits, group>, LowMap>;
        constexpr int bits = High ? hi_bits : lo_bits;
        uint16_t const word = data[Map::word_index(col, kk, n)];
        int slot;
        if constexpr (High && T == KType::Q5_K) slot = Map::word_slot(col, kk);
        else slot = Map::word_slot(kk);
        return (word >> (slot * bits)) & ((1 << bits) - 1);
    }

    CUTLASS_HOST_DEVICE static int code(uint16_t const* low, uint16_t const* high,
                                      int col, int kk, int n) {
        int q = plane<false>(low, col, kk, n);
        if constexpr (hi_bits != 0) q |= plane<true>(high, col, kk, n) << lo_bits;
        // Match the existing converter, then use its canonical affine zero.
        return q - (lo_bits == 4 ? 8 : 0);
    }

    CUTLASS_HOST_DEVICE static gguf_scale::GroupScale scale(
            uint8_t const* units, int col, int g, int n) {
        int const sb = g / U::kGroups;
        int64_t const offset = (int64_t(sb / U::kSbPerUnit) * n + col) * U::kUnitTotal
            + (sb % U::kSbPerUnit) * U::kSbBytes;
        return gguf_scale::packed_unit::unit_group<T,
            gguf_scale::packed_unit::kCanonicalPlacedZMul<T>>(units + offset, g % U::kGroups);
    }

    CUTLASS_HOST_DEVICE static Half weight(int code, gguf_scale::GroupScale s) {
        // Preserve the scale-first multiply and add's FP16 roundings. FP32
        // accumulation below intentionally differs from tensor-core ordering.
        return Half(float(Half(float(code) * float(s.scale))) + float(s.zero));
    }
};
} // namespace quactlize::execution
