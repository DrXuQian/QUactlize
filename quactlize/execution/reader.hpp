#pragma once
#include "gguf_packed_unit.hpp"
#include "gguf_scale_layout.hpp"
#include "kquant_kpack_offline.hpp"
#include "actlize_extensions/cutlass/gguf_packed_scale.h"

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
    using HighMap = kquant_kpack::HighPlaneMap<lo_bits, hi_bits, group>;

    CUTLASS_HOST_DEVICE static int raw_from_words(
            uint16_t low, uint16_t high, int col, int kk) {
        int q = (low >> (lo_bits * LowMap::word_slot(kk))) & ((1 << lo_bits) - 1);
        if constexpr (hi_bits != 0) {
            int slot;
            if constexpr (T == KType::Q5_K) slot = HighMap::word_slot(col, kk);
            else slot = HighMap::word_slot(kk);
            q |= ((high >> (hi_bits * slot)) & ((1 << hi_bits) - 1)) << lo_bits;
        }
        return q;
    }

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

    CUTLASS_HOST_DEVICE static uint32_t weight_pair(int raw0, int raw1, gguf_scale::GroupScale s) {
        using namespace cutlass::gguf_packed;
        uint32_t const codes=uint32_t(raw0) | (uint32_t(raw1)<<16) | UINT32_C(0x64006400);
        Half const offset(float(1024+(lo_bits==4 ? 8 : 0)));
        uint32_t const integer=sub_f16x2(codes,pack_h2(offset,offset));
        // Explicit experimental fused affine: one FP16 rounding, unlike
        // weight()'s two-rounding scalar oracle. Admit by independent dot
        // error, never by claiming bit identity with the scalar reader.
#if CUTLASS_GGUF_PACKED_F16X2_ASM
        return fma_f16x2(integer,pack_h2(s.scale,s.scale),pack_h2(s.zero,s.zero));
#else
        return pack_h2(Half(float(lo_h2(integer))*float(s.scale)+float(s.zero)),
                       Half(float(hi_h2(integer))*float(s.scale)+float(s.zero)));
#endif
    }
};
} // namespace quactlize::execution
