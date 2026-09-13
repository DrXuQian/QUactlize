#pragma once
#include "../execution/reader.hpp"
#include "cutlass/bfloat16.h"

namespace quactlize::dequant {
using gguf_scale::KType;

// This reconstructs raw GGUF values, not the rounded SF scale/zero planes.
// The canonical storage and all field/word addresses remain registry-owned.
template<KType T>
struct Reader : execution::Reader<T> {
    using R = execution::Reader<T>;
    using U = gguf_scale::packed_unit::Unit<T>;
    struct Affine { float scale, minimum; };

    CUTLASS_HOST_DEVICE static Affine affine(uint8_t const* units, int n, int g, int n_extent) {
        int const sb = g / U::kGroups;
        int64_t const offset = (int64_t(sb / U::kSbPerUnit) * n_extent + n) * U::kUnitTotal
            + (sb % U::kSbPerUnit) * U::kSbBytes;
        uint8_t const* p = units + offset;
        float const d = float(cutlass::half_t::bitcast(uint16_t(p[0]) | uint16_t(p[1]) << 8));
        int sc = gguf_scale::packed_unit::code_of<T>(p, g % U::kGroups, 0);
        if constexpr (U::kSigned) {
            if (sc & (1 << (U::kScaleBits - 1))) sc -= 1 << U::kScaleBits;
        }
        sc -= gguf_scale::Traits<T>::kScaleBias;
        float minimum = 0.f;
        if constexpr (U::kHasMin) {
            float const dm = float(cutlass::half_t::bitcast(uint16_t(p[2]) | uint16_t(p[3]) << 8));
            minimum = dm * gguf_scale::packed_unit::code_of<T>(p, g % U::kGroups, 1);
        }
        return {d * sc, minimum};
    }

    CUTLASS_HOST_DEVICE static uint16_t weight(int raw, Affine s) {
        constexpr int bias = T == KType::Q3_K ? 4 : T == KType::Q6_K ? 32 : 0;
        float const q = float(raw - bias);
        // Match the official dequant's distinct FP32 multiply/subtract.
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        float const value = __fsub_rn(__fmul_rn(q, s.scale), s.minimum);
#else
        float const product = q * s.scale;
        float const value = product - s.minimum;
#endif
        return cutlass::bfloat16_t(value).raw();
    }
};
} // namespace quactlize::dequant
