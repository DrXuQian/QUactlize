#pragma once
#include "reader.hpp"

namespace quactlize::dequant {

// Q4/Q5 canonical metadata is exactly one aligned 16-byte unit. Read it
// once, retain the original field map and rounding, and reuse all groups.
// This is a reader change, not a new offline representation.
template<KType T>
struct alignas(16) Unit16 {
    static_assert(T == KType::Q4_K || T == KType::Q5_K);
    using U = gguf_scale::packed_unit::Unit<T>;
    uint32_t x, y, z, w;

    CUTLASS_HOST_DEVICE static Unit16 load(uint8_t const* p) {
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        uint4 const v = *reinterpret_cast<uint4 const*>(p);
        return {v.x, v.y, v.z, v.w};
#else
        Unit16 v;
        __builtin_memcpy(&v, p, sizeof(v));
        return v;
#endif
    }
    CUTLASS_HOST_DEVICE uint32_t word(int i) const {
        return i == 0 ? x : i == 1 ? y : i == 2 ? z : w;
    }
    CUTLASS_HOST_DEVICE int code(int group, int field) const {
        int const bit = U::bit_of(group, field), shift = bit % 32;
        uint32_t v = word(bit / 32) >> shift;
        if (shift > 26) v |= word(bit / 32 + 1) << (32 - shift);
        return int(v & 63);
    }
    CUTLASS_HOST_DEVICE gguf_scale::GroupScale scale(int group) const {
        auto const d = cutlass::half_t::bitcast(uint16_t(x));
        auto const dm = cutlass::half_t::bitcast(uint16_t(x >> 16));
        auto v = gguf_scale::make_group_scale<T>(code(group, 0), code(group, 1), d, dm);
        constexpr int z = gguf_scale::packed_unit::kCanonicalPlacedZMul<T>;
        v.zero = cutlass::half_t(float(v.zero) + float(z) * float(v.scale));
        return v;
    }
    CUTLASS_HOST_DEVICE typename Reader<T>::Affine affine(int group) const {
        float const d = float(cutlass::half_t::bitcast(uint16_t(x)));
        float const dm = float(cutlass::half_t::bitcast(uint16_t(x >> 16)));
        return {d * code(group, 0), dm * code(group, 1)};
    }
};
static_assert(sizeof(Unit16<KType::Q4_K>) == 16);

} // namespace quactlize::dequant
