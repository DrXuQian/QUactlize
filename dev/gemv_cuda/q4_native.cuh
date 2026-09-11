// Development-only CUDA instructions for canonical Q4 K-pack words.
// No offline permutation, FP16 dot accumulation or expanded scale plane.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace quactlize::dev::q4_native {

struct ScaleZero {
    __half scale;
    __half zero;
};

__device__ __forceinline__ uint4 load_unit(uint8_t const* p) {
    if ((uintptr_t(p) & 15) == 0) return *reinterpret_cast<uint4 const*>(p);
    // Packed-unit pointers have no public 16-byte alignment requirement.
    uint32_t w[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i] = uint32_t(p[4*i]) | (uint32_t(p[4*i+1]) << 8) |
               (uint32_t(p[4*i+2]) << 16) | (uint32_t(p[4*i+3]) << 24);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ ScaleZero scale_zero(uint4 const m, unsigned group) {
    uint64_t const run = (group & 4)
        ? uint64_t(m.z >> 16) | (uint64_t(m.w) << 16)
        : uint64_t(m.y) | (uint64_t(m.z & 0xffff) << 32);
    unsigned const shift = 6 * (group & 3);
    unsigned const sc = unsigned(run >> shift) & 63;
    unsigned const mn = unsigned(run >> (24 + shift)) & 63;
    __half const scale = __hmul(__ushort_as_half(uint16_t(m.x)), __int2half_rn(sc));
    __half const zero = __hneg(__hmul(__ushort_as_half(uint16_t(m.x >> 16)),
                                    __int2half_rn(mn)));
    // Preserve unit_group<Q4_K,8>: round each metadata product to half,
    // then one FP32 expression and one half rounding for the +8 correction.
    return {scale, __float2half_rn(__half2float(zero) + 8.f * __half2float(scale))};
}

template<int Slot>
__device__ __forceinline__ __half2 codes(uint32_t words) {
    static_assert(Slot >= 0 && Slot < 4);
    // The two b16 halves own adjacent N columns, not adjacent K values.
    uint32_t const bits = ((words >> (4 * Slot)) & 0x000f000fu) | 0x64006400u;
    __half2_raw raw;
    raw.x = uint16_t(bits);
    raw.y = uint16_t(bits >> 16);
    return __hsub2(__half2(raw), __float2half2_rn(1032.f));
}

__device__ __forceinline__ float4 activation(void const* base, int64_t offset,
                                            int input_type) {
    if (input_type == 0) {
        auto p = static_cast<__half const*>(base) + offset;
        if ((uintptr_t(p) & 3) == 0) {
            float2 const lo = __half22float2(*reinterpret_cast<__half2 const*>(p));
            float2 const hi = __half22float2(*reinterpret_cast<__half2 const*>(p + 2));
            return make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        return make_float4(__half2float(p[0]), __half2float(p[1]),
                           __half2float(p[2]), __half2float(p[3]));
    }
    auto p = static_cast<float const*>(base) + offset;
    float4 v;
    if ((uintptr_t(p) & 15) == 0) v = *reinterpret_cast<float4 const*>(p);
    else v = make_float4(p[0], p[1], p[2], p[3]);
    // F32 endpoints still cross the same FP16-A boundary as the N2 reader.
    v.x = __half2float(__float2half_rn(v.x));
    v.y = __half2float(__float2half_rn(v.y));
    v.z = __half2float(__float2half_rn(v.z));
    v.w = __half2float(__float2half_rn(v.w));
    return v;
}

template<int Slot>
__device__ __forceinline__ void dot_slot(void const* a_base, int64_t group_offset,
    int input_type, uint32_t const (&words)[8], __half2 scale, __half2 zero,
    float (&x)[4], float (&y)[4]) {
    #pragma unroll
    for (int first = 0; first < 8; first += 4) {
        float4 const a = activation(a_base, group_offset + 8*Slot + first, input_type);
        float const av[4] = {a.x, a.y, a.z, a.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 const w = __half22float2(__hfma2(codes<Slot>(words[first+i]), scale, zero));
            x[i] = fmaf(av[i], w.x, x[i]);
            y[i] = fmaf(av[i], w.y, y[i]);
        }
    }
}

} // namespace quactlize::dev::q4_native
