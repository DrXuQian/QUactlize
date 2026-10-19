#pragma once

// Experiment only: canonical Q6_K metadata, no new packing or arithmetic.
// Words must expose nine named 32-bit entries. Selecting an entry through
// constant subscripts avoids materializing the 36-byte unit in thread-local
// shared memory. Keep this independent from the producer's CUDA types so
// host tests can exercise the actual extractor without CuTe/CUDA stubs.
namespace quactlize::execution::q6_metadata {

template<unsigned Begin, unsigned End, class Words>
__host__ __device__ __forceinline__ uint32_t word(Words const& m, unsigned index) {
    static_assert(Begin < End && End <= 9);
    if constexpr (End - Begin == 1) {
        return m.word[Begin];
    } else {
        constexpr unsigned Mid = (Begin + End) / 2;
        return index < Mid ? word<Begin, Mid>(m, index) : word<Mid, End>(m, index);
    }
}

template<class Words>
__host__ __device__ __forceinline__ float2 affine(Words const& m, unsigned group) {
    // Each unit contains two 18-byte records: FP16 d, then 16 signed scales.
    unsigned const sb = (group >> 4) & 1u;
    unsigned const byte = 2u + 18u * sb + (group & 15u);
    uint32_t const header = sb ? m.word[4] >> 16 : m.word[0];
    uint32_t const packed = word<0, 9>(m, byte >> 2);
    unsigned const code = (packed >> (8u * (byte & 3u))) & 255u;
    int const scale = int(code ^ 128u) - 128;
    float const d = __half2float(__ushort_as_half(uint16_t(header)));
    return make_float2(d * float(scale), 0.f);
}

} // namespace quactlize::execution::q6_metadata
