// Q4-specific instructions, but identical N2 group ownership and FP32 dot order.
// Other formats keep the generic N2 implementation in this development DSO.
template<class Reader>
__device__ __forceinline__ float2 pair_dot(qkg_call_v1 const& c, int64_t a_base,
    uint16_t const* low, uint16_t const* high, uint8_t const* units,
    int col, int worker, int workers, int partition, int split) {
#if QKG_QTYPE != 12
    return generic_n2_pair_dot<Reader>(c, a_base, low, high, units,
                                      col, worker, workers, partition, split);
#else
    using R = Reader;
    using namespace quactlize::dev::q4_native;
    static_assert(R::lo_bits == 4 && R::hi_bits == 0 && R::group == 32);
    float x[4]{}, y[4]{};
    for (int g = partition*workers + worker; g < c.k/32; g += split*workers) {
        auto const p = units + (int64_t(g/8)*c.n + col)*16;
        auto const s0 = scale_zero(load_unit(p), unsigned(g & 7));
        auto const s1 = scale_zero(load_unit(p + 16), unsigned(g & 7));
        __half2 const scale = __halves2half2(s0.scale, s1.scale);
        __half2 const zero = __halves2half2(s0.zero, s1.zero);
        uint32_t words[8];
        #pragma unroll
        for (int r = 0; r < 8; ++r)
            words[r] = n2_word(low + R::LowMap::word_index(col, g*32+r, c.n));
        // Constant slots keep paired-nibble extraction as a word operation.
        dot_slot<0>(c.a, a_base + g*32, c.input_type, words, scale, zero, x, y);
        dot_slot<1>(c.a, a_base + g*32, c.input_type, words, scale, zero, x, y);
        dot_slot<2>(c.a, a_base + g*32, c.input_type, words, scale, zero, x, y);
        dot_slot<3>(c.a, a_base + g*32, c.input_type, words, scale, zero, x, y);
    }
    return make_float2((x[0]+x[1])+(x[2]+x[3]), (y[0]+y[1])+(y[2]+y[3]));
#endif
}
