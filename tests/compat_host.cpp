#include "quactlize/compat/contract.hpp"
#include "quactlize/compat/reader.cuh"

extern "C" int compat_query(qpm_problem_v1 const *p, quactlize_ppu_placed_arrangement_v2 const *a,
                            qpm_sizes_v1 *s) {
    try {
        *s = quactlize::compat::layout(*p, a).sizes;
        return 0;
    } catch (std::exception const &) {
        return 20;
    }
}
extern "C" int compat_validate(qpm_problem_v1 const *p, qpm_buffers_v1 const *b) {
    try {
        auto l = quactlize::compat::layout(*p, nullptr);
        quactlize::compat::validate(*p, l, *b);
        return 0;
    } catch (std::exception const &) {
        return 20;
    }
}
template <int Q> void decode(uint8_t const *raw, uint16_t *out, int count, int dtype) {
    constexpr int block = Q < 10 ? 32 : 256;
    int bytes = quactlize::compat::block_bytes(Q);
    for (int i = 0; i < count; ++i) {
        float v = quactlize::compat::raw_value<Q>(raw + (i / block) * bytes, i % block);
        out[i] = dtype ? cutlass::bfloat16_t(v).raw() : cutlass::half_t(v).raw();
    }
}
extern "C" int compat_decode(int q, uint8_t const *raw, uint16_t *out, int count, int dtype) {
#define QPM_CASE(Q)                                                                                          \
    case Q:                                                                                                  \
        decode<Q>(raw, out, count, dtype);                                                                   \
        return 0
    switch (q) {
        QPM_CASE(7);
        QPM_CASE(8);
        QPM_CASE(10);
        QPM_CASE(11);
        QPM_CASE(12);
        QPM_CASE(13);
        QPM_CASE(14);
    default:
        return 20;
    }
#undef QPM_CASE
}
extern "C" int compat_abi_size(int type) {
    switch (type) {
    case 0:
        return sizeof(qpm_problem_v1);
    case 1:
        return sizeof(qpm_sizes_v1);
    case 2:
        return sizeof(qpm_buffers_v1);
    case 3:
        return sizeof(qpm_router_v1);
    case 4:
        return sizeof(qpm_swiglu_v1);
    case 5:
        return sizeof(qpm_merge_v1);
    default:
        return -1;
    }
}
template <int Q>
void decode_canonical(quactlize::compat::Weight const &w, uint16_t *out, int experts, int dtype) {
    for (int e = 0; e < experts; ++e)
        for (int n = 0; n < w.n; ++n)
            for (int k = 0; k < w.k; ++k) {
                float v = quactlize::compat::value<Q, true>(w, e, n, k);
                out[(int64_t(e) * w.n + n) * w.k + k] =
                    dtype ? cutlass::bfloat16_t(v).raw() : cutlass::half_t(v).raw();
            }
}
extern "C" int compat_decode_canonical(int q, uint8_t const *low, uint8_t const *high, uint8_t const *units,
                                       uint16_t *out, int n, int k, int experts, int dtype) {
    quactlize_ppu_placed_arrangement_v2 a{};
    quactlize_ppu_kpack_sizes_v1 s{};
    if (quactlize_ppu_kpack_canonical_arrangement_v1(q, &a) ||
        quactlize_ppu_kpack_sizes_for_arrangement_v1(n, k, experts, q, &a, &s))
        return 20;
    quactlize::compat::Weight w{
        nullptr, low, high, units, 0, s.low_bytes / experts, s.high_bytes / experts, s.units_bytes / experts,
        n,       k};
#define QPM_CASE(Q)                                                                                          \
    case Q:                                                                                                  \
        decode_canonical<Q>(w, out, experts, dtype);                                                         \
        return 0
    switch (q) {
        QPM_CASE(8);
        QPM_CASE(10);
        QPM_CASE(11);
        QPM_CASE(12);
        QPM_CASE(13);
        QPM_CASE(14);
    default:
        return 20;
    }
#undef QPM_CASE
}
