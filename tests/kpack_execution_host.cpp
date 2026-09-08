#include "reader.hpp"
#include "validation.hpp"

template<gguf_scale::KType T>
void read_host(uint8_t const* low, uint8_t const* high, uint8_t const* units,
               int n, int k, int experts, uint16_t* weights, uint16_t* scales, uint16_t* zeros) {
    using R = quactlize::execution::Reader<T>;
    for (int e = 0; e < experts; ++e) {
        int64_t const nk = int64_t(n)*k;
        auto l = reinterpret_cast<uint16_t const*>(low + e*nk*R::lo_bits/8);
        uint16_t const* h = nullptr;
        if constexpr (R::hi_bits) h = reinterpret_cast<uint16_t const*>(high + e*nk*R::hi_bits/8);
        auto u = units + e*nk/256*R::U::kSbBytes;
        for (int col = 0; col < n; ++col) for (int g = 0; g < k/R::group; ++g) {
            auto s = R::scale(u,col,g,n);
            int64_t const si = (int64_t(e)*(k/R::group)+g)*n+col;
            scales[si] = s.scale.raw(); zeros[si] = s.zero.raw();
            for (int j = 0; j < R::group; ++j) {
                int const kk = g*R::group+j;
                weights[(int64_t(e)*n+col)*k+kk] = R::weight(R::code(l,h,col,kk,n),s).raw();
            }
        }
    }
}
extern "C" int qkg_host_read(int q, uint8_t const* low, uint8_t const* high, uint8_t const* units,
                            int n, int k, int e, uint16_t* w, uint16_t* s, uint16_t* z) {
#define QKG_CASE(Q,T) case Q: read_host<gguf_scale::KType::T>(low,high,units,n,k,e,w,s,z); return 0
    switch (q) {
        QKG_CASE(10,Q2_K); QKG_CASE(11,Q3_K); QKG_CASE(12,Q4_K); QKG_CASE(13,Q5_K); QKG_CASE(14,Q6_K);
        default: return QKG_FORMAT;
    }
#undef QKG_CASE
}
extern "C" int qkg_host_query(qkg_call_v1 const* c, qkg_config_v1 const* f,
        quactlize_ppu_placed_arrangement_v2 const* a, qkg_sizes_v1* s) {
    return quactlize::execution::query(*c,*f,a,*s);
}
