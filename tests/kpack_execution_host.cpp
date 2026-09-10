#include "reader.hpp"
#include "validation.hpp"
#include "../quactlize/runtime/grouped_workspace.hpp"

template<class R, bool Pair = false>
void read_host(uint8_t const* low, uint8_t const* high, uint8_t const* units,
               int n, int k, int experts, uint16_t* weights, uint16_t* scales, uint16_t* zeros) {
    for (int e = 0; e < experts; ++e) {
        int64_t const nk = int64_t(n)*k;
        auto l = reinterpret_cast<uint16_t const*>(low + e*nk*R::lo_bits/8);
        uint16_t const* h = nullptr;
        if constexpr (R::hi_bits) h = reinterpret_cast<uint16_t const*>(high + e*nk*R::hi_bits/8);
        auto u = units + e*R::metadata_bytes(nk);
        for (int col = 0; col < n; ++col) for (int g = 0; g < k/R::group; ++g) {
            auto s = R::scale(u,col,g,n);
            int64_t const si = (int64_t(e)*(k/R::group)+g)*n+col;
            scales[si] = s.scale.raw(); zeros[si] = s.zero.raw();
            for (int j = 0; j < R::group; ++j) {
                int const kk = g*R::group+j;
                int code=R::code(l,h,col,kk,n);
                if constexpr (Pair) {
                    int raw=code+(R::lo_bits==8 ? 128 : R::lo_bits==4 ? 8 : 0);
                    weights[(int64_t(e)*n+col)*k+kk]=uint16_t(R::weight_pair(raw,raw,s));
                } else weights[(int64_t(e)*n+col)*k+kk] = R::weight(code,s).raw();
            }
        }
    }
}
extern "C" int qkg_host_read(int q, uint8_t const* low, uint8_t const* high, uint8_t const* units,
                            int n, int k, int e, uint16_t* w, uint16_t* s, uint16_t* z) {
#define QKG_CASE(Q,T) case Q: read_host<quactlize::execution::Reader<gguf_scale::KType::T>>(low,high,units,n,k,e,w,s,z); return 0
    switch (q) {
        case 8: read_host<quactlize::execution::Q8Reader>(low,high,units,n,k,e,w,s,z); return 0;
        QKG_CASE(10,Q2_K); QKG_CASE(11,Q3_K); QKG_CASE(12,Q4_K); QKG_CASE(13,Q5_K); QKG_CASE(14,Q6_K);
        default: return QKG_FORMAT;
    }
#undef QKG_CASE
}
extern "C" int qkg_host_pair_read(int q, uint8_t const* low, uint8_t const* high, uint8_t const* units,
                            int n, int k, int e, uint16_t* w, uint16_t* s, uint16_t* z) {
#define QKG_CASE(Q,T) case Q: read_host<quactlize::execution::Reader<gguf_scale::KType::T>,true>(low,high,units,n,k,e,w,s,z); return 0
    switch (q) {
        case 8: read_host<quactlize::execution::Q8Reader,true>(low,high,units,n,k,e,w,s,z); return 0;
        QKG_CASE(10,Q2_K); QKG_CASE(11,Q3_K); QKG_CASE(12,Q4_K); QKG_CASE(13,Q5_K); QKG_CASE(14,Q6_K);
        default: return QKG_FORMAT;
    }
#undef QKG_CASE
}
extern "C" int qkg_host_query(qkg_call_v1 const* c, qkg_config_v1 const* f,
        quactlize_ppu_placed_arrangement_v2 const* a, qkg_sizes_v1* s) {
    return quactlize::execution::query(*c,*f,a,*s);
}

template<gguf_scale::KType T>
int reuse_host(uint8_t const* low, uint8_t const* high, int n, int k, int experts, bool rotate) {
    using R=quactlize::execution::Reader<T>;
    int bad=0;
    for (int e=0; e<experts; ++e) {
        int64_t nk=int64_t(n)*k;
        auto lo=reinterpret_cast<uint16_t const*>(low+e*nk*R::lo_bits/8);
        uint16_t const* hi=nullptr;
        if constexpr (R::hi_bits) hi=reinterpret_cast<uint16_t const*>(high+e*nk*R::hi_bits/8);
        for (int col=0; col<n; ++col) for (int g=0; g<k/R::group; ++g) for (int r=0; r<8; ++r) {
            int begin=g*R::group+r;
            uint16_t lw=lo[R::LowMap::word_index(col,begin,n)], hw=0;
            if constexpr (R::hi_bits) hw=hi[R::HighMap::word_index(col,begin,n)];
            for (int slot=0; slot<R::group/8; ++slot) {
                int kk=begin+8*slot;
                int read_k=rotate ? begin+8*((slot+1)%(R::group/8)) : kk;
                int want=R::code(lo,hi,col,kk,n)+(R::lo_bits==4 ? 8 : 0);
                int got=R::raw_from_words(lw,hw,col,read_k);
                if (want!=got) ++bad;
            }
        }
    }
    return bad;
}
extern "C" int qkg_host_reuse(int q, uint8_t const* l, uint8_t const* h,
                              int n, int k, int e, bool rotate) {
#define QKG_REUSE(Q,T) case Q: return reuse_host<gguf_scale::KType::T>(l,h,n,k,e,rotate)
    switch(q) {
        QKG_REUSE(10,Q2_K); QKG_REUSE(11,Q3_K); QKG_REUSE(12,Q4_K);
        QKG_REUSE(13,Q5_K); QKG_REUSE(14,Q6_K);
        default: return -1;
    }
#undef QKG_REUSE
}
extern "C" int qkg_host_pair_query(qkg_call_v1 const* c, qkg_config_v1 const* f,
        quactlize_ppu_placed_arrangement_v2 const* a, qkg_sizes_v1* s) {
    return quactlize::execution::query(*c,*f,a,*s,true);
}
extern "C" bool qkg_host_grouped_workspace(int e, int split, int m, int n,
        uint64_t head, uint64_t shape, uint64_t stride, bool device_rows,
        quactlize::runtime::GroupedWorkspace* out) {
    return quactlize::runtime::grouped_workspace(e,split,m,n,head,shape,stride,device_rows,*out);
}
extern "C" int qkg_host_pair_arithmetic() {
    using R=quactlize::execution::Reader<gguf_scale::KType::Q4_K>;
    using Half=cutlass::half_t;
    int bad=0;
    for (uint16_t scale : {uint16_t(0),uint16_t(1),uint16_t(0x0200),uint16_t(0x211f),uint16_t(0x3c00)})
      for (uint16_t zero : {uint16_t(0),uint16_t(0x8000),uint16_t(0x2e66),uint16_t(0xc000)})
        for (int a=0; a<64; ++a) for (int b=0; b<64; ++b) {
            gguf_scale::GroupScale s{Half::bitcast(scale),Half::bitcast(zero)};
            uint32_t pair=R::weight_pair(a,b,s);
            Half w0(float(a-8)*float(s.scale)+float(s.zero));
            Half w1(float(b-8)*float(s.scale)+float(s.zero));
            uint32_t want=uint32_t(w0.raw())|(uint32_t(w1.raw())<<16);
            if (pair!=want) ++bad;
        }
    return bad;
}
