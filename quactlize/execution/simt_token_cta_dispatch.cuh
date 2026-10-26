#pragma once
#include "simt_token_cta_scope.hpp"
#include "simt_token_cta.cuh"
#include "simt_q8_token_cta.cuh"

namespace quactlize::execution::simt {
// Keep the experimental device bodies byte-for-byte; admission is entirely
// this exact-key launcher binding, not a global change to generic SIMT.
template<int Qtype>
int token_cta_dispatch(qkg_simt_call_v2 const& d, qkg_simt_config_v1 const& f) {
    if (token_cta_window(d, f) < 0) return QKG_SHAPE;
    auto const& c = d.call;
#define QTC(Q,N,K,M,V,C,W,P,S,Window) \
    if constexpr (Qtype==Q) { \
        if (c.n==N && c.k==K && c.rows==M && f.variant==V && f.columns==C && \
            f.warps==W && f.values==P && f.split==S) { \
            if constexpr (Q==8) return q8_token_cta_launch<M,C,W,P,Window>(d,f.split); \
            else return token_cta_launch<M,C,W,P,Q>(d,f.split); \
        } \
    }
#include "token_cta.inc"
#undef QTC
    return QKG_SHAPE;
}
}
