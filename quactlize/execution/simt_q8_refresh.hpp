#pragma once
#include "simt.h"

namespace quactlize::execution::simt {
// Body identity is independent of geometry. Only the measured dense/F32/F16
// exact keys can opt in; no MoE, split, BF16 or unseen-shape extrapolation.
inline int q8_refresh_body(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {
    auto const& c=d.call;
    if(c.qtype!=8 || c.mode!=QKG_DENSE || c.input_type!=QKG_F32 ||
       d.compute_type!=QKG_COMPUTE_F16 || c.experts!=1 || c.topk!=1 || c.channels!=1 ||
       f.variant!=5 || f.values!=4 || f.split!=1) return 0;
#define Q8R(N,K,M,C,W,Body) \
    if(c.n==N && c.k==K && c.rows==M && f.columns==C && f.warps==W) return Body;
#include "q8_refresh.inc"
#undef Q8R
    return 0;
}
}
