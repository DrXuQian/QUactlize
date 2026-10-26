#pragma once
#include "simt.h"

namespace quactlize::execution::simt {
// The v1 config ABI contains geometry only. Bind the measured body to the
// entire semantic key so a bucket donor cannot transfer token ownership.
inline int token_cta_window(qkg_simt_call_v2 const& d, qkg_simt_config_v1 const& f) {
    auto const& c = d.call;
    if (c.mode != QKG_DENSE || c.input_type != QKG_F32 ||
        d.compute_type != QKG_COMPUTE_F16 || c.experts != 1 || c.topk != 1 || c.channels != 1)
        return -1;
#define QTC(Q,N,K,M,V,C,W,P,S,Window) \
    if (c.qtype==Q && c.n==N && c.k==K && c.rows==M && f.variant==V && \
        f.columns==C && f.warps==W && f.values==P && f.split==S) return Window;
#include "token_cta.inc"
#undef QTC
    return -1;
}
}
