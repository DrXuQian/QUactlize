#pragma once
#include "simt.h"

namespace quactlize::execution::simt {
// The same implementation key is used by the host explanation and generated
// device entrypoints. A donor transfers geometry, never fixed dimensions.
enum class MeasuredDecode {
    None,
#define QDM(Id,...) Id,
#include "measured_decode.inc"
#undef QDM
};

inline MeasuredDecode measured_decode(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {
    auto const& c=d.call;
    if(c.input_type!=QKG_F32 || c.rows!=(c.mode==QKG_INDEXED?c.topk:1)) return MeasuredDecode::None;
#define QDM(Id,Q,Mode,N,K,E,Top,Ch,Compute,V,C,W,P,S,Changes,Hoist,Fixed) \
    if(c.qtype==Q && c.mode==Mode && c.n==N && c.k==K && c.experts==E && \
       c.topk==Top && c.channels==Ch && d.compute_type==Compute && f.variant==V && \
       f.columns==C && f.warps==W && f.values==P && f.split==S) return MeasuredDecode::Id;
#include "measured_decode.inc"
#undef QDM
    return MeasuredDecode::None;
}

inline char const* measured_decode_name(MeasuredDecode id) {
    switch(id) {
#define QDM(Id,...) case MeasuredDecode::Id:return #Id;
#include "measured_decode.inc"
#undef QDM
        default:return "none";
    }
}
}
