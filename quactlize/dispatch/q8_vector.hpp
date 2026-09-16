#pragma once
#include "api.h"
#include "../../policies/kpack_q8_vector_v1.hpp"

namespace quactlize::dispatch::q8_vector {
inline bool select(qkg_simt_call_v2 const& d,qkg_simt_config_v1& f) {
    auto const& c=d.call;
    if(c.qtype!=8 || c.input_type!=QKG_F32 || c.topk<=0) return false;
    int tokens=c.mode==QKG_INDEXED ? c.rows/c.topk : c.rows;
    for(auto const& r:quactlize::q8_vector_data::kRows) {
        if(r.mode!=c.mode || r.n!=c.n || r.k!=c.k || r.experts!=c.experts ||
           r.topk!=c.topk || r.channels!=c.channels || r.tokens!=tokens || r.compute!=d.compute_type)
            continue;
        auto const* a=r.baseline;
        if(f.variant!=a[0] || f.columns!=a[1] || f.warps!=a[2] || f.values!=a[3] || f.split!=a[4])
            return false;
        a=r.candidate;
        f={1,sizeof(f),a[0],a[1],a[2],a[3],a[4]};
        return true;
    }
    return false;
}
}
