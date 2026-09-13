#pragma once
#include "policy.hpp"
#include "../../policies/kpack_q4_decode_v1.hpp"

namespace quactlize::dispatch {
inline Selected select_decode_tc(qks_request_v1 const& r) {
    if (!valid(r) || r.qtype!=12) return {};
    int mode=-1, tokens=0;
    if (r.route==QK_DENSE_FQ && r.m<=8) { mode=0; tokens=r.m; }
    else if (r.route==QK_GROUPED_FQ && r.experts==256 && r.m<=64 &&
             r.m%8==0 && r.max_rows==r.m/8) { mode=2; tokens=r.m/8; }
    auto row=decode_policy::select(mode,r.n,r.k,tokens,1);
    // The automatic table may retain TC for a router-indistinguishable cohort;
    // its minimax choice need not equal the per-arm TC-only alternative.
    auto automatic=decode_policy::select(mode,r.n,r.k,tokens);
    if (automatic && !automatic->simt) row=automatic;
    return row ? Selected{&row->tc,QKS_DECODE_MEASURED} : Selected{};
}
} // namespace quactlize::dispatch
