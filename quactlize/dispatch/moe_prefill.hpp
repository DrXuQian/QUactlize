#pragma once
#include "api.h"
#include "../../policies/kpack_moe_prefill_exact_v1.hpp"

namespace quactlize::dispatch::moe_prefill {
namespace data = quactlize_moe_prefill_exact_v1;

// Canonical projections, including explicitly measured concatenated gate/up.
// Never infer a router distribution or obtain a merged row by doubling N.
// The selected call uses public prepare v2 and complete BF16/F32 endpoints.
inline data::Row const* select(qks_request_v1 const& r,int compute,int channels) {
    if(r.version!=1 || r.size!=sizeof(r) || (r.route!=2 && r.route!=3) ||
       compute!=QK_COMPUTE_BF16 || r.experts!=256 || r.max_rows<=8 || int64_t(r.max_rows)*8!=r.m ||
       r.mapping_id!=(r.qtype==12?UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001)))
        return nullptr;
    data::Row const* donor=nullptr;
    // Nearest log-M within the SAME weight/activation family. This preserves
    // every exact winner and transfers only the token count, never N/K,
    // precision, channel broadcasting or router distribution. Above the
    // final knot reuse that knot; module queries still bound real resources.
    for(auto const& row:data::kRows) {
        if(row.q!=r.qtype || row.n!=r.n || row.k!=r.k || row.channels!=channels) continue;
        if(row.tokens>=r.max_rows) {
            if(!donor || int64_t(r.max_rows)*r.max_rows>int64_t(donor->tokens)*row.tokens) donor=&row;
            break;
        }
        donor=&row;
    }
    return donor;
}

inline int query(qks_request_v1 const& r,unsigned mask,int compute,int channels,qks_prefill_choice_v2& out) {
    out={};
    if(mask!=1 && mask!=3 && mask!=7) return QKS_INVALID;
    auto row=select(r,compute,channels);if(!row) return QKS_MISS;
    int route=row->config.route-2;
    if(!(mask&(1U<<route))) return QKS_BINDING;
    bool predicted=row->tokens!=r.max_rows;
    out.version=2;out.size=sizeof(out);out.policy=predicted?QKS_MOE_PREFILL_BUCKET:QKS_MOE_PREFILL_CONFIRMED;
    out.compute_type=compute;out.channels=channels;
    // A router-independent row has no single latency prediction. Per-profile
    // costs remain in the evidence JSON; zero means unreported, not free work.
    out.base={1,sizeof(out.base),route,row->dequant,row->tokens,int(predicted),0.,0.};
    return QKS_OK;
}
} // namespace quactlize::dispatch::moe_prefill
