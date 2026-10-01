#pragma once
#include "api.h"
#include "../../policies/kpack_zw810_cost_v1.hpp"
#include <cstdint>
#include <cstdlib>

namespace quactlize::dispatch::cost {
namespace data = quactlize_kpack_cost_v1;

inline data::Knot const* knot(qks_request_v1 const& r) {
    if (r.version!=1 || r.size!=sizeof(r) || r.qtype<10 || r.qtype>14 ||
        r.route<0 || r.route>3 || r.max_rows<128 || r.max_rows>4096 ||
        r.mapping_id!=(r.qtype==12 ? UINT64_C(0x51344b5034540001) : UINT64_C(0x514b504b54000001)) ||
        (r.route<2 ? r.experts!=1 || r.m!=r.max_rows :
                    r.experts!=256 || int64_t(r.m)!=8LL*r.max_rows)) return nullptr;
    data::Knot const *lo=nullptr,*hi=nullptr;
    for (auto const& row:data::kKnots) {
        if (row.q!=r.qtype || row.n!=r.n || row.k!=r.k || row.experts!=r.experts) continue;
        if (row.tokens<=r.max_rows && (!lo || row.tokens>lo->tokens)) lo=&row;
        if (row.tokens>=r.max_rows && (!hi || row.tokens<hi->tokens)) hi=&row;
    }
    // No extrapolation to an unmeasured weight family or beyond its M domain.
    // Interior transfer is explicitly predicted, never a measured 5% claim.
    if (!lo || !hi) return nullptr;
    return r.max_rows-lo->tokens <= hi->tokens-r.max_rows ? lo : hi;
}

inline data::Config const* fixed_route(qks_request_v1 const& r) {
    auto row=knot(r);
    if (!row) return nullptr;
    auto c=row->choices[r.route%2];
    return c.config<0 ? nullptr : &data::kConfigs[c.config];
}

inline int query(qks_request_v1 const& r,unsigned mask,qks_prefill_choice_v1& out) {
    if (!(mask&1) || (mask&~7U) || ((mask&4) && !(mask&2))) return QKS_INVALID;
    auto row=knot(r);
    if (!row) return QKS_MISS;
    auto c=row->choices[mask==7 ? 3 : mask==3 ? 2 : 0];
    out={};out.version=1;out.size=sizeof(out);
    out.route=c.route;out.dequant_config=c.dequant;out.measured_tokens=row->tokens;
    out.predicted=row->tokens!=r.max_rows;
    out.gemm_us=c.gemm_us;out.dequant_us=c.dequant_us;
    return QKS_OK;
}
} // namespace quactlize::dispatch::cost
