#pragma once
#include "api.h"
#include "../../policies/kpack_prefill_refresh_v1.hpp"
#include "moe_prefill.hpp"

namespace quactlize::dispatch::prefill {
namespace data = quactlize_prefill_refresh_v1;

// Exact measured domain only. The grouped calibration used shared token A,
// E256/top8 and BF16; channels8, F16 grouped and BF16 dense are NOT aliases.
// Unknown keys retain the caller's explicitly labelled proposal path.
template<class Knot, size_t N>
inline Knot const* knot_in(qks_request_v1 const& r,int compute,int channels,Knot const (&knots)[N]) {
    if (r.version!=1 || r.size!=sizeof(r) || r.qtype<10 || r.qtype>14 ||
        r.route<0 || r.route>3 || channels!=1 || r.max_rows<128 ||
        r.mapping_id!=(r.qtype==12 ? UINT64_C(0x51344b5034540001) : UINT64_C(0x514b504b54000001)) ||
        (r.route<2 ? r.experts!=1 || r.m!=r.max_rows || compute!=QK_COMPUTE_F16 :
                    r.experts!=256 || int64_t(r.m)!=8LL*r.max_rows || compute!=QK_COMPUTE_BF16)) return nullptr;
    for (auto const& k:knots)
        if (k.q==r.qtype && k.n==r.n && k.k==r.k && k.experts==r.experts &&
            k.tokens==r.max_rows && k.compute==compute && k.channels==channels) return &k;
    return nullptr;
}

inline data::Knot const* knot(qks_request_v1 const& r,int compute,int channels) {
    return knot_in(r,compute,channels,data::kKnots);
}

template<class Knot, size_t N>
inline int query_in(qks_request_v1 const& r,unsigned mask,int compute,int channels,
                    qks_prefill_choice_v2& out,Knot const (&knots)[N]) {
    out={};
    if (mask!=1 && mask!=3 && mask!=7) return QKS_INVALID;
    auto k=knot_in(r,compute,channels,knots);
    if (!k) return QKS_MISS;
    // A covered key must not silently run an older slower recipe when a
    // required provider/helper is absent. Deployment can explicitly disable
    // this policy, but then it is not the confirmed strategy.
    if (!(mask & (1U<<k->route))) return QKS_BINDING;
    out.version=2;out.size=sizeof(out);out.policy=QKS_PREFILL_COMPONENT_MEASURED;
    out.compute_type=compute;out.channels=channels;
    // predicted describes shape transfer, not the latest timing gate. Six
    // exact keys keep their prior recipe/cost; the catalog records their open
    // refresh timing status separately instead of claiming a fresh speedup.
    out.base={1,sizeof(out.base),k->route,k->dequant,k->tokens,0,k->core_us,k->dequant_us};
    return QKS_OK;
}

inline int query(qks_request_v1 const& r,unsigned mask,int compute,int channels,qks_prefill_choice_v2& out) {
    int rc=moe_prefill::query(r,mask,compute,channels,out);
    if(rc!=QKS_MISS) return rc;
    return query_in(r,mask,compute,channels,out,data::kKnots);
}
} // namespace quactlize::dispatch::prefill
