#pragma once
#include "policy.hpp"
#include "../execution/simt_validation.hpp"
#include "../../policies/kpack_smallm_v1.hpp"

namespace quactlize::dispatch::smallm {
namespace data=quactlize::smallm_data;
struct Selection { data::Row const* row=nullptr; int policy=0; };

inline int log2(int n) { int b=0; while (n>1) { n/=2; ++b; } return b; }
inline int token_bin(int n) { return n==1 ? 0 : log2(n-1)+1; }
inline int tokens(qkg_call_v1 const& c) { return c.mode==QKG_INDEXED ? c.rows/8 : c.rows; }

inline bool eligible(data::Choice const& choice,qkg_call_v1 const& c) {
    if (choice.simt) return true;
    auto const& f=choice.tc;
    return f.qtype==c.qtype && (f.route>=2)==(c.mode==QKG_INDEXED) &&
        c.k%(f.tk*f.split)==0 && c.k/(f.tk*f.split)>=f.stages-1 &&
        (!f.ap || (c.mode==QKG_DENSE && c.rows==1)) &&
        (c.mode==QKG_INDEXED || f.tm!=8 || c.rows<=8);
}

inline Selection select(qkg_call_v1 const& c) {
    int m=tokens(c);
    if (c.qtype==12 || c.input_type!=QKG_F32 || m<1 || m>8 ||
        (c.mode!=QKG_DENSE && c.mode!=QKG_INDEXED) ||
        (c.mode==QKG_INDEXED && (c.experts!=256 || c.topk!=8 || c.rows%8 ||
                              (c.channels!=1 && c.channels!=8)))) return {};
    for (auto const& r:data::kExact)
        if (r.q==c.qtype && r.mode==c.mode && r.n==c.n && r.k==c.k &&
            r.tokens==m && r.channels==c.channels && eligible(data::kChoices[r.choice],c))
            return {&r,QKS_SMALLM_EXACT};
    data::Row const* best=nullptr;
    int distance=INT32_MAX;
    for (auto const& b:data::kBuckets) {
        if (b.q!=c.qtype || b.mode!=c.mode || b.channels!=c.channels) continue;
        auto const& r=data::kExact[b.row];
        if (!eligible(data::kChoices[r.choice],c)) continue;
        int d=4*std::abs(token_bin(m)-b.m)+std::abs(log2(c.n)-b.n)+std::abs(log2(c.k)-b.k);
        if (d<distance) { distance=d; best=&r; }
    }
    return {best,best ? QKS_SMALLM_BUCKET : 0};
}

inline int validate(qkg_call_v1 const& c,quactlize_ppu_placed_arrangement_v2 const* a) {
    qkg_sizes_v1 sizes{};
    int rc=quactlize::execution::query(c,{1,sizeof(qkg_config_v1),16,4,1},a,sizes);
    if (rc) return QKS_INVALID;
    if (c.a_row_stride%4 || (c.mode==QKG_INDEXED && c.a_token_stride%4) ||
        ((uintptr_t(c.a)|uintptr_t(c.low)|uintptr_t(c.high))&15)) return QKS_MISS;
    return QKS_OK;
}
} // namespace quactlize::dispatch::smallm
