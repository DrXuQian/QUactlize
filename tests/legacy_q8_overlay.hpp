// Test-only pre-refactor overlay oracle; production folds its data at compile time.
#pragma once
#include "quactlize/dispatch/api.h"
#include "policies/kpack_q8_vector_v1.hpp"
#include "quactlize/execution/model_gemv_scope.hpp"
#include <cstring>

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

template<class Config>
inline bool select_tc(qkg_simt_call_v2 const& d,Config const& tc,qkg_simt_config_v1& f) {
    auto const& c=d.call;
    if(c.qtype!=8 || d.compute_type!=QKG_COMPUTE_F16 || tc.qtype!=8 || tc.route!=1 ||
       tc.parent_persistent!=-1 || tc.grid_mode!=0 || tc.grid_b!=0 || !tc.symbol) return false;
    for(auto const& r:quactlize::q8_vector_data::kTcRows) {
        if(!execution::model_gemv::dense_m1(c,r.n,r.k) || std::strcmp(tc.symbol,r.symbol) ||
           tc.tm!=r.tm || tc.tn!=r.tn || tc.tk!=r.tk || tc.wm!=r.wm || tc.wn!=r.wn ||
           tc.stages!=r.stages || tc.ap!=r.ap || tc.dn!=r.dn || tc.split!=r.split) continue;
        auto a=r.candidate;f={1,sizeof(f),a[0],a[1],a[2],a[3],a[4]};return true;
    }
    return false;
}

// Bucket requests inherit their donor's implementation upgrades as well as
// its original geometry. This is a prediction, not a new measured row. Never
// choose a donor here or cross a format/compute/operator/token family.
template<class Row,class Choice>
inline bool select_bucket(qkg_simt_call_v2 const& d,Row const& donor,
                          Choice const& choice,qkg_simt_config_v1& out) {
    auto const& c=d.call;
    if(c.qtype!=8 || donor.q!=8 || c.input_type!=QKG_F32 ||
       c.mode!=donor.mode || c.experts!=donor.experts || c.topk!=donor.topk ||
       c.channels!=donor.channels || d.compute_type!=donor.compute ||
       c.rows<=0 || c.topk<=0 || donor.tokens<=0 || c.n<=0 || c.k<=0 ||
       donor.n<=0 || donor.k<=0 ||
       (c.mode==QKG_INDEXED && c.rows%c.topk)) return false;
    int tokens=c.mode==QKG_INDEXED ? c.rows/c.topk : c.rows;
    auto bin=[](int m) {int b=0;for(--m;m>0;m>>=1)++b;return b;};
    if(tokens>8 || donor.tokens>8 || bin(tokens)!=bin(donor.tokens) ||
       int64_t(c.n)>int64_t(donor.n)*2 || int64_t(donor.n)>int64_t(c.n)*2 ||
       int64_t(c.k)>int64_t(donor.k)*2 || int64_t(donor.k)>int64_t(c.k)*2)
        return false;
    auto source=d;
    source.call.n=donor.n;source.call.k=donor.k;
    source.call.rows=donor.tokens*(donor.mode==QKG_INDEXED ? donor.topk : 1);
    qkg_simt_config_v1 candidate{};
    bool upgraded=false;
    if(choice.kind==QKS_SMALLM_TC) upgraded=select_tc(source,choice.tc,candidate);
    else if(choice.kind==QKS_SMALLM_SIMT) {
        auto const& f=choice.reader;
        candidate={1,sizeof(candidate),f.variant,f.columns,f.warps,f.values,f.split};
        upgraded=select(source,candidate);
    }
    if(upgraded) out=candidate;
    return upgraded;
}
}
