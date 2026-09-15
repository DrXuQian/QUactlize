#pragma once
#include "smallm.hpp"
#include "../../policies/kpack_smallm_matched_v1.hpp"

namespace quactlize::dispatch::matched {
namespace data=quactlize::smallm_matched_data;
struct Selection { data::Row const* row=nullptr; int policy=0; };

inline bool domain(qkg_simt_call_v2 const& d) {
    auto const& c=d.call;
    if(d.version!=2 || d.size!=sizeof(d) || c.version!=1 || c.size!=sizeof(c) ||
       (d.compute_type!=QK_COMPUTE_F16 && d.compute_type!=QK_COMPUTE_BF16) ||
       c.input_type!=QKG_F32 || c.n<=0 || c.k<=0 || c.n%256 ||
       c.k%((c.qtype==11 || c.qtype==14)?512:256) ||
       (c.qtype!=8 && (c.qtype<10 || c.qtype>14))) return false;
    if(c.mode==QKG_DENSE) return c.experts==1 && c.topk==1 && c.channels==1 && c.rows>=1 && c.rows<=8;
    return c.mode==QKG_INDEXED && c.experts==256 && c.topk==8 && c.rows>=8 && c.rows<=64 &&
        c.rows%8==0 && (c.channels==1 || c.channels==8);
}

inline bool eligible(data::Choice const& choice,qkg_call_v1 const& c) {
    if(choice.kind!=0) return true;
    auto const& f=choice.tc;
    return f.qtype==c.qtype && (f.route>=2)==(c.mode==QKG_INDEXED) &&
        c.k%(f.tk*f.split)==0 && c.k/(f.tk*f.split)>=f.stages-1 &&
        (!f.ap || (c.mode==QKG_DENSE && c.rows==1));
}

inline Selection select(qkg_simt_call_v2 const& d) {
    if(!domain(d)) return {};
    auto const& c=d.call;int m=smallm::tokens(c);
    auto family=[&](auto const& r) {
        return r.q==c.qtype && r.mode==c.mode && r.experts==c.experts && r.topk==c.topk &&
               r.channels==c.channels && r.compute==d.compute_type;
    };
    for(auto const& r:data::kExact) if(family(r) && r.n==c.n && r.k==c.k && r.tokens==m &&
        eligible(data::kChoices[r.choice],c)) return {&r,r.router_sensitive?QKS_MATCHED_ROUTER:QKS_MATCHED_EXACT};
    for(auto const& r:data::kOpen) if(family(r) && r.n==c.n && r.k==c.k && r.tokens==m) return {};
    data::Row const* donor=nullptr;int distance=INT32_MAX;
    for(auto const& b:data::kBuckets) {
        if(!family(b) || b.m!=smallm::token_bin(m)) continue;
        auto const& r=data::kExact[b.row];
        if(int64_t(c.n)>int64_t(r.n)*2 || int64_t(r.n)>int64_t(c.n)*2 ||
           int64_t(c.k)>int64_t(r.k)*2 || int64_t(r.k)>int64_t(c.k)*2 ||
           !eligible(data::kChoices[r.choice],c)) continue;
        int delta=std::abs(smallm::log2(c.n)-b.n)+std::abs(smallm::log2(c.k)-b.k);
        if(delta<distance) {distance=delta;donor=&r;}
    }
    return {donor,donor?QKS_MATCHED_BUCKET:0};
}
} // namespace quactlize::dispatch::matched
