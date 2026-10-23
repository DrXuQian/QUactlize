#pragma once
#include "policy.hpp"

namespace quactlize::dispatch::dense_prefill {
// Whole-call F16 dense recipes. dequant is SF's scale-prepass config for TC,
// or the typed dense weight-helper ID for Full. io=-1 means original row casts.
struct Choice { int route, io, dequant; Config tc; };
struct Exact { int q,m,n,k,choice; };
struct Bucket { int q,mb,nb,kb,aspect,nlo,nhi,klo,khi,choice; };
#include "dense_prefill.inc"

struct Decision {
    Choice const* choice=nullptr;
    int policy=0;
    explicit operator bool() const { return choice!=nullptr; }
};
inline int band(int x) { int b=0;while(x>1) {x>>=1;++b;}return b; }
inline bool domain(qks_request_v1 const& r,int compute) {
    return valid(r) && compute==QK_COMPUTE_F16 && r.route<2 && r.experts==1 &&
        r.m==r.max_rows && r.m>=128 && r.m<=2048 && int64_t(r.m)*r.n<=INT32_MAX;
}
inline bool legal(Choice const& c,qks_request_v1 const& r) {
    if(c.route==2) return true; // typed Full checks actual byte extents at prepare
    auto const& t=c.tc;
    return t.qtype==r.qtype && t.mapping_id==r.mapping_id && t.split>0 && t.tk>0 &&
        r.k%(t.tk*t.split)==0 && r.k/(t.tk*t.split)>=t.stages-1 &&
        r.n%t.tn==0 && (t.tm!=8 || r.m<=64) && !t.ap;
}
// Called only after the complete provider verifies ZW810 and contiguous F32
// endpoints. No dtype transfer, kernel launch, runtime tuning or remote donor.
inline Decision select(qks_request_v1 const& r,int compute) {
    if(!domain(r,compute)) return {};
    for(auto const& p:kExact)
        if(p.q==r.qtype && p.m==r.m && p.n==r.n && p.k==r.k)
            return {&kChoices[p.choice],QKS_DENSE_PREFILL_EXACT};
    int mb=band(r.m),nb=band(r.n),kb=band(r.k);
    int aspect=int64_t(r.n)>=2LL*r.k?1:int64_t(r.k)>=2LL*r.n?2:r.n>=r.k?0:3;
    for(auto const& b:kBuckets)
        if(b.q==r.qtype && b.mb==mb && b.nb==nb && b.kb==kb && b.aspect==aspect &&
            r.n>=b.nlo && r.n<=b.nhi && r.k>=b.klo && r.k<=b.khi && legal(kChoices[b.choice],r))
            return {&kChoices[b.choice],QKS_DENSE_PREFILL_BUCKET};
    return {};
}
} // namespace quactlize::dispatch::dense_prefill
