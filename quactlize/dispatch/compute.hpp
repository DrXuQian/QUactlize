#pragma once
#include "policy.hpp"
#include <string>

namespace quactlize::dispatch {

inline bool compute_valid(int type) {
    return type==QK_COMPUTE_F16 || type==QK_COMPUTE_BF16;
}

// Geometry transferred to a different arithmetic contract is a proposal,
// never a measured BF16 winner. AP1 is an FP16-only packed-A provider.
inline Config compute_proposal(qks_request_v1 const& r, Selected donor, std::string& name) {
    Config c{};
    if (donor.config) {
        c=*donor.config;
        name=c.symbol;
        if (c.ap) {
            auto pos=name.find("_ap1_");
            if (c.ap!=1 || pos==std::string::npos) return {};
            name.replace(pos,5,"_ap0_");
            c.ap=0;
        }
    } else {
        // One small resource-bounded AP0 parent covers a missing shape family.
        // Resource query still decides admission; there is no runtime sweep.
        int tk=(r.qtype==11 || r.qtype==13) ? 256 : (r.qtype==10 || r.qtype==14) ? 128 : 64;
        c={"bf16-initial",nullptr,"COMPUTE_INITIAL",r.qtype,r.route,
           16,64,tk,16,16,2,0,16,r.route==2 ? 0 : -1,1,0,0,0,r.mapping_id};
        std::string q=std::to_string(r.qtype), layout=r.qtype==12 ? "1" : "2";
        std::string tile="tm16_tn64_tk"+std::to_string(tk)+"_wm16_wn16_s2";
        if (r.route==QK_DENSE_FQ) name="fqk_tc_q"+q+"_l"+layout+"_a0_"+tile+"_bc0_ap0_dn16";
        if (r.route==QK_DENSE_SF) name="sf_q"+q+"_a0_"+tile+"_bc0_ap0_dn16";
        if (r.route==QK_GROUPED_FQ) name="fqg_q"+q+"_l"+layout+"_"+tile+"_ap0_dn16_nonpersistent";
        if (r.route==QK_GROUPED_SF) name="sfg_q"+q+"_"+tile+"_ap0_dn16";
    }
    c.symbol=name.c_str();
    return c;
}
} // namespace quactlize::dispatch
