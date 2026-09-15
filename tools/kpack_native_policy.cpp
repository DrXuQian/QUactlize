#include "quactlize/dispatch/policy.hpp"
#include "quactlize/dispatch/decode.hpp"
#include "quactlize/dispatch/compute.hpp"
#include <cstring>
#include <iostream>

int main(int argc,char** argv) {
    bool decode=false,bf16=false;
    for (int i=1;i<argc;++i) {
        if (!std::strcmp(argv[i],"--decode")) decode=true;
        else if (!std::strcmp(argv[i],"--bf16")) bf16=true;
        else return 2;
    }
    int q,route,m,n,k,e,maximum;
    while (std::cin>>q>>route>>m>>n>>k>>e>>maximum) {
        qks_request_v1 r{1,sizeof(r),q,route,m,n,k,e,maximum,
            q==8 ? q8_kpack2::kMappingId : q==12 ? UINT64_C(0x51344b5034540001) : UINT64_C(0x514b504b54000001)};
        auto s=decode ? quactlize::dispatch::select_decode_tc(r) : quactlize::dispatch::select(r);
        quactlize::dispatch::Config proposal{};std::string name;
        if (bf16 && quactlize::dispatch::valid(r) && (route>=2 || m<=8)) {
            if (!s.config) s=quactlize::dispatch::select(r);
            proposal=quactlize::dispatch::compute_proposal(r,s,name);
            s={proposal.symbol ? &proposal : nullptr,QKS_COMPUTE_INITIAL};
        } else if (bf16) s={};
        if (!s.config) { std::cout<<"MISS\n"; continue; }
        auto& c=*s.config;
        std::cout<<c.symbol<<' '<<c.qtype<<' '<<c.route<<' '<<c.tm<<' '<<c.tn<<' '<<c.tk<<' '
            <<c.wm<<' '<<c.wn<<' '<<c.stages<<' '<<c.ap<<' '<<c.dn<<' '<<c.parent_persistent<<' '
            <<c.split<<' '<<c.grid_mode<<' '<<c.grid_b<<' '<<s.policy<<'\n';
    }
    return std::cin.eof() ? 0 : 1;
}
