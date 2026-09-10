#include "quactlize/dispatch/policy.hpp"
#include <iostream>

int main() {
    int q,route,m,n,k,e,maximum;
    while (std::cin>>q>>route>>m>>n>>k>>e>>maximum) {
        qks_request_v1 r{1,sizeof(r),q,route,m,n,k,e,maximum,
            q==8 ? q8_kpack2::kMappingId : q==12 ? UINT64_C(0x51344b5034540001) : UINT64_C(0x514b504b54000001)};
        auto s=quactlize::dispatch::select(r);
        if (!s.config) { std::cout<<"MISS\n"; continue; }
        auto& c=*s.config;
        std::cout<<c.symbol<<' '<<c.qtype<<' '<<c.route<<' '<<c.tm<<' '<<c.tn<<' '<<c.tk<<' '
            <<c.wm<<' '<<c.wn<<' '<<c.stages<<' '<<c.ap<<' '<<c.dn<<' '<<c.parent_persistent<<' '
            <<c.split<<' '<<c.grid_mode<<' '<<c.grid_b<<' '<<s.policy<<'\n';
    }
    return std::cin.eof() ? 0 : 1;
}
