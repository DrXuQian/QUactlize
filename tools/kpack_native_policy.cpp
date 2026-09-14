#include "quactlize/dispatch/policy.hpp"
#include "quactlize/dispatch/decode.hpp"
#include <cstring>
#include <iostream>

int main(int argc,char** argv) {
    bool decode=argc==2 && !std::strcmp(argv[1],"--decode");
    if (argc!=1 && !decode) return 2;
    int q,route,m,n,k,e,maximum;
    while (std::cin>>q>>route>>m>>n>>k>>e>>maximum) {
        qks_request_v1 r{1,sizeof(r),q,route,m,n,k,e,maximum,
            q==8 ? q8_kpack2::kMappingId : q==12 ? UINT64_C(0x51344b5034540001) : UINT64_C(0x514b504b54000001)};
        auto s=decode ? quactlize::dispatch::select_decode_tc(r) : quactlize::dispatch::select(r);
        if (!s.config) { std::cout<<"MISS\n"; continue; }
        auto& c=*s.config;
        std::cout<<c.symbol<<' '<<c.qtype<<' '<<c.route<<' '<<c.tm<<' '<<c.tn<<' '<<c.tk<<' '
            <<c.wm<<' '<<c.wn<<' '<<c.stages<<' '<<c.ap<<' '<<c.dn<<' '<<c.parent_persistent<<' '
            <<c.split<<' '<<c.grid_mode<<' '<<c.grid_b<<' '<<s.policy<<'\n';
    }
    return std::cin.eof() ? 0 : 1;
}
