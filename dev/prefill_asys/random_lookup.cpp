// Host-only replay of the same selector used by the complete dense provider.
#include "quactlize/dispatch/dense_prefill.hpp"
#include <iostream>

int main() {
    using namespace quactlize::dispatch::dense_prefill;
    int q,m,n,k;
    while (std::cin>>q>>m>>n>>k) {
        qks_request_v1 r{1,sizeof(r),q,q==8?1:0,m,n,k,1,m,
            q==8?UINT64_C(0x51384b5032540001):q==12?UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001)};
        auto d=select(r,QK_COMPUTE_F16);
        std::cout<<(d?int(d.choice-kChoices):-1)<<' '<<d.policy<<'\n';
    }
}
