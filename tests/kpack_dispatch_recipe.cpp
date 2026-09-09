#include "quactlize/dispatch/policy.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 12) return 2;
    int v[11];
    for (int i=0; i<11; ++i) v[i]=std::atoi(argv[i+1]);
    quactlize::dispatch::Config c{};
    c.tm=v[5]; c.tn=v[6]; c.split=v[7]; c.grid_mode=v[8]; c.grid_b=v[9];
    qks_request_v1 q{};
    q.route=v[0]; q.m=v[1]; q.n=v[2]; q.experts=v[3]; q.max_rows=v[4];
    auto r=quactlize::dispatch::recipe(c,q,v[10]);
    int64_t mt=q.route>=2 ? quactlize::moe_directory::bounded_entries(q.m,q.max_rows,q.experts,c.tm)
                         : (int64_t(q.m)+c.tm-1)/c.tm;
    int64_t work=mt*((int64_t(q.n)+c.tn-1)/c.tn)*(q.route>=2 ? c.split : 1);
    std::printf("algorithm=%d split=%d grid=%d work=%lld\n",r.algorithm,r.split,r.grid,
                static_cast<long long>(work));
    return 0;
}
