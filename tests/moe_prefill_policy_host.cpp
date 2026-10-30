#include "quactlize/dispatch/selection.hpp"
#include <cassert>
#include <string>

using namespace quactlize::dispatch;
int main() {
    for(auto const& row:moe_prefill::data::kRows) {
        auto const& c=row.config;
        qks_request_v1 r{1,sizeof(r),row.q,c.route,row.tokens*8,row.n,row.k,256,row.tokens,c.mapping_id};
        qks_prefill_choice_v2 out{};
        assert(valid(r));
        assert(prefill::query(r,3,1,row.channels,out)==QKS_OK && out.policy==22);
        assert(out.base.route==c.route-2 && out.base.dequant_config==row.dequant);
        assert(out.base.measured_tokens==row.tokens && !out.base.predicted);
        auto decision=select_tc(r,false,1,{&c,out.policy});
        assert(decision && decision.symbol==c.symbol && decision.policy==22);
        auto selected=decision.config();
        assert(selected.tm==c.tm && selected.tn==c.tn && selected.tk==c.tk && selected.split==c.split);
        for(int cu:{64,72}) for(int occupancy:{1,2,6}) {
            auto launch=recipe(selected,r,occupancy,cu);
            assert(launch.split==c.split && launch.algorithm==int(c.grid_mode>=2));
            assert(launch.algorithm?launch.grid>0 && launch.grid<=cu*occupancy:launch.grid==0);
        }
        if(c.route==3) assert(prefill::query(r,1,1,row.channels,out)==QKS_BINDING);
        assert(!moe_prefill::select(r,0,row.channels));
        assert(!moe_prefill::select(r,1,row.channels==1?8:1));
        auto bad=r;bad.m-=1;assert(!moe_prefill::select(bad,1,row.channels));
        bad=r;bad.mapping_id^=1;assert(!moe_prefill::select(bad,1,row.channels));
        bad=r;bad.experts=128;assert(!moe_prefill::select(bad,1,row.channels));
        bad=r;bad.n*=2;assert(!moe_prefill::select(bad,1,row.channels));
        bad=r;bad.max_rows=9;bad.m=72;assert(!moe_prefill::select(bad,1,row.channels));
        bad=r;bad.route=0;bad.experts=1;bad.m=bad.max_rows;assert(!moe_prefill::select(bad,1,row.channels));
    }
    // Neighbors and unmeasured formats retain the old confirmed path exactly.
    for(auto const& k:prefill::data::kKnots) {
        qks_request_v1 r{1,sizeof(r),k.q,k.experts==1?0:2,k.tokens*(k.experts==1?1:8),
            k.n,k.k,k.experts,k.tokens,k.q==12?UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001)};
        if(moe_prefill::select(r,k.compute,k.channels)) continue;
        qks_prefill_choice_v2 old{},now{};
        assert(prefill::query_in(r,7,k.compute,k.channels,old,prefill::data::kKnots)==
               prefill::query(r,7,k.compute,k.channels,now));
        assert(old.policy==now.policy && old.base.route==now.base.route &&
            old.base.dequant_config==now.base.dequant_config && old.base.gemm_us==now.base.gemm_us);
    }
}
