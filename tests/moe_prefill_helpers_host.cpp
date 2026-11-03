#include "dev/moe_prefill_helpers/contract.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>

int main() {
    using namespace quactlize::moe::prefill_helpers;
    alignas(64) unsigned char storage[128]{};
    int tested=0;
    for(int m : {1,2,7,8,9,17,512,2048,5120,8193,65536}) {
        for(int n : {4,32,512,516,2048,2052,3072}) {
            qzm_prepare_v1 p{};
            p.route.tokens=m;p.route.topk=8;p.route.channels=1;
            p.config.algorithm=QZR_TILED;p.config.tile_rows=512;
            p.dtype=QZM_BF16;p.columns=n;p.input_row_stride=n+4;
            p.input=reinterpret_cast<float const*>(storage);p.output=storage+64;
            assert(prepare_fast(p));auto sp=p;
            p.route.tokens=0;assert(!prepare_fast(p));p=sp;
            p.route.topk=6;assert(!prepare_fast(p));p=sp;
            p.route.channels=8;assert(!prepare_fast(p));p=sp;
            p.config.algorithm=QZR_AUTO;assert(!prepare_fast(p));p=sp;
            p.config.tile_rows=256;assert(!prepare_fast(p));p=sp;
            p.dtype=QZM_F16;assert(!prepare_fast(p));p=sp;
            p.columns+=1;assert(!prepare_fast(p));p=sp;
            p.input_row_stride+=1;assert(!prepare_fast(p));p=sp;
            p.input=reinterpret_cast<float const*>(storage+4);assert(!prepare_fast(p));p=sp;
            p.output=storage+2;assert(!prepare_fast(p));

            qzm_finish_v1 f{};
            f.tokens=m;f.topk=8;f.columns=n;f.output_stride=n+4;
            f.down={storage,QZM_BF16,QZM_GROUPED,QZM_PROJECTED,1,n+4,0,QZM_BF16,0};
            f.output=reinterpret_cast<float*>(storage+64);
            assert(finish_fast(f));auto sf=f;
            f.tokens=0;assert(!finish_fast(f));f=sf;
            f.topk=6;assert(!finish_fast(f));f=sf;
            f.columns+=1;assert(!finish_fast(f));f=sf;
            f.output_stride+=1;assert(!finish_fast(f));f=sf;
            f.down.dtype=QZM_F32;assert(!finish_fast(f));f=sf;
            f.down.round_type=QZM_F16;assert(!finish_fast(f));f=sf;
            f.down.order=QZM_TOKEN_SLOT;assert(!finish_fast(f));f=sf;
            f.down.state=QZM_PARTIALS;assert(!finish_fast(f));f=sf;
            f.down.partial_count=4;assert(!finish_fast(f));f=sf;
            f.down.row_stride+=1;assert(!finish_fast(f));f=sf;
            f.down.data=storage+2;assert(!finish_fast(f));f=sf;
            f.output=reinterpret_cast<float*>(storage+4);assert(!finish_fast(f));

            qzm_transition_v1 t{};
            t.tokens=m;t.topk=8;t.hidden=n;t.merged=1;
            t.gate={storage,QZM_BF16,QZM_GROUPED,QZM_PROJECTED,1,2*n+4,0,QZM_BF16,0};
            t.output=storage+64;t.output_dtype=QZM_BF16;t.output_order=QZM_GROUPED;t.output_stride=n+4;
            assert(transition_fast(t));auto st=t;
            t.tokens=0;assert(!transition_fast(t));t=st;
            t.hidden+=1;assert(!transition_fast(t));t=st;
            t.merged=0;assert(!transition_fast(t));t=st;
            t.gate.state=QZM_ACTIVATED;assert(!transition_fast(t));t=st;
            t.gate.dtype=QZM_F32;assert(!transition_fast(t));t=st;
            t.gate.round_type=QZM_F16;assert(!transition_fast(t));t=st;
            t.output_order=QZM_TOKEN_SLOT;assert(!transition_fast(t));t=st;
            t.output_dtype=QZM_F32;assert(!transition_fast(t));t=st;
            t.output_stride+=1;assert(!transition_fast(t));t=st;
            t.output=storage+2;assert(!transition_fast(t));
            ++tested;
        }
    }
    std::printf("PREFILL_HELPER_HOST PASS shapes=%d production_changed=0\n",tested);
}
