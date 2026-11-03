#include "dev/moe_decode_helpers/contract.hpp"
#include "quactlize/moe/selection.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>

int main() {
    using namespace quactlize::moe::decode_helpers;
    int tested=0;
    for(int m : {1,2,3,4,5,6,7,8,9,16,128,2048,65536}) {
        for(int n : {1,31,512,1024,2048,2051,3072,3073}) {
            qzm_finish_v1 f{};f.tokens=m;f.columns=n;f.topk=8;
            f.weights_stride=11;f.output_stride=n+3;
            f.down={nullptr,QZM_F32,QZM_TOKEN_SLOT,QZM_PROJECTED,1,n+3,0,QZM_F32,0};
            for(int dtype : {QZM_F32,QZM_F16,QZM_BF16}) {
                f.down.round_type=dtype;assert(finish_fast(f));++tested;
            }
            auto saved=f;
            f.topk=6;assert(!finish_fast(f));f=saved;
            f.down.dtype=QZM_BF16;assert(!finish_fast(f));f=saved;
            f.down.order=QZM_GROUPED;assert(!finish_fast(f));f=saved;
            f.down.state=QZM_PARTIALS;assert(!finish_fast(f));f=saved;
            f.down.partial_count=4;assert(!finish_fast(f));
            qzm_transition_v1 t{};t.tokens=m;t.topk=8;t.hidden=n;t.merged=1;
            t.gate={nullptr,QZM_F32,QZM_TOKEN_SLOT,QZM_PROJECTED,1,2*n+3,0,QZM_F32,0};
            t.output_dtype=QZM_F32;t.output_order=QZM_TOKEN_SLOT;t.output_stride=n+3;
            assert(transition_fast(t));auto st=t;
            t.merged=0;assert(!transition_fast(t));t=st;
            t.gate.round_type=QZM_BF16;assert(!transition_fast(t));t=st;
            t.gate.state=QZM_ACTIVATED;assert(!transition_fast(t));t=st;
            t.output_order=QZM_GROUPED;assert(!transition_fast(t));t=st;
            t.output_dtype=QZM_BF16;assert(!transition_fast(t));
        }
    }
    std::printf("DECODE_HELPER_HOST PASS legal_typed_finish=%d production_changed=0\n",tested);
}
