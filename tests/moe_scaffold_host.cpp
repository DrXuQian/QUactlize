#include "quactlize/moe/validation.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>

using namespace quactlize::moe;
static qzm_value_v1 value_at(uintptr_t address,int width,int state=QZM_PROJECTED) {
    qzm_value_v1 v{};v.data=reinterpret_cast<void*>(address);
    v.dtype=QZM_F32;v.order=QZM_GROUPED;v.state=state;
    v.partial_count=state==QZM_PARTIALS?4:1;v.row_stride=width;
    v.partial_stride=40960LL*width;v.round_type=state==QZM_ACTIVATED?QZM_F32:QZM_BF16;
    return v;
}
int main() {
    static_assert(sizeof(qzm_value_v1)==48);
    qzm_prepare_v1 p{};p.version=1;p.size=sizeof(p);
    p.route.version=1;p.route.size=sizeof(p.route);p.route.experts=256;p.route.topk=8;
    p.route.channels=1;p.route.ids_stride=8;p.route.source_stride=1;
    p.config={1,sizeof(p.config),QZR_TILED,512};p.columns=2048;p.dtype=QZM_BF16;p.input_row_stride=2048;
    quactlize::routing::Layout l;
    for(int m:{0,1,7,8,9,16,33,64,65,128,257,512,1024,2048,4096,5120,65537}) {
        p.route.tokens=m;assert(prepare(p,l,false)==0 && l.rows==m*8);
    }
    p.route.tokens=INT_MAX;assert(prepare(p,l,false)==QZR_OVERFLOW);
    p.route.tokens=5120;p.config.algorithm=QZR_AUTO;assert(prepare(p,l,false)==QZR_INVALID);
    p.config.algorithm=QZR_WARP;assert(prepare(p,l,false)==0);
    p.route.tokens=8193;assert(prepare(p,l,false)==QZR_CAPACITY);
    p.config.algorithm=QZR_TILED;p.route.tokens=9;p.route.inverse=1;
    assert(prepare(p,l,false)==QZR_INVALID);p.route.inverse=0;
    p.input_row_stride=INT64_MAX;assert(prepare(p,l,false)==QZR_OVERFLOW);

    qzm_transition_v1 c{};c.version=1;c.size=sizeof(c);c.topk=8;c.hidden=512;c.merged=1;
    c.gate=value_at(0x1000000000ULL,1024);c.output_dtype=QZM_BF16;
    c.output_stride=512;c.output=reinterpret_cast<void*>(0x3000000000ULL);
    uint32_t ops=0;
    for(int m:{0,1,8,9,16,64,128,5120}) {
        c.tokens=m;assert(transition(c,ops,true)==0 && ops==uint32_t(m?QZM_SWIGLU:QZM_NOOP));
        c.gate.state=QZM_PARTIALS;c.gate.partial_count=4;
        assert(transition(c,ops,true)==0 && ops==uint32_t(m?(QZM_REDUCE_GATE|QZM_SWIGLU):QZM_NOOP));
        c.gate.state=QZM_PROJECTED;c.gate.partial_count=1;
    }
    c.tokens=9;c.merged=0;c.gate=value_at(0x1000000000ULL,512,QZM_PARTIALS);
    c.up=value_at(0x2000000000ULL,512,QZM_PARTIALS);
    assert(transition(c,ops,true)==0 && ops==(QZM_REDUCE_GATE|QZM_REDUCE_UP|QZM_SWIGLU));
    c.up.order=QZM_TOKEN_SLOT;assert(transition(c,ops,true)==QZR_INVALID);
    c.row_ids=reinterpret_cast<int*>(0x4000000000ULL);assert(transition(c,ops,true)==0);
    c.output=const_cast<void*>(c.gate.data);assert(transition(c,ops,true)==QZR_INVALID);
    c.gate=value_at(0x1000000000ULL,512,QZM_ACTIVATED);
    c.output_dtype=QZM_F32;
    assert(transition(c,ops,true)==0 && ops==QZM_NOOP); // no double SwiGLU or cast
    c.output=reinterpret_cast<void*>(0x3000000000ULL);
    assert(transition(c,ops,true)==0 && ops==QZM_COPY_ACTIVATED);
    c.gate.partial_count=4;assert(transition(c,ops,false)==QZR_INVALID);
    c.gate.partial_count=1;c.gate.round_type=QZM_BF16;assert(transition(c,ops,false)==QZR_INVALID);
    c.gate.round_type=QZM_F32;c.output_order=QZM_TOKEN_SLOT;
    assert(transition(c,ops,true)==QZR_INVALID);
    c.inverse_rows=reinterpret_cast<int*>(0x5000000000ULL);assert(transition(c,ops,true)==0);

    qzm_finish_v1 f{};f.version=1;f.size=sizeof(f);f.tokens=5120;f.topk=8;f.columns=257;
    f.down=value_at(0x1000000000ULL,257,QZM_PARTIALS);
    f.weights=reinterpret_cast<float*>(0x3000000000ULL);f.weights_stride=11;
    f.output=reinterpret_cast<float*>(0x4000000000ULL);f.output_stride=263;
    assert(finish(f,true)==QZR_INVALID);
    f.inverse_rows=reinterpret_cast<int*>(0x5000000000ULL);assert(finish(f,true)==0);
    f.down.state=QZM_ACTIVATED;assert(finish(f,false)==QZR_INVALID);
    f.down.state=QZM_PARTIALS;f.down.partial_stride=1;assert(finish(f,false)==QZR_INVALID);
    f.down=value_at(0x1000000000ULL,257);f.down.order=QZM_TOKEN_SLOT;f.inverse_rows=nullptr;
    assert(finish(f,true)==0);
    f.output=const_cast<float*>(f.weights);assert(finish(f,true)==QZR_INVALID);
    f.tokens=0;f.down.data=nullptr;f.weights=f.output=nullptr;assert(finish(f,true)==0);
    uint64_t bytes=0;
    assert(matrix_bytes(INT_MAX,INT_MAX,INT64_MAX,QZM_F32,bytes)==QZR_OVERFLOW);
    assert(matrix_bytes(2,8,8,QZM_F32,bytes,INT_MAX,INT64_MAX)==QZR_OVERFLOW);
    puts("MOE_SCAFFOLD_HOST PASS");
}
