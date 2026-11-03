#include "quactlize/moe/selection.hpp"
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
static void projected_helpers() {
    qzm_finish_v1 f{};f.version=1;f.size=sizeof(f);f.topk=8;f.weights_stride=8;
    for(int m:{0,1,2,3,4,5,6,7,8,9,16,17,32,64,128,129,2048,5120}) {
        for(int n:{1,512,2048,2051,3072,4096}) {
            f.tokens=m;f.columns=n;f.output_stride=n;
            f.down=value_at(0x1000000000ULL,n);f.down.order=QZM_TOKEN_SLOT;f.down.round_type=QZM_F32;
            bool wanted=((m>=1 && m<=8) || m==16 || m==128) && (n==2048 || n==3072);
            assert(use_projected_finish(f)==wanted);
            qzm_plan_v1 plan{};
            assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_AUTO,&plan)==0);
            assert(plan.implementation==QZM_FINISH_SERIAL && plan.stages==int(m>0) && !plan.workspace_bytes);
            for(int dtype:{QZM_F16,QZM_BF16}) {
                auto bad=f;bad.down.dtype=dtype;assert(!use_projected_finish(bad));
                bad=f;bad.down.round_type=dtype;assert(!use_projected_finish(bad));
            }
            auto bad=f;bad.down.order=QZM_GROUPED;assert(!use_projected_finish(bad));
            bad=f;bad.down.state=QZM_PARTIALS;bad.down.partial_count=4;assert(!use_projected_finish(bad));
            bad=f;bad.topk=6;assert(!use_projected_finish(bad));
            // Legal strides and changing addresses are call bindings, not cached selection state.
            ++f.output_stride;++f.down.row_stride;++f.weights_stride;f.down.data=nullptr;
            assert(use_projected_finish(f)==wanted);
        }
    }
    qzm_transition_v1 t{};t.version=1;t.size=sizeof(t);t.merged=1;
    t.output_dtype=QZM_F32;t.output_order=QZM_TOKEN_SLOT;
    for(int m:{0,1,2,3,4,5,6,7,8,9,16,17,128,2048,5120}) {
        for(int h:{1,256,512,513,1024,2048}) for(int k:{1,2,6,8,16}) {
            t.tokens=m;t.topk=k;t.hidden=h;t.output_stride=h;
            t.gate=value_at(0x1000000000ULL,2*h);t.gate.order=QZM_TOKEN_SLOT;t.gate.round_type=QZM_F32;
            bool wanted=m>=1 && m<=8 && (h==512 || h==1024) && (k==1 || k==8);
            uint32_t ops=0;assert(transition(t,ops,false)==0);
            assert(ops==uint32_t(m?QZM_SWIGLU:QZM_NOOP));
            assert(use_projected_transition(t)==wanted);
            for(int dtype:{QZM_F16,QZM_BF16}) {
                auto bad=t;bad.gate.dtype=dtype;assert(!use_projected_transition(bad));
                bad=t;bad.gate.round_type=dtype;assert(!use_projected_transition(bad));
                bad=t;bad.output_dtype=dtype;assert(!use_projected_transition(bad));
            }
            auto bad=t;bad.gate.order=QZM_GROUPED;assert(!use_projected_transition(bad));
            bad=t;bad.output_order=QZM_GROUPED;assert(!use_projected_transition(bad));
            bad=t;bad.gate.state=QZM_PARTIALS;bad.gate.partial_count=4;assert(!use_projected_transition(bad));
            bad=t;bad.gate.state=QZM_ACTIVATED;assert(!use_projected_transition(bad));
            bad=t;bad.merged=0;assert(!use_projected_transition(bad));
            ++t.output_stride;++t.gate.row_stride;t.gate.data=nullptr;t.stream=reinterpret_cast<void*>(2);
            assert(use_projected_transition(t)==wanted);
        }
    }
}
int main() {
    projected_helpers();
    static_assert(sizeof(qzm_value_v1)==48);
    static_assert(sizeof(qzm_prepare_plan_v2)==48);
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

    static_assert(sizeof(qzm_plan_v1)==24);
    qzm_plan_v1 plan{},other{};
    p.input_row_stride=p.columns;p.route.tokens=16;p.config.algorithm=QZR_TILED;
    assert(quactlize_ppu_moe_prepare_select_v1(&p,QZM_AUTO,&plan)==0);
    assert(plan.implementation==QZM_PREPARE_FUSED && plan.stages==3);
    p.route.tokens=32;
    assert(quactlize_ppu_moe_prepare_select_v1(&p,QZM_AUTO,&plan)==0);
    assert(plan.implementation==QZM_PREPARE_HYBRID && plan.stages==4);
    p.route.tokens=17; // Not a measured M: retain the original implementation.
    assert(quactlize_ppu_moe_prepare_select_v1(&p,QZM_AUTO,&plan)==0 && plan.implementation==QZM_PREPARE_FUSED);
    p.config.algorithm=QZR_WARP;
    assert(quactlize_ppu_moe_prepare_select_v1(&p,QZM_PREPARE_SEPARATE,&plan)==0 && plan.stages==3 && !plan.workspace_bytes);
    assert(quactlize_ppu_moe_prepare_select_v1(&p,QZM_PREPARE_HYBRID,&plan)==QZR_INVALID);
    for(int tile:{256,512,1024}) {
        p.config={1,sizeof(p.config),QZR_TILED,tile};
        assert(quactlize_ppu_moe_prepare_select_v1(&p,QZM_PREPARE_SEPARATE,&plan)==0 && plan.stages==5);
        assert(quactlize_ppu_moe_prepare_select_v1(&p,QZM_PREPARE_HYBRID,&plan)==0 && plan.stages==4);
    }
    p.route.tokens=0;
    assert(quactlize_ppu_moe_prepare_select_v1(&p,QZM_AUTO,&plan)==0 && plan.stages==1 && !plan.workspace_bytes);

    f.tokens=2048;f.topk=8;f.columns=3072;f.weights_stride=8;f.output_stride=3072;
    f.down=value_at(0x1000000000ULL,3072,QZM_PARTIALS);
    f.down.partial_count=8;f.down.partial_stride=int64_t(f.tokens)*8*3072;
    f.weights=reinterpret_cast<float*>(0x3000000000ULL);
    f.output=reinterpret_cast<float*>(0x4000000000ULL);f.inverse_rows=reinterpret_cast<int*>(0x5000000000ULL);
    assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_AUTO,&plan)==0);
    assert(plan.implementation==QZM_FINISH_SEPARATE && plan.stages==3 && plan.workspace_bytes==402653184);
    assert(finish(f,true)==0);
    void* scratch=reinterpret_cast<void*>(0x6000000000ULL);
    assert(finish_workspace(f,plan.workspace_bytes,scratch,plan.workspace_bytes)==0);
    assert(finish_workspace(f,plan.workspace_bytes,scratch,plan.workspace_bytes-1)==QZR_CAPACITY);
    for(auto ptr:{f.down.data,static_cast<void const*>(f.weights),static_cast<void const*>(f.output),
                  static_cast<void const*>(f.inverse_rows)})
        assert(finish_workspace(f,plan.workspace_bytes,const_cast<void*>(ptr),plan.workspace_bytes)==QZR_INVALID);
    assert(finish_workspace(f,plan.workspace_bytes,reinterpret_cast<void*>(0x6000000004ULL),plan.workspace_bytes)==QZR_INVALID);
    assert(finish_workspace(f,plan.workspace_bytes,reinterpret_cast<void*>(UINTPTR_MAX-15),plan.workspace_bytes)==QZR_INVALID);
    // Dynamic addresses/streams are not cached in the plan.
    f.down.data=reinterpret_cast<void*>(0x7000000000ULL);f.stream=reinterpret_cast<void*>(2);
    assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_AUTO,&other)==0 && same_plan(plan,other));
    ++other.workspace_bytes;assert(!same_plan(plan,other));
    other=plan;++other.stages;assert(!same_plan(plan,other));
    other=plan;other.version=2;assert(!same_plan(plan,other));
    assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_PREPARE_FUSED,&other)==QZR_INVALID);
    // Original producer split is irrelevant after it returns final output.
    f.down.state=QZM_PROJECTED;f.down.partial_count=1;
    assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_AUTO,&other)==0 && other.implementation==QZM_FINISH_SERIAL);
    f.tokens=1;
    assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_AUTO,&other)==0 && other.implementation==QZM_FINISH_SLOTS4);
    f.tokens=8;
    assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_AUTO,&other)==0 && other.implementation==QZM_FINISH_SLOTS4);
    f.tokens=17;
    assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_AUTO,&other)==0 && other.implementation==QZM_FINISH_SERIAL);
    assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_FINISH_SLOTS4,&other)==0); // Explicit arbitrary M remains legal.
    f.tokens=0;f.down.data=nullptr;f.weights=f.output=nullptr;f.inverse_rows=nullptr;
    assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_FINISH_SEPARATE,&other)==0 && !other.stages && !other.workspace_bytes);
    f.tokens=INT_MAX;assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_AUTO,&other)==QZR_OVERFLOW);
    assert(quactlize_ppu_moe_finish_select_v1(nullptr,QZM_AUTO,&other)==QZR_INVALID);
    assert(quactlize_ppu_moe_finish_select_v1(&f,QZM_AUTO,nullptr)==QZR_INVALID);
    puts("MOE_SCAFFOLD_HOST PASS");
}
