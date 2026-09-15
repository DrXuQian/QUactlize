#include "quactlize/dispatch/binding.cpp"
#include <cassert>
#include <cstdio>

static std::vector<int> calls;
static qk_moe_projection_v1 part(uintptr_t base,int n,int k,int channels) {
    qk_moe_projection_v1 p{};p.version=1;p.size=sizeof(p);p.shape_size=12;p.stride_size=8;
    p.shape_offsets[0]=8;p.shape_offsets[1]=4;p.m=8;p.n=n;p.k=k;p.experts=256;p.tile_m=8;p.splits=1;
    p.a=(void*)base;p.output=(void*)(base+0x100000);p.workspace=(void*)(base+0x200000);p.workspace_bytes=0x100000;
    p.offsets=(int*)(base+0x300000);p.io.row_ids=(int*)(base+0x310000);p.rows=(int*)(base+0x203000);
    p.shapes=(void*)(base+0x200000);p.outputs=(void*)(base+0x201000);p.strides=(void*)(base+0x202000);
    p.directory_header=(void*)(base+0x204000);p.directory_entries=(void*)(base+0x205000);p.directory_capacity=8;
    p.io={1,sizeof(p.io),1,8,channels,0,8,k,int64_t(channels)*k,n,(int*)0x01000000,
        (float*)0x02000000,(float*)(base+0x400000),(int*)(base+0x310000)};
    return p;
}
static int project(void* h,qk_moe_projection_v1* out) { *out=*static_cast<qk_moe_projection_v1*>(h);return 0; }
static int stage_tc(void* h,qk_moe_plan_v1 const*,int phase,void*) {
    auto p=static_cast<qk_moe_projection_v1*>(h);calls.push_back(10+int(uintptr_t(p->a)/0x10000000)+phase*10);return 0;
}
static int bind_simt(qkg_call_v1 const* c,int,void* scratch,uint64_t,qk_moe_projection_v1* p) {
    *p=part(uintptr_t(scratch),c->n,c->k,c->channels);
    p->io.a=static_cast<float const*>(c->a);p->io.output=c->output;p->io.ids=c->ids;return 0;
}
static int select_simt(qkg_call_v1 const*,quactlize_ppu_placed_arrangement_v2 const*,qkg_q4_decode_config_v1* f,qkg_sizes_v1*) {
    *f={1,sizeof(*f),0,0,4,1,4};return 0;
}
static int stage_simt(qk_moe_plan_v1 const*,uint32_t mask,int phase,void*) {
    assert(mask && (phase==QK_MOE_PREPARE || phase==QK_MOE_ACTIVATE));calls.push_back(100+phase);return 0;
}
static int weighted_finish(qk_moe_plan_v1 const*,uint32_t,qk_llama_moe_finish_v1 const* f,void*) {
    assert(f->output==(float*)0x50000000 && f->weights==(float*)0x04000000);
    calls.push_back(900);return 0;
}
static int run_simt(qkg_call_v1 const* c,qkg_q4_decode_config_v1 const*,quactlize_ppu_placed_arrangement_v2 const*) {
    int i=int(uintptr_t(c->low))-1;
    uintptr_t base=uintptr_t(i+1)*0x10000000;
    assert(c->input_type==QKG_F32 && c->mode==QKG_INDEXED && c->ids==(int*)0x01000000);
    if (i==2) {
        assert(c->a==(void*)base && c->a_row_stride==c->k && c->a_token_stride==8*c->k);
        assert(c->output==(float*)(base+0x400000));
    } else {assert(c->a==(void*)0x02000000);assert(c->output==(float*)(base+0x100000));}
    calls.push_back(50+i);return 0;
}
int main() {
    for (bool merged:{false,true}) for (uint32_t mask=0;mask<8;++mask) {
        if (merged && (mask&2)) continue;
        Runtime r;r.device=0;r.cu=72;r.moe=std::make_shared<MoeExecution>();
        r.moe->bind=bind_simt;r.moe->select=select_simt;r.moe->q4=run_simt;r.moe->stage=stage_simt;
        qkg_q4_decode_config_v1 cfg{1,sizeof(cfg),0,0,4,1,4};
        qk_moe_projection_v1 ps[3]={part(0x10000000,merged?1024:512,2048,1),
            part(0x20000000,512,2048,1),part(0x30000000,2048,512,8)};
        Handle h[3];qks_moe_endpoint_v2 endpoints[3]{};qkg_call_v1 gc[3]{};
        quactlize_ppu_placed_arrangement_v2 arr{};
        for (int i=0;i<3;++i) {
            auto& e=endpoints[i];e.version=2;e.size=sizeof(e);
            auto& p=ps[i];h[i].module=std::make_shared<Module>();h[i].inner=&p;
            h[i].module->moe_projection=project;h[i].module->moe_stage=stage_tc;
            h[i].module->destroy=[](void*){};
            if (!(mask&(1u<<i))) {e.tc_handle=&h[i];continue;}
            auto& c=gc[i];c.version=1;c.size=sizeof(c);c.qtype=12;c.n=p.n;c.k=p.k;
            c.experts=256;c.rows=8;c.mode=QKG_INDEXED;c.input_type=QKG_F32;c.topk=8;c.channels=p.io.channels;
            c.a_row_stride=p.k;c.a_token_stride=p.io.a_token_stride;c.ids_stride=8;c.out_row_stride=p.n;
            c.a=p.io.a;c.output=p.io.output;c.ids=p.io.ids;c.low=(uint8_t*)(uintptr_t(i)+1);
            e.simt_call=&c;e.q4_config=&cfg;e.arrangement=&arr;e.scratch=p.a;e.scratch_bytes=0x400000;
        }
        void* chain=nullptr;
        auto create=[&] {return quactlize_kpack_dispatch_moe_create_v2(&r,&endpoints[0],merged?nullptr:&endpoints[1],&endpoints[2],&chain);};
        assert(create()==0 && chain);
        std::vector<int> expected{mask?100:11};
        expected.push_back(mask&1?50:21);
        if (!merged) expected.push_back(mask&2?51:22);
        expected.push_back(mask?102:31);expected.push_back(mask&4?52:23);
        if (!(mask&4)) expected.push_back(43);
        calls.clear();assert(quactlize_kpack_dispatch_moe_run_v1(chain,nullptr)==0);assert(calls==expected);
        qk_llama_router_v1 router{1,sizeof(router),0,1,0,0,1e-8f,1.f,(float*)0x03000000,nullptr,(float*)0x04000000};
        calls.clear();assert(quactlize_kpack_dispatch_moe_run_router_v1(chain,&router,nullptr)==0);assert(calls==expected);
        qk_llama_moe_finish_v1 finish{1,sizeof(finish),8,2048,(float*)0x04000000,(float*)0x50000000};
        assert(quactlize_kpack_dispatch_moe_bind_finish_v1(&r,chain,&finish)==QKS_MISS);
        r.moe->finish=weighted_finish;
        auto bad_finish=finish;bad_finish.output=(float*)ps[2].workspace;
        assert(quactlize_kpack_dispatch_moe_bind_finish_v1(&r,chain,&bad_finish)==QKS_MISS);
        assert(quactlize_kpack_dispatch_moe_bind_finish_v1(&r,chain,&finish)==QKS_OK);
        assert(quactlize_kpack_dispatch_moe_bind_finish_v1(&r,chain,&finish)==QKS_INVALID);
        if (!(mask&4)) expected.pop_back();
        expected.push_back(900);
        calls.clear();assert(quactlize_kpack_dispatch_moe_run_v1(chain,nullptr)==0);assert(calls==expected);
        calls.clear();assert(quactlize_kpack_dispatch_moe_run_router_v1(chain,&router,nullptr)==0);assert(calls==expected);
        router.weights=(float*)0x04100000;
        calls.clear();assert(quactlize_kpack_dispatch_moe_run_router_v1(chain,&router,nullptr)==QKS_MISS);assert(calls.empty());
        quactlize_kpack_dispatch_moe_destroy_v1(chain);chain=nullptr;
        if (!mask) continue;
        auto altered=cfg;altered.warps=8;int first=(mask&1)?0:(mask&2)?1:2;
        endpoints[first].q4_config=&altered;assert(create()==QKS_INVALID && !chain);endpoints[first].q4_config=&cfg;
        auto saved=endpoints[first].scratch;endpoints[first].scratch=(void*)0x02000000;
        assert(create()==QKS_MISS && !chain);endpoints[first].scratch=saved;
        if (mask&4) {gc[2].channels=1;assert(create()==QKS_MISS && !chain);}
    }
    std::puts("KPACK_MOE_MIXED_HOST PASS all masks, merged/separate, router, weighted finish, aliases and wrong recipes");
}
