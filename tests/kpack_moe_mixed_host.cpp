#include "quactlize/dispatch/binding.cpp"
#include <cassert>
#include <cstdio>

static std::vector<int> calls;
static qk_moe_projection_v1 part(uintptr_t base,int n,int k,int channels,int tokens=1) {
    qk_moe_projection_v1 p{};p.version=1;p.size=sizeof(p);p.shape_size=12;p.stride_size=8;
    p.shape_offsets[0]=8;p.shape_offsets[1]=4;p.m=8*tokens;p.n=n;p.k=k;p.experts=256;p.tile_m=8;p.splits=1;
    p.a=(void*)base;p.output=(void*)(base+0x100000);p.workspace=(void*)(base+0x200000);p.workspace_bytes=0x100000;
    p.offsets=(int*)(base+0x300000);p.io.row_ids=(int*)(base+0x310000);p.rows=(int*)(base+0x203000);
    p.shapes=(void*)(base+0x200000);p.outputs=(void*)(base+0x201000);p.strides=(void*)(base+0x202000);
    p.directory_header=(void*)(base+0x204000);p.directory_entries=(void*)(base+0x205000);p.directory_capacity=p.m;
    p.io={1,sizeof(p.io),tokens,8,channels,0,8,k,int64_t(channels)*k,n,(int*)0x01000000,
        (float*)0x02000000,(float*)(base+0x400000),(int*)(base+0x310000)};
    return p;
}
static int project(void* h,qk_moe_projection_v1* out) { *out=*static_cast<qk_moe_projection_v1*>(h);return 0; }
static int stage_tc(void* h,qk_moe_plan_v1 const*,int phase,void*) {
    auto p=static_cast<qk_moe_projection_v1*>(h);calls.push_back(10+int(uintptr_t(p->a)/0x10000000)+phase*10);return 0;
}
static int bind_simt(qkg_call_v1 const* c,int,void* scratch,uint64_t,qk_moe_projection_v1* p) {
    *p=part(uintptr_t(scratch),c->n,c->k,c->channels,c->rows/8);
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

static int query_reuse(qkg_call_v1 const* c,qkg_simt_config_v1 const* f,
    quactlize_ppu_placed_arrangement_v2 const* a,qkg_sizes_v1* sizes) {
    return quactlize::execution::simt::query(*c,*f,a,*sizes);
}
static int run_reuse(qkg_call_v1 const* c,qkg_simt_config_v1 const* f,
    quactlize_ppu_placed_arrangement_v2 const* a) {
    qkg_sizes_v1 sizes{};
    assert(query_reuse(c,f,a,&sizes)==QKG_OK);
    assert(quactlize::execution::simt::buffers(*c,sizes)==QKG_OK);
    int i=int(uintptr_t(c->low)/0x100000000ULL)-1;
    assert(i>=0 && i<3 && c->rows>=8 && c->rows<=64);
    uintptr_t base=uintptr_t(i+1)*0x10000000;
    if (i==2) {
        assert(c->a==(void*)base && c->a_token_stride==8*c->k);
        assert(c->output==(float*)(base+0x400000));
    } else {
        assert(c->a==(void*)0x02000000 && c->output==(float*)(base+0x100000));
    }
    calls.push_back(50+i);return QKG_OK;
}

static int project_bf16(void* h,qk_moe_projection_v2* out) {
    *out={2,sizeof(*out),*static_cast<qk_moe_projection_v1*>(h),QK_COMPUTE_BF16};return QK_OK;
}
static int stage_tc_bf16(void* h,qk_moe_plan_v2 const* p,int phase,void* stream) {
    assert(p->version==2 && p->size==sizeof(*p) && p->compute_type==QK_COMPUTE_BF16);
    return stage_tc(h,&p->plan,phase,stream);
}
static int select_q4_bf16(qkg_simt_call_v2 const* c,quactlize_ppu_placed_arrangement_v2 const* a,
    qkg_q4_decode_config_v1* f,qkg_sizes_v1* s) {
    assert(c->version==2 && c->size==sizeof(*c) && c->compute_type==QK_COMPUTE_BF16);
    return select_simt(&c->call,a,f,s);
}
static int run_q4_bf16(qkg_simt_call_v2 const* c,qkg_q4_decode_config_v1 const* f,
    quactlize_ppu_placed_arrangement_v2 const* a) {
    assert(c->version==2 && c->size==sizeof(*c) && c->compute_type==QK_COMPUTE_BF16);
    return run_simt(&c->call,f,a);
}
static int stage_mixed_bf16(qkg_moe_compute_v2 const* p,int phase,void* stream) {
    assert(p->version==2 && p->size==sizeof(*p) && p->compute_type==QK_COMPUTE_BF16);
    return stage_simt(&p->plan,p->simt_mask,phase,stream);
}
static int finish_bf16(qkg_moe_compute_v2 const* p,qk_llama_moe_finish_v1 const* f,void* stream) {
    assert(p->version==2 && p->size==sizeof(*p) && p->compute_type==QK_COMPUTE_BF16);
    return weighted_finish(&p->plan,p->simt_mask,f,stream);
}

static void test_q4_bf16_chains() {
    int checked=0;
    for(int tokens=1;tokens<=8;++tokens) for(bool merged:{false,true})
    for(unsigned mask=0;mask<8;++mask) {
        if(merged && (mask&2)) continue;
        Runtime r;r.device=0;r.cu=72;r.moe=std::make_shared<MoeExecution>();
        // Deliberately no generic or F16 producer: none is required by this chain.
        r.moe->bind=bind_simt;r.moe->select_compute=select_q4_bf16;r.moe->q4_compute=run_q4_bf16;
        r.moe->stage_compute=stage_mixed_bf16;r.moe->finish_compute=finish_bf16;
        qkg_q4_decode_config_v1 cfg{1,sizeof(cfg),0,0,4,1,4};
        qk_moe_projection_v1 ps[3]={part(0x10000000,merged?1024:512,2048,1,tokens),
            part(0x20000000,512,2048,1,tokens),part(0x30000000,2048,512,8,tokens)};
        Handle h[3];qks_moe_endpoint_v4 endpoints[3]{};qkg_call_v1 gc[3]{};
        quactlize_ppu_placed_arrangement_v2 arr{};
        for(int i=0;i<3;++i) {
            auto& t=endpoints[i];t.version=4;t.size=sizeof(t);t.compute_type=QK_COMPUTE_BF16;
            auto& e=t.endpoint;e.version=3;e.size=sizeof(e);
            auto& p=ps[i];h[i].module=std::make_shared<Module>();h[i].inner=&p;
            h[i].module->compute_type=QK_COMPUTE_BF16;
            h[i].module->moe_projection_compute=project_bf16;h[i].module->moe_stage_compute=stage_tc_bf16;
            h[i].module->destroy=[](void*){};
            if(!(mask&(1u<<i))) {e.tc_handle=&h[i];continue;}
            auto& c=gc[i];c.version=1;c.size=sizeof(c);c.qtype=12;c.n=p.n;c.k=p.k;
            c.experts=256;c.rows=8*tokens;c.mode=QKG_INDEXED;c.input_type=QKG_F32;c.topk=8;c.channels=p.io.channels;
            c.a_row_stride=p.k;c.a_token_stride=p.io.a_token_stride;c.ids_stride=8;c.out_row_stride=p.n;
            c.a=p.io.a;c.output=p.io.output;c.ids=p.io.ids;c.low=(uint8_t*)(uintptr_t(i)+1);
            e.simt_call=&c;e.q4_config=&cfg;e.arrangement=&arr;e.scratch=p.a;e.scratch_bytes=0x400000;
        }
        void* chain=nullptr;
        auto create=[&] {return quactlize_kpack_dispatch_moe_create_v4(&r,&endpoints[0],
            merged?nullptr:&endpoints[1],&endpoints[2],&chain);};
        assert(create()==QKS_OK && chain);
        std::vector<int> expected{mask?100:11,mask&1?50:21};
        if(!merged) expected.push_back(mask&2?51:22);
        expected.push_back(mask?102:31);expected.push_back(mask&4?52:23);
        if(!(mask&4)) expected.push_back(43);
        calls.clear();assert(quactlize_kpack_dispatch_moe_run_v1(chain,nullptr)==QKS_OK && calls==expected);
        qk_llama_router_v1 router{1,sizeof(router),0,1,0,0,1e-8f,1.f,(float*)0x03000000,nullptr,(float*)0x04000000};
        qk_llama_moe_finish_v1 finish{1,sizeof(finish),8,2048,(float*)0x04000000,(float*)0x50000000};
        assert(quactlize_kpack_dispatch_moe_bind_finish_v1(&r,chain,&finish)==QKS_OK);
        if(!(mask&4)) expected.pop_back();expected.push_back(900);
        calls.clear();assert(quactlize_kpack_dispatch_moe_run_router_v1(chain,&router,nullptr)==QKS_OK && calls==expected);
        auto snapshot=router;snapshot.logits=router.weights-8;
        calls.clear();
        assert(quactlize_kpack_dispatch_moe_run_router_v1(chain,&snapshot,nullptr)==(tokens==1?QKS_OK:QKS_MISS));
        assert(calls==(tokens==1?expected:std::vector<int>{}));
        calls.clear();assert(quactlize_kpack_dispatch_moe_run_router_v1(chain,&router,nullptr)==QKS_OK && calls==expected);
        quactlize_kpack_dispatch_moe_destroy_v1(chain);chain=nullptr;
        auto saved=endpoints[2].compute_type;endpoints[2].compute_type=QK_COMPUTE_F16;
        assert(create()==QKS_INVALID && !chain);endpoints[2].compute_type=saved;
        if(mask) {
            r.moe->select_compute=nullptr;assert(create()==QKS_MISS && !chain);r.moe->select_compute=select_q4_bf16;
            r.moe->q4_compute=nullptr;assert(create()==QKS_MISS && !chain);r.moe->q4_compute=run_q4_bf16;
            r.moe->stage_compute=nullptr;assert(create()==QKS_MISS && !chain);r.moe->stage_compute=stage_mixed_bf16;
            int first=mask&1?0:mask&2?1:2;
            auto bad=cfg;++bad.warps;endpoints[first].endpoint.q4_config=&bad;
            assert(create()==QKS_INVALID && !chain);endpoints[first].endpoint.q4_config=&cfg;
        }
        ++checked;
    }
    std::printf("KPACK_MOE_BF16_Q4_HOST PASS chains=%d typed-only all masks tokens=1..8 router finish missing-entry negatives\n",checked);
}

static void test_reuse_chains() {
    int checked=0;
    for (int q:{8,10,11,12,13,14}) for (int tokens=1;tokens<=8;++tokens)
    for (int split:{1,2,4,8}) for (bool merged:{false,true}) for (unsigned mask=1;mask<8;++mask) {
        if (merged && (mask&2)) continue;
        Runtime r;r.device=0;r.cu=72;r.moe=std::make_shared<MoeExecution>();
        r.moe->bind=bind_simt;r.moe->stage=stage_simt;r.moe->query_reuse=query_reuse;r.moe->reuse=run_reuse;
        qkg_simt_config_v1 config{1,sizeof(config),q==8?1:3,8,4,4,split};
        auto arrangement=q==8 ? q8_kpack2::arrangement() :
            q==12 ? ppu_arrangements::q4_kpack4_transpose_v1() :
                    ppu_arrangements::kquant_kpack_transpose_v1(q);
        qk_moe_projection_v1 ps[3]={part(0x10000000,merged?1024:512,2048,1,tokens),
            part(0x20000000,512,2048,1,tokens),part(0x30000000,2048,512,8,tokens)};
        Handle handles[3];qks_moe_endpoint_v3 endpoints[3]{};qkg_call_v1 gc[3]{};
        for (int i=0;i<3;++i) {
            auto& e=endpoints[i];e.version=3;e.size=sizeof(e);
            auto& p=ps[i];auto& h=handles[i];h.module=std::make_shared<Module>();h.inner=&p;
            h.module->moe_projection=project;h.module->moe_stage=stage_tc;h.module->destroy=[](void*){};
            if (!(mask&(1u<<i))) {e.tc_handle=&h;continue;}
            auto& c=gc[i];c.version=1;c.size=sizeof(c);c.qtype=q;c.n=p.n;c.k=p.k;
            c.experts=256;c.rows=8*tokens;c.mode=QKG_INDEXED;c.input_type=QKG_F32;c.topk=8;c.channels=p.io.channels;
            c.a_row_stride=p.k;c.a_token_stride=p.io.a_token_stride;c.ids_stride=8;c.out_row_stride=p.n;
            c.a=p.io.a;c.output=p.io.output;c.ids=p.io.ids;
            uintptr_t weight=0x100000000ULL*(i+1);
            c.low=(uint8_t*)weight;c.units=(uint8_t*)(weight+0x80000000);
            if (q==11 || q==13 || q==14) c.high=(uint8_t*)(weight+0x40000000);
            qkg_sizes_v1 sizes{};assert(query_reuse(&c,&config,&arrangement,&sizes)==QKG_OK);
            if (sizes.workspace_bytes) {
                c.workspace=(void*)(0x500000000ULL+i*0x1000000);c.workspace_bytes=sizes.workspace_bytes;
            }
            e.simt_call=&c;e.reuse_config=&config;e.arrangement=&arrangement;
            e.scratch=p.a;e.scratch_bytes=0x400000;
        }
        void* chain=nullptr;
        auto create=[&] {return quactlize_kpack_dispatch_moe_create_v3(&r,&endpoints[0],
            merged?nullptr:&endpoints[1],&endpoints[2],&chain);};
        assert(create()==QKS_OK && chain);
        std::vector<int> expected{100,mask&1?50:21};
        if (!merged) expected.push_back(mask&2?51:22);
        expected.push_back(102);expected.push_back(mask&4?52:23);
        if (!(mask&4)) expected.push_back(43);
        calls.clear();assert(quactlize_kpack_dispatch_moe_run_v1(chain,nullptr)==QKS_OK);assert(calls==expected);
        qk_llama_router_v1 router{1,sizeof(router),0,1,0,0,1e-8f,1.f,(float*)0x03000000,nullptr,(float*)0x04000000};
        calls.clear();assert(quactlize_kpack_dispatch_moe_run_router_v1(chain,&router,nullptr)==QKS_OK);assert(calls==expected);
        int first=(mask&1)?0:(mask&2)?1:2;
        if (split>1) {
            auto original=router.logits;router.logits=(float*)gc[first].workspace;
            calls.clear();assert(quactlize_kpack_dispatch_moe_run_router_v1(chain,&router,nullptr)==QKS_MISS);assert(calls.empty());
            router.logits=original;
            qk_llama_moe_finish_v1 finish{1,sizeof(finish),8,2048,(float*)gc[first].workspace,(float*)0x50000000};
            assert(quactlize_kpack_dispatch_moe_bind_finish_v1(&r,chain,&finish)==QKS_MISS);
        }
        quactlize_kpack_dispatch_moe_destroy_v1(chain);chain=nullptr;
        auto query=r.moe->query_reuse;r.moe->query_reuse=nullptr;
        assert(create()==QKS_MISS && !chain);r.moe->query_reuse=query;
        auto run=r.moe->reuse;r.moe->reuse=nullptr;
        assert(create()==QKS_MISS && !chain);r.moe->reuse=run;
        auto& e=endpoints[first];e.version=2;assert(create()==QKS_INVALID && !chain);e.version=3;
        qkg_config_v1 extra{1,sizeof(extra),16,4,1};e.simt_config=&extra;
        assert(create()==QKS_INVALID && !chain);e.simt_config=nullptr;
        if (split>1) {
            auto saved=gc[first].workspace;
            // A different projection's live scratch is not caught by the
            // standalone reader; composition must reject it explicitly.
            gc[first].workspace=ps[first==2?0:2].workspace;
            assert(create()==QKS_INVALID && !chain);gc[first].workspace=saved;
            --gc[first].workspace_bytes;assert(create()==QKS_INVALID && !chain);++gc[first].workspace_bytes;
        }
        ++checked;
    }
    std::printf("KPACK_MOE_REUSE_HOST PASS chains=%d formats=6 tokens=1..8 splits=1/2/4/8 missing-entry+alias negatives\n",checked);
}
int main() {
    test_q4_bf16_chains();
    test_reuse_chains();
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
