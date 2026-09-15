// Host-only dispatch protocol test. No GPU arithmetic is simulated here.
#include "quactlize/dispatch/binding.cpp"
#include <cassert>
#include <cstdio>

static int typed_prepares=0, stages=0, simt_runs=0, mixed_stages=0;
static int expected_compute=QK_COMPUTE_BF16;
static int dense_prepare(qkd_dense_call_v2 const* d,qk_recipe_v1 const*,void** h) {
    assert(d->version==2 && d->size==sizeof(*d) && d->compute_type==expected_compute);
    assert(d->dense.input_type==QKD_F32 && d->dense.output_type==QKD_F32);
    ++typed_prepares; *h=reinterpret_cast<void*>(1); return QK_OK;
}
static int grouped_prepare(qk_compute_device_call_v3 const* d,qk_recipe_v1 const*,void** h) {
    assert(d->version==3 && d->size==sizeof(*d) && d->compute_type==expected_compute);
    assert(d->device_call.max_rows==129);
    ++typed_prepares; *h=reinterpret_cast<void*>(1); return QK_OK;
}
static int stage_compute(void*,qk_moe_plan_v2 const* d,int,void*) {
    assert(d->version==2 && d->size==sizeof(*d) && d->compute_type==expected_compute);
    ++stages; return QK_OK;
}
static int reuse_compute(qkg_simt_call_v2 const* d,qkg_simt_config_v1 const*,
                         quactlize_ppu_placed_arrangement_v2 const*) {
    assert(d->version==2 && d->size==sizeof(*d) && d->compute_type==expected_compute);
    ++simt_runs;return QKG_OK;
}
static int mixed_compute(qkg_moe_compute_v2 const* d,int,void*) {
    assert(d->version==2 && d->size==sizeof(*d) && d->compute_type==expected_compute);
    ++mixed_stages;return QKG_OK;
}

int main() {
    using namespace quactlize::dispatch;
    for (int q:{8,10,11,12,13,14}) for (int route:{0,1,2,3}) {
        if (q==8 && !(route&1)) continue;
        for (int m:{1,8,9,129,4096}) {
            uint64_t map=q==8 ? q8_kpack2::kMappingId : q==12 ? UINT64_C(0x51344b5034540001) : UINT64_C(0x514b504b54000001);
            qks_request_v1 r{1,sizeof(r),q,route,m,768,1536,route>=2?256:1,m,map};
            assert(valid(r));
            std::string name;
            auto c=compute_proposal(r,{},name);
            assert(c.qtype==q && c.route==route && c.ap==0 && c.mapping_id==map && c.split==1);
            assert(name==c.symbol && c.tm==16 && c.tk%((q==11 || q==13) ? 256 : (q==10 || q==14) ? 128 : 64)==0);
            assert(key(r,0,0,0)!=key(r,0,0,1));
        }
    }
    Runtime runtime;runtime.device=0;runtime.cu=72;
    auto module=std::make_shared<Module>();module->compute_type=QK_COMPUTE_BF16;
    module->destroy=[](void*){};
    module->prepare_dense_compute=dense_prepare;
    module->prepare_compute=grouped_prepare;
    module->moe_stage_compute=stage_compute;
    for (bool grouped:{false,true}) {
        qks_request_v1 req{1,sizeof(req),14,grouped?2:0,grouped?528:1,5120,25600,
            grouped?256:1,grouped?129:1,UINT64_C(0x514b504b54000001)};
        qks_choice_v1 choice{};choice.version=1;choice.size=sizeof(choice);choice.ticket=runtime.plans.size()+1;
        qk_recipe_v1 rec{1,sizeof(rec),QK_ORDINARY,1,0};
        runtime.plans.push_back({req,choice,rec,module,grouped?0:QKD_F32,QK_COMPUTE_BF16});
        auto call=call_for(req,runtime);
        void* handle=nullptr;
        if (grouped) {
            assert(quactlize_kpack_dispatch_prepare_v1(&runtime,&choice,&call,&handle)==QKS_INVALID && !handle);
            qk_compute_device_call_v3 typed{3,sizeof(typed),{2,sizeof(qk_device_call_v2),call,129,0},QK_COMPUTE_BF16};
            assert(quactlize_kpack_dispatch_prepare_compute_v1(&runtime,&choice,&typed,&handle)==QKS_OK);
            quactlize_kpack_dispatch_destroy_v1(handle);
            for (int fault=0;fault<4;++fault) {
                auto bad=typed;
                if (fault==0) bad.compute_type=QK_COMPUTE_F16;
                if (fault==1) bad.device_call.max_rows=128;
                if (fault==2) bad.device_call.reserved=1;
                if (fault==3) bad.version=2;
                assert(quactlize_kpack_dispatch_prepare_compute_v1(&runtime,&choice,&bad,&handle)==QKS_INVALID && !handle);
            }
        } else {
            qkd_dense_call_v1 io{1,sizeof(io),call,QKD_F32,QKD_F32};
            assert(quactlize_kpack_dispatch_prepare_dense_io_v1(&runtime,&choice,&io,&handle)==QKS_INVALID && !handle);
            qkd_dense_call_v2 typed{2,sizeof(typed),io,QK_COMPUTE_BF16};
            assert(quactlize_kpack_dispatch_prepare_dense_io_v2(&runtime,&choice,&typed,&handle)==QKS_OK);
            quactlize_kpack_dispatch_destroy_v1(handle);
            typed.compute_type=QK_COMPUTE_F16;
            assert(quactlize_kpack_dispatch_prepare_dense_io_v2(&runtime,&choice,&typed,&handle)==QKS_INVALID && !handle);
        }
    }
    assert(typed_prepares==2);
    Handle gate,up,down;gate.module=up.module=down.module=module;
    MoeChain chain;chain.gate=&gate;chain.up=&up;chain.down=&down;chain.compute_type=QK_COMPUTE_BF16;
    assert(quactlize_kpack_dispatch_moe_run_v1(&chain,nullptr)==QKS_OK && stages==6);
    chain.execution=std::make_shared<MoeExecution>();
    chain.execution->reuse_compute=reuse_compute;chain.execution->stage_compute=mixed_compute;
    chain.simt_mask=chain.reuse_mask=3;
    assert(quactlize_kpack_dispatch_moe_run_v1(&chain,nullptr)==QKS_OK);
    assert(stages==8 && simt_runs==2 && mixed_stages==2);
    qks_moe_endpoint_v4 ep{4,sizeof(ep),{3,sizeof(qks_moe_endpoint_v3),&gate},QK_COMPUTE_BF16};
    auto bad=ep;bad.compute_type=QK_COMPUTE_F16;bad.endpoint.tc_handle=&down;
    void* out=nullptr;
    assert(quactlize_kpack_dispatch_moe_create_v4(&runtime,&ep,nullptr,&bad,&out)==QKS_INVALID && !out);
    qks_moe_endpoint_v4 legacy=ep;legacy.endpoint.tc_handle=nullptr;
    qkg_call_v1 call{};qkg_q4_decode_config_v1 old_reader{};
    legacy.endpoint.simt_call=&call;legacy.endpoint.q4_config=&old_reader;
    quactlize_ppu_placed_arrangement_v2 arrangement{};legacy.endpoint.arrangement=&arrangement;
    ep.endpoint.tc_handle=&down;
    assert(quactlize_kpack_dispatch_moe_create_v4(&runtime,&legacy,nullptr,&ep,&out)==QKS_MISS && !out);
    std::puts("KPACK_COMPUTE_DISPATCH PASS six formats, separate tickets, typed prepare, TC/mixed stages; wrong-type/version/legacy-reader RED");
}
