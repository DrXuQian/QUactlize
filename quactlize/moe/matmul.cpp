#include "matmul_validation.hpp"
#include "selection.hpp"
#include "../dispatch/selection.hpp"
#include "../runtime/library.hpp"
#include "../prefill/scatter.hpp"
#include "../dequant/api.h"
#include <hggc_runtime_api.h>
#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

namespace quactlize::moe::matmul {
using library::require;
using library::dispatch_check;
using dense::reserve;
thread_local std::string error;
struct Context;
struct Plan {
    Context* owner=nullptr;
    quactlize_ppu_moe_matmul_params_v1 params{};
    quactlize_ppu_moe_matmul_plan_info_v1 info{1,sizeof(info)};
    uint8_t const *low=nullptr,*high=nullptr,*units=nullptr;
    int kind=QKS_SMALLM_TC,dequant=-1,metadata=QK_METADATA_BF16;
    qks_smallm_choice_v2 small{};
    qks_choice_v1 tc{};
    qks_request_v1 request{};
    qzm_prepare_plan_v2 preparation{};
    void* handle=nullptr;
    uint64_t src=0,dst=0,offsets=0,inverse=0,counts=0,active=0,count=0;
    uint64_t route=0,input=0,output=0,status=0,kernel=0,kernel_bytes=0;
    uint64_t scale=0,zero=0,plane_bytes=0;
    ~Plan() { if(handle) quactlize_kpack_dispatch_destroy_v1(handle); }
};
struct Context {
    int device=0;
    std::filesystem::path root;
    std::string sdk;
    void* dispatcher=nullptr;
    bool jit=false;
    std::mutex mutex;
    std::vector<std::unique_ptr<Plan>> plans;
    ~Context() { plans.clear();if(dispatcher) quactlize_kpack_dispatch_close_v1(dispatcher); }
    void enable_jit() {
        if(!jit) { library::enable_jit(dispatcher,root,sdk);jit=true; }
    }
};
void device_check(int device) {
    int current=-1;
    require(hggcGetDevice(&current)==hggcSuccess && current==device,"MoE context/current device mismatch");
}
void outside_capture() {
    hggcStreamCaptureStatus capture;
    require(hggcStreamIsCapturing(nullptr,&capture)==hggcSuccess && capture==hggcStreamCaptureStatusNone,
            "MoE create/prepare during capture");
}
qkg_simt_call_v2 indexed(Plan const& p) {
    auto const& a=p.params;
    qkg_call_v1 c{1,sizeof(c)};
    c.qtype=a.qtype;c.n=a.n;c.k=a.k;c.experts=a.experts;c.rows=a.tokens*a.topk;
    c.mode=QKG_INDEXED;c.input_type=QKG_F32;c.channels=a.channels;c.topk=a.topk;
    c.a_row_stride=a.input_row_stride;c.a_token_stride=a.input_token_stride;
    c.ids_stride=a.ids_stride;c.out_row_stride=a.output_stride;
    c.low=p.low;c.high=p.high;c.units=p.units;
    return {2,sizeof(qkg_simt_call_v2),c,QK_COMPUTE_BF16};
}
qzm_prepare_v1 prepare_call(Plan const& p) {
    auto const& a=p.params;
    qzm_prepare_v1 c{1,sizeof(c)};
    c.route.version=1;c.route.size=sizeof(c.route);c.route.tokens=a.tokens;
    c.route.experts=a.experts;c.route.topk=a.topk;c.route.channels=a.channels;
    c.route.ids_stride=a.ids_stride;c.route.source_stride=a.input_token_stride/a.input_row_stride;
    c.config={1,sizeof(c.config),QZR_AUTO,0};c.columns=a.k;c.dtype=QZM_BF16;
    c.input_row_stride=a.input_row_stride;
    return c;
}
void select(Plan& p) {
    auto& ctx=*p.owner;auto const& a=p.params;
    p.request={1,sizeof(p.request),a.qtype,QK_GROUPED_SF,a.tokens*a.topk,a.n,a.k,a.experts,a.tokens,a.arrangement.mapping_id};
    if(a.tokens<=8) {
        auto c=indexed(p);
        auto decision=dispatch::select_smallm(c,a.arrangement);
        if(decision.status==QKS_OK && decision.choice.base.kind==QKS_SMALLM_TC) ctx.enable_jit();
        int rc=quactlize_kpack_dispatch_query_smallm_v3(ctx.dispatcher,&c,&a.arrangement,&p.small);
        require(rc==QKS_OK || rc==QKS_MISS,quactlize_kpack_dispatch_error_v1());
        if(rc==QKS_MISS) {
            // Retain the existing explicit all-format indexed SIMT coverage.
            // This floor is labelled a proposal, not a measured model winner.
            p.small.base.policy=QKS_COMPUTE_INITIAL;
            p.small.base.kind=QKS_SMALLM_SIMT;
            if(a.qtype==12) {
                rc=quactlize_kpack_q4_decode_select_v2(&c,&a.arrangement,&p.small.q4,&p.small.base.sizes);
                if(rc==QKG_OK) p.small.base.kind=QKS_SMALLM_Q4;
                else require(rc==QKG_SHAPE,"MoE Q4 selection failed");
            }
            if(p.small.base.kind==QKS_SMALLM_SIMT) {
                p.small.base.simt={1,sizeof(p.small.base.simt),0,4,4,4,1};
                rc=quactlize_kpack_simt_query_v2(&c,&p.small.base.simt,&a.arrangement,&p.small.base.sizes);
            }
            require(rc==QKG_OK,"MoE indexed SIMT coverage unavailable");
        }
        p.kind=p.small.base.kind;
        p.info.source_tokens=p.small.base.source_tokens;
        if(p.kind!=QKS_SMALLM_TC) {
            p.kernel_bytes=p.small.base.sizes.workspace_bytes;
            p.info.route=QUACTLIZE_MOE_SIMT;p.info.policy=p.small.base.policy;
            if(p.kind==QKS_SMALLM_Q4) {
                auto const& f=p.small.q4;p.info.split=1;
                std::snprintf(p.info.config,sizeof(p.info.config),"q4:r%d:v%d:w%d:x%d:c%d",f.reader,f.variant,f.warps,f.values,f.columns);
            } else {
                auto const& f=p.small.base.simt;p.info.split=f.split;
                std::snprintf(p.info.config,sizeof(p.info.config),"simt:v%d:c%d:w%d:x%d:s%d",f.variant,f.columns,f.warps,f.values,f.split);
            }
            return;
        }
        p.tc=p.small.base.tc;p.dequant=decision.dequant_config;
    } else {
        ctx.enable_jit();
        qks_prefill_choice_v2 proposal{};
        // Latest ordinary-MoE evidence is authoritative. Never let the dense
        // prefill table or older FP16/Full timings replace a confirmed winner.
        int rc=dispatch::moe_prefill::query(p.request,3,QK_COMPUTE_BF16,a.channels,proposal);
        if(rc==QKS_OK) {
            p.request.route=proposal.base.route+2;p.dequant=proposal.base.dequant_config;
            p.info.source_tokens=proposal.base.measured_tokens;
            dispatch_check(quactlize_kpack_dispatch_query_prefill_v1(ctx.dispatcher,&p.request,
                QK_COMPUTE_BF16,a.channels,&p.tc));
        } else {
            require(rc==QKS_MISS,"MoE prefill policy unavailable");
            // Out-of-family coverage is an explicit SF proposal. Resource
            // queries validate the selected same-format parent on this device.
            // No large-M legacy FQ fallback and no online timing/readback.
            dispatch_check(quactlize_kpack_dispatch_query_compute_v1(ctx.dispatcher,&p.request,
                QK_COMPUTE_BF16,0,0,&p.tc));
        }
    }
    int compute=-1;
    dispatch_check(quactlize_kpack_dispatch_grouped_contract_v1(ctx.dispatcher,&p.tc,&p.request,&compute,&p.metadata));
    require(compute==QK_COMPUTE_BF16,"MoE selected non-BF16 compute");
    p.info.route=p.request.route==QK_GROUPED_FQ?QUACTLIZE_MOE_FQ_TC:QUACTLIZE_MOE_SF_TC;
    p.info.policy=p.tc.policy;p.info.split=p.tc.split;p.kernel_bytes=p.tc.workspace_bytes;
    if(p.info.policy==QKS_MOE_PREFILL_CONFIRMED && (a.input_row_stride!=a.k ||
        a.input_token_stride!=int64_t(a.k)*a.channels || a.ids_stride!=a.topk || a.output_stride!=a.n))
        p.info.policy=QKS_MOE_PREFILL_BUCKET;
    std::snprintf(p.info.config,sizeof(p.info.config),"%s",p.tc.parent);
    if(p.request.route==QK_GROUPED_SF && a.qtype!=8)
        p.plane_bytes=uint64_t(a.n)*a.k*a.experts/a.arrangement.group_size*2;
}
void prepare(Plan& p) {
    select(p);p.info.compute_type=QK_COMPUTE_BF16;
    uint64_t cursor=0;auto const& a=p.params;uint64_t rows=uint64_t(a.tokens)*a.topk;
    if(p.kind==QKS_SMALLM_TC) {
        auto c=prepare_call(p);
        require(quactlize_ppu_moe_prepare_select_v2(&c,&p.preparation)==QZR_OK,"MoE prepare selection failed");
        p.info.prepare_algorithm=p.preparation.config.algorithm;p.info.prepare_helper=p.preparation.helper.implementation;
        p.src=reserve(cursor,rows*4);p.dst=reserve(cursor,rows*4);p.offsets=reserve(cursor,uint64_t(a.experts+1)*4);
        p.inverse=reserve(cursor,rows*4);p.counts=reserve(cursor,uint64_t(a.experts)*4);
        p.active=reserve(cursor,uint64_t(a.experts)*4);p.count=reserve(cursor,4);
        p.route=reserve(cursor,p.preparation.helper.workspace_bytes);
        p.input=reserve(cursor,rows*a.k*2);p.output=reserve(cursor,rows*a.n*2);p.status=reserve(cursor,4);
        if(p.plane_bytes) {p.scale=reserve(cursor,p.plane_bytes);p.zero=reserve(cursor,p.plane_bytes);}
        dispatch_check(quactlize_kpack_dispatch_prepare_grouped_dynamic_v1(p.owner->dispatcher,&p.tc,&p.handle));
    }
    p.kernel=reserve(cursor,p.kernel_bytes);p.info.workspace_bytes=cursor;
}
void run(Plan const& p,float const* input,int32_t const* ids,float* output,void* workspace,void* stream) {
    auto const& a=p.params;auto base=static_cast<uint8_t*>(workspace);
    if(p.kind!=QKS_SMALLM_TC) {
        auto c=indexed(p);c.call.a=input;c.call.ids=ids;c.call.output=output;c.call.stream=stream;
        c.call.workspace=p.kernel_bytes?base+p.kernel:nullptr;c.call.workspace_bytes=p.kernel_bytes;
        int rc=p.kind==QKS_SMALLM_Q4?quactlize_kpack_q4_decode_run_v2(&c,&p.small.q4,&a.arrangement):
            quactlize_kpack_simt_run_v2(&c,&p.small.base.simt,&a.arrangement);
        require(rc==QKG_OK,"MoE indexed SIMT launch failed");return;
    }
    auto ptr=[&](uint64_t offset) { return reinterpret_cast<int32_t*>(base+offset); };
    auto c=prepare_call(p);c.input=input;c.output=base+p.input;
    c.route.ids=ids;c.route.src_rows=ptr(p.src);c.route.dst_rows=ptr(p.dst);c.route.offsets=ptr(p.offsets);
    c.route.workspace=p.preparation.helper.workspace_bytes?base+p.route:nullptr;
    c.route.workspace_bytes=p.preparation.helper.workspace_bytes;c.route.stream=stream;
    c.inverse_rows=ptr(p.inverse);c.expert_rows=ptr(p.counts);c.active_experts=ptr(p.active);c.active_count=ptr(p.count);
    require(quactlize_ppu_moe_prepare_execute_v2(&c,&p.preparation)==QZR_OK,"MoE routing/gather failed");
    void const* metadata=p.units;void const* zero=nullptr;
    if(p.plane_bytes) {
        metadata=base+p.scale;zero=base+p.zero;int rc;
        if(p.dequant>=0) {
            qzd_call_v1 w{1,sizeof(w)};w.qtype=a.qtype;w.n=a.n;w.k=a.k;w.experts=a.experts;w.config=p.dequant;
            w.low=p.low;w.high=p.high;w.units=p.units;w.low_bytes=a.sizes.low_bytes;
            w.high_bytes=a.sizes.high_bytes;w.unit_bytes=a.sizes.units_bytes;
            w.output=base+p.scale;w.zero=base+p.zero;w.output_bytes=p.plane_bytes;w.stream=stream;
            qzd_call_v2 d{2,sizeof(d),w,p.metadata};rc=quactlize_kpack_dequant_v2(&d,&a.arrangement);
        } else rc=quactlize_kpack_sf_prepare_v2(a.qtype,a.n,a.k,a.experts,p.units,a.sizes.units_bytes,
            reinterpret_cast<uint16_t*>(base+p.scale),reinterpret_cast<uint16_t*>(base+p.zero),
            p.plane_bytes,&a.arrangement,p.metadata,stream);
        require(rc==QKG_OK,"MoE SF metadata expansion failed");
    }
    qk_call_v1 k{1,sizeof(k)};k.m=a.tokens*a.topk;k.n=a.n;k.k=a.k;k.experts=a.experts;
    k.group_size=a.arrangement.group_size;k.device=p.owner->device;k.compute_units=p.tc.compute_units;
    k.mapping_id=a.arrangement.mapping_id;k.a=base+p.input;k.low=p.low;k.high=p.high;k.metadata=metadata;k.zero=zero;
    k.output=base+p.output;k.offsets_device=ptr(p.offsets);k.workspace=base+p.kernel;
    k.workspace_bytes=p.kernel_bytes;k.stream=stream;
    qk_compute_device_call_v4 d{4,sizeof(d),{2,sizeof(qk_device_call_v2),k,a.tokens,0},QK_COMPUTE_BF16,p.metadata};
    dispatch_check(quactlize_kpack_dispatch_run_grouped_dynamic_v1(p.handle,&d,QK_GROUPED_COMPLETE));
    require(hggcMemsetAsync(ptr(p.status),0,4,static_cast<hggcStream_t>(stream))==hggcSuccess,"MoE output status reset failed");
    require(quactlize_prefill_scatter_bf16(base+p.output,output,ptr(p.dst),k.m,a.n,a.output_stride,ptr(p.status),stream)==QKG_OK,
            "MoE output scatter failed");
}
template<class F> int boundary(F fn) {
    error.clear();try {fn();return 0;} catch(std::exception const& e) {error=e.what();}
    catch(...) {error="unknown MoE runtime error";}return 1;
}
} // namespace quactlize::moe::matmul

extern "C" int quactlize_ppu_moe_matmul_supported_v1(quactlize_ppu_moe_matmul_params_v1 const* p) {
    quactlize::moe::matmul::error.clear();return quactlize::moe::matmul::admission(p);
}
extern "C" int quactlize_ppu_moe_matmul_create_v1(int32_t device,void** out) {
    using namespace quactlize::moe::matmul;if(out) *out=nullptr;
    return boundary([&] {
        require(out && device>=0,"invalid MoE context ABI");device_check(device);outside_capture();
        auto c=std::make_unique<Context>();c->device=device;
        c->root=quactlize::library::runtime_directory(reinterpret_cast<void*>(&quactlize_ppu_moe_matmul_create_v1));
        c->sdk=quactlize::library::environment("PPU_SDK","/usr/local/PPU_SDK");
        require(std::filesystem::path(c->sdk).is_absolute(),"PPU_SDK must be an absolute SDK root");
        dispatch_check(quactlize_kpack_dispatch_open_v1(c->root.c_str(),&c->dispatcher));*out=c.release();
    });
}
extern "C" int quactlize_ppu_moe_matmul_prepare_v1(void* context,quactlize_ppu_moe_matmul_params_v1 const* a,
    void const* low,void const* high,void const* units,void** out,quactlize_ppu_moe_matmul_plan_info_v1* info) {
    using namespace quactlize::moe::matmul;if(out) *out=nullptr;
    return boundary([&] {
        require(context && out && info && info->version==1 && info->size==sizeof(*info) &&
            admission(a)==1 && a->tokens>0,"invalid MoE prepare ABI/shape");
        require(low && units && !(uintptr_t(low)%16) && !(uintptr_t(units)%16) &&
            (a->sizes.high_bytes?(high && !(uintptr_t(high)%16)):!high),"invalid MoE weight planes");
        auto& c=*static_cast<Context*>(context);device_check(c.device);outside_capture();
        auto p=std::make_unique<quactlize::moe::matmul::Plan>();p->owner=&c;p->params=*a;
        p->low=static_cast<uint8_t const*>(low);p->high=static_cast<uint8_t const*>(high);p->units=static_cast<uint8_t const*>(units);
        std::lock_guard<std::mutex> lock(c.mutex);prepare(*p);
        auto ptr=p.get();c.plans.push_back(std::move(p));*info=ptr->info;*out=ptr;
    });
}
extern "C" int quactlize_ppu_moe_matmul_run_v1(void* plan,float const* input,int32_t const* ids,float* output,
    void* workspace,uint64_t bytes,void* stream) {
    using namespace quactlize::moe::matmul;
    return boundary([&] {
        require(plan,"null MoE plan");auto const& p=*static_cast<quactlize::moe::matmul::Plan const*>(plan);device_check(p.owner->device);
        require(buffers(p.params,p.low,p.high,p.units,input,ids,output,workspace,bytes,p.info.workspace_bytes),
                "invalid MoE input/IDs/output/workspace extent, alignment or overlap");
        run(p,input,ids,output,workspace,stream);
    });
}
extern "C" void quactlize_ppu_moe_matmul_destroy_v1(void* context) { delete static_cast<quactlize::moe::matmul::Context*>(context); }
extern "C" char const* quactlize_ppu_moe_matmul_error_v1(void) { return quactlize::moe::matmul::error.c_str(); }
