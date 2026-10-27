// Real provider, dispatcher and selectors; fake DEVICE launches/resources.
// Tests ABI/composition/selection only. This is not a GPU numerical oracle.
#include "quactlize/dispatch/binding.cpp"
#include "quactlize/dense/runtime.cpp"
#include "quactlize/prefill/dynamic.hpp"
#include "quactlize/prefill/dense_helpers.hpp"
#include "quactlize/runtime/dynamic.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>

namespace test {
std::vector<std::string> launches;
int device=0,capture=0;
bool helper_device=true;
struct Handle {qk_call_v1 shape;bool typed;};
int resource(qk_call_v1 const* c,qk_recipe_v1 const* r,qk_resources_v1* out) {
    *out={1,sizeof(*out),r->split>1 ? uint64_t(c->m)*c->n*r->split*4 : 0,4096,6,0,0};return 0;
}
int prepare(qk_call_v1 const* c,qk_recipe_v1 const*,void** out) {
    assert(quactlize::runtime::address_free(*c));*out=new Handle{*c,false};return 0;
}
int run(void* handle,qk_call_v1 const* c) {
    assert(quactlize::runtime::same_dense_shape(static_cast<Handle*>(handle)->shape,*c));
    launches.push_back("tc");return 0;
}
void install(quactlize::dense::Context& ctx,quactlize::dispatch::TcDecision const& d,bool typed,int compute) {
    if (!d) return;
    auto c=d.config();auto& r=*static_cast<Runtime*>(ctx.dispatcher);
    for (auto const& i:kImages) if (i.parent==c.symbol && i.dense_io==typed && i.compute_type==compute) return;
    std::ostringstream os;os<<std::setfill('0')<<std::setw(64)<<std::hex<<(kImages.size()+1);
    Image image{c.symbol,os.str(),std::string(64,'a'),c.qtype,c.route,c.tm,c.tn,c.tk,c.wm,c.wn,c.stages,c.ap,c.dn,{},typed,compute};
    auto mod=std::make_shared<Module>();mod->qtype=c.qtype;mod->compute_type=compute;
    mod->query=resource;
    mod->query_device=[](auto d,auto r,auto out){return resource(&d->call,r,out);};
    mod->query_compute=[](auto d,auto r,auto out){
        assert(d->compute_type==QK_COMPUTE_BF16 && d->metadata_type==QK_METADATA_BF16);
        return resource(&d->device_call.call,r,out);
    };
    mod->query_dense_io=[](auto d,auto r,auto out){return resource(&d->call,r,out);};
    mod->query_dense_compute=[](auto d,auto r,auto out){return resource(&d->dense.call,r,out);};
    mod->query_dense_general=mod->query_dense_compute;
    mod->prepare_dynamic=prepare;mod->run_dynamic=run;
    mod->prepare_dense_dynamic=[](auto d,auto r,auto out) {
        int rc=prepare(&d->dense.call,r,out);static_cast<Handle*>(*out)->typed=true;return rc;
    };
    mod->prepare_dense_general=mod->prepare_dense_dynamic;
    mod->run_dense_dynamic=[](auto h,auto d){assert(static_cast<Handle*>(h)->typed);return run(h,&d->dense.call);};
    mod->destroy=[](void* h){delete static_cast<Handle*>(h);};
    r.modules.emplace(image.key,mod);kImages.push_back(image);
}

quactlize_ppu_dense_params_v1 params(int q,int m,int n=1024,int k=5120) {
    quactlize_ppu_dense_params_v1 p{1,sizeof(p)};p.qtype=q;p.m=m;p.n=n;p.k=k;p.input_stride=k;p.output_stride=n;
    assert(!quactlize_ppu_kpack_canonical_arrangement_v1(q,&p.arrangement));
    assert(!quactlize_ppu_kpack_sizes_for_arrangement_v1(n,k,1,q,&p.arrangement,&p.sizes));return p;
}
// Preparing a host-only test runtime bypasses filesystem JIT and installs
// resource fakes for the actual selected parent, not a different tactic.
std::unique_ptr<quactlize::dense::Context> context(int compute,bool reviewed=false) {
    auto owner=std::make_unique<quactlize::dense::Context>();auto& ctx=*owner;
    ctx.device=0;ctx.compute=compute;ctx.jit=true;
    // Frozen PoC comparisons exclude the new whole-call policy; the separate
    // reviewed_prefill() suite exercises EVERY new row through the public ABI.
    ctx.dense_device=reviewed?1:0;
    auto r=new Runtime;r->device=0;r->cu=72;ctx.dispatcher=r;
    r->moe=std::make_shared<MoeExecution>();r->moe->select=&quactlize_kpack_q4_decode_select_v1;
    r->moe->select_compute=&quactlize_kpack_q4_decode_select_v2;
    r->moe->q4_compute=&quactlize_kpack_q4_decode_run_v2;
    return owner;
}
void install_for(quactlize::dense::Context& ctx,quactlize_ppu_dense_params_v1 const& p) {
    using namespace quactlize::dispatch;
    quactlize::dense::Plan plan;plan.params=p;plan.compute=ctx.compute;
    auto typed=quactlize::dense::simt_call(plan);
    auto matched=select_smallm(typed,p.arrangement);
    auto r=quactlize::dense::request_for(p);
    auto measured=dense_prefill::select(r,ctx.compute);
    if(ctx.dense_device==1 && measured && measured.choice->route!=2) {
        auto const& c=measured.choice->tc;r.route=c.route;
        install(ctx,select_tc(r,false,ctx.compute,{&c,measured.policy}),false,ctx.compute);
    }
    auto confirmed=prefill::knot(r,ctx.compute,1);
    if(confirmed && confirmed->config>=0) {
        auto const& c=prefill::data::kConfigs[confirmed->config];r.route=c.route;
        install(ctx,select_tc(r,false,ctx.compute,{&c,QKS_PREFILL_COMPONENT_MEASURED}),false,ctx.compute);
    }
    if (matched.status==0 && matched.tc) {
        r.route=matched.tc->route;
        install(ctx,select_tc(r,true,ctx.compute,{matched.tc,matched.choice.base.policy}),true,ctx.compute);
    }
    auto legacy=smallm::select(typed.call);
    if (legacy.row) {
        auto const& c=smallm::data::kChoices[legacy.row->choice];
        if (!c.simt) {r.route=c.tc.route;install(ctx,select_tc(r,true,ctx.compute,{&c.tc,legacy.policy}),true,ctx.compute);}
    }
    for (int route=0;route<2;++route) {
        r.route=route;
        install(ctx,select_tc(r,true,ctx.compute),true,ctx.compute);
        auto general=select_dense_dynamic(r,ctx.compute);
        install(ctx,general,(p.m<=8 && general.policy!=QKS_LEGACY_DENSE) || ctx.compute!=0,ctx.compute);
    }
}
void* plan(quactlize::dense::Context& ctx,quactlize_ppu_dense_params_v1 const& p) {
    install_for(ctx,p);
    // Sentinel weight spans are never dereferenced by these DEVICE fakes.
    void* result=nullptr;quactlize_ppu_dense_plan_info_v1 info{1,sizeof(info)};
    auto rc=quactlize_ppu_dense_prepare_v1(&ctx,&p,reinterpret_cast<void*>(UINT64_C(0x10000000000)),
        p.sizes.high_bytes?reinterpret_cast<void*>(UINT64_C(0x20000000000)):nullptr,
        reinterpret_cast<void*>(UINT64_C(0x30000000000)),&result,&info);
    if (rc) std::fprintf(stderr,"prepare q=%d m=%d n=%d k=%d: %s\n",p.qtype,p.m,p.n,p.k,quactlize_ppu_dense_error_v1());
    assert(!rc && result);assert(info.version==1 && info.size==sizeof(info));return result;
}
}

extern "C" hggcError_t hggcGetDevice(int* out) {*out=test::device;return hggcSuccess;}
extern "C" hggcError_t hggcGetDeviceProperties(hggcDeviceProp* out,int) {
    *out={};std::strcpy(out->name,test::helper_device?"PPU-ZW810":"PPU-ZW610");
    out->multiProcessorCount=test::helper_device?72:64;out->warpSize=32;return hggcSuccess;
}
extern "C" hggcError_t hggcStreamIsCapturing(hggcStream_t,hggcStreamCaptureStatus* out) {
    *out=test::capture?hggcStreamCaptureStatusActive:hggcStreamCaptureStatusNone;return hggcSuccess;
}
extern "C" int quactlize_dense_copy_f32(float const*,float*,int,int,int64_t,int64_t,void*) {test::launches.push_back("copy-f32");return 0;}
extern "C" int quactlize_dense_convert_f16(int input,void const*,void*,int,int,int64_t,int linear,void*) {
    test::launches.push_back(linear?(input?"legacy-in":"legacy-out"):(input?"f16-in":"f16-out"));return 0;
}
extern "C" int quactlize_kpack_dense_helper_policy_v1(qkp_call_v1 const* c,int storage,int* dq,int* io) {
    *dq=*io=-1;
    if(!test::helper_device || c->a_stride!=c->weight.k || c->output_stride!=c->weight.n)return QKG_SHAPE;
    auto choice=quactlize::prefill::dense_helpers(c->weight.qtype,c->m,c->weight.n,c->weight.k,storage);
    *dq=choice.dequant;*io=choice.io;return *dq<0?QKG_SHAPE:QKG_OK;
}
extern "C" int quactlize_kpack_dense_full_helper_policy_v1(qkp_call_v1 const* c,int storage,int* dq,int* io) {
    *dq=*io=-1;
    if(!test::helper_device || c->a_stride!=c->weight.k || c->output_stride!=c->weight.n)return QKG_SHAPE;
    auto choice=quactlize::prefill::dense_full_helpers(c->weight.qtype,c->m,c->weight.n,c->weight.k,storage);
    *dq=choice.dequant;*io=choice.io;return *dq<0?QKG_SHAPE:QKG_OK;
}
extern "C" int quactlize_dense_io_v1(int storage,int input,int vector,void const*,void*,int,int,int64_t,int const*,void*) {
    assert(!storage);test::launches.push_back(std::string(input?"vector-in-":"vector-out-")+std::to_string(vector));return 0;
}
struct TypedFull {qkp_call_v1 shape;int storage;};
extern "C" int quactlize_kpack_dense_full_workspace_v1(qkp_call_v1 const* c,quactlize_ppu_placed_arrangement_v2 const* a,uint64_t* out) {
    quactlize::prefill::Layout l;int rc=quactlize::prefill::layout(*c,a,l,true);
    *out=quactlize::prefill::dynamic_bytes(*c,l);return rc;
}
extern "C" int quactlize_kpack_dense_full_prepare_v1(qkp_call_v1 const* c,quactlize_ppu_placed_arrangement_v2 const*,qkp_dense_options_v1 const* o,void** out) {
    assert(quactlize::prefill::address_free(*c) && o->math_mode==16 && o->algorithm==-1);
    if(o->storage==1) {
        auto choice=quactlize::prefill::dense_full_helpers(c->weight.qtype,c->m,c->weight.n,c->weight.k,1);
        assert(o->vector_io==choice.io && o->dequant_variant==choice.dequant);
    } else {
        auto p=test::params(c->weight.qtype,c->m,c->weight.n,c->weight.k);
        auto d=quactlize::dispatch::dense_prefill::select(quactlize::dense::request_for(p),0);
        assert(d && d.choice->route==2 && o->vector_io==d.choice->io && o->dequant_variant==d.choice->dequant);
    }
    *out=new TypedFull{*c,o->storage};return 0;
}
int quactlize::prefill::run_dense_full(void* h,qkp_call_v1 const* c,int stage,bool dequant_first) {
    auto& p=*static_cast<TypedFull*>(h);
    assert(stage==-1 && quactlize::prefill::same_shape(p.shape,*c));
    assert(!dequant_first || p.storage==0);
    test::launches.push_back(std::string(p.storage?"full-typed-bf16":"full-typed-f16")+
        (dequant_first?":dq-first":""));return 0;
}
extern "C" int quactlize_kpack_dense_full_run_v1(void* h,qkp_call_v1 const* c,int stage) {
    return quactlize::prefill::run_dense_full(h,c,stage,false);
}
extern "C" void quactlize_kpack_dense_full_destroy_v1(void* h){delete static_cast<TypedFull*>(h);}
extern "C" char const* quactlize_kpack_dense_full_error_v1(){return "fake typed provider";}
extern "C" int qkg_q4_decode_launch(qkg_call_v1 const&,qkg_q4_decode_config_v1 const&) {test::launches.push_back("q4");return 0;}
extern "C" int qkg_q4_decode_launch_bf16(qkg_call_v1 const&,qkg_q4_decode_config_v1 const&) {test::launches.push_back("q4-bf16");return 0;}
#define QSTUB(Q) \
extern "C" int qkg_simt_launch_##Q(qkg_call_v1 const*,qkg_simt_config_v1 const*) {test::launches.push_back("simt");return 0;} \
extern "C" int qkg_simt_launch_v2_##Q(qkg_simt_call_v2 const*,qkg_simt_config_v1 const*) {test::launches.push_back("simt");return 0;} \
extern "C" bool qkg_simt_supported_##Q(qkg_simt_config_v1 const*) {return true;}
QSTUB(8) QSTUB(10) QSTUB(11) QSTUB(12) QSTUB(13) QSTUB(14)
#undef QSTUB
extern "C" int quactlize_kpack_sf_prepare_v2(int,int,int,int,uint8_t const*,uint64_t,uint16_t*,uint16_t*,uint64_t,
    quactlize_ppu_placed_arrangement_v2 const*,int32_t metadata,void*) {
    assert(metadata==QK_METADATA_F16);test::launches.push_back("sf-default");return 0;
}
extern "C" int quactlize_kpack_dequant_v2(qzd_call_v2 const* c,quactlize_ppu_placed_arrangement_v2 const*) {
    assert(c->metadata_type==QK_METADATA_F16 && c->call.operation==0);
    test::launches.push_back("sf-"+std::to_string(c->call.config));return 0;
}
extern "C" int quactlize_kpack_prefill_query_dynamic_v1(qkp_call_v1 const* c,quactlize_ppu_placed_arrangement_v2 const* a,uint64_t* out) {
    quactlize::prefill::Layout l;int rc=quactlize::prefill::layout(*c,a,l);
    *out=quactlize::prefill::dynamic_bytes(*c,l);return rc;
}
extern "C" int quactlize_kpack_prefill_prepare_dynamic_v1(qkp_call_v1 const* c,quactlize_ppu_placed_arrangement_v2 const*,qkp_options_v1 const*,void** out) {
    assert(quactlize::prefill::address_free(*c));*out=new qkp_call_v1(*c);return 0;
}
extern "C" int quactlize_kpack_prefill_run_dynamic_v1(void* h,qkp_call_v1 const* c) {
    assert(quactlize::prefill::same_shape(*static_cast<qkp_call_v1*>(h),*c));test::launches.push_back("full");return 0;
}
extern "C" void quactlize_kpack_prefill_destroy_v1(void* h) {delete static_cast<qkp_call_v1*>(h);}
extern "C" char const* quactlize_kpack_prefill_error_v1() {return "fake provider";}

#include "dense_poc_selection.hpp"

void promoted_helpers() {
    for(int compute:{0,1})for(bool target:{false,true}) {
        kImages.clear();auto ctx=test::context(compute);test::helper_device=target;
        auto p=test::params(12,2048);void* raw=nullptr;
        quactlize_ppu_dense_plan_info_v1 info{1,sizeof(info)};
        quactlize_ppu_dense_candidate_v1 candidate{1,sizeof(candidate)};
        candidate.kind=3;candidate.route=QUACTLIZE_DENSE_FULL_BF16;candidate.dequant_config=12;
        assert(!quactlize_ppu_dense_prepare_candidate_v1(ctx.get(),&p,
            reinterpret_cast<void*>(UINT64_C(0x10000000000)),nullptr,reinterpret_cast<void*>(UINT64_C(0x30000000000)),
            &candidate,&raw,&info));
        auto& plan=*static_cast<quactlize::dense::Plan*>(raw);
        assert(plan.typed_full==target && info.compute_type==QUACTLIZE_DENSE_BF16);
        auto full=quactlize::prefill::dense_full_helpers(p.qtype,p.m,p.n,p.k,1);
        assert(plan.helper_io==(target?full.io:-1) && plan.helper_dequant==(target?full.dequant:-1));
        for(uintptr_t replay:{1,2}) {
            test::launches.clear();
            assert(!quactlize_ppu_dense_run_v1(raw,reinterpret_cast<float*>(UINT64_C(0x40000000000)+replay*256),
                reinterpret_cast<float*>(UINT64_C(0x50000000000)+replay*256),
                reinterpret_cast<void*>(UINT64_C(0x60000000000)+replay*256),info.workspace_bytes,reinterpret_cast<void*>(replay)));
            assert((test::launches==std::vector<std::string>{target?"full-typed-bf16":"full"}));
        }
    }
    for(bool target:{false,true}) {
        kImages.clear();test::helper_device=target;auto ctx=test::context(0);auto p=test::params(12,2048);
        auto raw=test::plan(*ctx,p);auto& plan=*static_cast<quactlize::dense::Plan*>(raw);
        assert(plan.kind==quactlize::dense::Kind::Tc && plan.native_f16);
        assert(plan.helper_io==(target?2:-1));test::launches.clear();
        assert(!quactlize_ppu_dense_run_v1(raw,reinterpret_cast<float*>(UINT64_C(0x40000000000)),
            reinterpret_cast<float*>(UINT64_C(0x50000000000)),reinterpret_cast<void*>(UINT64_C(0x60000000000)),
            plan.info.workspace_bytes,nullptr));
        assert(test::launches[test::launches.size()-3]==(target?"vector-in-2":"f16-in"));
        assert(test::launches.back()==(target?"vector-out-2":"f16-out"));
    }
    test::helper_device=true;
}

void explicit_candidates() {
    using namespace quactlize::dispatch;
    for (int compute:{0,1}) for (int q:{8,10,11,12,13,14}) for (int m:{1,128}) {
        kImages.clear();auto owner=test::context(compute);auto& ctx=*owner;
        auto p=test::params(q,m);auto request=quactlize::dense::request_for(p);
        auto d=select_dense_dynamic(request,compute);assert(d);
        bool native=compute==0 && m>8;
        test::install(ctx,d,!native,compute);auto const& c=d.value;
        quactlize_ppu_dense_candidate_v1 candidate{1,sizeof(candidate)};
        candidate.kind=0;candidate.route=request.route;candidate.native_f16=native;
        candidate.dequant_config=-1;candidate.tm=c.tm;candidate.tn=c.tn;candidate.tk=c.tk;
        candidate.wm=c.wm;candidate.wn=c.wn;candidate.stages=c.stages;candidate.ap=c.ap;
        candidate.delivery_n=c.dn;candidate.split=c.split;candidate.grid_mode=c.grid_mode;candidate.grid_b=c.grid_b;
        std::snprintf(candidate.parent,sizeof(candidate.parent),"%s",d.symbol.c_str());
        auto low=reinterpret_cast<void const*>(UINT64_C(0x10000000000));
        auto high=p.sizes.high_bytes?reinterpret_cast<void const*>(UINT64_C(0x20000000000)):nullptr;
        auto units=reinterpret_cast<void const*>(UINT64_C(0x30000000000));
        void* out=nullptr;quactlize_ppu_dense_plan_info_v1 info{1,sizeof(info)};
        auto& runtime=*static_cast<Runtime*>(ctx.dispatcher);
        size_t before=runtime.requests.size();
        assert(!quactlize_ppu_dense_prepare_candidate_v1(&ctx,&p,low,high,units,&candidate,&out,&info));
        assert(out && info.compute_type==compute && info.split==candidate.split);
        assert(std::string(info.config)==d.symbol && info.policy==QKS_PREDICTED);
        assert(runtime.requests.size()==before); // cannot poison default selection
        auto& plan=*static_cast<quactlize::dense::Plan*>(out);
        assert(plan.native_f16==native && plan.metadata_type==QK_METADATA_F16);
        candidate.variant=1;
        assert(quactlize_ppu_dense_prepare_candidate_v1(&ctx,&p,low,high,units,&candidate,&out,&info) && !out);
        candidate.variant=0;candidate.native_f16=1;
        if (compute) assert(quactlize_ppu_dense_prepare_candidate_v1(&ctx,&p,low,high,units,&candidate,&out,&info) && !out);
        candidate.native_f16=native;candidate.split=0;
        assert(quactlize_ppu_dense_prepare_candidate_v1(&ctx,&p,low,high,units,&candidate,&out,&info) && !out);
        candidate.split=1;candidate.parent[0]=';';
        assert(quactlize_ppu_dense_prepare_candidate_v1(&ctx,&p,low,high,units,&candidate,&out,&info) && !out);
        auto normal=test::plan(ctx,p);
        // The frozen llama PoC fixture is explicitly F16, not a BF16 oracle.
        auto& automatic=*static_cast<quactlize::dense::Plan*>(normal);
        if(automatic.confirmed_prefill) assert(automatic.info.policy==19);
        else if (!compute) compare_poc(poc_dense(ctx.dispatcher,p),automatic);
        else assert(static_cast<quactlize::dense::Plan*>(normal)->compute==compute);
    }
    kImages.clear();auto owner=test::context(0);auto& ctx=*owner;
    auto p=test::params(12,1);
    auto low=reinterpret_cast<void const*>(UINT64_C(0x10000000000));
    auto units=reinterpret_cast<void const*>(UINT64_C(0x30000000000));
    void* out=nullptr;quactlize_ppu_dense_plan_info_v1 info{1,sizeof(info)};
    quactlize_ppu_dense_candidate_v1 candidate{1,sizeof(candidate)};
    candidate.kind=1;candidate.dequant_config=-1;candidate.variant=1;candidate.columns=4;
    candidate.warps=4;candidate.values=4;candidate.split=1;
    assert(!quactlize_ppu_dense_prepare_candidate_v1(&ctx,&p,low,nullptr,units,&candidate,&out,&info));
    assert(info.route==QUACTLIZE_DENSE_SIMT && info.policy==QKS_PREDICTED);
    candidate={1,sizeof(candidate)};candidate.kind=3;candidate.route=QUACTLIZE_DENSE_FULL_BF16;candidate.dequant_config=4;
    assert(quactlize_ppu_dense_prepare_candidate_v1(&ctx,&p,low,nullptr,units,&candidate,&out,&info) && !out);
    p.m=128;
    assert(!quactlize_ppu_dense_prepare_candidate_v1(&ctx,&p,low,nullptr,units,&candidate,&out,&info));
    assert(info.route==QUACTLIZE_DENSE_FULL_BF16 && info.compute_type==QUACTLIZE_DENSE_BF16);
    for (int q:{10,11,12,13,14}) for (int config:{0,4,5,6,10,11,12,13}) {
        p=test::params(q,128);candidate.dequant_config=config;
        auto high=p.sizes.high_bytes?reinterpret_cast<void const*>(UINT64_C(0x20000000000)):nullptr;
        int rc=quactlize_ppu_dense_prepare_candidate_v1(&ctx,&p,low,high,units,&candidate,&out,&info);
        bool supported=config<=12 && (config<=5 || q==12 || q==13);
        assert((rc==0)==supported);
        if (supported) assert(static_cast<quactlize::dense::Plan*>(out)->prefill.dequant_config==config);
        else assert(!out);
    }
    std::puts("DENSE_CANDIDATES_HOST PASS precision+default-cache-isolation+admission timing=NOT_MEASURED");
}

void reviewed_prefill() {
    namespace data=quactlize::dispatch::dense_prefill;
    int exact=0,buckets=0;
    auto check=[&](quactlize_ppu_dense_params_v1 const& p,int policy) {
        kImages.clear();auto ctx=test::context(0,true);
        auto d=data::select(quactlize::dense::request_for(p),0);
        assert(d && d.policy==policy);
        auto raw=test::plan(*ctx,p);auto& plan=*static_cast<quactlize::dense::Plan*>(raw);
        auto const& c=*d.choice;
        assert(plan.info.policy==policy && plan.info.compute_type==0);
        assert(plan.helper_io==c.io && plan.helper_dequant==c.dequant);
        if(c.route==2) {
            assert(plan.kind==quactlize::dense::Kind::Full && plan.full_storage==0 && plan.typed_full);
            assert(!ctx->dispatcher || static_cast<Runtime*>(ctx->dispatcher)->plans.empty());
            assert(std::string(plan.info.config).find("full-f16:")==0);
        } else {
            assert(plan.native_f16 && plan.request.route==c.route && plan.tc.policy==policy);
            assert(std::string(plan.tc.parent)==c.tc.symbol && plan.tc.split==c.tc.split);
            assert(plan.tc.grid==quactlize::dispatch::recipe(c.tc,plan.request,6,72).grid);
        }
        for(uintptr_t replay:{1,2}) {
            test::launches.clear();
            assert(!quactlize_ppu_dense_run_v1(raw,reinterpret_cast<float*>(UINT64_C(0x40000000000)+replay*256),
                reinterpret_cast<float*>(UINT64_C(0x50000000000)+replay*256),
                reinterpret_cast<void*>(UINT64_C(0x60000000000)+replay*256),plan.info.workspace_bytes,
                reinterpret_cast<void*>(replay)));
            std::vector<std::string> expected;
            if(c.route==2) {
                expected={c.dequant_first?"full-typed-f16:dq-first":"full-typed-f16"};
                assert((std::strstr(plan.info.config,":dq-first")!=nullptr)==c.dequant_first);
            }
            else {
                if(c.route==1 && p.qtype!=8) expected.push_back("sf-"+std::to_string(c.dequant));
                expected.push_back(c.io<0?"f16-in":"vector-in-"+std::to_string(c.io));
                expected.push_back("tc");
                expected.push_back(c.io<0?"f16-out":"vector-out-"+std::to_string(c.io));
            }
            assert(test::launches==expected);
        }
    };
    for(auto const& row:data::kExact) {
        auto p=test::params(row.q,row.m,row.n,row.k);
        check(p,QKS_DENSE_PREFILL_EXACT);++exact;
        // Same shape, different storage stride or explicit BF16 must not be
        // relabelled as the F16 contiguous measured contract.
        for(int mode:{0,1,2}) {
            kImages.clear();auto ctx=test::context(mode==0?1:0,true);auto other=p;
            if(mode==1) other.input_stride+=8;
            if(mode==2) other.output_stride+=8;
            auto raw=test::plan(*ctx,other);
            assert(!static_cast<quactlize::dense::Plan*>(raw)->dense_prefill);
        }
        // An interior M exercises every fitted interval, including large M,
        // without shadowing the exact row.
        if(row.m<8192) {
            auto neighbor=p;++neighbor.m;
            if(data::select(quactlize::dense::request_for(neighbor),0)) {
                check(neighbor,QKS_DENSE_PREFILL_BUCKET);++buckets;
            }
        }
    }
    test::helper_device=false;
    auto ctx=test::context(0,true);ctx->dense_device=-1;
    assert(!ctx->dense_policy_device());test::helper_device=true;
    ctx->dense_device=-1;assert(ctx->dense_policy_device());
    std::cout<<"DENSE_PREFILL_HOST exact="<<exact<<" bucket_replays="<<buckets<<" gpu_execution=0\n";
}

void dense_m_controls() {
    std::set<std::tuple<int,int,int>> families;
    for(auto const& r:quactlize::dispatch::dense_prefill::kExact) families.emplace(r.q,r.n,r.k);
    kImages.clear();auto ctx=test::context(0,true);
    for(auto [q,n,k]:families) for(int m=1;m<=8;++m) {
        auto p=test::params(q,m,n,k);auto raw=test::plan(*ctx,p);
        auto const& plan=*static_cast<quactlize::dense::Plan*>(raw);
        int kind=plan.kind==quactlize::dense::Kind::Tc?0:plan.kind==quactlize::dense::Kind::Simt?1:2;
        std::cout<<"DENSE_M_CONTROL {\"request\":["<<q<<",0,"<<n<<','<<k<<",1,1,1,"<<m
            <<",0],\"kind\":"<<kind<<",\"policy\":"<<plan.info.policy
            <<",\"parent\":"<<std::quoted(plan.tc.parent)<<",\"split\":"<<plan.info.split
            <<",\"native_f16\":"<<(plan.native_f16?"true":"false");
        auto const& s=plan.small.base.simt;auto const& f=plan.small.q4;
        if(kind==1)std::cout<<",\"config\":["<<s.variant<<','<<s.columns<<','<<s.warps<<','<<s.values<<','<<s.split<<']';
        if(kind==2)std::cout<<",\"config\":["<<f.reader<<','<<f.variant<<','<<f.columns<<','<<f.warps<<','<<f.values<<']';
        std::cout<<"}\n";
        // Prepared plans are owned by ctx, not separately destroyable handles.
    }
}

int main() {
    dense_m_controls();
    reviewed_prefill();
    using namespace quactlize::dispatch;
    size_t matched_count=0,cost_count=0,component_sf_count=0,token_count=0;
    // Each context has a separate device module cache. The test catalog is
    // reset between contexts; no untracked modules can satisfy a selection.
    for (int compute:{0,1}) {
        kImages.clear();auto owner=test::context(compute);auto& ctx=*owner;
        for (auto const& row:matched::data::kExact) if (row.mode==QKG_DENSE && row.compute==compute) {
            auto p=test::params(row.q,row.tokens,row.n,row.k);
            auto raw=test::plan(ctx,p);auto& plan=*static_cast<quactlize::dense::Plan*>(raw);
            auto d=select_smallm(quactlize::dense::simt_call(plan),p.arrangement);
            assert(d.status==0);
            if (d.tc) {
                assert(plan.kind==quactlize::dense::Kind::Tc);
                assert(std::string(plan.tc.parent)==d.tc->symbol && plan.tc.split==d.tc->split);
                assert(plan.request.route==d.tc->route && plan.compute==compute);
                assert(plan.metadata_type==QK_METADATA_F16);
                if(d.dequant_config>=0) {
                    assert(plan.prefill.version && plan.prefill.route==1);
                    assert(plan.prefill.dequant_config==d.dequant_config);
                    assert(d.choice.base.policy==QKS_DECODE_COMPONENT_MEASURED);
                    // Actual provider helper dispatch (device stub): a table
                    // change without changing the launched expansion is wrong.
                    void const *metadata=nullptr,*zero=nullptr;
                    std::vector<uint8_t> scratch(plan.info.workspace_bytes);
                    test::launches.clear();
                    quactlize::dense::expand_sf(plan,scratch.data(),nullptr,metadata,zero);
                    assert((test::launches==std::vector<std::string>{"sf-"+std::to_string(d.dequant_config)}));
                    ++component_sf_count;
                }
            } else {
                assert(plan.info.route==QUACTLIZE_DENSE_SIMT);
                assert(plan.kind==(d.choice.base.kind==QKS_SMALLM_Q4?quactlize::dense::Kind::Q4:quactlize::dense::Kind::Simt));
                if(plan.kind==quactlize::dense::Kind::Simt &&
                   quactlize::execution::simt::token_cta_window(quactlize::dense::simt_call(plan),d.choice.base.simt)>=0) {
                    assert(std::strstr(plan.info.config,":tokens") && plan.info.split==d.choice.base.simt.split);
                    assert(plan.kernel_bytes==(plan.info.split>1 ?
                        uint64_t(row.tokens)*row.n*plan.info.split*sizeof(float) : 0));
                    ++token_count;
                }
            }
            assert(!plan.copy_input && !plan.copy_output);++matched_count;
        }
        for (auto const& knot:cost::data::kKnots) if (knot.experts==1) {
            auto p=test::params(knot.q,knot.tokens,knot.n,knot.k);
            auto raw=test::plan(ctx,p);auto& plan=*static_cast<quactlize::dense::Plan*>(raw);
            qks_prefill_choice_v1 expected{};
            auto req=quactlize::dense::request_for(p);
            qks_prefill_choice_v2 confirmed{};
            bool fresh=quactlize_kpack_dispatch_prefill_v2(&req,7,compute,1,&confirmed)==0;
            if(fresh) expected=confirmed.base;else assert(cost::query(req,7,expected)==0);
            assert(plan.prefill.route==expected.route && plan.prefill.dequant_config==expected.dequant_config);
            assert(plan.info.route==(expected.route==2?QUACTLIZE_DENSE_FULL_BF16:expected.route));
            if (expected.route!=2) {
                auto r=quactlize::dense::request_for(p);r.route=expected.route;
                auto row=prefill::knot(r,compute,1);
                Selected preferred{};
                if(row) preferred={&prefill::data::kConfigs[row->config],QKS_PREFILL_COMPONENT_MEASURED};
                auto old=select_tc(r,false,compute,preferred);assert(old);
                assert(std::string(plan.tc.parent)==old.symbol && plan.tc.split==old.value.split);
                assert(plan.native_f16==(compute==0));
            }
            ++cost_count;
        }
        for (int q:{8,10,11,12,13,14}) for (int m:{0,1,8,9,16,64,127,128,512,4096,8192}) {
            auto p=test::params(q,m,768,512);
            assert(quactlize_ppu_dense_supported_v1(&p)==1);
            if (m) test::plan(ctx,p);
            p.input_stride=INT64_MAX;assert(quactlize_ppu_dense_supported_v1(&p)==0);
            p=test::params(q,m);p.sizes.units_bytes++;assert(quactlize_ppu_dense_supported_v1(&p)==0);
        }
    }
    assert(component_sf_count==51); // Later complete-call winners supersede only exact SF entries.
    assert(token_count==121);
    kImages.clear();auto owner=test::context(0);auto& ctx=*owner;
    // Replay all measured dense weight families over boundary M, not only
    // exact table keys. This exercises both exact and bucket/fallback order.
    std::set<std::tuple<int,int,int>> weights;
    for (auto const& r:matched::data::kExact) if (r.mode==QKG_DENSE) weights.emplace(r.q,r.n,r.k);
    for (auto const& r:smallm::data::kExact) if (r.mode==QKG_DENSE) weights.emplace(r.q,r.n,r.k);
    for (auto const& r:cost::data::kKnots) if (r.experts==1) weights.emplace(r.q,r.n,r.k);
    for (auto const& r:quactlize::decode_policy::kRanges) if (r.mode==QKG_DENSE) weights.emplace(12,r.n,r.k);
    size_t equivalent=0,legacy=0,replaced=0;
    for (auto [q,n,k]:weights) for (int m:{1,2,3,4,5,6,7,8,9,16,32,64,127,128,256,512,1024,2048,4096}) {
        auto p=test::params(q,m,n,k);auto raw=test::plan(ctx,p);
        auto old=poc_dense(ctx.dispatcher,p);
        auto& now=*static_cast<quactlize::dense::Plan*>(raw);
        if(now.confirmed_prefill) {
            auto row=prefill::knot(quactlize::dense::request_for(p),0,1);assert(row);
            assert(now.prefill.route==row->route && now.prefill.dequant_config==row->dequant && now.info.policy==19);
            std::cout<<"REFRESH_DENSE {\"request\":["<<q<<",0,"<<n<<','<<k<<",1,1,1,"<<m
                <<",0],\"route\":"<<now.prefill.route<<",\"dequant\":"<<now.prefill.dequant_config
                <<",\"compute\":"<<now.info.compute_type<<",\"parent\":"<<std::quoted(now.tc.parent)
                <<",\"split\":"<<now.tc.split<<",\"grid\":"<<now.tc.grid
                <<",\"native_f16\":"<<(now.native_f16?"true":"false")<<"}\n";
            ++replaced;
        } else {compare_poc(old,now);++equivalent;}
        if (old.legacy) ++legacy;
    }
    assert(replaced==280);
    std::printf("DENSE_POC_SELECTION PASS equivalent=%zu confirmed_replacements=%zu old_legacy=%zu weights=%zu fields=route+parent+recipe+precision+prepass+endpoints timing=NOT_MEASURED\n",equivalent,replaced,legacy,weights.size());
    // Exercise the REAL exported query + materializer for every confirmed TC
    // key, including BF16 indexed. Device resources are explicitly fake here.
    size_t confirmed_count=0;
    for(auto const& row:prefill::data::kKnots) if(row.config>=0) {
        auto const& c=prefill::data::kConfigs[row.config];
        qks_request_v1 request{1,sizeof(request),row.q,c.route,row.tokens*(row.experts==1?1:8),
            row.n,row.k,row.experts,row.tokens,c.mapping_id};
        test::install(ctx,select_tc(request,false,row.compute,{&c,QKS_PREFILL_COMPONENT_MEASURED}),false,row.compute);
        qks_choice_v1 choice{};
        assert(!quactlize_kpack_dispatch_query_prefill_v1(ctx.dispatcher,&request,row.compute,1,&choice));
        assert(choice.policy==19 && std::string(choice.parent)==c.symbol && choice.split==c.split);
        auto& runtime=*static_cast<Runtime*>(ctx.dispatcher);
        assert(runtime.plans.at(choice.ticket-1).compute_type==row.compute);
        auto wanted=recipe(c,request,6,72);
        assert(choice.algorithm==wanted.algorithm && choice.grid==wanted.grid);
        auto const& materialized=runtime.plans.at(choice.ticket-1);
        std::cout<<"REFRESH_TICKET {\"request\":["<<row.q<<','<<(row.experts==1?0:2)<<','<<row.n<<','<<row.k
            <<','<<row.experts<<','<<(row.experts==1?1:8)<<",1,"<<row.tokens<<','<<row.compute
            <<"],\"parent\":"<<std::quoted(choice.parent)<<",\"compute\":"<<materialized.compute_type
            <<",\"split\":"<<materialized.recipe.split<<",\"grid\":"<<materialized.recipe.grid
            <<",\"algorithm\":"<<materialized.recipe.algorithm<<",\"endpoint\":"<<materialized.endpoint_type<<"}\n";
        auto ticket=choice.ticket;
        assert(!quactlize_kpack_dispatch_query_prefill_v1(ctx.dispatcher,&request,row.compute,1,&choice) && choice.ticket==ticket);
        assert(quactlize_kpack_dispatch_query_prefill_v1(ctx.dispatcher,&request,row.compute,8,&choice)==QKS_MISS && !choice.ticket);
        request.route^=1;
        assert(quactlize_kpack_dispatch_query_prefill_v1(ctx.dispatcher,&request,row.compute,1,&choice)==QKS_INVALID);
        ++confirmed_count;
    }
    assert(confirmed_count==257);
    // Run order and pointer/stream mutation. All
    // actual arithmetic is deliberately excluded from this host test.
    auto p=test::params(12,512);
    auto raw=test::plan(ctx,p);auto& plan=*static_cast<quactlize::dense::Plan*>(raw);
    assert(plan.info.route==QUACTLIZE_DENSE_SF_TC && plan.native_f16);
    for (int iteration=0;iteration<2;++iteration) {
        void *a=nullptr,*c=nullptr,*scratch=nullptr;
        assert(!posix_memalign(&a,256,uint64_t(p.m)*p.k*4));assert(!posix_memalign(&c,256,uint64_t(p.m)*p.n*4));
        assert(!posix_memalign(&scratch,256,plan.info.workspace_bytes));test::launches.clear();
        assert(quactlize_ppu_dense_run_v1(raw,static_cast<float*>(a),static_cast<float*>(c),scratch,
            plan.info.workspace_bytes,reinterpret_cast<void*>(uintptr_t(iteration+1)))==0);
        assert((test::launches==std::vector<std::string>{"sf-"+std::to_string(plan.prefill.dequant_config),"f16-in","tc","f16-out"}));
        test::launches.clear();
        assert(quactlize_ppu_dense_run_v1(raw,static_cast<float*>(a),static_cast<float*>(a),scratch,plan.info.workspace_bytes,nullptr)!=0);
        assert(test::launches.empty());
        std::free(a);std::free(c);std::free(scratch);
    }
    auto lp=test::params(12,3,51200,5120);
    auto lr=test::plan(ctx,lp);auto& legacy_plan=*static_cast<quactlize::dense::Plan*>(lr);
    void *la=nullptr,*lc=nullptr,*lw=nullptr;
    assert(!posix_memalign(&la,256,uint64_t(lp.m)*lp.k*4));
    assert(!posix_memalign(&lc,256,uint64_t(lp.m)*lp.n*4));
    assert(!posix_memalign(&lw,256,legacy_plan.info.workspace_bytes));
    test::launches.clear();
    assert(!quactlize_ppu_dense_run_v1(lr,static_cast<float*>(la),static_cast<float*>(lc),lw,
        legacy_plan.info.workspace_bytes,nullptr));
    assert((test::launches==std::vector<std::string>{"legacy-in","tc","legacy-out"}));
    std::free(la);std::free(lc);std::free(lw);
    // A new shape that cannot use the old TK256/stage3 parent retains an
    // explicitly unmeasured coverage floor instead of pretending equivalence.
    auto uncovered=test::params(13,16,768,256);
    auto covered=select_dense_dynamic(quactlize::dense::request_for(uncovered),0);
    assert(!select_legacy_dense(quactlize::dense::request_for(uncovered)));
    assert(covered && covered.policy==QKS_DENSE_FALLBACK && covered.value.split==1);
    // Prepare failure must not publish a dangling plan or enqueue any work.
    void* failed=reinterpret_cast<void*>(1);quactlize_ppu_dense_plan_info_v1 info{1,sizeof(info)};
    test::capture=1;
    assert(quactlize_ppu_dense_prepare_v1(&ctx,&p,plan.low,plan.high,plan.units,&failed,&info)!=0 && !failed);
    test::capture=0;
    std::printf("DENSE_PROVIDER_HOST PASS matched=%zu prefill=%zu all_m=1 metadata=F16 default=POC_F16 timing=NOT_MEASURED\n",matched_count,cost_count);
    explicit_candidates();
    promoted_helpers();
}
