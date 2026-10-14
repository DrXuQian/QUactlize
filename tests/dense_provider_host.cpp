// Real provider, dispatcher and selectors; fake DEVICE launches/resources.
// Tests ABI/composition/selection only. This is not a GPU numerical oracle.
#include "quactlize/dispatch/binding.cpp"
#include "quactlize/dense/runtime.cpp"
#include "quactlize/prefill/dynamic.hpp"
#include "quactlize/runtime/dynamic.hpp"
#include <cassert>
#include <iomanip>
#include <sstream>
#include <set>

namespace test {
std::vector<std::string> launches;
int device=0,capture=0;
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
std::unique_ptr<quactlize::dense::Context> context(int compute) {
    auto owner=std::make_unique<quactlize::dense::Context>();auto& ctx=*owner;
    ctx.device=0;ctx.compute=compute;ctx.jit=true;
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
extern "C" hggcError_t hggcStreamIsCapturing(hggcStream_t,hggcStreamCaptureStatus* out) {
    *out=test::capture?hggcStreamCaptureStatusActive:hggcStreamCaptureStatusNone;return hggcSuccess;
}
extern "C" int quactlize_dense_copy_f32(float const*,float*,int,int,int64_t,int64_t,void*) {test::launches.push_back("copy-f32");return 0;}
extern "C" int quactlize_dense_convert_f16(int input,void const*,void*,int,int,int64_t,int linear,void*) {
    test::launches.push_back(linear?(input?"legacy-in":"legacy-out"):(input?"f16-in":"f16-out"));return 0;
}
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
        if (!compute) compare_poc(poc_dense(ctx.dispatcher,p),*static_cast<quactlize::dense::Plan*>(normal));
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

int main() {
    using namespace quactlize::dispatch;
    size_t matched_count=0,cost_count=0;
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
            } else {
                assert(plan.info.route==QUACTLIZE_DENSE_SIMT);
                assert(plan.kind==(d.choice.base.kind==QKS_SMALLM_Q4?quactlize::dense::Kind::Q4:quactlize::dense::Kind::Simt));
            }
            assert(!plan.copy_input && !plan.copy_output);++matched_count;
        }
        for (auto const& knot:cost::data::kKnots) if (knot.experts==1) {
            auto p=test::params(knot.q,knot.tokens,knot.n,knot.k);
            auto raw=test::plan(ctx,p);auto& plan=*static_cast<quactlize::dense::Plan*>(raw);
            qks_prefill_choice_v1 expected{};assert(cost::query(quactlize::dense::request_for(p),7,expected)==0);
            assert(plan.prefill.route==expected.route && plan.prefill.dequant_config==expected.dequant_config);
            assert(plan.info.route==(expected.route==2?QUACTLIZE_DENSE_FULL_BF16:expected.route));
            if (expected.route!=2) {
                auto r=quactlize::dense::request_for(p);r.route=expected.route;
                auto old=select_tc(r,false,compute);assert(old);
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
    kImages.clear();auto owner=test::context(0);auto& ctx=*owner;
    // Replay all measured dense weight families over boundary M, not only
    // exact table keys. This exercises both exact and bucket/fallback order.
    std::set<std::tuple<int,int,int>> weights;
    for (auto const& r:matched::data::kExact) if (r.mode==QKG_DENSE) weights.emplace(r.q,r.n,r.k);
    for (auto const& r:smallm::data::kExact) if (r.mode==QKG_DENSE) weights.emplace(r.q,r.n,r.k);
    for (auto const& r:cost::data::kKnots) if (r.experts==1) weights.emplace(r.q,r.n,r.k);
    for (auto const& r:quactlize::decode_policy::kRanges) if (r.mode==QKG_DENSE) weights.emplace(12,r.n,r.k);
    size_t equivalent=0,legacy=0;
    for (auto [q,n,k]:weights) for (int m:{1,2,3,4,5,6,7,8,9,16,32,64,127,128,256,512,1024,2048,4096}) {
        auto p=test::params(q,m,n,k);auto raw=test::plan(ctx,p);
        auto old=poc_dense(ctx.dispatcher,p);
        compare_poc(old,*static_cast<quactlize::dense::Plan*>(raw));++equivalent;
        if (old.legacy) ++legacy;
    }
    std::printf("DENSE_POC_SELECTION PASS equivalent=%zu old_legacy=%zu weights=%zu fields=route+parent+recipe+precision+prepass+endpoints timing=NOT_MEASURED\n",equivalent,legacy,weights.size());
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
}
