// Real provider, dispatcher and selectors; DEVICE doubles only, no numeric
// or performance claims. Every current call binding is checked before enqueue.
#include "quactlize/dispatch/binding.cpp"
#include "quactlize/moe/matmul.cpp"
#include "quactlize/runtime/grouped_dynamic.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace test {
int device=0,capture=0;
bool ffn=false;
void const* last_a=nullptr;int32_t const* last_ids=nullptr;void* last_stream=nullptr;
std::vector<std::string> launches;
struct Handle {qk_compute_device_call_v4 shape;int split;};
int resource(qk_call_v1 const* c,qk_recipe_v1 const* r,qk_resources_v1* out) {
    *out={1,sizeof(*out),256+(r->split>1?uint64_t(c->m)*c->n*r->split*4:0),4096,2,0,0};return 0;
}
void install(quactlize::moe::matmul::Context& ctx,quactlize::dispatch::TcDecision const& d) {
    if(!d) return;auto c=d.config();auto& r=*static_cast<Runtime*>(ctx.dispatcher);
    for(auto const& i:kImages) if(i.parent==c.symbol) return;
    std::ostringstream key;key<<std::setfill('0')<<std::setw(64)<<std::hex<<(kImages.size()+1);
    Image image{c.symbol,key.str(),std::string(64,'a'),c.qtype,c.route,c.tm,c.tn,c.tk,c.wm,c.wn,c.stages,c.ap,c.dn,{},false,1};
    auto mod=std::make_shared<Module>();mod->qtype=c.qtype;mod->compute_type=1;
    mod->query=resource;mod->query_device=[](auto d,auto r,auto out){return resource(&d->call,r,out);};
    mod->query_compute=[](auto d,auto r,auto out){assert(d->compute_type==1);return resource(&d->device_call.call,r,out);};
    mod->prepare_grouped_dynamic=[](auto d,auto r,void** out){
        assert(quactlize::runtime::address_free(d->device_call.call));
        assert(d->compute_type==1 && (d->metadata_type==0 || d->metadata_type==1));
        *out=new Handle{*d,r->split};return 0;
    };
    mod->run_grouped_dynamic=[](void* h,auto d,int mode){
        auto& handle=*static_cast<Handle*>(h);auto const& c=d->device_call.call;
        assert(quactlize::runtime::same_grouped_shape(*d,handle.shape));
        assert(mode==(ffn?QK_GROUPED_PRODUCER:QK_GROUPED_COMPLETE) && c.a && c.low && c.metadata && c.offsets_device && c.output);
        assert(!c.rows_host && !c.rows_device && c.stream==last_stream);
        launches.push_back(handle.split>1?(ffn?"tc-partials":"tc+reducer"):"tc");return 0;
    };
    mod->grouped_output=[](void* h,auto c,int mode,auto out){
        quactlize::runtime::GroupedWorkspace layout{};layout.partials=256;
        return quactlize::runtime::grouped_result(*c,static_cast<Handle*>(h)->split,mode,layout,*out);
    };
    mod->destroy=[](void* h){delete static_cast<Handle*>(h);};
    r.modules.emplace(image.key,mod);kImages.push_back(image);
}
auto context() {
    auto c=std::make_unique<quactlize::moe::matmul::Context>();c->device=0;c->jit=true;
    auto r=new Runtime;r->device=0;r->cu=64;c->dispatcher=r;
    r->moe=std::make_shared<MoeExecution>();r->moe->select=&quactlize_kpack_q4_decode_select_v1;
    r->moe->select_compute=&quactlize_kpack_q4_decode_select_v2;r->moe->q4_compute=&quactlize_kpack_q4_decode_run_v2;
    return c;
}
auto params(int q,int m,int n,int k,int e,int topk,int ch) {
    quactlize_ppu_moe_matmul_params_v1 p{1,sizeof(p)};
    p.qtype=q;p.tokens=m;p.n=n;p.k=k;p.experts=e;p.topk=topk;p.channels=ch;
    p.input_row_stride=k;p.input_token_stride=k*ch;p.ids_stride=topk;p.output_stride=n;
    assert(!quactlize_ppu_kpack_canonical_arrangement_v1(q,&p.arrangement));
    assert(!quactlize_ppu_kpack_sizes_for_arrangement_v1(n,k,e,q,&p.arrangement,&p.sizes));return p;
}
void install_for(quactlize::moe::matmul::Context& c,quactlize_ppu_moe_matmul_params_v1 const& p) {
    using namespace quactlize::dispatch;
    quactlize::moe::matmul::Plan plan;plan.params=p;
    auto selected=select_smallm(quactlize::moe::matmul::indexed(plan),p.arrangement);
    qks_request_v1 r{1,sizeof(r),p.qtype,3,p.tokens*p.topk,p.n,p.k,p.experts,p.tokens,p.arrangement.mapping_id};
    if(selected.status==0 && selected.tc) {
        r.route=selected.tc->route;install(c,select_tc(r,true,1,{selected.tc,selected.choice.base.policy}));
    }
    if(auto row=moe_prefill::select(r,1,p.channels)) {
        r.route=row->config.route;install(c,select_tc(r,false,1,{&row->config,row->tokens==p.tokens?22:23}));
    }
    r.route=3;install(c,select_tc(r,false,1));
}
void binding(qkg_call_v1 const& c) {
    assert(c.mode==QKG_INDEXED && (ffn?c.a!=nullptr:c.a==last_a) && c.ids==last_ids && c.stream==last_stream);
    assert(c.input_type==QKG_F32 && c.output);launches.push_back("simt");
}
}
extern "C" hggcError_t hggcGetDevice(int* out) {*out=test::device;return hggcSuccess;}
extern "C" hggcError_t hggcDeviceGetAttribute(int* out,hggcDeviceAttr attr,int device) {
    assert(device==test::device && attr==hggcDevAttrMultiProcessorCount);*out=64;return hggcSuccess;
}
extern "C" hggcError_t hggcPeekAtLastError() {return hggcSuccess;}
extern "C" hggcError_t hggcStreamIsCapturing(hggcStream_t,hggcStreamCaptureStatus* out) {
    *out=test::capture?hggcStreamCaptureStatusActive:hggcStreamCaptureStatusNone;return hggcSuccess;
}
extern "C" hggcError_t hggcMemsetAsync(void*,int,size_t,hggcStream_t stream) {
    assert(stream==test::last_stream);test::launches.push_back("status");return hggcSuccess;
}
extern "C" int quactlize_ppu_moe_prepare_execute_v2(qzm_prepare_v1 const* c,qzm_prepare_plan_v2 const* p) {
    auto selected=*c;selected.config=p->config;quactlize::routing::Layout l;
    assert(!quactlize::moe::prepare(selected,l,true));
    assert(c->input==test::last_a && c->route.ids==test::last_ids && c->route.stream==test::last_stream && c->dtype==QZM_BF16);
    test::launches.push_back("prepare");return 0;
}
extern "C" int quactlize_prefill_scatter_bf16(void const*,float*,int32_t const*,int,int,int64_t,int32_t*,void*) {
    test::launches.push_back("scatter");return 0;
}
extern "C" int quactlize_kpack_sf_prepare_v2(int,int,int,int,uint8_t const*,uint64_t,uint16_t*,uint16_t*,uint64_t,
    quactlize_ppu_placed_arrangement_v2 const*,int32_t metadata,void*) {
    assert(metadata==QK_METADATA_BF16);test::launches.push_back("sf");return 0;
}
extern "C" int quactlize_kpack_dequant_v2(qzd_call_v2 const* c,quactlize_ppu_placed_arrangement_v2 const*) {
    assert(c->metadata_type==QK_METADATA_BF16 && c->call.operation==0);test::launches.push_back("sf");return 0;
}
extern "C" int qkg_q4_decode_launch(qkg_call_v1 const&,qkg_q4_decode_config_v1 const&) {assert(false);return 1;}
extern "C" int qkg_q4_decode_launch_bf16(qkg_call_v1 const& c,qkg_q4_decode_config_v1 const&) {test::binding(c);return 0;}
#define STUB(Q) \
extern "C" int qkg_simt_launch_##Q(qkg_call_v1 const*,qkg_simt_config_v1 const*) {assert(false);return 1;} \
extern "C" int qkg_simt_launch_v2_##Q(qkg_simt_call_v2 const* c,qkg_simt_config_v1 const*) {assert(c->compute_type==1);test::binding(c->call);return 0;} \
extern "C" bool qkg_simt_supported_##Q(qkg_simt_config_v1 const*) {return true;}
STUB(8) STUB(10) STUB(11) STUB(12) STUB(13) STUB(14)
#undef STUB

int main() {
    using namespace quactlize::moe::matmul;
    int cases=0;
    if(auto root=std::getenv("QZ_TEST_MXROOT")) {
        auto ctx=test::context();ctx->root=root;ctx->sdk=root;
        auto p=test::params(39,17,256,512,256,6,1);
        void* handles[2]{};quactlize_ppu_moe_matmul_plan_info_v1 info{1,sizeof(info)};
        for(auto& handle:handles) {
            assert(!quactlize_ppu_moe_matmul_prepare_v1(ctx.get(),&p,(void*)(1ull<<40),nullptr,
                (void*)(3ull<<40),&handle,&info));
            assert(info.route==QUACTLIZE_MOE_MXFP4_W4FA16 && info.compute_type==1 && info.split==1);
        }
        using MoEPlan=quactlize::moe::matmul::Plan;
        assert(static_cast<MoEPlan*>(handles[0])->mxfp4==static_cast<MoEPlan*>(handles[1])->mxfp4);
        for(int i=0;i<2;++i) {
            auto a=reinterpret_cast<float*>(uint64_t(5+4*i)<<40);
            auto ids=reinterpret_cast<int32_t*>(uint64_t(6+4*i)<<40);
            auto out=reinterpret_cast<float*>(uint64_t(7+4*i)<<40);
            auto scratch=reinterpret_cast<void*>(uint64_t(4+4*i)<<40);
            test::last_a=a;test::last_ids=ids;test::last_stream=reinterpret_cast<void*>(i+1);
            test::capture=i;test::launches.clear();
            assert(!quactlize_ppu_moe_matmul_run_v1(handles[i],a,ids,out,scratch,info.workspace_bytes,test::last_stream));
            assert((test::launches==std::vector<std::string>{"prepare","status","scatter"}));
            assert(quactlize_ppu_moe_matmul_run_v1(handles[i],a,ids,out,scratch,info.workspace_bytes-1,test::last_stream));
        }
        test::capture=0;
    }
    auto check=[&](auto p) {
        assert(quactlize_ppu_moe_matmul_supported_v1(&p)==1);
        auto admission=p;admission.tokens=0;assert(quactlize_ppu_moe_matmul_supported_v1(&admission)==1);
        admission.arrangement.mapping_id^=1;assert(quactlize_ppu_moe_matmul_supported_v1(&admission)==0);
        auto ctx=test::context();test::install_for(*ctx,p);
        void* handle=nullptr;quactlize_ppu_moe_matmul_plan_info_v1 info{1,sizeof(info)};
        int rc=quactlize_ppu_moe_matmul_prepare_v1(ctx.get(),&p,(void*)(1ULL<<40),p.sizes.high_bytes?(void*)(2ULL<<40):nullptr,
            (void*)(3ULL<<40),&handle,&info);
        if(rc) std::cerr<<p.qtype<<' '<<p.tokens<<' '<<p.n<<' '<<p.k<<' '<<quactlize_ppu_moe_matmul_error_v1()<<'\n';
        assert(!rc && handle && info.compute_type==1);
        auto& plan=*static_cast<quactlize::moe::matmul::Plan*>(handle);
        if(plan.kind==QKS_SMALLM_TC) assert(plan.metadata==(p.qtype==8?0:1));
        if(auto row=quactlize::dispatch::moe_prefill::select(plan.request,1,p.channels)) {
            assert(info.policy==(row->tokens==p.tokens?22:23));
            assert(std::string(info.config)==row->config.symbol && info.split==row->config.split);
            assert(plan.tc.compute_units==64 && info.source_tokens==row->tokens);
        }
        for(int iteration=0;iteration<2;++iteration) {
            auto a=reinterpret_cast<float*>(uint64_t(5+iteration*4)<<40);
            auto ids=reinterpret_cast<int32_t*>(uint64_t(6+iteration*4)<<40);
            auto out=reinterpret_cast<float*>(uint64_t(7+iteration*4)<<40);
            auto scratch=reinterpret_cast<void*>(uint64_t(4+iteration*4)<<40);
            test::last_a=a;test::last_ids=ids;test::last_stream=reinterpret_cast<void*>(iteration+1);test::launches.clear();
            test::capture=iteration;
            rc=quactlize_ppu_moe_matmul_run_v1(handle,a,ids,out,scratch,info.workspace_bytes,test::last_stream);
            if(rc) std::cerr<<quactlize_ppu_moe_matmul_error_v1()<<'\n';
            assert(!rc && !test::launches.empty());
            if(plan.kind!=QKS_SMALLM_TC) assert(test::launches==std::vector<std::string>{"simt"});
            else {
                assert(test::launches.front()=="prepare" && test::launches.back()=="scatter");
                assert(test::launches[test::launches.size()-3]==(info.split>1?"tc+reducer":"tc"));
            }
            test::launches.clear();
            assert(quactlize_ppu_moe_matmul_run_v1(handle,a,ids,a,scratch,info.workspace_bytes,test::last_stream));
            assert(test::launches.empty());
            if(info.workspace_bytes) {
                assert(quactlize_ppu_moe_matmul_run_v1(handle,a,ids,out,scratch,info.workspace_bytes-1,test::last_stream));
                assert(test::launches.empty());
            }
        }
        test::capture=0;++cases;kImages.clear();
    };
    for(auto const& r:quactlize::dispatch::moe_prefill::data::kRows) check(test::params(r.q,r.tokens,r.n,r.k,256,8,r.channels));
    for(int q:{8,10,11,12,13,14}) for(int m:{1,2,3,4,5,6,7,8,9,16,32,64,128,256,512,1024,2048,4096,5120,8192})
        for(int ch:{1,2}) check(test::params(q,m,256,512,4,2,ch));
    for(auto const& family:std::vector<std::vector<int>>{{12,512,2048,1},{12,512,3072,1},{13,2048,512,8},{13,3072,512,8},
                                                       {12,1024,2048,1},{12,1024,3072,1}})
        for(int m:{1,2,3,4,5,6,7,8,9,17,33,65,129,257,513,1025,2049,4097,5121})
            check(test::params(family[0],m,family[1],family[2],256,8,family[3]));
    std::cout<<"MOE_MATMUL_HOST PASS cases="<<cases<<" device_execution=0\n";
    return 0;
}
