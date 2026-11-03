// Reuse the real-dispatcher fixture; only GPU entrypoints are doubled.
#define main independent_projection_main
#include "moe_matmul_host.cpp"
#undef main
#include "quactlize/ffn/ffn.cpp"
#include <algorithm>

static_assert(sizeof(qzm_value_v1)==48 && offsetof(qzm_value_v1,round_type)==40,"test transition value ABI");
static_assert(sizeof(qzm_transition_v1)==168 && offsetof(qzm_transition_v1,gate)==24 &&
              offsetof(qzm_transition_v1,stream)==160,"test transition call ABI");

namespace test {
std::vector<qzm_transition_v1> transitions;
std::vector<qzm_finish_v1> finishes;
std::vector<std::tuple<float const*,int64_t,float*>> snapshots;
struct Dense {quactlize_ppu_dense_params_v1 shape;};
struct DenseContext {std::vector<std::unique_ptr<Dense>> plans;};
}
extern "C" int quactlize_ppu_dense_create_v1(int,void** out) {*out=new test::DenseContext;return 0;}
extern "C" void quactlize_ppu_dense_destroy_v1(void* ctx) {delete static_cast<test::DenseContext*>(ctx);}
extern "C" char const* quactlize_ppu_dense_error_v1() {return "host dense double";}
extern "C" int quactlize_ppu_dense_prepare_v1(void* ctx,quactlize_ppu_dense_params_v1 const* p,
    void const*,void const*,void const*,void** out,quactlize_ppu_dense_plan_info_v1* info) {
    assert(quactlize::dense::admission(p)==1);
    auto plan=std::make_unique<test::Dense>();plan->shape=*p;*out=plan.get();
    static_cast<test::DenseContext*>(ctx)->plans.push_back(std::move(plan));
    *info={1,sizeof(*info),256,2,0,1,0,"host-dense-double"};return 0;
}
extern "C" int quactlize_ppu_dense_run_v1(void*,float const* input,float* out,void*,uint64_t,void* stream) {
    assert(input && out && stream==test::last_stream);test::launches.push_back("dense");return 0;
}
extern "C" int quactlize_ppu_moe_route_run_v1(qzr_call_v1 const* c,qzr_config_v1 const* config) {
    quactlize::routing::Layout l;assert(!quactlize::routing::layout(*c,*config,l));
    assert(c->inverse==1 && c->ids==test::last_ids && c->stream==test::last_stream);
    test::launches.push_back("route-only");return 0;
}
extern "C" int quactlize_ppu_moe_transition_run_v1(qzm_transition_v1 const* c) {
    uint32_t ops=0;assert(!quactlize::moe::transition(*c,ops,true));assert(ops & QZM_SWIGLU);
    assert(c->stream==test::last_stream);test::transitions.push_back(*c);
    test::launches.push_back("transition");return 0;
}
int quactlize::ffn::dense_swiglu(float const* gate,float const* up,float* output,int rows,int hidden,
    int64_t gate_stride,int64_t up_stride,int64_t output_stride,void* stream) {
    assert(gate && up && output && stream==test::last_stream);
    qzm_transition_v1 t{1,sizeof(t)};t.tokens=rows;t.topk=1;t.hidden=hidden;
    t.merged=up==gate+hidden && gate_stride==up_stride;
    t.gate={gate,QZM_F32,QZM_TOKEN_SLOT,QZM_PROJECTED,1,gate_stride,0,QZM_F32,0};
    if(!t.merged) t.up={up,QZM_F32,QZM_TOKEN_SLOT,QZM_PROJECTED,1,up_stride,0,QZM_F32,0};
    t.output=output;t.output_dtype=QZM_F32;t.output_order=QZM_TOKEN_SLOT;t.output_stride=output_stride;t.stream=stream;
    uint32_t ops=0;assert(!quactlize::moe::transition(t,ops,true));assert(ops==QZM_SWIGLU);
    test::transitions.push_back(t);test::launches.push_back("dense-swiglu");return 0;
}
int quactlize::moe::transition_snapshot(qzm_transition_v1 const* c,float const* weights,int64_t stride,float* retained) {
    assert(weights && retained && stride>=c->topk);
    test::snapshots.emplace_back(weights,stride,retained);
    return quactlize_ppu_moe_transition_run_v1(c);
}
extern "C" int quactlize_ppu_moe_finish_execute_v1(qzm_finish_v1 const* c,qzm_plan_v1 const* p,void* scratch,uint64_t bytes) {
    assert(!quactlize::moe::finish(*c,true));
    assert(!quactlize::moe::finish_workspace(*c,p->workspace_bytes,scratch,bytes));
    assert(c->stream==test::last_stream);test::finishes.push_back(*c);test::launches.push_back("finish");return 0;
}
namespace test {
auto weight(int q,int n,int k,int e,int slot) {
    auto a=params(q,1,n,k,e,1,1);
    return quactlize_ppu_ffn_weight_v1{q,n,k,e,a.arrangement,a.sizes,
        reinterpret_cast<void*>(uint64_t(slot+1)<<40),
        a.sizes.high_bytes?reinterpret_cast<void*>(uint64_t(slot+2)<<40):nullptr,
        reinterpret_cast<void*>(uint64_t(slot+3)<<40)};
}
auto ffn_params(int q,int m,bool merged,bool dense) {
    quactlize_ppu_ffn_params_v1 a{1,sizeof(a)};
    a.kind=dense?QUACTLIZE_FFN_DENSE:QUACTLIZE_FFN_MOE;a.tokens=m;a.topk=dense?1:2;
    a.channels=1;a.merged=merged;a.input_row_stride=512;a.input_token_stride=512;
    a.ids_stride=dense?0:2;a.weights_stride=dense?0:2;a.output_stride=256;
    a.gate=weight(q,merged?1024:512,512,dense?1:4,0);
    if(!merged) a.up=weight(q,512,512,dense?1:4,3);
    a.down=weight(q,256,512,dense?1:4,6);return a;
}
void exercise(quactlize::ffn::Plan& p) {
    auto const& a=p.params;
    for(int repeat=0;repeat<2;++repeat) {
        auto at=[&](int n){return reinterpret_cast<void*>(uint64_t(16+repeat*6+n)<<40);};
        auto input=static_cast<float*>(at(0)),out=static_cast<float*>(at(1));
        auto scratch=at(2);auto weights=a.kind==QUACTLIZE_FFN_MOE?static_cast<float*>(at(3)):nullptr;
        auto ids=a.kind==QUACTLIZE_FFN_MOE?static_cast<int32_t*>(at(4)):nullptr;
        last_a=input;last_ids=ids;last_stream=reinterpret_cast<void*>(repeat+1);capture=repeat;
        launches.clear();transitions.clear();finishes.clear();
        int rc=quactlize_ppu_ffn_run_v1(&p,input,ids,weights,out,scratch,p.info.workspace_bytes,last_stream);
        if(rc) std::cerr<<quactlize_ppu_ffn_error_v1()<<'\n';assert(!rc);
        assert(transitions.size()==1 &&
            (a.kind==QUACTLIZE_FFN_DENSE?finishes.empty():finishes.size()==1));
        assert(std::count(launches.begin(),launches.end(),"dense-swiglu")==int(a.kind==QUACTLIZE_FFN_DENSE));
        assert(std::count(launches.begin(),launches.end(),"transition")==int(a.kind==QUACTLIZE_FFN_MOE));
        assert(std::count(launches.begin(),launches.end(),"prepare")==int(p.info.prepare_stage>=0 && p.info.prepare_stage<2));
        assert(std::count(launches.begin(),launches.end(),"route-only")==int(p.info.prepare_stage==2));
        assert(std::count(launches.begin(),launches.end(),"scatter")==0);
        auto const& t=transitions.back();
        assert(t.gate.order==(p.stages[0].tc()?QZM_GROUPED:QZM_TOKEN_SLOT));
        assert(t.output_order==(p.stages[2].tc()?QZM_GROUPED:QZM_TOKEN_SLOT));
        assert(t.output_dtype==(p.stages[2].tc()?QZM_BF16:QZM_F32));
        assert(t.merged==a.merged);
        for(auto const& f:finishes) assert(f.down.partial_count==p.stages[2].value.partial_count && f.output==out && f.weights==weights);
        launches.clear();
        assert(quactlize_ppu_ffn_run_v1(&p,input,ids,weights,input,scratch,p.info.workspace_bytes,last_stream));
        assert(quactlize_ppu_ffn_run_v1(&p,input,ids,weights,out,scratch,p.info.workspace_bytes-1,last_stream));
        assert(quactlize_ppu_ffn_run_v1(&p,input,ids,weights,const_cast<float*>(static_cast<float const*>(a.gate.low)),scratch,p.info.workspace_bytes,last_stream));
        assert(quactlize_ppu_ffn_run_v1(&p,input,ids,weights,out,const_cast<float*>(input),p.info.workspace_bytes,last_stream));
        assert(launches.empty());
        // v2 permits inputs reused by the final output, without another launch.
        for(int alias=0;alias<(a.kind==QUACTLIZE_FFN_MOE?4:1);++alias) {
            auto target=alias==0?input:alias==1?reinterpret_cast<float*>(ids):weights;
            auto source_ids=alias==3?reinterpret_cast<int32_t*>(target):ids;
            auto source_a=alias==3?target:input;
            last_ids=source_ids;last_a=source_a;
            launches.clear();transitions.clear();finishes.clear();snapshots.clear();
            assert(!quactlize_ppu_ffn_run_v2(&p,source_a,source_ids,weights,target,scratch,p.info.workspace_bytes,last_stream));
            const auto sequence=launches;
            bool retain=a.kind==QUACTLIZE_FFN_MOE && alias>=2;
            assert(snapshots.size()==size_t(retain));
            if(retain) {
                auto saved=static_cast<float*>(reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(scratch)+p.retained_weights));
                assert(std::get<0>(snapshots[0])==weights && std::get<1>(snapshots[0])==a.weights_stride &&
                    std::get<2>(snapshots[0])==saved && finishes[0].weights==saved && finishes[0].weights_stride==a.topk);
            }
            launches.clear();
            assert(!quactlize_ppu_ffn_run_v2(&p,source_a,source_ids,weights,out,scratch,p.info.workspace_bytes,last_stream));
            assert(launches==sequence);
        }
        last_ids=ids;last_a=input;launches.clear();
        assert(quactlize_ppu_ffn_run_v2(&p,input,ids,weights,out,input,p.info.workspace_bytes,last_stream));
        assert(quactlize_ppu_ffn_run_v2(&p,input,ids,weights,const_cast<float*>(static_cast<float const*>(a.gate.low)),scratch,p.info.workspace_bytes,last_stream));
        assert(launches.empty());
        device=1;assert(quactlize_ppu_ffn_run_v1(&p,input,ids,weights,out,scratch,p.info.workspace_bytes,last_stream));device=0;
    }
    capture=0;
}
}
int main() {
    using namespace quactlize::ffn;test::ffn=true;int cases=0;
    for(bool dense:{false,true}) for(bool merged:{false,true}) for(int q:{8,10,11,12,13,14})
        for(int m:{1,2,3,4,5,6,7,8,9,16,32,128,512,4096,5120,8192}) {
            auto a=test::ffn_params(q,m,merged,dense);assert(admission(&a)==1);
            if(m==9) {
                a.input_row_stride+=16;a.input_token_stride+=16;a.output_stride+=8;
                if(!dense) {a.ids_stride+=1;a.weights_stride+=3;}
                assert(admission(&a)==1);
            }
            if(!merged && m==16) {
                a.up=test::weight(q==14?8:q==8?10:q+1,512,512,dense?1:4,3);
                assert(admission(&a)==1);
            }
            auto bad=a;bad.gate.arrangement.mapping_id^=1;assert(!admission(&bad));
            bad=a;bad.down.k*=2;assert(!admission(&bad));
            bad=a;bad.tokens=INT_MAX;assert(!admission(&bad));
            bad=a;bad.tokens=0;assert(admission(&bad)==1);
            auto ctx=std::make_unique<Context>();ctx->device=0;
            if(!dense) {
                ctx->moe=test::context();for(int i=0;i<3;++i) if(!(merged && i==1)) test::install_for(*ctx->moe,moe_params(a,i));
            }
            void* handle=nullptr;quactlize_ppu_ffn_plan_info_v1 info{1,sizeof(info)};
            test::capture=1;
            assert(quactlize_ppu_ffn_prepare_v1(ctx.get(),&a,&handle,&info) && !handle);
            test::capture=0;
            int rc=quactlize_ppu_ffn_prepare_v1(ctx.get(),&a,&handle,&info);
            if(rc) std::cerr<<quactlize_ppu_ffn_error_v1()<<'\n';assert(!rc && handle);
            test::exercise(*static_cast<quactlize::ffn::Plan*>(handle));++cases;kImages.clear();
        }
    // Real merged Gate/Up families must reach the same selector as independent
    // projections. Keep Down's original hidden width and channel broadcasting.
    for(int k:{2048,3072}) for(int m:{9,16,32,64,128,256,512,513,1024,2048,4096,5120,5121}) {
        auto a=test::ffn_params(12,m,true,false);
        a.topk=8;a.ids_stride=8;a.weights_stride=8;
        a.input_row_stride=k;a.input_token_stride=k;a.output_stride=k;
        a.gate=test::weight(12,1024,k,256,0);a.down=test::weight(13,k,512,256,6);
        assert(admission(&a)==1);
        auto ctx=std::make_unique<Context>();ctx->moe=test::context();
        for(int i:{0,2}) test::install_for(*ctx->moe,moe_params(a,i));
        void* handle=nullptr;quactlize_ppu_ffn_plan_info_v1 info{1,sizeof(info)};
        int rc=quactlize_ppu_ffn_prepare_v1(ctx.get(),&a,&handle,&info);
        if(rc) std::cerr<<quactlize_ppu_ffn_error_v1()<<'\n';assert(!rc && handle);
        auto& p=*static_cast<quactlize::ffn::Plan*>(handle);
        assert(!p.stages[1].moe && info.projections[1].route==-1 && info.prepare_stage==0);
        for(int i:{0,2}) {
            auto const& projection=*p.stages[i].moe;
            auto const& params=projection.params;
            auto row=quactlize::dispatch::moe_prefill::select(projection.request,1,params.channels);
            assert(row && params.n==(i==0?1024:k) && params.k==(i==0?k:512));
            assert(params.channels==(i==0?1:8));
            assert(info.projections[i].policy==(m==row->tokens?22:23));
            assert(std::string(info.projections[i].config)==row->config.symbol);
            assert(info.projections[i].split==row->config.split && projection.dequant==row->dequant);
        }
        test::exercise(p);++cases;kImages.clear();
    }
    // Merged dense families keep their measured 2N projection and the original
    // hidden-width Down. No fused epilogue is admitted for these families.
    for(auto family:std::vector<std::pair<int,int>>{{12,25600},{11,17408},{12,17408},{13,17408},{14,17408}})
    for(int m:{1,2,3,4,5,6,7,8,16,32,64,128,256,512,1024,2048,4096,5120}) {
        auto [q,n]=family;auto a=test::ffn_params(q,m,true,true);
        a.input_row_stride=a.input_token_stride=a.output_stride=5120;
        a.gate=test::weight(q,2*n,5120,1,0);a.down=test::weight(14,5120,n,1,6);
        auto ctx=std::make_unique<Context>();
        void* handle=nullptr;quactlize_ppu_ffn_plan_info_v1 info{1,sizeof(info)};
        int rc=quactlize_ppu_ffn_prepare_v1(ctx.get(),&a,&handle,&info);
        if(rc) std::cerr<<quactlize_ppu_ffn_error_v1()<<'\n';assert(!rc && handle);
        auto& p=*static_cast<quactlize::ffn::Plan*>(handle);
        assert(!p.stages[1].dense && info.projections[1].route==-1);
        auto const& gate=static_cast<test::Dense*>(p.stages[0].dense)->shape;
        auto const& down=static_cast<test::Dense*>(p.stages[2].dense)->shape;
        assert(gate.qtype==q && gate.m==m && gate.n==2*n && gate.k==5120);
        assert(down.qtype==14 && down.m==m && down.n==5120 && down.k==n);
        test::exercise(p);++cases;kImages.clear();
    }
    for(bool dense:{false,true}) for(int k:{2048,3072}) for(int m:{1,2,3,4,5,6,7,8}) {
        int n=dense && k==3072?1024:512;
        auto a=test::ffn_params(dense?8:12,m,true,dense);
        a.topk=dense?1:8;a.ids_stride=dense?0:8;a.weights_stride=dense?0:8;
        a.input_row_stride=k;a.input_token_stride=k;a.output_stride=k;
        a.gate=test::weight(dense?8:12,2*n,k,dense?1:256,0);
        a.down=test::weight(dense?8:13,k,n,dense?1:256,6);
        auto ctx=std::make_unique<Context>();
        if(!dense) {
            ctx->moe=test::context();
            for(int i:{0,2}) test::install_for(*ctx->moe,moe_params(a,i));
        }
        void* handle=nullptr;quactlize_ppu_ffn_plan_info_v1 info{1,sizeof(info)};
        int rc=quactlize_ppu_ffn_prepare_v1(ctx.get(),&a,&handle,&info);
        if(rc) std::cerr<<quactlize_ppu_ffn_error_v1()<<'\n';assert(!rc && handle);
        auto& p=*static_cast<quactlize::ffn::Plan*>(handle);
        assert(info.projections[1].route==-1);
        for(int i:{0,2}) {
            auto const& actual=info.projections[i];
            if(dense) assert(std::string(actual.config)=="host-dense-double");
            else {
                auto const& selected=p.stages[i].moe->info;
                assert(actual.route==selected.route && actual.policy==selected.policy &&
                    actual.split==selected.split && actual.compute_type==selected.compute_type &&
                    std::string(actual.config)==selected.config);
            }
        }
        test::exercise(p);++cases;kImages.clear();
    }
    // All mixed producer boundaries, including only Down TC. Explicit host
    // fixtures do NOT promote these choices into a measured production table.
    for(int mask=0;mask<8;++mask) for(int split:{1,2,4,8}) {
        auto ctx=std::make_unique<Context>();ctx->moe=test::context();
        auto a=test::ffn_params(12,4,false,false);quactlize::ffn::Plan p;p.owner=ctx.get();p.params=a;
        uint64_t cursor=0;p.info.prepare_stage=-1;
        for(int i=0;i<3;++i) {
            auto& s=p.stages[i];s.moe=std::make_unique<mm::Plan>();auto& mp=*s.moe;
            mp.owner=ctx->moe.get();mp.params=moe_params(a,i);auto const& w=weight(a,i);
            mp.low=static_cast<uint8_t const*>(w.low);mp.high=static_cast<uint8_t const*>(w.high);mp.units=static_cast<uint8_t const*>(w.units);
            if(mask&(1<<i)) {
                // Resolve a real SF parent, then use its actual dynamic handle.
                qks_request_v1 request{1,sizeof(request),w.qtype,QK_GROUPED_SF,a.tokens*a.topk,w.n,w.k,w.experts,a.tokens,w.arrangement.mapping_id};
                auto decision=quactlize::dispatch::select_tc(request,false,1);test::install(*ctx->moe,decision);
                dispatch_check(quactlize_kpack_dispatch_query_compute_v1(ctx->moe->dispatcher,&request,1,0,0,&mp.tc));
                // Fixture module chooses the tested Split-K producer layout.
                auto& runtime=*static_cast<Runtime*>(ctx->moe->dispatcher);
                runtime.plans.at(mp.tc.ticket-1).recipe.split=split;
                dispatch_check(quactlize_kpack_dispatch_prepare_grouped_dynamic_v1(ctx->moe->dispatcher,&mp.tc,&mp.handle));
                mp.request=request;mp.kind=QKS_SMALLM_TC;mp.kernel_bytes=256+uint64_t(a.tokens)*a.topk*w.n*split*4;
                uint64_t local=0;auto rows=uint64_t(a.tokens)*a.topk;
                mp.src=reserve(local,rows*4);mp.dst=reserve(local,rows*4);mp.offsets=reserve(local,20);mp.inverse=reserve(local,rows*4);
                mp.counts=reserve(local,16);mp.active=reserve(local,16);mp.count=reserve(local,4);
                auto c=mm::prepare_call(mp);assert(!quactlize_ppu_moe_prepare_select_v2(&c,&mp.preparation));
                mp.route=reserve(local,mp.preparation.helper.workspace_bytes);mp.input=reserve(local,rows*w.k*2);
                mp.output=reserve(local,rows*w.n*2);mp.kernel=reserve(local,mp.kernel_bytes);s.bytes=local;
                s.value={nullptr,split>1?QZM_F32:QZM_BF16,QZM_GROUPED,split>1?QZM_PARTIALS:QZM_PROJECTED,
                    split,w.n,split>1?int64_t(rows)*w.n:0,QZM_BF16,0};
                if(p.info.prepare_stage<0)p.info.prepare_stage=i;
            } else {
                mp.kind=QKS_SMALLM_SIMT;mp.small.base.simt={1,sizeof(mp.small.base.simt),0,4,4,4,split};
                auto c=mm::indexed(mp);qkg_sizes_v1 size{};
                assert(!quactlize_kpack_simt_query_v2(&c,&mp.small.base.simt,&w.arrangement,&size));
                mp.kernel_bytes=size.workspace_bytes;s.bytes=mp.kernel_bytes;s.value=projected(w.n);
            }
            s.scratch=reserve(cursor,s.bytes);s.output=s.tc()?s.scratch+mp.output:reserve(cursor,uint64_t(a.tokens)*a.topk*w.n*4);
        }
        auto const& down=p.stages[2];p.activation=down.tc()?down.scratch+down.moe->input:reserve(cursor,uint64_t(a.tokens)*a.topk*a.down.k*4);
        auto f=finish_call(p,down.value);assert(!quactlize_ppu_moe_finish_select_v1(&f,QZM_AUTO,&p.finish));
        p.finish_scratch=reserve(cursor,p.finish.workspace_bytes);
        p.retained_weights=reserve(cursor,uint64_t(a.tokens)*a.topk*4);p.info.workspace_bytes=cursor;
        test::exercise(p);++cases;kImages.clear();
    }
    std::cout<<"FFN_HOST PASS cases="<<cases<<" device_execution=0\n";
}
