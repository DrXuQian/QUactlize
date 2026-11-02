#include "validation.hpp"
#include "../moe/matmul_internal.hpp"
#include "../moe/grouped.hpp"
#include "../fusion/validation.hpp"
#include <cstdio>

namespace quactlize::ffn {
using library::require;
using library::dispatch_check;
using dense::reserve;
namespace mm=moe::matmul;
thread_local std::string error;
struct Context;
struct Stage {
    void* dense=nullptr;
    std::unique_ptr<mm::Plan> moe;
    uint64_t scratch=0,bytes=0,output=0;
    qzm_value_v1 value{};
    bool tc() const {return moe && moe->kind==QKS_SMALLM_TC;}
};
struct Plan {
    Context* owner=nullptr;
    quactlize_ppu_ffn_params_v1 params{};
    quactlize_ppu_ffn_plan_info_v1 info{1,sizeof(info)};
    Stage stages[3];
    uint64_t activation=0,finish_scratch=0;
    qzm_plan_v1 finish{};
    bool fused=false;
    qkg_gate_up_call_v2 gate_up{};
    qkg_gate_up_recipe_v2 recipe{};
    qkg_gate_up_layout_v1 layout{};
};
struct Context {
    int device=0;
    void* dense=nullptr;
    std::unique_ptr<mm::Context> moe;
    std::mutex mutex;
    std::vector<std::unique_ptr<Plan>> plans;
    ~Context() {
        // Dynamic module handles in plans must die before the dispatcher.
        plans.clear();if(dense) quactlize_ppu_dense_destroy_v1(dense);
    }
};
void dense_check(int rc) {require(!rc,quactlize_ppu_dense_error_v1());}
void moe_check(int rc) {require(!rc,quactlize_ppu_moe_matmul_error_v1());}
qzm_value_v1 projected(int columns) {
    return {nullptr,QZM_F32,QZM_TOKEN_SLOT,QZM_PROJECTED,1,columns,0,QZM_F32,0};
}
bool same_value(qzm_value_v1 const& a,qzm_value_v1 const& b) {
    return a.dtype==b.dtype && a.order==b.order && a.state==b.state && a.partial_count==b.partial_count &&
        a.row_stride==b.row_stride && a.partial_stride==b.partial_stride && a.round_type==b.round_type;
}
qzm_finish_v1 finish_call(Plan const& p,qzm_value_v1 value) {
    qzm_finish_v1 c{1,sizeof(c)};auto const& a=p.params;
    c.tokens=a.tokens;c.topk=a.topk;c.columns=a.down.n;c.down=value;
    c.weights_stride=a.weights_stride;c.output_stride=a.output_stride;return c;
}
// Reuse only one projection's map storage. Other TC producers need their
// metadata/output/partials, not another A gather or a second set of routes.
// Handles are address-free: this changes scratch offsets, never a recipe.
void compact_scratch(mm::Plan& m,bool maps,bool input) {
    uint64_t cursor=0,rows=uint64_t(m.params.tokens)*m.params.topk;
    if(maps) {
        m.src=reserve(cursor,rows*4);m.dst=reserve(cursor,rows*4);m.inverse=reserve(cursor,rows*4);
        m.offsets=reserve(cursor,uint64_t(m.params.experts+1)*4);
        m.counts=reserve(cursor,uint64_t(m.params.experts)*4);m.active=reserve(cursor,uint64_t(m.params.experts)*4);
        m.count=reserve(cursor,4);m.route=reserve(cursor,m.preparation.helper.workspace_bytes);
    }
    if(input) m.input=reserve(cursor,rows*m.params.k*2);
    m.output=reserve(cursor,rows*m.params.n*2);
    if(m.plane_bytes) {m.scale=reserve(cursor,m.plane_bytes);m.zero=reserve(cursor,m.plane_bytes);}
    m.kernel=reserve(cursor,m.kernel_bytes);m.info.workspace_bytes=cursor;
}
void prepare_stage(Plan& p,int i,uint64_t& cursor) {
    auto& ctx=*p.owner;auto& s=p.stages[i];auto const& w=weight(p.params,i);
    auto& info=p.info.projections[i];s.value=projected(w.n);
    if(p.params.kind==QUACTLIZE_FFN_DENSE) {
        auto a=dense_params(p.params,i);quactlize_ppu_dense_plan_info_v1 selected{1,sizeof(selected)};
        dense_check(quactlize_ppu_dense_prepare_v1(ctx.dense,&a,w.low,w.high,w.units,&s.dense,&selected));
        info={selected.route,selected.policy,selected.split,selected.compute_type,{}};
        std::snprintf(info.config,sizeof(info.config),"%s",selected.config);s.bytes=selected.workspace_bytes;
    } else {
        s.moe=std::make_unique<mm::Plan>();auto& m=*s.moe;
        m.owner=ctx.moe.get();m.params=moe_params(p.params,i);
        m.low=static_cast<uint8_t const*>(w.low);m.high=static_cast<uint8_t const*>(w.high);m.units=static_cast<uint8_t const*>(w.units);
        mm::prepare(m);
        if(s.tc()) compact_scratch(m,p.info.prepare_stage<0,i==2 || p.info.prepare_stage<0);
        auto const& selected=m.info;
        info={selected.route,selected.policy,selected.split,selected.compute_type,{}};
        std::snprintf(info.config,sizeof(info.config),"%s",selected.config);s.bytes=selected.workspace_bytes;
        if(s.tc()) {
            // Producer-mode ABI leaves only actual TC split partials. Check
            // this planned schema against the module's live output descriptor
            // on every invocation; never infer a second SIMT reducer.
            bool split=selected.split>1;
            s.value={nullptr,split?QZM_F32:QZM_BF16,QZM_GROUPED,split?QZM_PARTIALS:QZM_PROJECTED,
                split?selected.split:1,w.n,split?int64_t(p.params.tokens)*p.params.topk*w.n:0,QZM_BF16,0};
        }
    }
    s.scratch=reserve(cursor,s.bytes);
    if(s.tc()) s.output=s.scratch+s.moe->output;
    else if(i!=2 || p.params.kind==QUACTLIZE_FFN_MOE)
        s.output=reserve(cursor,uint64_t(p.params.tokens)*p.params.topk*w.n*4);
}

bool prepare_gate_up(Plan& p,uint64_t& cursor) {
    auto const& a=p.params;
    if(!a.merged || a.tokens>8 || a.channels!=1) return false;
    auto& d=p.gate_up;d={2,sizeof(d)};
    d.call={1,sizeof(d.call)};d.call.input={2,sizeof(d.call.input)};
    auto& c=d.call.input.call;
    c.version=1;c.size=sizeof(c);c.qtype=a.gate.qtype;c.n=a.down.k;c.k=a.gate.k;
    c.experts=a.gate.experts;c.rows=a.tokens*a.topk;c.topk=a.topk;c.channels=a.channels;
    c.mode=a.kind==QUACTLIZE_FFN_MOE?QKG_INDEXED:QKG_DENSE;c.input_type=QKG_F32;
    c.a_row_stride=a.input_row_stride;c.a_token_stride=a.input_token_stride;c.ids_stride=a.ids_stride;
    c.out_row_stride=a.down.k;c.low=static_cast<uint8_t const*>(a.gate.low);
    c.high=static_cast<uint8_t const*>(a.gate.high);c.units=static_cast<uint8_t const*>(a.gate.units);
    d.call.input.compute_type=p.info.projections[0].compute_type;
    d.call.output_type=p.stages[2].tc()?QKG_SIMT_BF16:QKG_F32;
    d.call.round_projection=p.stages[0].tc()?1:0;
    int consumer=a.kind==QUACTLIZE_FFN_DENSE?QKG_GATE_UP_STANDALONE:
        p.stages[2].tc()?QKG_GATE_UP_TC_DOWN:QKG_GATE_UP_SIMT_DOWN;
    auto request=fusion::request(d.call,p.stages[2].tc()?QKG_GATE_UP_COMPACT_ROWS:QKG_GATE_UP_ORIGINAL_ROWS,
        consumer,a.down.qtype,a.down.n);
    // Reuse the bounded legacy geometry, not its interleaved storage or its
    // projection rounding. The canonical reader preserves the selected stage.
    if(a.kind==QUACTLIZE_FFN_MOE) request.round_projection=1;
    int selected=fusion::select(request,&p.recipe);
    require(selected==QKG_OK || selected==QKG_SHAPE,"FFN fused selection failed");
    if(selected!=QKG_OK) return false;
    p.layout={1,sizeof(p.layout),QKG_GATE_UP_CANONICAL_V1,a.gate.arrangement};
    qkg_sizes_v1 sizes{};
    if(fusion::query_recipe(d,p.recipe,p.layout,sizes)!=QKG_OK) return false;
    p.fused=true;cursor=0;
    auto& gate=p.stages[0];gate.bytes=sizes.workspace_bytes;gate.scratch=reserve(cursor,gate.bytes);
    auto& down=p.stages[2];
    if(down.tc()) {
        compact_scratch(*down.moe,true,true);down.bytes=down.moe->info.workspace_bytes;
    }
    down.scratch=reserve(cursor,down.bytes);
    if(down.tc()) down.output=down.scratch+down.moe->output;
    else if(a.kind==QUACTLIZE_FFN_MOE) down.output=reserve(cursor,uint64_t(a.tokens)*a.topk*a.down.n*4);
    p.info.prepare_stage=down.tc()?2:-1;
    auto& info=p.info.projections[0];
    info.route=p.recipe.config.backend==QKG_GATE_UP_TC?0:2;
    info.policy=QKS_PREDICTED;info.split=p.recipe.config.split;
    std::snprintf(info.config,sizeof(info.config),"gate-up-canonical:%s:recipe=%016llx:round=%d",
        p.recipe.config.backend==QKG_GATE_UP_TC?"tc":"simt",
        static_cast<unsigned long long>(p.recipe.recipe_id),d.call.round_projection);
    return true;
}
void prepare(Plan& p) {
    auto& ctx=*p.owner;auto const& a=p.params;
    if(a.kind==QUACTLIZE_FFN_DENSE && !ctx.dense) dense_check(quactlize_ppu_dense_create_v1(ctx.device,&ctx.dense));
    if(a.kind==QUACTLIZE_FFN_MOE && !ctx.moe) {
        void* out=nullptr;moe_check(quactlize_ppu_moe_matmul_create_v1(ctx.device,&out));
        ctx.moe.reset(static_cast<mm::Context*>(out));
    }
    uint64_t cursor=0;p.info.prepare_stage=-1;p.info.projections[1].route=-1;
    for(int i=0;i<3;++i) {
        if(i==1 && a.merged) continue;
        prepare_stage(p,i,cursor);
        if(p.stages[i].tc() && p.info.prepare_stage<0) p.info.prepare_stage=i;
    }
    prepare_gate_up(p,cursor);
    auto const& down=p.stages[2];
    if(down.tc()) p.activation=down.scratch+down.moe->input;
    else p.activation=reserve(cursor,uint64_t(a.tokens)*a.topk*a.down.k*4);
    if(p.info.prepare_stage>=0) {
        auto const& m=*p.stages[p.info.prepare_stage].moe;
        // If only Down is TC, build maps without gathering the original A.
        p.info.prepare_helper=p.info.prepare_stage==2?0:m.preparation.helper.implementation;
    }
    if(a.kind==QUACTLIZE_FFN_MOE) {
        auto c=finish_call(p,down.value);
        require(quactlize_ppu_moe_finish_select_v1(&c,QZM_AUTO,&p.finish)==QZR_OK,"FFN finish selection failed");
        p.info.finish_helper=p.finish.implementation;p.finish_scratch=reserve(cursor,p.finish.workspace_bytes);
    }
    p.info.workspace_bytes=cursor;
}
struct Maps {int32_t const *rows=nullptr,*inverse=nullptr,*offsets=nullptr;void const* input=nullptr;};
Maps prepare_maps(Plan const& p,float const* input,int32_t const* ids,uint8_t* base,void* stream) {
    if(p.info.prepare_stage<0) return {};
    auto const& stage=p.stages[p.info.prepare_stage];auto const& m=*stage.moe;auto scratch=base+stage.scratch;
    auto ptr=[&](uint64_t offset){return reinterpret_cast<int32_t*>(scratch+offset);};
    if(p.info.prepare_stage!=2) mm::prepare_input(m,input,ids,scratch,stream);
    else {
        auto c=mm::prepare_call(m).route;c.inverse=1;c.ids=ids;
        c.src_rows=ptr(m.inverse);c.dst_rows=ptr(m.dst);c.offsets=ptr(m.offsets);
        c.workspace=m.preparation.helper.workspace_bytes?scratch+m.route:nullptr;
        c.workspace_bytes=m.preparation.helper.workspace_bytes;c.stream=stream;
        require(quactlize_ppu_moe_route_run_v1(&c,&m.preparation.config)==QZR_OK,"FFN Down route failed");
    }
    return {ptr(m.dst),ptr(m.inverse),ptr(m.offsets),scratch+m.input};
}
qzm_value_v1 produce(Plan const& p,int i,float const* input,int32_t const* ids,void const* grouped_input,
                    Maps const& maps,uint8_t* base,void* stream,float* final_output=nullptr) {
    auto const& s=p.stages[i];auto output=final_output?final_output:reinterpret_cast<float*>(base+s.output);
    auto scratch=s.bytes?base+s.scratch:nullptr;
    qzm_value_v1 value=s.value;value.data=output;
    if(s.dense) dense_check(quactlize_ppu_dense_run_v1(s.dense,input,output,scratch,s.bytes,stream));
    else if(!s.tc()) mm::run(*s.moe,input,ids,output,scratch,stream);
    else {
        auto call=mm::grouped(*s.moe,grouped_input,maps.offsets,scratch,stream,QK_GROUPED_PRODUCER);
        qk_grouped_result_v1 result{};
        dispatch_check(quactlize_kpack_dispatch_grouped_output_v1(s.moe->handle,&call,QK_GROUPED_PRODUCER,&result));
        require(moe::grouped_value(result,value)==QZR_OK && same_value(value,s.value),"FFN producer output contract changed");
    }
    return value;
}
void run(Plan const& p,float const* input,int32_t const* ids,float const* weights,float* output,void* workspace,void* stream) {
    auto const& a=p.params;auto base=static_cast<uint8_t*>(workspace);
    auto maps=prepare_maps(p,input,ids,base,stream);
    if(p.fused) {
        auto d=p.gate_up;auto& c=d.call.input.call;
        c.a=input;c.ids=ids;c.output=reinterpret_cast<float*>(base+p.activation);c.stream=stream;
        c.workspace=p.stages[0].bytes?base+p.stages[0].scratch:nullptr;
        c.workspace_bytes=p.stages[0].bytes;d.input_rows=p.stages[2].tc()?maps.rows:nullptr;
        require(quactlize_gate_up_run_v3(&d,&p.recipe,&p.layout)==QKG_OK,"FFN canonical fused gate/up failed");
    } else {
    auto gate=produce(p,0,input,ids,maps.input,maps,base,stream);
    qzm_value_v1 up{};if(!a.merged) up=produce(p,1,input,ids,maps.input,maps,base,stream);
    qzm_transition_v1 c{1,sizeof(c)};c.tokens=a.tokens;c.topk=a.topk;c.hidden=a.down.k;c.merged=a.merged;
    c.gate=gate;c.up=up;c.output_dtype=p.stages[2].tc()?QZM_BF16:QZM_F32;
    c.output_order=p.stages[2].tc()?QZM_GROUPED:QZM_TOKEN_SLOT;c.output_stride=a.down.k;
    c.output=base+p.activation;c.row_ids=maps.rows;c.inverse_rows=maps.inverse;c.stream=stream;
    require(quactlize_ppu_moe_transition_run_v1(&c)==QZR_OK,"FFN SwiGLU/transition failed");
    }
    auto activation=reinterpret_cast<float const*>(base+p.activation);
    auto down=produce(p,2,activation,ids,base+p.activation,maps,base,stream,
                      a.kind==QUACTLIZE_FFN_DENSE?output:nullptr);
    if(a.kind==QUACTLIZE_FFN_MOE) {
        auto f=finish_call(p,down);f.inverse_rows=maps.inverse;f.weights=weights;f.output=output;f.stream=stream;
        require(quactlize_ppu_moe_finish_execute_v1(&f,&p.finish,
            p.finish.workspace_bytes?base+p.finish_scratch:nullptr,p.finish.workspace_bytes)==QZR_OK,
            "FFN weighted finish failed");
    }
}
template<class F> int boundary(F fn) {
    error.clear();try {fn();return 0;} catch(std::exception const& e) {error=e.what();}
    catch(...) {error="unknown FFN runtime error";}return 1;
}
}
extern "C" int quactlize_ppu_ffn_supported_v1(quactlize_ppu_ffn_params_v1 const* p) {
    quactlize::ffn::error.clear();return quactlize::ffn::admission(p);
}
extern "C" int quactlize_ppu_ffn_create_v1(int32_t device,void** out) {
    using namespace quactlize::ffn;if(out) *out=nullptr;
    return boundary([&]{require(out && device>=0,"invalid FFN context ABI");mm::device_check(device);mm::outside_capture();
        auto p=std::make_unique<Context>();p->device=device;*out=p.release();});
}
extern "C" int quactlize_ppu_ffn_prepare_v1(void* context,quactlize_ppu_ffn_params_v1 const* a,
                                          void** out,quactlize_ppu_ffn_plan_info_v1* info) {
    using namespace quactlize::ffn;if(out) *out=nullptr;
    return boundary([&]{
        require(context && out && info && info->version==1 && info->size==sizeof(*info) && admission(a)==1 &&
                a->tokens>0,"invalid FFN prepare ABI/shape");
        quactlize::moe::Spans spans;
        for(int i=0;i<3;++i) if(!(i==1 && a->merged)) require(planes(weight(*a,i),spans),"invalid FFN weight planes");
        auto& ctx=*static_cast<Context*>(context);mm::device_check(ctx.device);mm::outside_capture();
        auto p=std::make_unique<quactlize::ffn::Plan>();p->owner=&ctx;p->params=*a;
        std::lock_guard<std::mutex> lock(ctx.mutex);prepare(*p);
        auto ptr=p.get();ctx.plans.push_back(std::move(p));*info=ptr->info;*out=ptr;
    });
}
extern "C" int quactlize_ppu_ffn_run_v1(void* plan,float const* input,int32_t const* ids,float const* weights,
                                      float* output,void* workspace,uint64_t bytes,void* stream) {
    using namespace quactlize::ffn;
    return boundary([&]{
        require(plan,"null FFN plan");auto const& p=*static_cast<quactlize::ffn::Plan const*>(plan);mm::device_check(p.owner->device);
        require(buffers(p.params,input,ids,weights,output,workspace,bytes,p.info.workspace_bytes),
                "invalid FFN input/IDs/weights/output/workspace extent, alignment or overlap");
        run(p,input,ids,weights,output,workspace,stream);
    });
}
extern "C" void quactlize_ppu_ffn_destroy_v1(void* context) {delete static_cast<quactlize::ffn::Context*>(context);}
extern "C" char const* quactlize_ppu_ffn_error_v1(void) {return quactlize::ffn::error.c_str();}
