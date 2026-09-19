// Policy/ABI only: fake resource queries, never simulated GPU arithmetic.
#include "quactlize/dispatch/binding.cpp"
#include <cassert>
#include <cstdio>
static int compute=-1,queries=0;
static int resource(qk_resources_v1* out) {
  ++queries;*out={};out->workspace_bytes=8192;out->shared_bytes=1234;out->occupancy=3;return QK_OK;
}
int main() {
  using namespace quactlize::dispatch;
  int count=0;
  for(auto const& row:matched::data::kExact) {
    auto const& f=matched::data::kChoices[row.choice];if(f.kind!=0)continue;
    auto const& c=f.tc;bool dense=row.mode==QKG_DENSE;compute=row.compute;
    Runtime runtime;runtime.device=0;runtime.cu=72;
    auto module=std::make_shared<Module>();module->compute_type=compute;
    module->query_dense_io=[](qkd_dense_call_v1 const* d,qk_recipe_v1 const*,qk_resources_v1* out){assert(compute==0 && d->input_type==QKD_F32);return resource(out);};
    module->query_dense_compute=[](qkd_dense_call_v2 const* d,qk_recipe_v1 const*,qk_resources_v1* out){assert(compute==1 && d->compute_type==compute && d->dense.input_type==QKD_F32);return resource(out);};
    module->query_device=[](qk_device_call_v2 const*,qk_recipe_v1 const*,qk_resources_v1* out){assert(compute==0);return resource(out);};
    module->query_compute=[](qk_compute_device_call_v4 const* d,qk_recipe_v1 const*,qk_resources_v1* out){assert(compute==1 && d->compute_type==compute);return resource(out);};
    Image image{c.symbol,std::string(64,'a'),std::string(64,'b'),c.qtype,c.route,c.tm,c.tn,c.tk,c.wm,c.wn,c.stages,c.ap,c.dn,{},dense,compute};
    kImages={image};runtime.modules[image.key]=module;
    qkg_call_v1 call{};call.version=1;call.size=sizeof(call);call.mode=row.mode;call.qtype=row.q;
    call.n=row.n;call.k=row.k;call.experts=row.experts;call.topk=row.topk;call.channels=row.channels;
    call.rows=row.tokens*(dense?1:8);call.input_type=QKG_F32;call.a_row_stride=row.k;
    call.a_token_stride=int64_t(row.channels)*row.k;call.ids_stride=row.topk;call.out_row_stride=row.n;
    auto arr=row.q==8?q8_kpack2::arrangement():row.q==12?
      ppu_arrangements::q4_kpack4_transpose_v1():ppu_arrangements::kquant_kpack_transpose_v1(row.q);
    qkg_simt_call_v2 typed{2,sizeof(typed),call,compute};qks_smallm_choice_v2 out{};
    int rc=quactlize_kpack_dispatch_query_smallm_v3(&runtime,&typed,&arr,&out);
    assert(rc==QKS_OK && out.base.kind==QKS_SMALLM_TC && out.compute_type==compute);
    assert(out.base.policy==(row.router_sensitive?QKS_MATCHED_ROUTER:QKS_MATCHED_EXACT));
    assert(out.base.tc.policy==out.base.policy && std::string(out.base.tc.parent)==c.symbol);
    assert(out.base.tc.split==c.split && out.base.tc.workspace_bytes==8192 && out.base.tc.shared_bytes==1234);
    auto const& plan=runtime.plans.back();assert(plan.compute_type==compute && plan.choice.ticket==out.base.tc.ticket);
    auto rec=recipe(c,plan.request,3);
    assert(rec.algorithm==out.base.tc.algorithm && rec.grid==out.base.tc.grid && rec.split==out.base.tc.split);
    int before=queries;qks_smallm_choice_v2 again{};
    assert(quactlize_kpack_dispatch_query_smallm_v3(&runtime,&typed,&arr,&again)==QKS_OK);
    assert(queries==before && !std::memcmp(&out,&again,sizeof(out)));++count;
  }
  assert(count>500);
  printf("MATCHED_TC_DISPATCH PASS rows=%d exact compute/geometry/split/grid/ticket preserved\n",count);
}
