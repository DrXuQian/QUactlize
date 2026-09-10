#include "quactlize/dispatch/moe.hpp"
#include "quactlize/integrations/llama/moe_graph.hpp"
#include <cstdio>
#include <stdexcept>

static void require(bool ok,char const* what) { if (!ok) throw std::runtime_error(what); }

static qk_moe_projection_v1 projection(uintptr_t base,int n,int k,int tm,int splits) {
  qk_moe_projection_v1 p{};
  p.version=1; p.size=sizeof(p); p.shape_size=12; p.stride_size=8;
  p.shape_offsets[0]=8; p.shape_offsets[1]=4; p.shape_offsets[2]=0;
  p.m=8; p.n=n; p.k=k; p.experts=256; p.tile_m=tm; p.splits=splits;
  p.a=(void*)base; p.output=(void*)(base+0x100000); p.workspace=(void*)(base+0x200000);
  p.workspace_bytes=0x100000; p.partials=(void*)(base+0x280000);
  p.shapes=(void*)(base+0x200000); p.outputs=(void*)(base+0x201000); p.strides=(void*)(base+0x202000);
  p.offsets=(int32_t*)(base+0x300000); p.rows=(int32_t*)(base+0x203000);
  p.directory_header=(void*)(base+0x204000); p.directory_entries=(void*)(base+0x205000); p.directory_capacity=8;
  p.io={1,sizeof(p.io),1,8,1,0,256,k,k,n,(int32_t*)0x01000000,(float*)0x02000000,
      (float*)(base+0x400000),(int32_t*)(base+0x310000)};
  return p;
}
static void composition() {
  using quactlize::dispatch::compatible_moe;
  qk_moe_plan_v1 plan{1,sizeof(plan),0,0,projection(0x10000000,512,2048,8,4),
      projection(0x20000000,512,2048,16,2),projection(0x30000000,2048,512,32,1)};
  require(compatible_moe(plan),"separate plan rejected");
  qk_llama_router_v1 router{1,sizeof(router),0,1,0,0,0.0001f,1.f,
      (float*)0x03000000,nullptr,(float*)0x04000000};
  using quactlize::dispatch::compatible_router;
  require(compatible_router(plan,router),"router rejected");
  auto router_negative=[&](auto edit) { auto bad=router; edit(bad);
    require(!compatible_router(plan,bad),"router negative missed"); };
  router_negative([](auto& r){r.weights=(float*)0x01000000;});
  router_negative([](auto& r){r.weights=(float*)0x02000000;});
  router_negative([](auto& r){r.logits=(float*)0x10200000;});
  router_negative([](auto& r){r.weights=(float*)r.logits;});
  router_negative([](auto& r){r.clamp=0.f;});
  router_negative([](auto& r){r.delayed_softmax=1;});
  auto negative=[&](auto edit) { auto bad=plan; edit(bad); require(!compatible_moe(bad),"composition negative missed"); };
  negative([](auto& p){p.up.workspace=p.gate.workspace;});
  negative([](auto& p){p.up.a=p.gate.a;});
  negative([](auto& p){p.up.output=p.gate.partials;});
  negative([](auto& p){p.down.io.ids+=1;});
  negative([](auto& p){p.down.io.tokens=2;});
  negative([](auto& p){p.down.k=768;});
  negative([](auto& p){p.up.shape_offsets[0]=0;});
  negative([](auto& p){p.up.stride_size=16;});
  negative([](auto& p){p.up.io.a+=1;});
  negative([](auto& p){p.down.device=1;});
  negative([](auto& p){p.down.io.output=(float*)p.gate.workspace;});
  negative([](auto& p){p.up.workspace=(void*)(UINTPTR_MAX-255);});
  plan.merged=1; plan.gate.n*=2;
  require(compatible_moe(plan),"merged plan rejected");
  std::puts("KPACK_MOE_COMPOSITION PASS separate+merged twelve negatives RED");
}
static void graph(bool merged,bool extra_consumer,bool bad_view,bool same_ids,bool silu) {
  auto * ctx=ggml_init({8*1024*1024,nullptr,true});
  require(ctx!=nullptr,"ggml context");
  auto * a=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,2048,1,1);
  auto * ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,8,1);
  auto * weight=ggml_new_tensor_3d(ctx,GGML_TYPE_Q4_K,2048,merged?1024:512,256);
  auto * gate=ggml_mul_mat_id(ctx,weight,a,ids);
  ggml_tensor * up;
  if (merged) {
    auto * both=gate;
    gate=ggml_view_3d(ctx,both,512,8,1,both->nb[1],both->nb[2],bad_view?4:0);
    up=ggml_view_3d(ctx,both,512,8,1,both->nb[1],both->nb[2],512*4);
  } else up=ggml_mul_mat_id(ctx,weight,a,ids);
  auto * glu=silu?ggml_swiglu_split(ctx,gate,up):ggml_geglu_split(ctx,gate,up);
  auto * dw=ggml_new_tensor_3d(ctx,GGML_TYPE_Q5_K,512,2048,256);
  auto * out=ggml_mul_mat_id(ctx,dw,glu,same_ids?ids:ggml_dup_tensor(ctx,ids));
  auto * graph=ggml_new_graph(ctx);
  ggml_build_forward_expand(graph,out);
  if (extra_consumer) ggml_build_forward_expand(graph,ggml_scale(ctx,gate,2.f));
  bool found=false;
  for (int i=0;i<graph->n_nodes;++i) found |= quactlize::llama::match_moe(graph,i).count!=0;
  require(found==(!extra_consumer&&!bad_view&&same_ids&&silu),"graph match disagrees");
  ggml_free(ctx);
}
int main() {
  try {
    composition();
    for (bool merged:{false,true}) {
      graph(merged,false,false,true,true); graph(merged,true,false,true,true);
      graph(merged,false,false,false,true); graph(merged,false,false,true,false);
    }
    graph(true,false,true,true,true);
    std::puts("KPACK_MOE_GRAPH PASS exact GGML separate+merged; shared consumer, wrong view, IDs, activation RED");
    return 0;
  } catch (std::exception const& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
