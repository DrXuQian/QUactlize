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

static void router_graph(bool merged,int fault,int tokens=1) {
  auto * ctx=ggml_init({8*1024*1024,nullptr,true});
  auto * input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2048,tokens);
  auto * logits=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,tokens);
  auto * probs=ggml_soft_max(ctx,logits);
  auto * ids=ggml_argsort_top_k(ctx,probs,8);
  auto * view=ggml_reshape_3d(ctx,probs,1,256,tokens);
  auto * weights=ggml_get_rows(ctx,view,ids);
  auto * g=ggml_new_graph(ctx);
  ggml_build_forward_expand(g,weights);
  // The real model expands router weights before the input's reshape.
  auto * a=ggml_reshape_3d(ctx,input,2048,1,tokens);
  auto * weight=ggml_new_tensor_3d(ctx,GGML_TYPE_Q4_K,2048,merged?1024:512,256);
  auto * gate=ggml_mul_mat_id(ctx,weight,a,ids);
  ggml_tensor * up;
  if (merged) {
    auto * both=gate;
    gate=ggml_view_3d(ctx,both,512,8,tokens,both->nb[1],both->nb[2],0);
    up=ggml_view_3d(ctx,both,512,8,tokens,both->nb[1],both->nb[2],512*4);
  } else up=ggml_mul_mat_id(ctx,weight,a,ids);
  auto * glu=ggml_swiglu_split(ctx,gate,up);
  auto * dw=ggml_new_tensor_3d(ctx,GGML_TYPE_Q5_K,512,2048,256);
  auto * out=ggml_mul_mat_id(ctx,dw,glu,fault==2?ggml_dup_tensor(ctx,ids):ids);
  ggml_build_forward_expand(g,out);
  if (fault==1) ggml_build_forward_expand(g,ggml_scale(ctx,gate,2.f));
  if (fault==3) ggml_build_forward_expand(g,ggml_scale(ctx,a,2.f));
  int start=-1,ii=-1,wi=-1,end=-1;
  for (int i=0;i<g->n_nodes;++i) {
    if (g->nodes[i]==probs) start=i;
    if (g->nodes[i]==ids) ii=i;
    if (g->nodes[i]==weights) wi=i;
    if (g->nodes[i]==out) end=i;
  }
  require(start>=0 && ii>start && wi>=ii && end>wi,"router fixture indices");
  std::vector<ggml_op> prefix,whole;
  for (int i=start;i<=wi;++i) prefix.push_back(g->nodes[i]->op);
  for (int i=start;i<=end;++i) whole.push_back(g->nodes[i]->op);
  int old_outputs[]={ii,wi,end};
  bool legacy=ggml_can_fuse_subgraph(g,start,int(whole.size()),whole.data(),old_outputs,3);
  auto span=quactlize::llama::match_moe_router(g,start,prefix,ii,wi);
  require(!legacy,"legacy predicate unexpectedly accepted external input view");
  require(bool(span.count)==(tokens<=4 && (fault==0 || fault==3)),"router/input-view fusion disagrees");
  if (span.count) require(start+span.count-1==end,"router fusion skipped wrong nodes");
  require(!quactlize::llama::match_moe_router(g,start,prefix,wi,wi).count,"wrong router IDs accepted");
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
    for (int tokens:{1,2,3,4,5,8,16,32,64,128,512,2048})
      for (bool merged:{false,true}) for (int fault=0;fault<4;++fault) router_graph(merged,fault,tokens);
    std::puts("KPACK_MOE_GRAPH_TOKEN_SCOPE PASS topk=8 fused_tokens=1,2,3,4 declined_tokens=5,8,16,32,64,128,512,2048");
    std::puts("KPACK_MOE_ROUTER_GRAPH PASS input views retained; legacy predicate RED; projection uses/IDs rejected");
    std::puts("KPACK_MOE_GRAPH PASS exact GGML separate+merged; shared consumer, wrong view, IDs, activation RED");
    return 0;
  } catch (std::exception const& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
