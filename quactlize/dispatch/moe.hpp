#pragma once
#include "../runtime/moe_protocol.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>

namespace quactlize::dispatch {
struct MoeSpan { uintptr_t begin, end; };
inline bool moe_span(void const* pointer,uint64_t bytes,MoeSpan& out) {
  uintptr_t begin=reinterpret_cast<uintptr_t>(pointer);
  if (!begin || !bytes || bytes>UINTPTR_MAX-begin) return false;
  out={begin,begin+bytes}; return true;
}
inline bool moe_overlap(MoeSpan a,MoeSpan b) { return a.begin<b.end && b.begin<a.end; }

// Validates composition, not a second config selector. A miss does not mutate
// the existing standalone handles. All buffers are owned by their callers.
inline bool compatible_moe(qk_moe_plan_v1 const& plan) {
  if (plan.version!=1 || plan.size!=sizeof(plan) || plan.reserved || plan.merged>1) return false;
  auto const& g=plan.gate; auto const& u=plan.up; auto const& d=plan.down;
  if (g.m<=0 || g.m>32 || g.experts<=0 || g.experts>1024 ||
      int64_t(d.k)*(plan.merged?2:1)!=g.n ||
      (!plan.merged && (u.n!=g.n || u.k!=g.k || u.io.a!=g.io.a ||
       u.io.channels!=g.io.channels || u.io.a_row_stride!=g.io.a_row_stride ||
       u.io.a_token_stride!=g.io.a_token_stride))) return false;
  std::vector<MoeSpan> writes;
  auto add=[&](void const* ptr,uint64_t bytes) {
    MoeSpan span;
    if (!moe_span(ptr,bytes,span)) return false;
    for (auto old:writes) if (moe_overlap(old,span)) return false;
    writes.push_back(span); return true;
  };
  for (auto p:{&g,plan.merged?nullptr:&u,&d}) {
    if (!p) continue;
    if (p->version!=1 || p->size!=sizeof(*p) || p->reserved || p->device!=g.device ||
        p->m!=g.m || p->experts!=g.experts || p->n<=0 || p->n>INT32_MAX-8192 ||
        p->k<=0 || p->k>INT32_MAX-8192 || p->tile_m<8 || p->tile_m>256 ||
        (p->tile_m&(p->tile_m-1)) ||
        (p->splits!=1 && p->splits!=2 && p->splits!=4 && p->splits!=8) ||
        p->shape_size!=g.shape_size || p->stride_size!=g.stride_size ||
        p->stride_offset!=g.stride_offset ||
        std::memcmp(p->shape_offsets,g.shape_offsets,sizeof(g.shape_offsets)) ||
        p->io.version!=1 || p->io.size!=sizeof(p->io) || p->io.reserved ||
        p->io.ids!=g.io.ids || p->io.tokens!=g.io.tokens || p->io.topk!=g.io.topk ||
        p->io.ids_stride!=g.io.ids_stride || p->io.tokens<=0 || p->io.topk<=0 ||
        int64_t(p->io.tokens)*p->io.topk!=p->m || !p->io.row_ids ||
        !p->shapes || !p->outputs || !p->strides || !p->rows || !p->directory_header ||
        !p->directory_entries || !p->directory_capacity || (p->splits>1 && !p->partials)) return false;
    // Per-stream standalone scratch deliberately aliases between projections.
    // A fused chain MUST use disjoint retained storage or partials get clobbered.
    if (!add(p->a,uint64_t(p->m)*p->k*2) || !add(p->output,uint64_t(p->m)*p->n*2) ||
        !add(p->offsets,uint64_t(p->experts+1)*4) || !add(p->io.row_ids,uint64_t(p->m)*4) ||
        !add(p->workspace,p->workspace_bytes)) return false;
  }
  MoeSpan input,ids,output;
  if (g.io.channels<=0 || g.io.a_token_stride<=0 || g.io.ids_stride<g.io.topk ||
      d.io.out_row_stride<d.n || uint64_t(g.io.a_token_stride)>UINT64_MAX/4/g.io.tokens ||
      uint64_t(g.io.ids_stride)>UINT64_MAX/4/g.io.tokens ||
      uint64_t(d.io.out_row_stride)>UINT64_MAX/4/d.m ||
      !moe_span(g.io.a,uint64_t(g.io.a_token_stride)*g.io.tokens*4,input) ||
      !moe_span(g.io.ids,uint64_t(g.io.ids_stride)*g.io.tokens*4,ids) ||
      !moe_span(d.io.output,uint64_t(d.io.out_row_stride)*d.m*4,output)) return false;
  for (auto span:writes)
    if (moe_overlap(span,input) || moe_overlap(span,ids) || moe_overlap(span,output)) return false;
  // External output may alias original input: its only write follows all GEMMs.
  return !moe_overlap(output,ids);
}

inline bool compatible_router(qk_moe_plan_v1 const& p,qk_llama_router_v1 const& r) {
  if (r.version!=1 || r.size!=sizeof(r) || r.reserved || p.gate.experts!=256 ||
      (r.use_sigmoid!=0 && r.use_sigmoid!=1) || (r.with_norm!=0 && r.with_norm!=1) ||
      (r.delayed_softmax!=0 && r.delayed_softmax!=1) || (r.with_norm && r.delayed_softmax) ||
      !std::isfinite(r.scale) || (r.with_norm && !(r.clamp>0.f && std::isfinite(r.clamp))) ||
      (uintptr_t(r.logits)|uintptr_t(r.bias)|uintptr_t(r.weights))%4) return false;
  MoeSpan logits,bias,weights,ids,input;
  if (!moe_span(r.logits,uint64_t(p.gate.io.tokens)*256*4,logits) ||
      !moe_span(r.weights,uint64_t(p.gate.m)*4,weights) ||
      !moe_span(p.gate.io.ids,uint64_t(p.gate.io.tokens)*p.gate.io.ids_stride*4,ids) ||
      !moe_span(p.gate.io.a,uint64_t(p.gate.io.tokens)*p.gate.io.a_token_stride*4,input) ||
      (r.bias && !moe_span(r.bias,256*4,bias))) return false;
  if (moe_overlap(weights,logits) || moe_overlap(weights,ids) || moe_overlap(weights,input) ||
      moe_overlap(ids,logits) || moe_overlap(ids,input) ||
      (r.bias && (moe_overlap(weights,bias) || moe_overlap(ids,bias)))) return false;
  for (auto part:{&p.gate,p.merged?nullptr:&p.up,&p.down}) {
    if (!part) continue;
    MoeSpan writes[5];
    if (!moe_span(part->workspace,part->workspace_bytes,writes[0]) ||
        !moe_span(part->a,uint64_t(part->m)*part->k*2,writes[1]) ||
        !moe_span(part->output,uint64_t(part->m)*part->n*2,writes[2]) ||
        !moe_span(part->offsets,uint64_t(part->experts+1)*4,writes[3]) ||
        !moe_span(part->io.row_ids,uint64_t(part->m)*4,writes[4])) return false;
    for (auto span:writes) if (moe_overlap(span,logits) || moe_overlap(span,weights) ||
        (r.bias && moe_overlap(span,bias))) return false;
  }
  MoeSpan output;
  if (!moe_span(p.down.io.output,uint64_t(p.down.io.out_row_stride)*p.down.m*4,output)) return false;
  return !moe_overlap(output,weights); // Both remain graph outputs after the fused segment.
}
} // namespace quactlize::dispatch
