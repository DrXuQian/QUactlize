#pragma once
#include "moe_protocol.h"
#include "../integrations/llama/router.cuh"

namespace quactlize::runtime {
namespace moe = quactlize::moe_directory;

// All participants use the production GroupShape/DStride types. Their member
// offsets, not an assumed packed tuple representation, are checked at bind.
template<class Shape,class Stride>
CUTLASS_DEVICE void moe_descriptors(qk_moe_projection_v1 p,int const* ids,
    int const* ranks,int const* starts,int const* counts,bool valid) {
  int tid=int(threadIdx.x);
  auto shapes=static_cast<Shape*>(p.shapes);
  auto strides=static_cast<Stride*>(p.strides);
  for (int e=tid;e<p.experts;e+=int(blockDim.x)) {
    int begin=valid ? expert_begin(ids,p.m,e) : 0;
    int count=valid ? expert_count(ids,p.m,e) : 0;
    p.offsets[e]=begin; p.rows[e]=count; shapes[e]=cute::make_shape(count,p.n,p.k);
    for (int s=0;s<p.splits;++s) {
      int entry=e+s*p.experts;
      int64_t offset=(int64_t(s)*p.m+begin)*p.n;
      if (p.splits==1) static_cast<Half**>(p.outputs)[entry]=static_cast<Half*>(p.output)+offset;
      else static_cast<float**>(p.outputs)[entry]=static_cast<float*>(p.partials)+offset;
      strides[entry]=cutlass::make_cute_packed_stride(Stride{},cute::make_shape(count,p.n,1));
    }
  }
  if (tid<p.m) {
    p.io.row_ids[ranks[tid]]=tid;
    if (valid && starts[tid]==ranks[tid]) {
      int begin=0;
      for (int j=0;j<p.m;++j)
        if (starts[j]==ranks[j] && ids[j]<ids[tid]) begin+=(counts[j]+p.tile_m-1)/p.tile_m;
      auto entry=moe::make_entry(ids[tid],counts[tid],begin,starts[tid]);
      for (int b=0;b<(counts[tid]+p.tile_m-1)/p.tile_m;++b)
        static_cast<moe::BlockEntry*>(p.directory_entries)[begin+b]=entry;
    }
  }
  if (tid==0) {
    int total=0;
    if (valid) for (int j=0;j<p.m;++j)
      if (starts[j]==ranks[j]) total+=(counts[j]+p.tile_m-1)/p.tile_m;
    p.offsets[p.experts]=valid ? p.m : 0;
    *static_cast<moe::Header*>(p.directory_header)={total,
        valid ? 0 : int(moe::BuildStatus::InvalidArgument),p.tile_m,p.experts};
  }
}

template<class Shape,class Stride>
__global__ void moe_chain_prepare(qk_moe_plan_v1 plan) {
  auto p=plan.gate;
  __shared__ int ids[32], ranks[32], starts[32], counts[32];
  __shared__ bool valid;
  int tid=int(threadIdx.x);
  if (plan.router.version) quactlize::llama::router_256(plan.router,p.io,ids,blockIdx.x==0&&blockIdx.y==0);
  else if (tid<p.m) ids[tid]=p.io.ids[int64_t(tid/p.io.topk)*p.io.ids_stride+tid%p.io.topk];
  __syncthreads();
  if (tid==0) valid=valid_route(ids,p.m,p.io.topk,p.experts);
  __syncthreads();
  if (tid<p.m) {
    ranks[tid]=valid ? ranked_row(ids,p.m,tid) : tid;
    starts[tid]=valid ? expert_begin(ids,p.m,ids[tid]) : 0;
    counts[tid]=valid ? expert_count(ids,p.m,ids[tid]) : 0;
  }
  __syncthreads();
  if (blockIdx.x==0 && blockIdx.y==0) {
    moe_descriptors<Shape,Stride>(plan.gate,ids,ranks,starts,counts,valid);
    if (!plan.merged) moe_descriptors<Shape,Stride>(plan.up,ids,ranks,starts,counts,valid);
    moe_descriptors<Shape,Stride>(plan.down,ids,ranks,starts,counts,valid);
  }
  if (!valid) return;
  int r=int(blockIdx.y), to=ranks[r];
  int64_t from=int64_t(r/p.io.topk)*p.io.a_token_stride+(r%p.io.topk%p.io.channels)*p.io.a_row_stride;
  for (int64_t col=int64_t(blockIdx.x)*blockDim.x+tid;col<p.k;col+=int64_t(gridDim.x)*blockDim.x) {
    Half value=Half(p.io.a[from+col]);
    static_cast<Half*>(p.a)[int64_t(to)*p.k+col]=value;
    if (!plan.merged) static_cast<Half*>(plan.up.a)[int64_t(to)*p.k+col]=value;
  }
}

CUTLASS_DEVICE float moe_projection_value(qk_moe_projection_v1 const& p,int row,int col) {
  int64_t index=int64_t(row)*p.n+col;
  if (p.splits==1) return float(static_cast<Half const*>(p.output)[index]);
  float sum=0.f;
  // Ordered exactly like the original reducer. Round BEFORE the activation.
  for (int s=0;s<p.splits;++s) sum+=static_cast<float const*>(p.partials)[int64_t(s)*p.m*p.n+index];
  return float(Half(sum));
}

__global__ void moe_chain_swiglu(qk_moe_plan_v1 plan) {
  int row=int(blockIdx.y), n=plan.down.k;
  bool valid=!static_cast<moe::Header const*>(plan.gate.directory_header)->status;
  for (int64_t col=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;col<n;col+=int64_t(gridDim.x)*blockDim.x) {
    float gate=moe_projection_value(plan.gate,row,int(col));
    float up=moe_projection_value(plan.merged ? plan.gate : plan.up,row,int(col)+(plan.merged?n:0));
    // llama SWIGLU: gate * sigmoid(gate) * up, then down's F32->F16 gather.
    float value=(gate/(1.f+expf(-gate)))*up;
    if (!valid) value=__int_as_float(0x7fffffff);
    static_cast<Half*>(plan.down.a)[int64_t(row)*n+col]=Half(value);
  }
}
} // namespace quactlize::runtime
