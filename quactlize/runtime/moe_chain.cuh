#pragma once
#include "moe_protocol.h"
#include "../integrations/llama/router.cuh"

namespace quactlize::runtime {
namespace moe = quactlize::moe_directory;

// All participants use the production GroupShape/DStride types. Their member
// offsets, not an assumed packed tuple representation, are checked at bind.
template<class Shape,class Stride>
CUTLASS_DEVICE void moe_descriptors(qk_moe_projection_v1 const& p,int const* ids,
    int const* ranks,int const* starts,int const* counts,
    int const* expert_starts,int const* expert_counts,bool valid) {
  int tid=int(threadIdx.x),lane=tid%32,warp=tid/32;
  auto shapes=static_cast<Shape*>(p.shapes);
  auto strides=static_cast<Stride*>(p.strides);
  int e=int(blockIdx.x)*32+lane;
  if (e<p.experts) {
    int begin=expert_starts[lane],count=expert_counts[lane];
    if (warp==0) {
      p.offsets[e]=begin; p.rows[e]=count; shapes[e]=cute::make_shape(count,p.n,p.k);
    }
    // Split descriptors are independent; warps write slices in parallel.
    for (int s=warp;s<p.splits;s+=int(blockDim.x)/32) {
      int entry=e+s*p.experts;
      int64_t offset=(int64_t(s)*p.m+begin)*p.n;
      if (p.splits==1) static_cast<Half**>(p.outputs)[entry]=static_cast<Half*>(p.output)+offset;
      else static_cast<float**>(p.outputs)[entry]=static_cast<float*>(p.partials)+offset;
      strides[entry]=cutlass::make_cute_packed_stride(Stride{},cute::make_shape(count,p.n,1));
    }
  }
  if (blockIdx.x==0 && tid<32) {
    int id=tid<p.m ? ids[tid] : p.experts;
    int tiles=valid && tid<p.m && starts[tid]==ranks[tid] ? (counts[tid]+p.tile_m-1)/p.tile_m : 0;
    int begin=0,total=0;
    // Divide once per active expert, then prefix the tile counts in a warp.
    #pragma unroll
    for (int j=0;j<32;++j) {
      int other=__shfl_sync(0xffffffff,id,j,32);
      int blocks=__shfl_sync(0xffffffff,tiles,j,32);
      begin+=other<id ? blocks : 0; total+=blocks;
    }
    if (tid<p.m) {
      p.io.row_ids[ranks[tid]]=tid;
      auto entry=moe::make_entry(id,counts[tid],begin,starts[tid]);
      for (int b=0;b<tiles;++b) static_cast<moe::BlockEntry*>(p.directory_entries)[begin+b]=entry;
    }
    if (tid==0) {
      p.offsets[p.experts]=valid ? p.m : 0;
      *static_cast<moe::Header*>(p.directory_header)={total,
          valid ? 0 : int(moe::BuildStatus::InvalidArgument),p.tile_m,p.experts};
    }
  }
}

template<class Shape,class Stride>
__global__ void moe_chain_prepare(qk_moe_plan_v1 plan) {
  auto const& p=plan.gate;
  __shared__ int ids[32], ranks[32], starts[32], counts[32];
  __shared__ int expert_starts[32], expert_counts[32];
  int tid=int(threadIdx.x);
  if (plan.router.version) quactlize::llama::router_256(plan.router,p.io,ids,blockIdx.x==0);
  else if (tid<p.m) ids[tid]=p.io.ids[int64_t(tid/p.io.topk)*p.io.ids_stride+tid%p.io.topk];
  __syncthreads();
  bool invalid=false;
  if (tid<32) {
    int id=tid<p.m ? ids[tid] : -1;
    int e=int(blockIdx.x)*32+tid;
    int rank=0,start=0,count=0,eb=0,ec=0;
    int token_begin=tid/p.io.topk*p.io.topk;
    invalid=tid<p.m && (id<0 || id>=p.experts);
    // The bounded route has at most 32 rows. Parallel row validation avoids
    // one lane's quadratic, dynamically divided duplicate scan.
    #pragma unroll
    for (int j=0;j<32;++j) {
      int other=__shfl_sync(0xffffffff,id,j,32);
      if (j<p.m) {
        start+=other<id; count+=other==id;
        rank+=other<id || (other==id && j<tid);
        eb+=other<e; ec+=other==e;
        invalid|=tid<p.m && j>=token_begin && j<tid && other==id;
      }
    }
    if (tid<p.m) { ranks[tid]=rank; starts[tid]=start; counts[tid]=count; }
    expert_starts[tid]=eb; expert_counts[tid]=ec;
  }
  bool valid=__syncthreads_or(invalid)==0;
  if (!valid) {
    if (tid<p.m) { ranks[tid]=tid; starts[tid]=0; counts[tid]=0; }
    if (tid<32) { expert_starts[tid]=0; expert_counts[tid]=0; }
    __syncthreads();
  }
  moe_descriptors<Shape,Stride>(plan.gate,ids,ranks,starts,counts,expert_starts,expert_counts,valid);
  if (!plan.merged) moe_descriptors<Shape,Stride>(plan.up,ids,ranks,starts,counts,expert_starts,expert_counts,valid);
  moe_descriptors<Shape,Stride>(plan.down,ids,ranks,starts,counts,expert_starts,expert_counts,valid);
  if (!valid) return;
  int r=int(blockIdx.x)%p.m, to=ranks[r];
  int chunk=int(blockIdx.x)/p.m;
  int chunks=(int(gridDim.x)-1-r)/p.m+1;
  int64_t from=int64_t(r/p.io.topk)*p.io.a_token_stride+(r%p.io.topk%p.io.channels)*p.io.a_row_stride;
  for (int64_t col=int64_t(chunk)*blockDim.x+tid;col<p.k;col+=int64_t(chunks)*blockDim.x) {
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
