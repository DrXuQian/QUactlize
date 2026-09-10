#pragma once
#include "indexed.h"
#include <cfloat>
#include <cmath>

namespace quactlize::llama {
// Reduction/order and tie semantics match llama.cpp's MIT-licensed
// ggml-cuda/topk-moe.cu. Each CTA derives its own IDs: no inter-CTA readback.
CUTLASS_DEVICE float router_sum(float value) {
  for (int mask=16;mask;mask/=2) value+=__shfl_xor_sync(0xffffffff,value,mask,32);
  return value;
}
CUTLASS_DEVICE float router_max(float value) {
  for (int mask=16;mask;mask/=2) value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,mask,32));
  return value;
}
CUTLASS_DEVICE void router_softmax(float (&v)[8],int limit,int lane) {
  float maximum=-INFINITY,sum=0.f;
  for (int j=0;j<8;++j) if (lane+32*j<limit) maximum=fmaxf(maximum,v[j]);
  maximum=router_max(maximum);
  for (int j=0;j<8;++j) {
    v[j]=lane+32*j<limit ? expf(v[j]-maximum) : 0.f;
    sum+=v[j];
  }
  float inv=1.f/router_sum(sum);
  for (int j=0;j<8;++j) if (lane+32*j<limit) v[j]*=inv;
}
CUTLASS_DEVICE void router_256(qk_llama_router_v1 r,qk_llama_indexed_v1 io,
    int* shared_ids,bool publish) {
  int lane=int(threadIdx.x)%32,warp=int(threadIdx.x)/32;
  // Whole warps take this branch, and every thread reaches the caller's CTA
  // barrier even when tokens<8. No token-tail return before __syncthreads.
  for (int token=warp;token<io.tokens;token+=int(blockDim.x)/32) {
    float value[8],selection[8],chosen[8]={};
    for (int j=0;j<8;++j) value[j]=r.logits[int64_t(token)*256+lane+32*j];
    if (!r.delayed_softmax) {
      if (r.use_sigmoid) for (int j=0;j<8;++j) value[j]=1.f/(1.f+expf(-value[j]));
      else router_softmax(value,256,lane);
    }
    for (int j=0;j<8;++j) {
      if (isnan(value[j])) value[j]=-FLT_MAX;
      selection[j]=value[j]+(r.bias?r.bias[lane+32*j]:0.f);
    }
    float sum=0.f;
    for (int slot=0;slot<io.topk;++slot) {
      float best=selection[0],weight=value[0]; int expert=lane;
      for (int j=1;j<8;++j) if (selection[j]>best) {
        best=selection[j]; weight=value[j]; expert=lane+32*j;
      }
      for (int mask=16;mask;mask/=2) {
        float other=__shfl_xor_sync(0xffffffff,best,mask,32);
        float other_weight=__shfl_xor_sync(0xffffffff,weight,mask,32);
        int other_expert=__shfl_xor_sync(0xffffffff,expert,mask,32);
        if (other>best || (other==best && other_expert<expert)) {
          best=other; weight=other_weight; expert=other_expert;
        }
      }
      if (lane==expert%32) {
        selection[expert/32]=-INFINITY;
        if (r.with_norm) sum+=weight;
      }
      if (lane==slot%32) {
        shared_ids[token*io.topk+slot]=expert; chosen[slot/32]=weight;
        if (publish) const_cast<int32_t*>(io.ids)[int64_t(token)*io.ids_stride+slot]=expert;
      }
    }
    if (r.with_norm) {
      float inv=1.f/fmaxf(router_sum(sum),r.clamp);
      for (int j=0;j<8;++j) chosen[j]*=inv;
    }
    if (r.delayed_softmax) router_softmax(chosen,io.topk,lane);
    if (publish) for (int j=0;j<8;++j) if (lane+32*j<io.topk)
      r.weights[int64_t(token)*io.topk+lane+32*j]=chosen[j]*r.scale;
  }
}

// Single-token top-8 keeps selected slots in registers. Dynamic indexing of
// the per-lane arrays in the general router otherwise requires local storage.
template<bool HasBias>
CUTLASS_DEVICE void router_256_top8(qk_llama_router_v1 r,qk_llama_indexed_v1 io,int* shared_ids) {
  if (threadIdx.x>=32) return;
  int lane=int(threadIdx.x);
  float value[8],selection[8],chosen=0.f,sum=0.f;
  #pragma unroll
  for (int j=0;j<8;++j) value[j]=r.logits[lane+32*j];
  if (!r.delayed_softmax) {
    if (r.use_sigmoid) {
      #pragma unroll
      for (int j=0;j<8;++j) value[j]=1.f/(1.f+expf(-value[j]));
    } else router_softmax(value,256,lane);
  }
  #pragma unroll
  for (int j=0;j<8;++j) {
    if (isnan(value[j])) value[j]=-FLT_MAX;
    selection[j]=value[j];
    if constexpr (HasBias) selection[j]+=r.bias[lane+32*j];
  }
  #pragma unroll
  for (int slot=0;slot<8;++slot) {
    float best=selection[0],weight=value[0]; int expert=lane;
    #pragma unroll
    for (int j=1;j<8;++j) if (selection[j]>best) {
      best=selection[j]; weight=value[j]; expert=lane+32*j;
    }
    if constexpr (!HasBias) {
      // Values were sanitized above. Ordered integer keys enable native warp
      // reductions; canonicalize signed zero to retain floating-point ties.
      unsigned bits=best==0.f ? 0u : __float_as_uint(best);
      unsigned key=bits ^ ((bits>>31) ? 0xffffffffu : 0x80000000u);
      unsigned maximum=__reduce_max_sync(0xffffffff,key);
      expert=__reduce_min_sync(0xffffffff,key==maximum ? expert : 256);
      weight=__shfl_sync(0xffffffff,weight,expert%32,32);
    } else {
      #pragma unroll
      for (int mask=16;mask;mask/=2) {
        float other=__shfl_xor_sync(0xffffffff,best,mask,32);
        int other_expert=__shfl_xor_sync(0xffffffff,expert,mask,32);
        float other_weight=__shfl_xor_sync(0xffffffff,weight,mask,32);
        if (other>best || (other==best && other_expert<expert)) {
          best=other; weight=other_weight; expert=other_expert;
        }
      }
    }
    #pragma unroll
    for (int j=0;j<8;++j) if (expert==lane+32*j) selection[j]=-INFINITY;
    if (r.with_norm && lane==expert%32) sum+=weight;
    if (lane==slot) {
      shared_ids[slot]=expert;
      const_cast<int32_t*>(io.ids)[slot]=expert;
      chosen=weight;
    }
  }
  if (r.with_norm) chosen*=1.f/fmaxf(router_sum(sum),r.clamp);
  if (r.delayed_softmax) {
    float maximum=router_max(lane<8?chosen:-INFINITY);
    chosen=lane<8?expf(chosen-maximum):0.f;
    chosen*=1.f/router_sum(chosen);
  }
  if (lane<8) r.weights[lane]=chosen*r.scale;
}
} // namespace quactlize::llama
