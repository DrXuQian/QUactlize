#pragma once
#include "../integrations/llama/router.cuh"

namespace quactlize::llama {
template<bool HasBias>
CUTLASS_DEVICE void router_256_top8_warp(qk_llama_router_v1 r,qk_llama_indexed_v1 io,int* shared_ids) {
  int lane=int(threadIdx.x)%32;
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
}

