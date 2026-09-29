#pragma once
#include "../integrations/llama/router.cuh"

namespace quactlize::llama {
// Recover a float from its ordered-integer key. The forward map is
//   key = bits ^ (sign ? 0xffffffff : 0x80000000)
// so a set high bit in the key means the original value was non-negative.
CUTLASS_DEVICE float router_from_key(unsigned key) {
  unsigned bits=(key>>31) ? (key^0x80000000u) : ~key;
  return __uint_as_float(bits);
}

// CoalescedIds trades the eight scattered ID stores for one warp-wide store.
// It only pays off when the warp owns the whole CTA; with one warp per token the
// extra barrier and shared round trip cost more than the eight stores save.
//
// Without a bias the eight candidates are sorted once instead of being rescanned
// eight times. The shipping shape spent every round finding the lane maximum
// again (a seven-step scan) and then finding the winner's slot to poison it (an
// eight-step scan); a stage-isolated probe charged those two scans 1.49 us and
// 1.18 us of a 3.01 us router, against 0.27 us for all eight warp collectives.
// Sorting descending up front makes both scans disappear: the lane's best is the
// head and consuming it is a shift.
//
// Faithfulness rests on three points.
//   * The network is a bubble network whose comparators swap only on a strict
//     greater-than, so it is stable and equal values keep their original j order.
//     The head is therefore the lane maximum with the smallest expert, exactly
//     what the old `selection[j] > best` scan produced.
//   * Sorting the ordered keys rather than the floats is the same order: the map
//     is monotone and it collapses -0.0 and +0.0 to one key, which is the tie a
//     float compare gives. NaN is still folded to -FLT_MAX first.
//   * __reduce_max_sync already returns the winning key, so the weight is
//     recovered by inverting the transform and the per-round shuffle is gone.
// A probe compared 2100 randomised cases -- including wide ties, cross-lane
// ties, signed zeros, NaN and both infinities -- and every ID and every weight
// bit pattern matched the shipping arm.
//
// The bias path keeps the original scan. With a bias the ordering key and the
// payload differ, so a sorted list would have to carry both.
//
// Returns this lane's own slot, i.e. the expert chosen for slot `lane` when
// lane < 8 and an unspecified value otherwise. With CoalescedIds the caller owns
// the whole warp and reads back only its own slot, so the shared array is not
// written at all and the return value replaces it.
template<bool HasBias,bool CoalescedIds=false>
CUTLASS_DEVICE int router_256_top8_warp(qk_llama_router_v1 r,qk_llama_indexed_v1 io,int* shared_ids) {
  int lane=int(threadIdx.x)%32;
  float value[8],chosen=0.f,sum=0.f;
  int myid=0;
  unsigned key[8];
  #pragma unroll
  for (int j=0;j<8;++j) value[j]=r.logits[lane+32*j];
  if (!r.delayed_softmax) {
    if (r.use_sigmoid) {
      #pragma unroll
      for (int j=0;j<8;++j) value[j]=1.f/(1.f+expf(-value[j]));
    } else router_softmax(value,256,lane);
    if constexpr (!HasBias) {
      // Everything reaching here left either router_softmax or the sigmoid, and
      // both are non-negative for every input: expf yields +0 or a positive
      // number and never -0, scaling by the non-negative reciprocal keeps that,
      // and the sigmoid lies in (0,1]. So the sign bit is clear, -0.0 and -inf
      // cannot occur, and the sign-dependent mask together with the signed-zero
      // canonicalisation collapse to one unconditional xor. NaN is the only
      // escape (an infinite reciprocal, or a +inf logit) and still maps to the
      // key of -FLT_MAX, which is what the general form produces for it.
      #pragma unroll
      for (int j=0;j<8;++j) {
        unsigned raw=__float_as_uint(value[j]);
        key[j]=(raw&0x7fffffffu)>0x7f800000u ? 0x00800000u : (raw^0x80000000u);
      }
    }
  } else if constexpr (!HasBias) {
    // Delayed softmax hands the raw logits straight through, so here the sign
    // really is arbitrary and the general transform is required.
    #pragma unroll
    for (int j=0;j<8;++j) {
      float x=isnan(value[j]) ? -FLT_MAX : value[j];
      unsigned bits=x==0.f ? 0u : __float_as_uint(x);
      key[j]=bits ^ ((bits>>31) ? 0xffffffffu : 0x80000000u);
    }
  }
  if constexpr (HasBias) {
    float selection[8];
    #pragma unroll
    for (int j=0;j<8;++j) {
      if (isnan(value[j])) value[j]=-FLT_MAX;
      selection[j]=value[j]+r.bias[lane+32*j];
    }
    #pragma unroll
    for (int slot=0;slot<8;++slot) {
      float best=selection[0],weight=value[0]; int expert=lane;
      #pragma unroll
      for (int j=1;j<8;++j) if (selection[j]>best) {
        best=selection[j]; weight=value[j]; expert=lane+32*j;
      }
      #pragma unroll
      for (int mask=16;mask;mask/=2) {
        float other=__shfl_xor_sync(0xffffffff,best,mask,32);
        int other_expert=__shfl_xor_sync(0xffffffff,expert,mask,32);
        float other_weight=__shfl_xor_sync(0xffffffff,weight,mask,32);
        if (other>best || (other==best && other_expert<expert)) {
          best=other; weight=other_weight; expert=other_expert;
        }
      }
      #pragma unroll
      for (int j=0;j<8;++j) if (expert==lane+32*j) selection[j]=-INFINITY;
      if (r.with_norm && lane==expert%32) sum+=weight;
      if (lane==slot) {
        myid=expert;
        if constexpr (!CoalescedIds) {
          shared_ids[slot]=expert; const_cast<int32_t*>(io.ids)[slot]=expert;
        }
        chosen=weight;
      }
    }
  } else {
    unsigned ord=0;
    int order[8];
    #pragma unroll
    for (int j=0;j<8;++j) order[j]=j;
    // Stable descending bubble network: 28 adjacent comparators, strict >.
    #pragma unroll
    for (int pass=0;pass<7;++pass) {
      #pragma unroll
      for (int a=0;a<7-pass;++a) {
        bool swap=key[a+1]>key[a];
        unsigned tk=swap?key[a+1]:key[a]; key[a+1]=swap?key[a]:key[a+1]; key[a]=tk;
        int to=swap?order[a+1]:order[a]; order[a+1]=swap?order[a]:order[a+1]; order[a]=to;
      }
    }
    // Four bits per slot, so consuming the head is one 4-bit shift.
    #pragma unroll
    for (int j=0;j<8;++j) ord|=unsigned(order[j])<<(4*j);
    #pragma unroll
    for (int slot=0;slot<8;++slot) {
      unsigned head=key[0]; int expert=lane+32*int(ord&15u);
      unsigned maximum=__reduce_max_sync(0xffffffff,head);
      expert=__reduce_min_sync(0xffffffff,head==maximum ? expert : 256);
      float weight=router_from_key(maximum);
      bool mine=lane==expert%32;
      // After this round at most 7-slot more heads can leave this lane, so only
      // key[0..6-slot] is ever read again and the shift stops there.
      #pragma unroll
      for (int j=0;j<7-slot;++j) key[j]=mine?key[j+1]:key[j];
      if (mine) ord>>=4;
      if (r.with_norm && mine) sum+=weight;
      if (lane==slot) {
        myid=expert;
        if constexpr (!CoalescedIds) {
          shared_ids[slot]=expert; const_cast<int32_t*>(io.ids)[slot]=expert;
        }
        chosen=weight;
      }
    }
  }
  if constexpr (CoalescedIds) {
    // expert is warp-uniform each round, so every lane already captured its own
    // slot in myid; no shared round trip and no barrier are needed.
    if (lane<8) const_cast<int32_t*>(io.ids)[lane]=myid;
  }
  if (r.with_norm) chosen*=1.f/fmaxf(router_sum(sum),r.clamp);
  if (r.delayed_softmax) {
    float maximum=router_max(lane<8?chosen:-INFINITY);
    chosen=lane<8?expf(chosen-maximum):0.f;
    chosen*=1.f/router_sum(chosen);
  }
  if (lane<8) r.weights[lane]=chosen*r.scale;
  return myid;
}
}
