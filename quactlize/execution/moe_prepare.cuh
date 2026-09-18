#pragma once
#include "../runtime/moe_chain.cuh"
#include "moe_router_warp.cuh"

// Decode preparation with one router evaluation per token.
namespace quactlize::runtime::prepare_detail {

// Reuse the exact, register-resident shipping top8 arithmetic for each token.
// A warp has private logits/weights/ID views; no CTA recomputes another's router.
template<bool HasBias,bool CoalescedIds>
CUTLASS_DEVICE int token_router(qk_moe_plan_v1 const& plan,int token,int* ids) {
  auto router=plan.router;
  auto io=plan.gate.io;
  router.logits+=int64_t(token)*256;
  router.weights+=token*8;
  io.ids+=int64_t(token)*io.ids_stride;
  return quactlize::llama::router_256_top8_warp<HasBias,CoalescedIds>(router,io,ids+token*8);
}

template<class Compute>
CUTLASS_DEVICE uint32_t pair(float x,float y) {
  return uint32_t(Compute(x).raw())|(uint32_t(Compute(y).raw())<<16);
}

template<class Compute>
CUTLASS_DEVICE uint4 pack8(float const* a) {
  float4 lo=*reinterpret_cast<float4 const*>(a);
  float4 hi=*reinterpret_cast<float4 const*>(a+4);
  return make_uint4(pair<Compute>(lo.x,lo.y),pair<Compute>(lo.z,lo.w),
                    pair<Compute>(hi.x,hi.y),pair<Compute>(hi.z,hi.w));
}

// F32 input is converted once per token, then replicated to selected TC rows.
// Aligned production buffers get 128-bit stores; guarded/strided inputs keep
// the scalar path. The offline weight planes are neither read nor modified.
template<class Plan>
CUTLASS_DEVICE void gather(Plan const& plan,int const* ranks) {
  using Compute=typename MoeCompute<Plan>::type;
  auto const& p=plan.gate;
  uint32_t mask=moe_simt_mask(plan);
  bool gate_tc=!(mask&1),up_tc=!plan.merged && !(mask&2);
  if (!gate_tc && !up_tc) return;
  int lane=int(threadIdx.x)%32,warp=int(threadIdx.x)/32;
  int token=warp%p.io.tokens,stripe=warp/p.io.tokens;
  int warps=(int(blockDim.x)/32-1-token)/p.io.tokens+1;
  float const* source=p.io.a+int64_t(token)*p.io.a_token_stride;
  bool vector=p.k%8==0 && !(uintptr_t(source)&15) &&
      (!gate_tc || !(uintptr_t(p.a)&15)) && (!up_tc || !(uintptr_t(plan.up.a)&15));
  if (vector) {
    for (int col=(stripe*32+lane)*8;col<p.k;col+=warps*32*8) {
      uint4 value=pack8<Compute>(source+col);
      #pragma unroll
      for (int slot=0;slot<8;++slot) {
        int row=ranks[token*8+slot];
        if (gate_tc) *reinterpret_cast<uint4*>(static_cast<Compute*>(p.a)+int64_t(row)*p.k+col)=value;
        if (up_tc) *reinterpret_cast<uint4*>(static_cast<Compute*>(plan.up.a)+int64_t(row)*p.k+col)=value;
      }
    }
  } else {
    for (int col=stripe*32+lane;col<p.k;col+=warps*32) {
      Compute value=Compute(source[col]);
      #pragma unroll
      for (int slot=0;slot<8;++slot) {
        int row=ranks[token*8+slot];
        if (gate_tc) static_cast<Compute*>(p.a)[int64_t(row)*p.k+col]=value;
        if (up_tc) static_cast<Compute*>(plan.up.a)[int64_t(row)*p.k+col]=value;
      }
    }
  }
}

template<class Shape,class Stride,class Compute,bool AllSimt>
CUTLASS_DEVICE void descriptors(qk_moe_projection_v1 const& p,
    int id,int rank,int start,int count,int first,int total,int expert_begin,
    int expert_count,bool valid,bool simt) {
  int tid=int(threadIdx.x);
  if constexpr(!AllSimt) {
    if (!simt) {
      p.offsets[tid]=valid?expert_begin:0;
      p.rows[tid]=valid?expert_count:0;
      static_cast<Shape*>(p.shapes)[tid]=cute::make_shape(valid?expert_count:0,p.n,p.k);
      for (int s=0;s<p.splits;++s) {
        int entry=s*256+tid;
        int64_t offset=(int64_t(s)*p.m+(valid?expert_begin:0))*p.n;
        if (p.splits==1) static_cast<Compute**>(p.outputs)[entry]=static_cast<Compute*>(p.output)+offset;
        else static_cast<float**>(p.outputs)[entry]=static_cast<float*>(p.partials)+offset;
        static_cast<Stride*>(p.strides)[entry]=
            cutlass::make_cute_packed_stride(Stride{},cute::make_shape(valid?expert_count:0,p.n,1));
      }
      if (!tid) p.offsets[256]=valid?p.m:0;
    }
  }
  if (tid<p.m) {
    p.io.row_ids[valid?rank:tid]=tid;
    if (valid && rank==start)
      static_cast<moe::BlockEntry*>(p.directory_entries)[first]=moe::make_entry(id,count,first,start);
  }
  if (!tid) *static_cast<moe::Header*>(p.directory_header)=
      {valid?total:0,valid?0:int(moe::BuildStatus::InvalidArgument),p.tile_m,256};
}

// At most eight top8 tokens, and TileM >= tokens: every nonempty expert has
// exactly one M tile in every projection. Sort/prefix once, share the result.
// All-SIMT has no expert-sized descriptor arrays and needs one warp/token.
template<class Shape,class Stride,int Capacity,bool AllSimt,class Plan>
__global__ void once(Plan plan) {
  using Compute=typename MoeCompute<Plan>::type;
  auto const& p=plan.gate;
  __shared__ int ids[Capacity],ranks[Capacity],starts[Capacity];
  int tid=int(threadIdx.x),warp=tid/32;
  // Capacity 8 is only launched for the admitted tokens==1 all-SIMT case, whose
  // CTA is exactly one warp. Cross-lane agreement is then whatever the warp
  // collectives already give, and the router's ID publication can be a single
  // warp-wide store.
  constexpr bool OneWarp=AllSimt && Capacity==8;
  // With one warp per CTA the only later read of `ids` is lane `tid` fetching
  // slot `tid`, which is the slot that same lane just published, so the round
  // trip through shared memory moves nothing between lanes. Keep it in a
  // register instead: the eight stores, the eight loads and the barrier all go
  // away, and nothing after the router reads another lane's slot.
  //
  // A router without bias consumes one distinct (lane,slot) head per round: the
  // winner is the smallest expert among the lanes holding the round's maximum
  // key, and exactly that lane shifts its head off. So the eight IDs are eight
  // distinct `lane+32*j` in [0,256) whatever the logits are -- the duplicate
  // scan below is vacuous, and then none of the all-SIMT outputs depend on the
  // router at all. Issue them first: the stores retire underneath the router's
  // own latency, and the kernel's last dependent instruction becomes the
  // router's weight store rather than two warp collectives and a header write
  // chained behind it. The bias path keeps the scan, where duplicates really
  // are reachable: once a lane's remaining selections are all -INFINITY its
  // local argmax falls back to `expert=lane`, a slot it may already have spent.
  //
  // One warp per CTA is excluded on purpose. There the scan is already just the
  // __match_any_sync above, so skipping it buys almost nothing, while the second
  // copy of the publish and the runtime test cost 760 bytes and a measured
  // 0.15us. The multi-warp capacities pay a block-wide __syncthreads_or plus an
  // eight-step shared rescan instead, and dropping that is worth 0.75us.
  bool const proven=AllSimt && !OneWarp && plan.router.version && !plan.router.bias;
  auto publish=[&](bool valid) {
    auto one=[&](qk_moe_projection_v1 const& v) {
      if(tid<p.m) v.io.row_ids[tid]=tid;
      if(!tid) *static_cast<moe::Header*>(v.directory_header)=
          {0,valid?0:int(moe::BuildStatus::InvalidArgument),v.tile_m,256};
    };
    one(plan.gate);if(!plan.merged)one(plan.up);one(plan.down);
  };
  if constexpr(AllSimt) if (proven) publish(true);
  int mine=0;
  if (plan.router.version) {
    if (warp<p.io.tokens) {
      if (plan.router.bias) mine=token_router<true,OneWarp>(plan,warp,ids);
      else mine=token_router<false,OneWarp>(plan,warp,ids);
    }
  } else if (tid<p.m) {
    int id=p.io.ids[int64_t(tid/8)*p.io.ids_stride+tid%8];
    if constexpr(OneWarp) mine=id; else ids[tid]=id;
  }
  if constexpr(!OneWarp) if (!proven) __syncthreads();
  if constexpr(AllSimt) {
    // No TC endpoint consumes expert-sorted matrices or a block directory.
    // Keep every private activation in the original [token,slot] order;
    // the existing SIMT producers, SwiGLU and weighted finish agree on it.
    // Identity maps preserve the stage ABI without sorting unused metadata.
    if (proven) return;
    bool invalid=false,valid;
    if constexpr(OneWarp) {
      // Distinct sentinels on the unused lanes let one match cover all eight
      // slots, replacing the eight-step shared rescan of the duplicate check.
      int id=tid<p.m?mine:-1-tid;
      unsigned same=__match_any_sync(0xffffffff,id);
      if(tid<p.m) invalid=id<0 || id>=256 || __popc(same)>1;
      valid=__any_sync(0xffffffff,invalid)==0;
    } else {
      if(tid<p.m) {
        int id=ids[tid];invalid=id<0 || id>=256;
        #pragma unroll
        for(int s=0;s<8;++s) invalid|=s<tid%8 && ids[tid/8*8+s]==id;
      }
      valid=__syncthreads_or(invalid)==0;
    }
    publish(valid);
    return;
  }
  int id=tid<p.m?ids[tid]:256;
  int start=0,count=0,rank=0,eb=0,ec=0;
  bool invalid=tid<p.m && (id<0 || id>=256);
  #pragma unroll
  for (int j=0;j<Capacity;++j) if (j<p.m) {
    int other=ids[j];
    if (tid<p.m) {
      start+=other<id;count+=other==id;
      rank+=other<id || (other==id && j<tid);
      invalid|=j>=tid/8*8 && j<tid && other==id;
    }
    if constexpr(!AllSimt) {eb+=other<tid;ec+=other==tid;}
  }
  if (tid<p.m) {ranks[tid]=rank;starts[tid]=start;}
  bool valid=__syncthreads_or(invalid)==0;
  int first=0,total=0;
  if (tid<p.m) {
    #pragma unroll
    for (int j=0;j<Capacity;++j) if (j<p.m) {
      int head=ranks[j]==starts[j];
      first+=(ids[j]<id && head);total+=head;
    }
  }
  uint32_t simt=moe_simt_mask(plan);
  descriptors<Shape,Stride,Compute,AllSimt>(plan.gate,id,rank,start,count,first,total,eb,ec,valid,simt&1);
  if (!plan.merged) descriptors<Shape,Stride,Compute,AllSimt>(plan.up,id,rank,start,count,first,total,eb,ec,valid,simt&2);
  descriptors<Shape,Stride,Compute,AllSimt>(plan.down,id,rank,start,count,first,total,eb,ec,valid,simt&4);
  if constexpr(!AllSimt) if (valid) gather(plan,ranks);
}

template<class Plan>
CUTLASS_HOST_DEVICE bool supported(Plan const& plan) {
  auto const& p=plan.gate;
  int tokens=p.io.tokens;
  return tokens>=1 && tokens<=8 && p.experts==256 && p.io.topk==8 &&
      p.m==8*tokens && p.io.channels==1 && p.tile_m>=tokens &&
      plan.down.tile_m>=tokens && (plan.merged || plan.up.tile_m>=tokens);
}

// Measured merged/all-SIMT domain. TC and mixed plans retain their incumbent.
template<class Plan>
CUTLASS_HOST_DEVICE bool admitted(Plan const& plan) {
  int tokens=plan.gate.io.tokens;
  auto const& r=plan.router;
  return supported(plan) && plan.merged && moe_simt_mask(plan)==5 &&
      (tokens==1 || tokens==2 || tokens==4 || tokens==8) &&
      (plan.gate.k==512 || plan.gate.k==2048) &&
      plan.gate.n==1024 && plan.down.n==2048 && plan.down.k==512 &&
      r.version==1 && !r.use_sigmoid && r.with_norm && !r.delayed_softmax && !r.bias;
}

template<class Shape,class Stride,class Plan>
void launch(Plan const& plan,hggcStream_t stream) {
  if(admitted(plan)) {
    int tokens=plan.gate.io.tokens;
    if(tokens==1) once<Shape,Stride,8,true><<<1,32,0,stream>>>(plan);
    else if(tokens<=4) once<Shape,Stride,32,true><<<1,tokens*32,0,stream>>>(plan);
    else once<Shape,Stride,64,true><<<1,256,0,stream>>>(plan);
  } else if(moe_prepare_m1_supported(plan))
    moe_chain_prepare_m1<Shape,Stride><<<1,256,0,stream>>>(plan);
  else if(plan.gate.m>32)
    moe_chain_prepare<Shape,Stride,64><<<moe_prepare_blocks(256,plan.gate.m),256,0,stream>>>(plan);
  else moe_chain_prepare<Shape,Stride><<<moe_prepare_blocks(256,plan.gate.m),256,0,stream>>>(plan);
}
} // namespace quactlize::runtime::prepare_detail
