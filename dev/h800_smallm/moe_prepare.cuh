// CUDA experiment only. Includes the generated, production-derived
// router_top8_warp and prepare_fast_router before this file.
namespace quactlize::runtime {

template<class Shape,class Stride>
CUTLASS_DEVICE void small_projection(qk_moe_projection_v1 const& p,
    int const* ids,int const* ranks,int const* starts,int const* counts,
    int begin,int count,bool valid) {
  int tid=int(threadIdx.x);
  p.offsets[tid]=begin; p.rows[tid]=count;
  static_cast<Shape*>(p.shapes)[tid]=cute::make_shape(count,p.n,p.k);
  for (int s=0;s<p.splits;++s) {
    int entry=tid+s*256;
    int64_t offset=(int64_t(s)*p.m+begin)*p.n;
    if (p.splits==1) static_cast<Half**>(p.outputs)[entry]=static_cast<Half*>(p.output)+offset;
    else static_cast<float**>(p.outputs)[entry]=static_cast<float*>(p.partials)+offset;
    static_cast<Stride*>(p.strides)[entry]=
        cutlass::make_cute_packed_stride(Stride{},cute::make_shape(count,p.n,1));
  }
  if (tid<32) {
    int id=tid<p.m ? ids[tid] : p.experts;
    int tiles=valid && tid<p.m && starts[tid]==ranks[tid] ? (counts[tid]+p.tile_m-1)/p.tile_m : 0;
    int first=0,total=0;
    #pragma unroll
    for (int j=0;j<32;++j) {
      int other=__shfl_sync(0xffffffff,id,j);
      int blocks=__shfl_sync(0xffffffff,tiles,j);
      first+=other<id ? blocks : 0; total+=blocks;
    }
    if (tid<p.m) {
      p.io.row_ids[ranks[tid]]=tid;
      auto entry=moe::make_entry(id,counts[tid],first,starts[tid]);
      for (int b=0;b<tiles;++b)
        static_cast<moe::BlockEntry*>(p.directory_entries)[first+b]=entry;
    }
    if (!tid) {
      p.offsets[256]=valid?p.m:0;
      *static_cast<moe::Header*>(p.directory_header)={total,
          valid?0:int(moe::BuildStatus::InvalidArgument),p.tile_m,256};
    }
  }
}

// The bounded chain has at most four top-8 tokens. One warp per token
// computes routing exactly once. No global flag, host readback or weight load.
template<class Shape,class Stride,bool Wide=false>
__global__ void prepare_once(qk_moe_plan_v1 plan) {
  auto const& p=plan.gate;
  __shared__ int ids[32],ranks[32],starts[32],counts[32];
  int tid=int(threadIdx.x),lane=tid%32,warp=tid/32;
  if (plan.router.version) {
    if (warp<p.io.tokens) {
      auto router=plan.router; auto io=p.io;
      router.logits+=int64_t(warp)*256; router.weights+=warp*8;
      io.ids+=int64_t(warp)*io.ids_stride;
      if (router.bias) quactlize::llama::router_top8_warp<true>(router,io,ids+warp*8,true);
      else quactlize::llama::router_top8_warp<false>(router,io,ids+warp*8,true);
    }
  } else if (tid<p.m) ids[tid]=p.io.ids[int64_t(tid/8)*p.io.ids_stride+tid%8];
  __syncthreads();
  bool invalid=tid<p.m && (ids[tid]<0 || ids[tid]>=256);
  int begin=0,count=0,rank=0,start=0,matches=0;
  #pragma unroll
  for (int j=0;j<32;++j) if (j<p.m) {
    int other=ids[j];
    begin+=other<tid; count+=other==tid;
    if (tid<p.m) {
      start+=other<ids[tid]; matches+=other==ids[tid];
      rank+=other<ids[tid] || (other==ids[tid] && j<tid);
      invalid|=j>=tid/8*8 && j<tid && other==ids[tid];
    }
  }
  bool valid=__syncthreads_or(invalid)==0;
  if (tid<p.m) {
    ranks[tid]=valid?rank:tid; starts[tid]=valid?start:0; counts[tid]=valid?matches:0;
  }
  if (!valid) begin=count=0;
  __syncthreads();
  small_projection<Shape,Stride>(plan.gate,ids,ranks,starts,counts,begin,count,valid);
  if (!plan.merged) small_projection<Shape,Stride>(plan.up,ids,ranks,starts,counts,begin,count,valid);
  small_projection<Shape,Stride>(plan.down,ids,ranks,starts,counts,begin,count,valid);
  if (!valid) return;
  // Distinct token activations; eight selected experts share only their own
  // token's vector. Scalar half stores intentionally handle weak alignment.
  int token=Wide?warp%p.io.tokens:warp;
  int first=Wide?(warp/p.io.tokens)*32+lane:lane;
  int step=Wide?((7-token)/p.io.tokens+1)*32:32;
  if (token<p.io.tokens) {
    for (int col=first;col<p.k;col+=step) {
      Half value=Half(p.io.a[int64_t(token)*p.io.a_token_stride+col]);
      #pragma unroll
      for (int slot=0;slot<8;++slot) {
        int row=ranks[token*8+slot];
        static_cast<Half*>(p.a)[int64_t(row)*p.k+col]=value;
        if (!plan.merged) static_cast<Half*>(plan.up.a)[int64_t(row)*p.k+col]=value;
      }
    }
  }
}
} // namespace quactlize::runtime
