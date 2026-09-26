#pragma once
#include "io.cuh"

namespace quactlize::ppu17 {

template<class T>
__device__ __forceinline__ T cached_dequant(int code, Affine<T> const &a) {
  if constexpr(std::is_same_v<T,cutlass::half_t>) {
    auto product=__hmul_rn(T(code).to_half(),a.scale.to_half());
    return T(__hadd_rn(product,a.zero.to_half()));
  } else {
    auto product=__hmul_rn(T(code).to_nv_bfloat16(),a.scale.to_nv_bfloat16());
    return T(__hadd_rn(product,a.zero.to_nv_bfloat16()));
  }
}

// One N tile, four independent K32-group strips. Consume each packed b16
// exactly once per logical output row; retain typed affine metadata across
// its four slots. The original canonical word address and arithmetic stay
// owned by Format<12>. This is a decode-only alternative, never a fallback
// for unmeasured large M.
template<class T, bool SF, int LanesN, class Input, int Warps>
__global__ __launch_bounds__(Warps*32) void q4_simt_cached(q17_call_v1 c) {
  static_assert(LanesN==4 || LanesN==8 || LanesN==16);
  static_assert(Warps==4 || Warps==16);
  constexpr int lanes_k=32/LanesN, warps=Warps;
  auto rows=row_range(c,blockIdx.z);
  int row=blockIdx.y;
  if (row>=rows.size) return;
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  int ni=lane%LanesN, ki=lane/LanesN;
  int col=int(blockIdx.x)*LanesN+ni;
  WeightReader<12,T,SF> weight(c,blockIdx.z);
  auto activation=static_cast<Input const *>(c.a)+int64_t(rows.begin+row)*c.a_stride;
  using F=Format<12>;
  float sum=0;
  if (col<c.problem.n) {
    for (int group=warp; group<c.problem.k/32; group+=warps) {
      Affine<T> affine;
      if constexpr (SF)
        affine={weight.scale[size_t(group)*c.problem.n+col],
                weight.zero[size_t(group)*c.problem.n+col]};
      else
        affine=F::template affine<T>(weight.units,c.problem.n,col,group);
      #pragma unroll
      for (int w=ki; w<8; w+=lanes_k) {
        uint16_t packed=weight.low[size_t(group*8+w)*c.problem.n+col];
        #pragma unroll
        for (int slot=0; slot<4; ++slot) {
          int kk=group*32+w+slot*8;
          int code=int((packed>>(slot*4))&15)-8;
          // Preserve TWO typed roundings. Native MUL then ADD avoids the
          // unnecessary fp16/bf16 -> fp32 -> typed round trips in the loop.
          T b=cached_dequant(code,affine);
          T a;
          if constexpr(std::is_same_v<Input,T>) a=activation[kk];
          else a=T(float(activation[kk]));
          sum=fmaf(float(a),float(b),sum);
        }
      }
    }
  }
  #pragma unroll
  for (int delta=LanesN; delta<32; delta*=2)
    sum+=__shfl_xor_sync(0xffffffffu,sum,delta);
  __shared__ float partial[warps][LanesN];
  if (ki==0) partial[warp][ni]=sum;
  __syncthreads();
  if (threadIdx.x<LanesN && col<c.problem.n) {
    float result=0;
    #pragma unroll
    for(int w=0;w<warps;++w) result+=partial[w][ni];
    store_element(c.output,c.problem.output_type,
                  int64_t(rows.begin+row)*c.output_stride+col,result);
  }
}

template<class T, bool SF, int LanesN, int Warps=4>
inline void launch_q4_cached(q17_call_v1 const &c,dim3 grid,cudaStream_t stream) {
  if(c.problem.input_type==Q17_F32)
    q4_simt_cached<T,SF,LanesN,float,Warps><<<grid,Warps*32,0,stream>>>(c);
  else if(c.problem.input_type==Q17_BF16)
    q4_simt_cached<T,SF,LanesN,cutlass::bfloat16_t,Warps><<<grid,Warps*32,0,stream>>>(c);
  else
    q4_simt_cached<T,SF,LanesN,cutlass::half_t,Warps><<<grid,Warps*32,0,stream>>>(c);
}
} // namespace quactlize::ppu17
