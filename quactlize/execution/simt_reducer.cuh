#pragma once
#include "cutlass/cutlass.h"

// Keep the measured symbol stable. This row-major partial layout belongs to
// SIMT execution, not the TC split-major reducer or its JIT source contract.
namespace quactlize::decode {
// One lane owns an aligned pair within one row. Add splits in their original
// order and never vectorize across row padding or an expert output boundary.
template<int Splits>
__global__ void reduce_decode_rows(float const* partials,float* output,int rows,int n,int64_t stride) {
  static_assert(Splits==2 || Splits==4 || Splits==8);
  int64_t pair=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(pair>=int64_t(rows)*(n/2)) return;
  int64_t row=pair/(n/2),col=(pair%(n/2))*2;
  float2 sum{0.f,0.f};
  CUTLASS_PRAGMA_UNROLL
  for(int s=0;s<Splits;++s) {
    float2 v=*reinterpret_cast<float2 const*>(partials+(row*Splits+s)*n+col);
    sum.x+=v.x;sum.y+=v.y;
  }
  *reinterpret_cast<float2*>(output+row*stride+col)=sum;
}
} // namespace quactlize::decode
