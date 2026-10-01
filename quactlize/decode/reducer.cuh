#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include <type_traits>

namespace quactlize::decode {
template<int Splits,class Output>
__global__ void reduce_decode(float const* partials,Output* output,int count) {
  int i=(int(blockIdx.x)*32+int(threadIdx.x))*2;
  if (i>=count) return;
  float2 sum{0.f,0.f};
  CUTLASS_PRAGMA_UNROLL
  for (int s=0;s<Splits;++s) {
    float2 v=*reinterpret_cast<float2 const*>(partials+int64_t(s)*count+i);
    sum.x+=v.x;sum.y+=v.y;
  }
  if constexpr(std::is_same_v<Output,float>) *reinterpret_cast<float2*>(output+i)=sum;
  else {
    cutlass::AlignedArray<Output,2,4> value;
    value[0]=Output(sum.x);value[1]=Output(sum.y);
    __builtin_memcpy(output+i,&value,sizeof(value));
  }
}
} // namespace quactlize::decode
