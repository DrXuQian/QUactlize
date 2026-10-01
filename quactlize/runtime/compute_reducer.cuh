#pragma once
#include "../decode/reducer.cuh"
#include "cutlass/bfloat16.h"
#include <limits>

namespace quactlize::runtime {
template<int Splits>
__global__ void reduce_bf16_grouped(float const* partials,cutlass::bfloat16_t* output,int64_t count) {
  int64_t i=(int64_t(blockIdx.x)*32+int(threadIdx.x))*2;
  if(i>=count) return;
  float2 sum{0.f,0.f};
  CUTLASS_PRAGMA_UNROLL
  for(int s=0;s<Splits;++s) {
    float2 value=*reinterpret_cast<float2 const*>(partials+int64_t(s)*count+i);
    sum.x+=value.x;sum.y+=value.y;
  }
  cutlass::AlignedArray<cutlass::bfloat16_t,2,4> converted;
  converted[0]=cutlass::bfloat16_t(sum.x);converted[1]=cutlass::bfloat16_t(sum.y);
  __builtin_memcpy(output+i,&converted,sizeof(converted));
}

// Same contiguous [split,row,N] order as the admitted compact reducer. Only
// the final rounding changes; accumulation remains ordered F32 and the
// projection boundary is BF16. No extra staging buffer is introduced.
class Bf16CompactReduction {
 public:
  struct Arguments {
    int64_t rows, columns;
    int split_k_slices;
    float const* workspace;
    uint64_t workspace_bytes;
    cutlass::bfloat16_t* destination;
    int64_t destination_stride;
  };
  cutlass::Status initialize(Arguments const& a) {
    ready_=false;
    if(a.rows<=0 || a.columns<=0 || a.columns%64 ||
        a.destination_stride!=a.columns ||
        (a.split_k_slices!=2 && a.split_k_slices!=4 && a.split_k_slices!=8) ||
        a.rows>INT64_MAX/a.columns/a.split_k_slices/sizeof(float) ||
        uint64_t(a.rows)*a.columns/64>INT32_MAX ||
        !a.workspace || !a.destination || (uintptr_t(a.workspace)|uintptr_t(a.destination))%16 ||
        a.workspace_bytes<uint64_t(a.rows)*a.columns*a.split_k_slices*sizeof(float))
      return cutlass::Status::kErrorInvalidProblem;
    args_=a;ready_=true;return cutlass::Status::kSuccess;
  }
  cutlass::Status run(hggcStream_t stream) {
    if(!ready_) return cutlass::Status::kErrorInvalidProblem;
    int64_t count=args_.rows*args_.columns;
    if(hggcGetLastError()!=hggcSuccess) return cutlass::Status::kErrorInternal;
#define QK_BF16_REDUCE(S) case S: reduce_bf16_grouped<S><<<unsigned(count/64),32,0,stream>>>(args_.workspace,args_.destination,count);break
    switch(args_.split_k_slices) {QK_BF16_REDUCE(2);QK_BF16_REDUCE(4);QK_BF16_REDUCE(8);}
#undef QK_BF16_REDUCE
    return hggcGetLastError()==hggcSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
  }
 private:
  Arguments args_{};
  bool ready_=false;
};
} // namespace quactlize::runtime
