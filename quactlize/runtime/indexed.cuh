#pragma once
#include "indexed_rows.hpp"
#include "../integrations/llama/indexed.h"

namespace quactlize::runtime {

template<int TileM,class Shape,class Stride,class Output>
__global__ void indexed_prepare(qk_llama_indexed_v1 io,
    Half* a, int* offsets, int* rows, Shape* shapes, Output** outputs, Stride* strides,
    Output* destination, int m, int n, int k, int experts, int splits,
    quactlize::moe_directory::View directory) {
  __shared__ int ids[32];
  __shared__ int ranks[32], counts[32], starts[32], tiles[32];
  __shared__ bool valid;
  int tid=int(threadIdx.x);
  if (tid<m) ids[tid]=io.ids[int64_t(tid/io.topk)*io.ids_stride+tid%io.topk];
  __syncthreads();
  if (tid==0) {
    valid=valid_route(ids,m,io.topk,experts);
  }
  __syncthreads();
  if (tid<m) {
    ranks[tid]=valid ? ranked_row(ids,m,tid) : tid;
    counts[tid]=valid ? expert_count(ids,m,ids[tid]) : 0;
    starts[tid]=valid ? expert_begin(ids,m,ids[tid]) : 0;
    tiles[tid]=(valid && starts[tid]==ranks[tid]) ? (counts[tid]+TileM-1)/TileM : 0;
  }
  __syncthreads();

  // Every CTA derives the SAME ranks from the immutable input. Gather never
  // reads a prefix written by another CTA in this launch.
  if (blockIdx.x==0 && blockIdx.y==0) {
    for (int e=tid;e<experts;e+=int(blockDim.x)) {
      int begin=valid ? expert_begin(ids,m,e) : 0;
      int count=valid ? expert_count(ids,m,e) : 0;
      offsets[e]=begin; rows[e]=count; shapes[e]=cute::make_shape(count,n,k);
      for (int s=0;s<splits;++s) {
        int entry=e+s*experts;
        outputs[entry]=destination+(int64_t(s)*m+begin)*n;
        strides[entry]=cutlass::make_cute_packed_stride(Stride{},cute::make_shape(count,n,1));
      }
    }
    // Reuse per-route counts. Recounting all earlier experts for each entry
    // makes directory construction cubic in the small routed row count.
    if (tid<m && tiles[tid]) {
      int first=0;
      for (int j=0;j<m;++j) if (ids[j]<ids[tid]) first+=tiles[j];
      auto entry=quactlize::moe_directory::make_entry(ids[tid],counts[tid],first,starts[tid]);
      for (int b=0;b<tiles[tid];++b) directory.entries[first+b]=entry;
    }
    if (tid==0) {
      int total=0;
      for (int j=0;j<m;++j) total+=tiles[j];
      offsets[experts]=valid ? m : 0;
      *directory.header={valid ? total : 0,
          valid ? 0 : int(quactlize::moe_directory::BuildStatus::InvalidArgument),TileM,experts};
    }
    if (tid<m) io.row_ids[ranks[tid]]=tid;
  }
  if (!valid) return;
  int r=int(blockIdx.y), to=ranks[r];
  int64_t from=int64_t(r/io.topk)*io.a_token_stride+(r%io.topk%io.channels)*io.a_row_stride;
  for (int col=int(blockIdx.x)*int(blockDim.x)+tid;col<k;col+=int(gridDim.x)*int(blockDim.x))
    a[int64_t(to)*k+col]=Half(io.a[from+col]);
}

template<int Splits>
__global__ void indexed_finish(float const* partials, Half const* completed,
    float* output, int const* row_ids, int m, int n, int64_t output_stride,
    quactlize::moe_directory::Header const* directory) {
  int row=int(blockIdx.y), to=row_ids[row];
  for (int col=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);col<n;
       col+=int(gridDim.x)*int(blockDim.x)) {
    int64_t index=int64_t(row)*n+col;
    float value;
    if constexpr (Splits==1) value=float(completed[index]);
    else {
      float sum=0.f;
      CUTLASS_PRAGMA_UNROLL
      for (int s=0;s<Splits;++s) sum+=partials[int64_t(s)*m*n+index];
      value=float(Half(sum));  // Preserve the original reducer's FP16 boundary.
    }
    if (directory->status) value=__int_as_float(0x7fffffff);
    output[int64_t(to)*output_stride+col]=value;
  }
}
}  // namespace quactlize::runtime
