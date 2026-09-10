#pragma once
#include "cutlass/cutlass.h"

namespace quactlize::runtime {
// Stable expert-major row permutation, independent of thread/warp topology.
CUTLASS_HOST_DEVICE constexpr int expert_begin(int const* ids, int rows, int expert) {
  int result=0;
  for (int i=0;i<rows;++i) result+=ids[i]<expert;
  return result;
}
CUTLASS_HOST_DEVICE constexpr int expert_count(int const* ids, int rows, int expert) {
  int result=0;
  for (int i=0;i<rows;++i) result+=ids[i]==expert;
  return result;
}
CUTLASS_HOST_DEVICE constexpr int ranked_row(int const* ids, int rows, int source_row) {
  int result=0, expert=ids[source_row];
  for (int i=0;i<rows;++i) result+=ids[i]<expert || (ids[i]==expert && i<source_row);
  return result;
}
CUTLASS_HOST_DEVICE constexpr int directory_begin(int const* ids,int rows,int expert,int tile_m) {
  int result=0;
  for (int i=0;i<rows;++i) {
    bool first=ids[i]<expert;
    for (int j=0;j<i;++j) first&=ids[j]!=ids[i];
    if (first) result+=(expert_count(ids,rows,ids[i])+tile_m-1)/tile_m;
  }
  return result;
}
CUTLASS_HOST_DEVICE constexpr bool valid_route(int const* ids,int rows,int topk,int experts) {
  for (int i=0;i<rows;++i) {
    if (ids[i]<0 || ids[i]>=experts) return false;
    for (int j=i-i%topk;j<i;++j) if (ids[i]==ids[j]) return false;
  }
  return true;
}
}  // namespace quactlize::runtime
