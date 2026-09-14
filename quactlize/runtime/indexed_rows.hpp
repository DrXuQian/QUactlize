#pragma once
#include "cutlass/cutlass.h"

namespace quactlize::runtime {
// Preserve the original <=32-row contract. The new 33..64-row arm is decode
// only (tokens1..8), not an implicit prefill admission.
CUTLASS_HOST_DEVICE constexpr bool fused_indexed_rows(int rows,int tokens) {
  return rows>0 && tokens>0 && (rows<=32 || (rows<=64 && tokens<=8));
}
// One warp-wide expert range per CTA. Keep at least one CTA per routed row
// without repeating the router for every K/256 activation chunk.
CUTLASS_HOST_DEVICE constexpr int moe_prepare_blocks(int experts,int rows) {
  int ranges=(experts+31)/32;
  return ranges>rows ? ranges : rows;
}

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
