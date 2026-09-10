#include "quactlize/runtime/indexed_rows.hpp"
#include "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp"
#include "cute/tensor.hpp"
#include "cutlass/detail/layout.hpp"
#include "quactlize/integrations/llama/indexed.h"

extern "C" int row_map(int const* ids,int m,int topk,int experts,int tm,
    int* rank,int* offsets,int* shapes,int* directory) {
  using namespace quactlize::runtime;
  if (!valid_route(ids,m,topk,experts)) return 1;
  for (int r=0;r<m;++r) rank[r]=ranked_row(ids,m,r);
  for (int e=0;e<=experts;++e) {
    int begin=expert_begin(ids,m,e), count=expert_count(ids,m,e);
    offsets[e]=begin;
    if (e==experts) break;
    // Use the same actual packed CuTe shape as the grouped module.
    cute::Shape<int,int,int> shape=cute::make_shape(count,512,2048);
    shapes[e*3]=cute::get<0>(shape);
    shapes[e*3+1]=cute::get<1>(shape);
    shapes[e*3+2]=cute::get<2>(shape);
    int first=directory_begin(ids,m,e,tm);
    auto entry=quactlize::moe_directory::make_entry(e,count,first,begin);
    for (int b=0;b<(count+tm-1)/tm;++b) {
      int* out=directory+4*(first+b);
      out[0]=entry.expert; out[1]=entry.expert_rows;
      out[2]=entry.expert_block_begin; out[3]=entry.row_begin;
    }
  }
  return 0;
}
extern "C" int indexed_io_size() { return sizeof(qk_llama_indexed_v1); }
extern "C" int prepare_blocks(int experts,int rows) {
  return quactlize::runtime::moe_prepare_blocks(experts,rows);
}
