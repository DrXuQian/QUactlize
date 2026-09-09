#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>
#include "actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp"

namespace md = quactlize::moe_directory;
using Key = std::tuple<int,int,int,int>;

template<int TM>
bool check(std::vector<int> const& rows, int nt, int splits, int grid, bool persistent, int plant) {
  int const maximum=*std::max_element(rows.begin(),rows.end());
  int const m=std::accumulate(rows.begin(),rows.end(),0);
  int const capacity=md::bounded_entries(m,maximum,int(rows.size()),TM);
  std::vector<md::BlockEntry> entries;
  std::set<Key> expected,observed;
  int begin=0;
  for (int e=0;e<int(rows.size());++e) {
    int const count=(rows[e]+TM-1)/TM;
    auto entry=md::make_entry(e,rows[e],int(entries.size()),begin);
    for (int mt=0;mt<count;++mt) {
      entries.push_back(entry);
      for (int n=0;n<nt;++n) for (int s=0;s<splits;++s) expected.emplace(e,mt,n,s);
    }
    begin+=rows[e];
  }
  if (int(entries.size())>capacity) return false;
  if (!persistent) grid=capacity*nt*(plant==4 ? 1 : splits);
  uint64_t const total=uint64_t(entries.size())*nt*splits;
  size_t publications=0;
  bool exact=true;
  for (int cta=0;cta<grid;++cta) {
    for (uint64_t i=cta;i<total;i+=grid) {
      if (plant==1 && i==total/2) { if (!persistent) break; else continue; }
      uint64_t linear=plant==2 && i==total/2 ? 0 : i;
      auto split=md::decode_split_work(linear,splits);
      auto entry=entries.at(split.tile/nt);
      int const local=int(split.tile)-entry.expert_block_begin*nt;
      auto work=md::decode_swizzled(local,(entry.expert_rows+TM-1)/TM,nt,TM==16 ? 1 : 2);
      int const destination_slice=plant==3 ? (split.slice+1)%splits : split.slice;
      int source_sum=0,expected_sum=0;
      for (int k=split.slice;k<16;k+=splits) source_sum+=k+1;
      for (int k=destination_slice;k<16;k+=splits) expected_sum+=k+1;
      exact &= source_sum==expected_sum;
      observed.emplace(entry.expert,work.m_tile,work.n_tile,destination_slice);
      ++publications;
      if (!persistent) break;
    }
  }
  return exact && observed==expected && publications==expected.size();
}

int main() {
  static_assert(md::bounded_entries(8,1,256,8)==8);
  static_assert(md::bounded_entries(8,8,256,8)==8);
  static_assert(md::bounded_entries(13,9,4,8)==5);
  static_assert(md::bounded_entries(0,1,256,8)==0);
  static_assert(md::bounded_entries(INT32_MAX,INT32_MAX,1,8)==268435456);
  for (int a=0;a<=17;++a) for (int b=0;b<=17;++b)
    for (int c=0;c<=17;++c) for (int d=0;d<=17;++d) {
      int const m=a+b+c+d, maximum=std::max({a,b,c,d});
      for (int tm : {8,16,32}) {
        int const actual=(a+tm-1)/tm+(b+tm-1)/tm+(c+tm-1)/tm+(d+tm-1)/tm;
        if (actual>md::bounded_entries(m,maximum,4,tm)) return 1;
      }
    }
  std::vector<std::vector<int>> profiles={{9,0,3,1},{0,3,1,9},{0,0,0,13},
      {0,1,7,8,9,15,16,17,31,32,33,65},std::vector<int>(256)};
  for (int i=0;i<8;++i) profiles.back()[17*i]=1;
  for (auto const& rows : profiles) for (int nt : {1,3,8,32})
    for (int split : {1,2,4,8}) for (int grid : {1,7,72,144}) for (bool persistent : {false,true}) {
      if (!check<8>(rows,nt,split,grid,persistent,0) ||
          !check<16>(rows,nt,split,grid,persistent,0)) return 2;
      for (int plant : {1,2,3}) {
        if (plant==3 && split==1) continue;
        if (check<8>(rows,nt,split,grid,persistent,plant)) return 3;
      }
    }
  if (check<8>({1,1,1,1},8,4,1,false,4)) return 4;
  std::puts("GPU_COMPACT_HOST PASS bounded-grid exact-once persistent+ordinary S1/2/4/8; missing/duplicate/slice/short-grid negatives RED");
}
