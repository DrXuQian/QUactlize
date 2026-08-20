// Exact Q4/A32 scale/zero consumer map for the first remaining device failure:
//   TM64 TN64 TK128 WM16 WN32 Stages8, gs32 => four metadata groups/tile.
// This follows the production scale shared layout, make_tiled_copy_B,
// partition_S, retile_D and MMA fragment layout.  Values encode (stage,group,n)
// so an accidental broadcast, group rotation or flattened-stage alias is red.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

using namespace cute;

struct L211F16Atom {};
namespace cute {
template <> struct MMA_Traits<L211F16Atom> {
  using ValTypeD=float; using ValTypeA=cutlass::half_t;
  using ValTypeB=cutlass::half_t; using ValTypeC=float;
  using Shape_MNK=Shape<_16,_16,_16>; using ThrID=Layout<_32>;
  using ALayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,
                       Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using BLayout=ALayout;
  using CLayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2>>,
                       Stride<Stride<_16,_1>,Stride<_64,_8>>>;
};
}

namespace {
constexpr int TM=64,TN=64,TK=128,WM=16,WN=32,Groups=4,Stages=8;
constexpr int WOM=TM/WM,WON=TN/WN;
using Mma=TiledMMA<MMA_Atom<L211F16Atom>,Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                   Tile<Int<WOM*16>,Int<WON*16>,Int<TK>>>;
using Atom=Layout<Shape<_8,_1>>;
using Storage=decltype(tile_to_shape(
    Atom{},make_shape(Int<TN>{},Int<Groups>{},Int<Stages>{})));
using Flat=decltype(tile_to_shape(
    Atom{},make_shape(Int<TN>{},_1{},Int<Groups*Stages>{})));

constexpr int value(int stage,int group,int n) {
  return 10000*stage+100*group+n;
}

bool storage_flat_same() {
  Storage s; Flat f;
  if (cosize(s)!=cosize(f)) return false;
  for(int st=0;st<Stages;++st)for(int g=0;g<Groups;++g)for(int n=0;n<TN;++n)
    if(int(s(n,g,st))!=int(f(n,0,st*Groups+g))) return false;
  return true;
}

bool metadata_map() {
  std::vector<int> smem(cosize(Storage{}),-1);
  for(int st=0;st<Stages;++st)for(int g=0;g<Groups;++g)for(int n=0;n<TN;++n)
    smem[Storage{}(n,g,st)]=value(st,g,n);
  auto flat=make_tensor(smem.data(),Flat{});
  auto identity=make_identity_tensor(make_shape(Int<TN>{},Int<TK>{}));
  auto tiled=make_tiled_copy_B(Copy_Atom<DefaultCopy,int>{},Mma{});
  int bad=0,total=0,shown=0;
  for(int thread=0;thread<int(size(Mma{}));++thread) {
    auto thr=Mma{}.get_thread_slice(thread);
    auto frag_ref=thr.partition_fragment_B(
        make_tensor((int*)nullptr,Storage{})(_,_,0));
    auto frag=make_tensor<int>(make_layout_like(frag_ref.layout()));
    auto copy_thr=tiled.get_thread_slice(thread);
    auto src=copy_thr.partition_S(flat);
    auto dst=copy_thr.retile_D(frag);
    auto logical=thr.partition_B(identity);
    auto pi=right_inverse(frag.layout());
    for(int st=0;st<Stages;++st)for(int g=0;g<Groups;++g) {
      clear(frag);
      copy(tiled,src(_,_,0,st*Groups+g),dst(_,_,0));
      for(int i=0;i<int(size(frag));++i) {
        auto nk=logical(pi(i));
        int n=int(get<0>(nk));
        ++total;
        if(frag(i)!=value(st,g,n)) {
          ++bad;
          if(shown++<8) std::printf(
              "L211 bad thread=%d stage=%d group=%d slot=%d n=%d got=%d want=%d\n",
              thread,st,g,i,n,frag(i),value(st,g,n));
        }
      }
    }
  }
  std::printf("L211 metadata-map correct=%d/%d bad=%d\n",total-bad,total,bad);
  return bad==0;
}
}

int main() {
  bool a=storage_flat_same(),b=metadata_map();
  std::printf("L211 Q4-A32 TM64/TN64/TK128/WM16/WN32/S8 flat=%s map=%s %s\n",
              a?"EXACT":"FAIL",b?"EXACT":"FAIL",a&&b?"PASS":"FAIL");
  return a&&b?0:1;
}
