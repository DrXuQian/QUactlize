// L143 -- independent audit of the production WK4 B consumer.
//
// L138 proved that each 64-value compute fragment needs 32 int4 codes from
// each of two K2 shadow fragments.  Its destination was a compact row-major
// (N,K) tensor, however.  Production does not use that layout: it allocates B
// with CollectiveMma::SmemLayoutB and asks the real TiledMma for
// partition_fragment_B.  This oracle instantiates exactly those production
// objects, then runs the real partition_S -> retile_D algebra of the K2
// shadow reader.  The only hand-written piece is the L138-proved selector.
//
// Three things are load-bearing:
//   * production and compact destinations differ in 12,288 / 16,384 slots;
//   * all selected source codes map bijectively to the production MMA wants;
//   * the tempting "convert first 32 codes from each source and concatenate"
//     is red.  Selection is spread across all four source vregs.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <tuple>
#include <type_traits>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cutlass/numeric_types.h"
#include "quactlize_extensions/cutlass/quactlize_mix_gemm_convert.h"

namespace {
using namespace cute;

constexpr int TM=16,TN=128,TK=128,WM=16,WN=64,WarpK=32;
constexpr int ArtifactTK=64,Bits=4,Stages=4;
constexpr int WOM=TM/WM,WON=TN/WN,WK=TK/WarpK;
constexpr int RowBytes=TK*Bits/8;

using F16 = PPU0010_16x16x16_F32F16F16F32_TN;
using Compute = TiledMMA<MMA_Atom<F16>,Layout<Shape<Int<WOM>,Int<WON>,Int<WK>>>,
                         Tile<Int<WOM*16>,Int<WON*16>,Int<64>>>;
using SmemLayoutB = decltype(tile_to_shape(
    Layout<Shape<_8,Int<128>>,Stride<Int<128>,_1>>{},
    make_shape(Int<TN>{},Int<TK>{},Int<Stages>{})));
using S8Inst = PPU0010_16x16x32_S32S8S8S32_TN;
using Shadow = TiledMMA<MMA_Atom<S8Inst>,Layout<Shape<Int<WOM>,Int<WON>,_2>>,
                        Tile<Int<WOM*16>,Int<WON*16>,Int<64>>>;
using ShadowOp = PPU0010_TSM_LD_SWZL<int8_t,TN,32,true,false,1>;
using ShadowAtom = Copy_Atom<ShadowOp,int8_t>;
using ShadowTraits = Copy_Traits<ShadowOp>;

static_assert(size(Compute{})==256 && size(Shadow{})==128);
static_assert(std::is_same_v<ShadowOp,
    PPU0010_TSM_LD_SWZL<int8_t,TN,32,true,false,1>>);

struct Source {
  std::vector<int> physical,vreg,code;
  bool valid=true;
};

Source source_fragment(int thread) {
  auto s4=make_tensor(make_smem_ptr((cutlass::int4b_t*)nullptr),SmemLayoutB{});
  auto s8=recast<int8_t>(s4);
  auto sid=make_counting_tensor(s8.layout());
  auto load=Shadow{}.get_thread_slice(thread).partition_fragment_B(s8(_,_,0));
  auto cp=make_tiled_copy_B(ShadowAtom{},Shadow{}).get_thread_slice((thread/32)*32);
  auto src=cp.partition_S(sid);
  auto view=cp.retile_D(load);
  Source out;
  out.physical.assign(2*cosize(load.layout()),-1);
  out.vreg.assign(out.physical.size(),-1);out.code.assign(out.physical.size(),-1);
  constexpr int CN=size<1>(decltype(view.layout()){}),CK=size<2>(decltype(view.layout()){});
  constexpr int WPR=ShadowTraits::LogicalWordsPerRow;
  for(int ck=0;ck<CK;++ck)for(int cn=0;cn<CN;++cn)
    for(int v=0;v<4;++v)for(int c=0;c<8;++c){
      auto base=src(0,cn,ck);
      int word=int(typename ShadowTraits::LogicalTV{}(
          make_coord(make_coord(thread%4,(thread%32)/4),make_coord(v%2,v/2),_0{})));
      // base is an int8 counting offset in production SmemLayoutB.  LogicalTV
      // contributes the byte displacement inside the 16x32B cube.
      int byte=int(base)+4*word+c/2;
      int dst=2*int(view.layout()(4*v+c/2,cn,ck))+c%2;
      int physical=2*byte+c%2;
      bool in=dst>=0&&dst<int(out.physical.size());out.valid&=in;if(!in)continue;
      out.valid&=out.physical[dst]<0||out.physical[dst]==physical;
      out.physical[dst]=physical;out.vreg[dst]=v;out.code[dst]=c;
    }
  out.valid&=std::none_of(out.physical.begin(),out.physical.end(),[](int x){return x<0;});
  return out;
}

struct Ref {int logical,frag,wk;};
std::vector<Ref> wants(int thread,bool compact) {
  auto mma=Compute{};auto tc=mma.get_thr_layout_vmnk().get_flat_coord(thread);
  // Destination ownership is a property of the fp16 MMA fragment.  Borrow
  // the production layout but label its storage offsets with the chosen
  // anchor; partition_B exposes the logical value at every fragment slot.
  auto prod=make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),SmemLayoutB{});
  auto frag=mma.get_thread_slice(thread).partition_fragment_B(prod(_,_,0));
  auto pi=right_inverse(frag.layout());
  auto compact_id=make_counting_tensor(make_layout(Shape<Int<TN>,Int<TK>>{},Stride<Int<TK>,_1>{}));
  auto production_id=make_counting_tensor(SmemLayoutB{});
  auto compact_part=mma.get_thread_slice(thread).partition_B(compact_id);
  auto production_part=mma.get_thread_slice(thread).partition_B(production_id(_,_,0));
  std::vector<Ref> r; r.reserve(size(frag));
  for(int i=0;i<int(size(frag));++i){
    int logical=compact?int(compact_part(pi(i))):int(production_part(pi(i)));
    r.push_back(Ref{logical,i,int(get<3>(tc))});
  }
  return r;
}

constexpr bool keep(int wk,int v,int c){return v/2==wk/2&&(c/2)%2==wk%2;}

struct Result {std::vector<int> map;int holes=0,pdup=0,ldup=0,fragbad=0;uint64_t hash=1469598103934665603ull;};
void hw(uint64_t&h,uint64_t x){for(int i=0;i<8;++i){h^=(x>>(8*i))&255;h*=1099511628211ull;}}

Result derive(bool compact,bool first32=false) {
  Result out;out.map.assign(TN*TK,-1);std::vector<int>ph(TN*TK),lh(TN*TK);
  auto ctl=Compute{}.get_thr_layout_vmnk();auto stl=Shadow{}.get_thr_layout_vmnk();
  for(int ct=0;ct<int(size(Compute{}));++ct){auto c=ctl.get_flat_coord(ct);int wk=int(get<3>(c));
    std::vector<int> chosen;
    for(int sk=0;sk<2;++sk){int st=int(stl(make_coord(get<0>(c),get<1>(c),get<2>(c),sk)));
      auto sf=source_fragment(st);if(!sf.valid){++out.fragbad;continue;}
      for(int d=0;d<int(sf.physical.size());++d){
        bool take=first32 ? d<32 : keep(wk,sf.vreg[d],sf.code[d]);
        if(take)chosen.push_back(sf.physical[d]);
      }}
    auto refs=wants(ct,compact);if(chosen.size()!=refs.size()){++out.fragbad;continue;}
    for(int i=0;i<int(refs.size());++i){int p=chosen[i],l=refs[i].logical;
      if(p<0||p>=int(out.map.size())||l<0||l>=int(out.map.size())){++out.fragbad;continue;}
      ++ph[p];++lh[l];if(out.map[p]>=0&&out.map[p]!=l)++out.fragbad;out.map[p]=l;}
  }
  for(int i=0;i<int(out.map.size());++i){out.holes+=out.map[i]<0;out.pdup+=std::max(0,ph[i]-1);out.ldup+=std::max(0,lh[i]-1);hw(out.hash,i);hw(out.hash,out.map[i]+1);}
  return out;
}

int diffs(std::vector<int>const&a,std::vector<int>const&b){int n=0;for(int i=0;i<int(a.size());++i)n+=a[i]!=b[i];return n;}
bool clean(Result const&r){return !r.holes&&!r.pdup&&!r.ldup&&!r.fragbad;}

int main(){
  auto production=derive(false),compact=derive(true),bad=derive(false,true);
  int pc=diffs(production.map,compact.map),pb=diffs(production.map,bad.map);
  std::printf("L143 production entries=%zu hash=%016llx holes=%d pdup=%d ldup=%d fragbad=%d\n",production.map.size(),(unsigned long long)production.hash,production.holes,production.pdup,production.ldup,production.fragbad);
  std::printf("L143 compact    hash=%016llx diff-production=%d/%zu (EXPECTED-RED)\n",(unsigned long long)compact.hash,pc,production.map.size());
  std::printf("L143 first32x2  hash=%016llx diff-production=%d/%zu clean=%d (EXPECTED-RED)\n",(unsigned long long)bad.hash,pb,production.map.size(),int(clean(bad)));
  bool ok=clean(production)&&clean(compact)&&production.hash==UINT64_C(0xea96e6b4155759c3)&&compact.hash==UINT64_C(0x17dfe6248fc38143)&&pc==12288&&(!clean(bad)||pb>0);
  std::printf("L143 real-production-destination=%s compact-negative=%s first32-negative=%s result=%s\n",clean(production)?"BIJECTIVE":"FAIL",pc?"RED":"GREEN",(!clean(bad)||pb)?"RED":"GREEN",ok?"PASS":"FAIL");
  return ok?0:1;
}
} // namespace
