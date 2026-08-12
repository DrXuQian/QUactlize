// L143 -- independent audit of the production WK4 B consumer.
//
// The shipping xplane bytes are the independent anchor.  For every real
// production K2 shadow fragment this oracle follows
//
//   SmemLayoutB -> make_mix_tensor_like -> partition_S -> retile_D
//
// and then proves that the two nibbles consumed by one f16x2 conversion,
// (c,c+4), land in one adjacent/even production MMA half2.  It does not call
// the collective's consumer implementation.  Across the tile, all 16,384
// source codes and all 8,192 production half2 destinations must be covered
// exactly once.
//
// Two formerly plausible artifact orders are permanent negatives:
//   * sequentially concatenating selected codes in the production fragment
//     order (ea96...) is not the shipping map;
//   * L138's compact-s16 destination order (17df...) is not production.
// A third negative converts the first 32 codes of each source.  Finally WK1
// goes through the public WarpK API and must remain byte-identical to the
// shipping writer.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
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
#include "xplane_offline.hpp"

namespace {
using namespace cute;

constexpr int TM=16,TN=128,TK=128,WM=16,WN=64,WarpK=32;
constexpr int ArtifactTK=64,Bits=4,Stages=4;
constexpr int WOM=TM/WM,WON=TN/WN,WK=TK/WarpK;
constexpr int RowBytes=TK*Bits/8;

using F16 = PPU0010_16x16x16_F32F16F16F32_TN;
using Compute = TiledMMA<MMA_Atom<F16>,
    Layout<Shape<Int<WOM>,Int<WON>,Int<WK>>>,
    Tile<Int<WOM*16>,Int<WON*16>,Int<64>>>;

// This is the exact ordinary-int4 operand atom produced by
// MixGemm_AIU_Operand<int4b_t,...> for ArtifactTileK=64.  The stage layout
// spelling is the collective's SmemLayoutB formula verbatim.
using SmemLayoutAtomB = Layout<Shape<_8,Int<64>>,Stride<Int<64>,_1>>;
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},make_shape(Int<TN>{},Int<TK>{},Int<Stages>{})));

using S8Inst = PPU0010_16x16x32_S32S8S8S32_TN;
using Shadow = TiledMMA<MMA_Atom<S8Inst>,
    Layout<Shape<Int<WOM>,Int<WON>,_2>>,
    Tile<Int<WOM*16>,Int<WON*16>,Int<64>>>;
using ShadowOp = PPU0010_TSM_LD_SWZL<int8_t,TN,32,true,false,1>;
using ShadowAtom = Copy_Atom<ShadowOp,int8_t>;
using ShadowTraits = Copy_Traits<ShadowOp>;

using ProductionFragLayout = decltype(Compute{}.get_thread_slice(0)
    .partition_fragment_B(make_tensor(
        make_smem_ptr((cutlass::half_t*)nullptr),SmemLayoutB{})(_,_,0)).layout());
using ExpectedProductionFragLayout =
    Layout<Shape<Shape<_2,_2,_2>,_4,_2>,
           Stride<Stride<_1,_2,_4>,_8,_32>>;
static_assert(std::is_same_v<ProductionFragLayout,ExpectedProductionFragLayout>,
              "oracle must use the real production B fragment layout");
static_assert(size(Compute{})==256 && size(Shadow{})==128);
static_assert(ShadowTraits::LogicalWordsPerRow==8 &&
              ShadowTraits::LogicalSlices==1);

struct Source {
  std::vector<int> physical,vreg,code,ni;
  bool valid=true;
};

Source production_source_fragment(int thread) {
  auto s4=make_tensor(make_smem_ptr((cutlass::int4b_t*)nullptr),SmemLayoutB{});
  auto s8=recast<int8_t>(s4);
  auto load=Shadow{}.get_thread_slice(thread).partition_fragment_B(s8(_,_,0));
  auto cp=make_tiled_copy_B(ShadowAtom{},Shadow{})
      .get_thread_slice((thread/32)*32);
  // The real collective uses make_mix_tensor_like.  Its rank-4 coordinate is
  // (byte-within-32B, row, 32B-half, stage), not a resolved linear offset.
  auto src=cp.partition_S(make_mix_tensor_like(s8));
  auto view=cp.retile_D(load);
  static_assert(rank(decltype(src.layout()){})==4,
                "production partition_S must retain its stage coordinate");
  static_assert(size<1>(decltype(view.layout()){})==4 &&
                size<2>(decltype(view.layout()){})==1);

  Source out;
  out.physical.assign(2*cosize(load.layout()),-1);
  out.vreg.assign(out.physical.size(),-1);
  out.code.assign(out.physical.size(),-1);
  out.ni.assign(out.physical.size(),-1);
  constexpr int CN=size<1>(decltype(view.layout()){});
  constexpr int CK=size<2>(decltype(view.layout()){});
  constexpr int WPR=ShadowTraits::LogicalWordsPerRow;
  for(int ck=0;ck<CK;++ck)for(int cn=0;cn<CN;++cn)
    for(int v=0;v<4;++v)for(int c=0;c<8;++c){
      auto base=src(_,cn,ck,0).data().coord_;
      int word=int(typename ShadowTraits::LogicalTV{}(
          make_coord(make_coord(thread%4,(thread%32)/4),
                     make_coord(v%2,v/2),_0{})));
      int row=int(get<1>(base))+word/WPR;
      int byte_in_row=int(get<0>(base))+32*int(get<2>(base));
      int byte=row*RowBytes+byte_in_row+4*(word%WPR)+c/2;
      int dst=2*int(view.layout()(4*v+c/2,cn,ck))+c%2;
      int physical=2*byte+c%2;
      bool in=dst>=0&&dst<int(out.physical.size()); out.valid&=in;
      if(!in)continue;
      out.valid&=out.physical[dst]<0||out.physical[dst]==physical;
      out.physical[dst]=physical; out.vreg[dst]=v;
      out.code[dst]=c; out.ni[dst]=cn;
    }
  out.valid&=std::none_of(out.physical.begin(),out.physical.end(),
                         [](int x){return x<0;});
  return out;
}

Source compact_source_fragment(int thread) {
  auto s8=make_tensor(make_smem_ptr((int8_t*)nullptr),
      make_layout(Shape<Int<TN>,Int<RowBytes>>{},Stride<Int<RowBytes>,_1>{}));
  auto sid=make_identity_tensor(make_shape(Int<TN>{},Int<RowBytes>{}));
  auto load=Shadow{}.get_thread_slice(thread).partition_fragment_B(s8);
  auto cp=make_tiled_copy_B(ShadowAtom{},Shadow{})
      .get_thread_slice((thread/32)*32);
  auto src=cp.partition_S(sid); auto view=cp.retile_D(load);
  Source out;
  out.physical.assign(2*cosize(load.layout()),-1);
  out.vreg.assign(out.physical.size(),-1);
  out.code.assign(out.physical.size(),-1);
  out.ni.assign(out.physical.size(),-1);
  constexpr int CN=size<1>(decltype(view.layout()){});
  constexpr int CK=size<2>(decltype(view.layout()){});
  constexpr int WPR=ShadowTraits::LogicalWordsPerRow;
  for(int ck=0;ck<CK;++ck)for(int cn=0;cn<CN;++cn)
    for(int v=0;v<4;++v)for(int c=0;c<8;++c){
      auto base=src(0,cn,ck);
      int word=int(typename ShadowTraits::LogicalTV{}(
          make_coord(make_coord(thread%4,(thread%32)/4),
                     make_coord(v%2,v/2),_0{})));
      int byte=(int(get<0>(base))+word/WPR)*RowBytes+int(get<1>(base))+
               4*(word%WPR)+c/2;
      int dst=2*int(view.layout()(4*v+c/2,cn,ck))+c%2;
      int physical=2*byte+c%2;
      bool in=dst>=0&&dst<int(out.physical.size()); out.valid&=in;
      if(!in)continue;
      out.valid&=out.physical[dst]<0||out.physical[dst]==physical;
      out.physical[dst]=physical; out.vreg[dst]=v;
      out.code[dst]=c; out.ni[dst]=cn;
    }
  out.valid&=std::none_of(out.physical.begin(),out.physical.end(),
                         [](int x){return x<0;});
  return out;
}

struct Ref {int logical,frag,wk;};

std::vector<Ref> production_refs(int thread) {
  auto mma=Compute{}; auto tc=mma.get_thr_layout_vmnk().get_flat_coord(thread);
  auto prod=make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),SmemLayoutB{});
  auto frag=mma.get_thread_slice(thread).partition_fragment_B(prod(_,_,0));
  auto bid=make_identity_tensor(make_shape(Int<TN>{},Int<TK>{}));
  auto part=mma.get_thread_slice(thread).partition_B(bid);
  auto pi=right_inverse(frag.layout());
  std::vector<Ref> out;
  for(int i=0;i<int(size(frag));++i){auto x=part(pi(i));
    out.push_back({int(get<0>(x))*TK+int(get<1>(x)),i,int(get<3>(tc))});}
  return out;
}

std::vector<Ref> compact_refs(int thread) {
  auto mma=Compute{}; auto tc=mma.get_thr_layout_vmnk().get_flat_coord(thread);
  auto s16=make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<TN>,Int<TK>>{},Stride<Int<TK>,_1>{}));
  auto frag=mma.get_thread_slice(thread).partition_fragment_B(s16);
  auto bid=make_identity_tensor(make_shape(Int<TN>{},Int<TK>{}));
  auto part=mma.get_thread_slice(thread).partition_B(bid);
  auto pi=right_inverse(frag.layout());
  std::vector<Ref> out;
  for(int i=0;i<int(size(frag));++i){auto x=part(pi(i));
    out.push_back({int(get<0>(x))*TK+int(get<1>(x)),i,int(get<3>(tc))});}
  return out;
}

int source_slot(Source const& s,int ni,int v,int code) {
  int found=-1;
  for(int d=0;d<int(s.physical.size());++d)
    if(s.ni[d]==ni&&s.vreg[d]==v&&s.code[d]==code){if(found>=0)return -2;found=d;}
  return found;
}
int output_slot(std::vector<Ref> const& r,int logical) {
  int found=-1;
  for(int i=0;i<int(r.size());++i)
    if(r[i].logical==logical){if(found>=0)return -2;found=i;}
  return found;
}
constexpr bool keep(int wk,int v,int c){return v/2==wk/2&&(c/2)%2==wk%2;}

void hash_word(uint64_t&h,uint64_t x){
  for(int i=0;i<8;++i){h^=(x>>(8*i))&255;h*=UINT64_C(1099511628211);}
}
uint64_t hash_map(std::vector<int> const& m){
  uint64_t h=UINT64_C(1469598103934665603);
  for(int i=0;i<int(m.size());++i){hash_word(h,i);hash_word(h,m[i]+1);}return h;
}
int differences(std::vector<int>const&a,std::vector<int>const&b){
  if(a.size()!=b.size())return -1; int n=0;
  for(int i=0;i<int(a.size());++i)n+=a[i]!=b[i];return n;
}

struct DirectResult {
  std::vector<int> map;
  int pairs=0,source_holes=0,source_dup=0,logical_holes=0,logical_dup=0;
  int destination_holes=0,destination_dup=0,bad_pairs=0,bad_fragments=0;
  int formula_mismatch=0;
};

enum class DirectMode { Correct, AdjacentNibble, SwapSources };

DirectResult direct_pair_scatter(DirectMode mode=DirectMode::Correct) {
  auto const shipping=xplane::plane_map<Bits,TM,TN,TK,WM,WN,1,ArtifactTK>();
  DirectResult out; out.map.assign(TN*TK,-1);
  std::vector<int> ph(TN*TK),lh(TN*TK),dh(size(Compute{})*32);
  auto ctl=Compute{}.get_thr_layout_vmnk(); auto stl=Shadow{}.get_thr_layout_vmnk();
  for(int ct=0;ct<int(size(Compute{}));++ct){
    auto c=ctl.get_flat_coord(ct); int wk=int(get<3>(c)); auto refs=production_refs(ct);
    for(int sk=0;sk<2;++sk){
      int source_k=mode==DirectMode::SwapSources?1-sk:sk;
      int st=int(stl(make_coord(get<0>(c),get<1>(c),get<2>(c),source_k)));
      auto src=production_source_fragment(st); out.bad_fragments+=!src.valid;
      int const v0=wk/2,v1=v0+2,phase=wk%2;
      for(int ni=0;ni<4;++ni)for(int vi=0;vi<2;++vi)
        for(int ti=0;ti<2;++ti){
          int v=vi?v1:v0,t=2*phase+ti;
          int second=t+(mode==DirectMode::AdjacentNibble?1:4);
          int d0=source_slot(src,ni,v,t),d1=source_slot(src,ni,v,second);
          bool source_ok=d0>=0&&d1>=0&&d0/8==d1/8&&d0%8==t&&d1%8==second;
          if(!source_ok){++out.bad_pairs;continue;}
          int p0=src.physical[d0],p1=src.physical[d1];
          int desired0=output_slot(refs,shipping[p0]);
          int desired1=output_slot(refs,shipping[p1]);
          int expected_half2=16*sk+4*ni+2*vi+ti;
          int actual0=2*expected_half2,actual1=actual0+1;
          bool destination_ok=desired0==actual0&&desired1==actual1;
          out.bad_pairs+=!destination_ok;
          out.formula_mismatch+=!destination_ok;
          ++out.pairs; ++dh[ct*32+expected_half2];
          for(auto [p,o]:{std::pair<int,int>{p0,actual0},std::pair<int,int>{p1,actual1}}){
            int logical=refs[o].logical; ++ph[p]; ++lh[logical];
            if(out.map[p]>=0&&out.map[p]!=logical)++out.bad_pairs;
            out.map[p]=logical;
          }
        }
    }
  }
  for(int i=0;i<TN*TK;++i){out.source_holes+=ph[i]==0;out.source_dup+=std::max(0,ph[i]-1);
    out.logical_holes+=lh[i]==0;out.logical_dup+=std::max(0,lh[i]-1);}
  for(int n:dh){out.destination_holes+=n==0;out.destination_dup+=std::max(0,n-1);}
  return out;
}

struct OrderResult {std::vector<int> map;int bad=0,holes=0,pdup=0,ldup=0;};

OrderResult sequential_order(bool compact,bool first32=false) {
  OrderResult out;out.map.assign(TN*TK,-1);std::vector<int> ph(TN*TK),lh(TN*TK);
  auto ctl=Compute{}.get_thr_layout_vmnk();auto stl=Shadow{}.get_thr_layout_vmnk();
  for(int ct=0;ct<int(size(Compute{}));++ct){auto c=ctl.get_flat_coord(ct);int wk=int(get<3>(c));
    std::vector<int> chosen;
    for(int sk=0;sk<2;++sk){int st=int(stl(make_coord(get<0>(c),get<1>(c),get<2>(c),sk)));
      auto src=compact?compact_source_fragment(st):production_source_fragment(st);out.bad+=!src.valid;
      for(int d=0;d<int(src.physical.size());++d)
        if(first32?d<32:keep(wk,src.vreg[d],src.code[d]))chosen.push_back(src.physical[d]);}
    auto refs=compact?compact_refs(ct):production_refs(ct);
    if(chosen.size()!=refs.size()){++out.bad;continue;}
    for(int i=0;i<int(refs.size());++i){int p=chosen[i],l=refs[i].logical;
      if(p<0||p>=TN*TK||l<0||l>=TN*TK){++out.bad;continue;}
      ++ph[p];++lh[l];if(out.map[p]>=0&&out.map[p]!=l)++out.bad;out.map[p]=l;}
  }
  for(int i=0;i<TN*TK;++i){out.holes+=out.map[i]<0;out.pdup+=std::max(0,ph[i]-1);out.ldup+=std::max(0,lh[i]-1);}
  return out;
}
bool clean(OrderResult const&r){return !r.bad&&!r.holes&&!r.pdup&&!r.ldup;}

bool wk1_byte_anchor(int& map_diff,int& byte_diff) {
  auto shipping=xplane::plane_map<Bits,TM,TN,TK,WM,WN,1,ArtifactTK>();
  auto wk1=xplane::plane_map_warp_k<Bits,TM,TN,TK,WM,WN,1,TK,ArtifactTK>();
  map_diff=differences(shipping,wk1);
  constexpr int K=256; std::vector<uint8_t> q(size_t(TN)*K);
  for(int k=0;k<K;++k)for(int n=0;n<TN;++n)q[size_t(k)*TN+n]=uint8_t((k*131+n*17+(k^n))&15);
  std::vector<int8_t>a(size_t(TN)*K/2),b(a.size());
  xplane::place_derived<Bits,TM,TN,TK,WM,WN,1,ArtifactTK>(a.data(),q,TN,K);
  xplane::place_derived_warp_k<Bits,TM,TN,TK,WM,WN,1,TK,ArtifactTK>(b.data(),q,TN,K);
  byte_diff=0;for(size_t i=0;i<a.size();++i)byte_diff+=a[i]!=b[i];
  return map_diff==0&&byte_diff==0;
}

int production_compact_source_diff() {
  int diff=0;
  for(int t=0;t<int(size(Shadow{}));++t){
    auto p=production_source_fragment(t),c=compact_source_fragment(t);
    if(!p.valid||!c.valid||p.physical.size()!=c.physical.size())return -1;
    for(int i=0;i<int(p.physical.size());++i)diff+=p.physical[i]!=c.physical[i];
  }
  return diff;
}

} // namespace

int main(){
  auto shipping=xplane::plane_map<Bits,TM,TN,TK,WM,WN,1,ArtifactTK>();
  auto direct=direct_pair_scatter();
  auto wrong_pair=direct_pair_scatter(DirectMode::AdjacentNibble);
  auto swapped=direct_pair_scatter(DirectMode::SwapSources);
  auto artifact=sequential_order(false),compact=sequential_order(true),first32=sequential_order(false,true);
  int direct_diff=differences(direct.map,shipping);
  int wrong_pair_diff=differences(wrong_pair.map,shipping);
  int swapped_diff=differences(swapped.map,shipping);
  int artifact_diff=differences(artifact.map,shipping),compact_diff=differences(compact.map,shipping);
  int first32_diff=differences(first32.map,shipping),wk1_map_diff=0,wk1_byte_diff=0;
  int source_layout_diff=production_compact_source_diff();
  uint64_t shipping_hash=hash_map(shipping),direct_hash=hash_map(direct.map);
  uint64_t artifact_hash=hash_map(artifact.map),compact_hash=hash_map(compact.map);
  bool direct_clean=direct.pairs==8192&&!direct.source_holes&&!direct.source_dup&&
      !direct.logical_holes&&!direct.logical_dup&&!direct.destination_holes&&
      !direct.destination_dup&&!direct.bad_pairs&&!direct.bad_fragments&&
      !direct.formula_mismatch;
  auto control_red=[&](DirectResult const& r,int diff){
    return diff>0||r.pairs!=8192||r.source_holes||r.source_dup||
        r.logical_holes||r.logical_dup||r.destination_holes||
        r.destination_dup||r.bad_pairs||r.bad_fragments||r.formula_mismatch;
  };
  bool wrong_pair_red=control_red(wrong_pair,wrong_pair_diff);
  bool swapped_red=control_red(swapped,swapped_diff);
  bool wk1_ok=wk1_byte_anchor(wk1_map_diff,wk1_byte_diff);
  bool ok=direct_clean&&direct_diff==0&&direct_hash==shipping_hash&&
      shipping_hash==UINT64_C(0xb89b157b5b1bd6c3)&&
      clean(artifact)&&artifact_hash==UINT64_C(0xea96e6b4155759c3)&&artifact_diff>0&&
      clean(compact)&&compact_hash==UINT64_C(0x17dfe6248fc38143)&&compact_diff>0&&
      (!clean(first32)||first32_diff>0)&&wrong_pair_red&&swapped_red&&
      wk1_ok&&source_layout_diff==0;
  std::printf("L143 direct-pair pairs=%d/8192 codes=%d/16384 destinations=%d/8192 "
              "bad-pairs=%d formula-mismatch=%d bad-fragments=%d map-diff=%d shipping-hash=%016llx\n",
      direct.pairs,16384-direct.source_holes,8192-direct.destination_holes,
      direct.bad_pairs,direct.formula_mismatch,direct.bad_fragments,direct_diff,
      (unsigned long long)direct_hash);
  std::printf("L143 production-order hash=%016llx shipping-diff=%d/16384 %s\n",
      (unsigned long long)artifact_hash,artifact_diff,artifact_diff>0?"EXPECTED-RED":"UNEXPECTED-GREEN");
  std::printf("L143 compact-order hash=%016llx shipping-diff=%d/16384 %s\n",
      (unsigned long long)compact_hash,compact_diff,compact_diff>0?"EXPECTED-RED":"UNEXPECTED-GREEN");
  std::printf("L143 first32x2 shipping-diff=%d/16384 clean=%d %s\n",first32_diff,
      int(clean(first32)),(!clean(first32)||first32_diff>0)?"EXPECTED-RED":"UNEXPECTED-GREEN");
  std::printf("L143 adjacent-nibble shipping-diff=%d/16384 pairs=%d bad-pairs=%d %s\n",
      wrong_pair_diff,wrong_pair.pairs,wrong_pair.bad_pairs,
      wrong_pair_red?"EXPECTED-RED":"UNEXPECTED-GREEN");
  std::printf("L143 swapped-sources shipping-diff=%d/16384 pairs=%d bad-pairs=%d %s\n",
      swapped_diff,swapped.pairs,swapped.bad_pairs,
      swapped_red?"EXPECTED-RED":"UNEXPECTED-GREEN");
  std::printf("L143 WK1 shipping map-diff=%d byte-diff=%d result=%s\n",
      wk1_map_diff,wk1_byte_diff,wk1_ok?"BIT-IDENTICAL":"FAIL");
  std::printf("L143 production-vs-compact-source-fragment-diff=%d\n",source_layout_diff);
  std::printf("L143 shipping-pair-scatter=%s artifact-order=RED compact-order=RED "
              "first32=RED wrong-pair=RED source-swap=RED WK1-BYTES=UNCHANGED result=%s\n",
      direct_clean&&direct_diff==0?"EXACT":"FAIL",ok?"PASS":"FAIL");
  return ok?0:1;
}
