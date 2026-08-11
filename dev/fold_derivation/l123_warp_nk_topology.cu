// L123 -- test Marlin's (N,K) warp grid without changing the shipping builder.
// S068 is 16x32x256/w16x16: 2N x 1K today, 2N x 4K in the candidate.
// The equal-four-warp 2Nx2K/1Nx4K pair proves N and K are separate axes.
// B's physical oracle composes the real partition_S -> retile_D -> int4
// converter-emission -> compute-fragment chain.  WK1 must reproduce the
// shipping xplane byte map; WK>1 tests the direct shadow-K construction, not
// every possible future address/scatter remap.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "quactlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#if defined(L123_TYPE_ONLY)
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl"
#else
#include "xplane_offline.hpp"
#endif
struct L123F16Atom {};
namespace cute {
template <> struct MMA_Traits<L123F16Atom> {
  using ValTypeD=float; using ValTypeA=cutlass::half_t;
  using ValTypeB=cutlass::half_t; using ValTypeC=float;
  using Shape_MNK=Shape<_16,_16,_16>; using ThrID=Layout<_32>;
  using ALayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,
                       Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using BLayout=ALayout;
  using CLayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2>>,
                       Stride<Stride<_16,_1>,Stride<_64,_8>>>;
};
} // namespace cute
namespace {
using namespace cute;
constexpr int TM=16, TN=32, TK=256, WM=16, Bits=4, Fold=1, ArtifactTK=64;
constexpr int BasePermK=cutlass::MixGemmMmaPermK<Bits,TK,Fold>::value;
using TileShape=Shape<Int<TM>,Int<TN>,Int<TK>>;
template <int WN, int WK, class Inst_=L123F16Atom>
struct Pair {
  static_assert(TN%WN==0 && WN%16==0 && WK>0 && BasePermK*WK<=TK);
  static_assert(TK%(BasePermK*WK)==0, "warp-K requires whole expanded MMA permutations");
  using Inst=Inst_;
  static constexpr int IM=size<0>(typename MMA_Traits<Inst>::Shape_MNK{});
  static constexpr int IN=size<1>(typename MMA_Traits<Inst>::Shape_MNK{});
  static constexpr int WOM=TM/WM, WON=TN/WN;
  using WarpLayout=Layout<Shape<Int<WOM>,Int<WON>,Int<WK>>>;
#if defined(L123_BREAK_PERMK)
  using PermutationK=Int<BasePermK>;
#else
  using PermutationK=Int<BasePermK*WK>;
#endif
  using Mma=TiledMMA<MMA_Atom<Inst>,WarpLayout,
      Tile<Int<WOM*IM>,Int<WON*IN>,PermutationK>>;
  // This catches the silent half-change: AtomLayout.K=WK with the old PermK.
  static_assert(PermutationK{}==Int<BasePermK*WK>{},
                "warp-K and PermutationK must change together");
  static_assert(decltype(Mma{}.template permutation_mnk<2>()){}==PermutationK{});
  static_assert(size(Mma{})==32*WOM*WON*WK);
};
static_assert(Pair<16,1>::PermutationK{}==Int<BasePermK>{});
static_assert(Pair<16,4>::PermutationK{}==Int<TK>{});
#if defined(L123_TYPE_ONLY)
template <int WN>
using RealBuilder=cutlass::gemm::collective::quactlize_detail::get_tiled_mma<
    cutlass::arch::PPU0010,cutlass::half_t,cutlass::half_t,float,
    TileShape,Shape<Int<WM>,Int<WN>,Int<TK>>,Int<BasePermK>>;
template <int WN,int WK>
using RealMma=typename Pair<WN,WK,typename RealBuilder<WN>::MmaInst>::Mma;
static_assert(std::is_same_v<RealMma<16,1>,typename RealBuilder<16>::TiledMma>);
static_assert(std::is_same_v<RealMma<32,1>,typename RealBuilder<32>::TiledMma>);
static_assert(!std::is_same_v<RealMma<16,2>,typename RealBuilder<16>::TiledMma>);
static_assert(!std::is_same_v<RealMma<16,2>,RealMma<32,4>>,
              "equal total warp count must not collapse the independent N/K axes");
static_assert(std::is_same_v<typename RealMma<16,4>::ThrLayoutVMNK,
                             typename Pair<16,4>::Mma::ThrLayoutVMNK>);
static_assert(std::is_same_v<decltype(RealMma<16,4>{}.get_layoutA_TV()),
                             decltype(typename Pair<16,4>::Mma{}.get_layoutA_TV())>);
static_assert(std::is_same_v<decltype(RealMma<16,4>{}.get_layoutB_TV()),
                             decltype(typename Pair<16,4>::Mma{}.get_layoutB_TV())>);
static_assert(std::is_same_v<decltype(RealMma<16,4>{}.get_layoutC_TV()),
                             decltype(typename Pair<16,4>::Mma{}.get_layoutC_TV())>);
static_assert(std::is_same_v<typename RealMma<32,4>::ThrLayoutVMNK,
                             typename Pair<32,4>::Mma::ThrLayoutVMNK>);
static_assert(std::is_same_v<decltype(RealMma<32,4>{}.get_layoutB_TV()),
                             decltype(typename Pair<32,4>::Mma{}.get_layoutB_TV())>,
              "the equal-four-warp 1Nx4K control must use the real PPU layout");
using Shadow10=MMA_Traits<PPU0010_16x16x32_S32S8S8S32_TN>;
using Shadow15=MMA_Traits<PPU0015_16x16x32_S32S8S8S32_TN>;
static_assert(std::is_same_v<typename Shadow10::Shape_MNK,typename Shadow15::Shape_MNK>);
static_assert(std::is_same_v<typename Shadow10::ThrID,typename Shadow15::ThrID>);
static_assert(std::is_same_v<typename Shadow10::BLayout,typename Shadow15::BLayout>,
              "host oracle and ppu001 target must use the same shadow-B register layout");
} // namespace
int main() {
  std::puts("L123 types: WK1 is exactly shipping; compute PermK=64*WK; shadow PermK=32*WK; real/stub layouts agree PASS");
  return 0;
}
#else
template <int WN,int WK>
bool topology() {
  using P=Pair<WN,WK>; constexpr int Threads=32*P::WOM*P::WON*WK;
  std::vector<uint8_t> a[WK],b[WK],c[WK];
  for(int w=0;w<WK;++w){a[w].assign(TM*TK,0);b[w].assign(TN*TK,0);c[w].assign(TM*TN,0);}
  auto iA=make_identity_tensor(make_shape(Int<TM>{},Int<TK>{}));
  auto iB=make_identity_tensor(make_shape(Int<TN>{},Int<TK>{}));
  auto iC=make_identity_tensor(make_shape(Int<TM>{},Int<TN>{}));
  bool ok=true;
  for(int t=0;t<Threads;++t){
    auto mma=typename P::Mma{}; auto tl=mma.get_thr_layout_vmnk(); auto tc=tl.get_flat_coord(t);
    int wk=int(get<3>(tc)); auto th=mma.get_thread_slice(t);
    auto ref=mma.get_thread_slice(int(tl(make_coord(get<0>(tc),get<1>(tc),get<2>(tc),Int<0>{}))));
    auto pA=th.partition_A(iA); auto pB=th.partition_B(iB); auto pC=th.partition_C(iC);
    auto rC=ref.partition_C(iC); ok&=size(pC)==size(rC);
    for(int i=0;i<int(size(pA));++i){auto x=pA(i);++a[wk][int(get<0>(x))*TK+int(get<1>(x))];}
    for(int i=0;i<int(size(pB));++i){auto x=pB(i);++b[wk][int(get<0>(x))*TK+int(get<1>(x))];}
    for(int i=0;i<int(size(pC));++i){auto x=pC(i),r=rC(i);++c[wk][int(get<0>(x))*TN+int(get<1>(x))];
      ok&=int(get<0>(x))==int(get<0>(r))&&int(get<1>(x))==int(get<1>(r));}
  }
  for(int w=0;w<WK;++w){
    for(int m=0;m<TM;++m)for(int k=0;k<TK;++k)
      ok&=a[w][m*TK+k]==(((k/16)%WK==w)?P::WON:0);
    for(int n=0;n<TN;++n)for(int k=0;k<TK;++k)
      ok&=b[w][n*TK+k]==(((k/16)%WK==w)?P::WOM:0);
    for(auto v:c[w])ok&=v==1;
  }
  return ok;
}
template <int WN,int WK> struct Shadow {
  static constexpr int WOM=TM/WM,WON=TN/WN,RowB=TK*Bits/8,AiuElem=ArtifactTK,InstNum=TK/AiuElem;
  using Inst=PPU0015_16x16x32_S32S8S8S32_TN;
  using PermutationK=Int<32*WK>;
  using Mma=TiledMMA<MMA_Atom<Inst>,Layout<Shape<Int<WOM>,Int<WON>,Int<WK>>>,
      Tile<Int<WOM*16>,Int<WON*16>,PermutationK>>;
  static_assert(decltype(Mma{}.template permutation_mnk<2>()){}==PermutationK{},
                "shadow warp-K and its 32-code permutation must change together");
  using Op=PPU0010_TSM_LD_SWZL<int8_t,TN,AiuElem*Bits/8,true,false,InstNum>;
};
template <int WN,int WK>
std::vector<int> delivery_map(std::vector<int>* cohort_out=nullptr,std::vector<int>* vreg_out=nullptr){
  using P=Pair<WN,WK>; using S=Shadow<WN,WK>; constexpr int Threads=size(typename P::Mma{});
  using Tr=Copy_Traits<typename S::Op>; constexpr int WPR=Tr::LogicalWordsPerRow;
  static_assert(Threads==size(typename S::Mma{}) && Tr::LogicalSlices==1);
  std::vector<int> map(TN*TK,-1),cohort(TN*TK,-1),vreg(TN*TK,-1); bool valid=true;
  auto s8=make_tensor(make_smem_ptr((int8_t*)nullptr),
      make_layout(Shape<Int<TN>,Int<S::RowB>>{},Stride<Int<S::RowB>,_1>{}));
  auto sid=make_identity_tensor(make_shape(Int<TN>{},Int<S::RowB>{}));
  auto s16=make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<TN>,Int<TK>>{},Stride<Int<TK>,_1>{}));
  auto bid=make_identity_tensor(make_shape(Int<TN>{},Int<TK>{}));
  for(int t=0;t<Threads;++t){
    auto mma=typename P::Mma{};
    auto tc=mma.get_thr_layout_vmnk().get_flat_coord(t);
    int wn=int(get<2>(tc)),wk=int(get<3>(tc));
    auto load=typename S::Mma{}.get_thread_slice(t).partition_fragment_B(s8);
    // Shipping uses one AIU copy-thread slice per warp group, selected by the
    // warp's first lane rather than by the individual consumer lane.
    auto cp=make_tiled_copy_B(Copy_Atom<typename S::Op,int8_t>{},typename S::Mma{})
                .get_thread_slice((t/32)*32);
    auto src=cp.partition_S(sid); auto view=cp.retile_D(load);
    // On the 2N anchor partition_S itself already is the proposed low-cost
    // K-base: cohort wk starts at 32*wk bytes and advances by 32*WK.  The
    // simple-base failure diagnosed below therefore occurs after this seam.
    if constexpr(WN==16)for(int ck=0;ck<int(size<2>(src));++ck){auto z=src(0,0,ck);
      valid&=int(get<0>(z))==16*wn&&int(get<1>(z))==32*(wk+WK*ck);
    }
    // Compose the production objects into register ownership.  `src(0,...)`
    // supplies the cube base, LogicalTV the lane/vreg word, and retile_D the
    // destination byte.  The scalar byte/nibble walk remains explicit; WK1's
    // exact production-map anchor below makes any baseline drift fail closed.
    std::vector<int> owner(2*cosize(load.layout()),-1);
    constexpr int CN=size<1>(decltype(view.layout()){}),CK=size<2>(decltype(view.layout()){});
    for(int ck=0;ck<CK;++ck)for(int cn=0;cn<CN;++cn)for(int v=0;v<4;++v)for(int c=0;c<8;++c){
      auto base=src(0,cn,ck); int word=int(typename Tr::LogicalTV{}(
          make_coord(make_coord(t%4,(t%32)/4),make_coord(v%2,v/2),_0{})));
      int byte=(int(get<0>(base))+word/WPR)*S::RowB+int(get<1>(base))+4*(word%WPR)+c/2;
      int dst=2*int(view.layout()(4*v+c/2,cn,ck))+c%2;
      valid&=dst>=0&&dst<int(owner.size())&&(owner[dst]<0||owner[dst]==2*byte+c%2);
      if(dst>=0&&dst<int(owner.size()))owner[dst]=2*byte+c%2;
    }
    auto frag=mma.get_thread_slice(t).partition_fragment_B(s16);
    auto part=mma.get_thread_slice(t).partition_B(bid); auto pi=right_inverse(frag.layout());
    constexpr int MMAK=size<2>(decltype(frag.layout()){});
    static_assert(MMAK%CK==0); constexpr int KAPC=MMAK/CK;
    for(int kb=0;kb<CK;++kb){
      auto in=recast<cutlass::uint4b_t>(load(_,_,kb));
      constexpr int NI=size(decltype(in.layout()){})/32; static_assert(NI==CN);
      int rb=2*int(load.layout()(0,0,kb)),ob=int(frag.layout()(0,0,kb*KAPC));
      for(int ii=0;ii<NI;++ii)for(int v=0;v<4;++v)for(int c=0;c<8;++c){
        int ri=rb+int(in.layout()(0,ii))+v*8+c;
        int oi=ob+int(in.layout()(0,ii))+cutlass::MixGemmEmit<4>::index(c,v);
        valid&=ri>=0&&ri<int(owner.size())&&owner[ri]>=0;
        if(ri<0||ri>=int(owner.size())||owner[ri]<0)continue;
        auto x=part(pi(oi)); int logical=int(get<0>(x))*TK+int(get<1>(x)),slot=owner[ri];
        valid&=slot>=0&&slot<int(map.size())&&(map[slot]<0||map[slot]==logical)&&
               (cohort[slot]<0||cohort[slot]==wk)&&(vreg[slot]<0||vreg[slot]==v);
        if(slot>=0&&slot<int(map.size())){map[slot]=logical;cohort[slot]=wk;vreg[slot]=v;}
      }
    }
  }
  if(!valid)return {}; if(cohort_out)*cohort_out=cohort;if(vreg_out)*vreg_out=vreg;return map;
}
std::size_t diff(std::vector<int> const&a,std::vector<int> const&b){
  if(a.size()!=b.size())return std::max(a.size(),b.size()); std::size_t d=0;
  for(std::size_t i=0;i<a.size();++i)d+=a[i]!=b[i]; return d;
}
bool bijective(std::vector<int> m){
  std::sort(m.begin(),m.end()); if(m.size()!=std::size_t(TN*TK))return false;
  for(int i=0;i<TN*TK;++i)if(m[i]!=i)return false; return true;
}
struct KBaseResult { bool works=true; int cross_row=0,max_deltas=0,max_vreg_deltas=0; };
KBaseResult k_base_oracle(std::vector<int> const& map,std::vector<int> const& cohort,
                          std::vector<int> const& vreg,std::vector<int> const& shipping,int wk_count){
  if(map.size()!=shipping.size()||cohort.size()!=map.size()||vreg.size()!=map.size())
    return {false,int(map.size()),0,0};
  std::vector<int> inv(shipping.size(),-1); for(int p=0;p<int(shipping.size());++p)inv[shipping[p]]=p;
  KBaseResult r;
  for(int wk=0;wk<wk_count;++wk){std::vector<uint8_t> delta(TK,0);
    for(int p=0;p<int(map.size());++p)if(cohort[p]==wk){
      if(map[p]<0||map[p]>=int(inv.size())||inv[map[p]]<0){r.works=false;++r.cross_row;continue;}
      int q=inv[map[p]];
      if(q/TK!=p/TK){++r.cross_row;continue;} delta[(q-p+TK)%TK]=1;
    }
    int nd=0;for(auto x:delta)nd+=x;r.max_deltas=std::max(r.max_deltas,nd);
    r.works&=r.cross_row==0&&nd==1;
    for(int v=0;v<4;++v){std::vector<uint8_t> vd(TK,0);
      for(int p=0;p<int(map.size());++p)if(cohort[p]==wk&&vreg[p]==v){
        if(map[p]<0||map[p]>=int(inv.size())||inv[map[p]]<0)continue;int q=inv[map[p]];
        if(q/TK==p/TK)vd[(q-p+TK)%TK]=1;
      }
      int nv=0;for(auto x:vd)nv+=x;r.max_vreg_deltas=std::max(r.max_vreg_deltas,nv);
    }
  }
  return r;
}
template <int WN,int WK>
bool row(std::vector<int> const& m,std::vector<int> const& base){
  auto d=diff(m,base); bool ok=topology<WN,WK>()&&bijective(m);
  if constexpr(WN==16&&WK==1){int s=0;for(int i=0;i<int(m.size())&&s<16;++i)if(m[i]!=base[i]){
    std::printf(" D%d:%d/%d",i,base[i],m[i]);++s;}std::puts("");}
  std::printf("L123 %dNx%dK=%d warps A/B=sharded C=slot-identical Bchain=actual map=bijective diff=%zu %s\n",
              TN/WN,WK,(TN/WN)*WK,d,ok?"PASS":"FAIL"); return ok;
}
} // namespace
int main(){
  auto b=xplane::plane_map<Bits,TM,TN,TK,WM,16,Fold,ArtifactTK>();
  std::vector<int> c2,c4,v2,v4;
  auto w1=delivery_map<16,1>(),n1=delivery_map<32,1>(),k2=delivery_map<16,2>(&c2,&v2),n2=delivery_map<32,2>();
  auto k4=delivery_map<16,4>(&c4,&v4),n4=delivery_map<32,4>(); bool ok=true;
  ok&=row<16,1>(w1,b); ok&=row<32,1>(n1,b); ok&=row<16,2>(k2,b);
  ok&=row<32,2>(n2,b); ok&=row<16,4>(k4,b); ok&=row<32,4>(n4,b);
  bool same_wn=diff(k2,n2)==0&&diff(k4,n4)==0&&diff(w1,n1)==0;
  bool anchored=diff(w1,b)==0&&diff(n1,b)==0;
  // This falsifies the direct builder-shaped extension only.  A future loader
  // may preserve the artifact with an explicit address/scatter remap, but that
  // would be a new seam to prove rather than a tactic-only AtomLayout change.
  bool direct_remap=diff(k2,b)==6144&&diff(n2,b)==6144&&diff(k4,b)==6144&&diff(n4,b)==6144;
  auto fixed4_diff=diff(k2,n4); bool two_dim=fixed4_diff>0;
  auto kb2=k_base_oracle(k2,c2,v2,b,2),kb4=k_base_oracle(k4,c4,v4,b,4);
  // partition_S passed the one-base shape above.  Yet the shipping inverse
  // requires four within-row offsets per cohort (and two even after fixing a
  // vreg): the interleaved compute partition consumes the fixed converter
  // emission in an order that one pointer base cannot repair.
  bool kbase_excluded=!kb2.works&&!kb4.works&&kb2.cross_row==0&&kb4.cross_row==0&&
                      kb2.max_deltas==4&&kb4.max_deltas==4&&
                      kb2.max_vreg_deltas==2&&kb4.max_vreg_deltas==2;
  ok&=same_wn&&anchored&&direct_remap&&two_dim&&kbase_excluded;
  std::printf("L123 current=2Nx1K candidate=2Nx4K fixed4={2Nx2K,1Nx4K} "
              "shipping-anchor=%zu WN-invariant=%s direct-shadow-artifact-invariance=REFUTED(d2=%zu,d4=%zu) "
              "fixed4-diff=%zu remap-required=%s simple-K-base=%s "
              "{wk2:cross-row=%d,max-delta-set=%d,max-vreg-set=%d;"
              "wk4:cross-row=%d,max-delta-set=%d,max-vreg-set=%d} result=%s\n",
              diff(w1,b),same_wn?"YES":"NO",diff(k2,b),diff(k4,b),
              fixed4_diff,direct_remap?"DETECTED(6144/8192)":"MISSED",kbase_excluded?"A/EXCLUDED":"NOT-EXCLUDED",
              kb2.cross_row,kb2.max_deltas,kb2.max_vreg_deltas,
              kb4.cross_row,kb4.max_deltas,kb4.max_vreg_deltas,ok?"PASS":"FAIL");
  return ok?0:1;
}
#endif
