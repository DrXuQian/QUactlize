// L123 -- test Marlin's (N,K) warp grid without changing the shipping builder.
// S068 is 16x32x256/w16x16: 2N x 1K today, 2N x 4K in the candidate.
// The equal-four-warp 2Nx2K/1Nx4K pair proves N and K are separate axes.
// B's physical oracle composes the real partition_S -> retile_D -> int4
// converter-emission -> compute-fragment chain.  WK1 must reproduce the
// shipping xplane byte map.  WK is an offline-packer/artifact-descriptor axis
// (the same kind of axis as TileK and fold), not a new quantization format.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <type_traits>
#include <vector>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "quactlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#if defined(__HGGCCC__)
// The real shipping-builder identity arm is a PPU-device contract.  Keep its
// definition visible to that arm (and to the device-aware include-closure
// gate) without pulling device-only CUTLASS bodies into the host topology run.
#include "quactlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl"
#endif
#if defined(L123_TYPE_ONLY)
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#else
#include "xplane_offline.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/detail/ppu_2plane_source_layout.hpp"
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
template <int WN, int WK, class Inst_=L123F16Atom, bool ExpandComputePermK=true>
struct Pair {
  static_assert(TN%WN==0 && WN%16==0 && WK>0 && BasePermK*WK<=TK);
  static_assert(TK%(BasePermK*WK)==0, "warp-K requires whole MMA permutations");
  using Inst=Inst_;
  static constexpr int IM=size<0>(typename MMA_Traits<Inst>::Shape_MNK{});
  static constexpr int IN=size<1>(typename MMA_Traits<Inst>::Shape_MNK{});
  static constexpr int WOM=TM/WM, WON=TN/WN;
  using WarpLayout=Layout<Shape<Int<WOM>,Int<WON>,Int<WK>>>;
  // Expanding the unfolded int4 compute permutation is a construction choice,
  // not a proved requirement: CuTe repeats the 64-K permutation and the fixed
  // and expanded forms are extensionally equal below.  The shadow int8 loader
  // has a separate, real 32*WK coupling.
  using PermutationK=Int<ExpandComputePermK ? BasePermK*WK : BasePermK>;
  using Mma=TiledMMA<MMA_Atom<Inst>,WarpLayout,
      Tile<Int<WOM*IM>,Int<WON*IN>,PermutationK>>;
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
template <int WN, int WK>
using RealBuilderNK=cutlass::gemm::collective::quactlize_detail::get_tiled_mma<
    cutlass::arch::PPU0010,cutlass::half_t,cutlass::half_t,float,
    TileShape,Shape<Int<WM>,Int<WN>,Int<TK/WK>>,Int<BasePermK>>;
template <int WN,int WK,bool Expand=true>
using RealMma=typename Pair<WN,WK,typename RealBuilder<WN>::MmaInst,Expand>::Mma;
static_assert(std::is_same_v<RealMma<16,1>,typename RealBuilder<16>::TiledMma>);
static_assert(std::is_same_v<RealMma<32,1>,typename RealBuilder<32>::TiledMma>);
static_assert(!std::is_same_v<RealMma<16,2>,typename RealBuilder<16>::TiledMma>);
static_assert(!std::is_same_v<RealMma<16,2,true>,RealMma<16,2,false>>,
              "fixed and expanded K permutations are distinct types even where their maps repeat");
static_assert(std::is_same_v<typename RealBuilderNK<16,2>::TiledMma, RealMma<16,2,false>>);
static_assert(std::is_same_v<typename RealBuilderNK<16,4>::TiledMma, RealMma<16,4,false>>,
              "the shipping builder must consume WarpShape.K as the real K-cohort axis");
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
  std::puts("L123 types: WK1 is exactly shipping; fixed/expanded compute PermK are distinct types; "
            "shadow PermK=32*WK; real/stub layouts agree PASS");
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
  using PermutationK=Int<
#if defined(L123_BREAK_SHADOW_PERMK)
      32
#else
      32*WK
#endif
      >;
  using Mma=TiledMMA<MMA_Atom<Inst>,Layout<Shape<Int<WOM>,Int<WON>,Int<WK>>>,
      Tile<Int<WOM*16>,Int<WON*16>,PermutationK>>;
  static_assert(decltype(Mma{}.template permutation_mnk<2>()){}==PermutationK{},
                "shadow warp-K and its 32-code permutation must change together");
  using Op=PPU0010_TSM_LD_SWZL<int8_t,TN,AiuElem*Bits/8,true,false,InstNum>;
};
template <int WN,int WK,bool ExpandComputePermK=true>
std::vector<int> delivery_map(std::vector<int>* cohort_out=nullptr,std::vector<int>* vreg_out=nullptr,
                              std::vector<int>* code_out=nullptr,
                              std::array<int,4> const* emit_vreg_permutation=nullptr){
  using P=Pair<WN,WK,L123F16Atom,ExpandComputePermK>; using S=Shadow<WN,WK>; constexpr int Threads=size(typename P::Mma{});
  using Tr=Copy_Traits<typename S::Op>; constexpr int WPR=Tr::LogicalWordsPerRow;
  static_assert(Threads==size(typename S::Mma{}) && Tr::LogicalSlices==1);
  std::vector<int> map(TN*TK,-1),cohort(TN*TK,-1),vreg(TN*TK,-1),code(TN*TK,-1); bool valid=true;
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
        int const ev=emit_vreg_permutation?(*emit_vreg_permutation)[v]:v;
        int ri=rb+int(in.layout()(0,ii))+v*8+c;
        int oi=ob+int(in.layout()(0,ii))+cutlass::MixGemmEmit<4>::index(c,ev);
        valid&=ri>=0&&ri<int(owner.size())&&owner[ri]>=0;
        if(ri<0||ri>=int(owner.size())||owner[ri]<0)continue;
        auto x=part(pi(oi)); int logical=int(get<0>(x))*TK+int(get<1>(x)),slot=owner[ri];
        valid&=slot>=0&&slot<int(map.size())&&(map[slot]<0||map[slot]==logical)&&
               (cohort[slot]<0||cohort[slot]==wk)&&(vreg[slot]<0||vreg[slot]==v)&&
               (code[slot]<0||code[slot]==c);
        if(slot>=0&&slot<int(map.size())){map[slot]=logical;cohort[slot]=wk;vreg[slot]=v;
          code[slot]=c;}
      }
    }
  }
  if(!valid)return {}; if(cohort_out)*cohort_out=cohort;if(vreg_out)*vreg_out=vreg;
  if(code_out)*code_out=code;return map;
}
std::size_t diff(std::vector<int> const&a,std::vector<int> const&b){
  if(a.size()!=b.size())return std::max(a.size(),b.size()); std::size_t d=0;
  for(std::size_t i=0;i<a.size();++i)d+=a[i]!=b[i]; return d;
}
bool bijective(std::vector<int> m){
  std::sort(m.begin(),m.end());
  for(int i=0;i<int(m.size());++i)if(m[i]!=i)return false; return true;
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
struct PhaseResult {bool exact=true;std::array<int,4> bad{},total{};};
PhaseResult phase_oracle(std::vector<int> const& map,std::vector<int> const& cohort,
                         std::vector<int> const& vreg,std::vector<int> const& code,
                         std::vector<int> const& shipping,int wk_count){
  PhaseResult r;
  for(int p=0;p<int(map.size());++p){int v=vreg[p],w=cohort[p],c=code[p];
    if(v<0||v>=4||w<0||w>=wk_count||c<0){r.exact=false;continue;}
    // The exact four-phase source of 6144/8192: every vreg contains all four
    // phases; one phase per cohort stays fixed and three move. Thus all four
    // vregs are 1536/2048 wrong -- this is not "three bad vregs" and cannot be
    // repaired by one global vreg permutation.
    int phase=2*(v&1)+((c>>1)&1);
    bool expected=wk_count==4 ? w==phase : (w==0&&phase==0)||(w==1&&phase==3);
    bool equal=map[p]==shipping[p];r.exact&=equal==expected;++r.total[v];r.bad[v]+=!equal;
  }
  return r;
}
struct VregRemapResult {
  bool common_base=false,paired_hist=true,identity_unique_best=false;
  int best_direct=1000000,best_count=0;std::array<int,4> best{};std::map<int,int> histogram;
};
VregRemapResult fixed_vreg_remap_oracle(std::vector<int> const& shipping){
  VregRemapResult out;std::array<int,4> p{0,1,2,3};
  do {std::vector<int> c2,c4,v2,v4;
    auto k2=delivery_map<16,2>(&c2,&v2,nullptr,&p),k4=delivery_map<16,4>(&c4,&v4,nullptr,&p);
    int d2=int(diff(k2,shipping)),d4=int(diff(k4,shipping));out.paired_hist&=d2==d4;++out.histogram[d2];
    int direct=d2+d4;if(direct<out.best_direct){out.best_direct=direct;out.best=p;out.best_count=1;}
    else if(direct==out.best_direct)++out.best_count;
    auto r2=k_base_oracle(k2,c2,v2,shipping,2),r4=k_base_oracle(k4,c4,v4,shipping,4);
    out.common_base|=r2.works&&r4.works;
  } while(std::next_permutation(p.begin(),p.end()));
  out.identity_unique_best=out.best_direct==12288&&out.best_count==1&&out.best==std::array<int,4>{0,1,2,3};
  return out;
}
template <int WN,int WK>
bool row(std::vector<int> const& m,std::vector<int> const& base){
  auto d=diff(m,base); bool ok=topology<WN,WK>()&&bijective(m);
  if constexpr(WN==16&&WK==1){int s=0;for(int i=0;i<int(m.size())&&s<16;++i)if(m[i]!=base[i]){
    std::printf(" D%d:%d/%d",i,base[i],m[i]);++s;}std::puts("");}
  std::printf("L123 %dNx%dK=%d warps A/B=sharded C=slot-identical Bchain=actual map=bijective diff=%zu %s\n",
              TN/WN,WK,(TN/WN)*WK,d,ok?"PASS":"FAIL"); return ok;
}

// Folded formats use the production F*T physical row and keep the compute
// permutation at T.  Warp-K changes the shadow loader's 32-code permutation,
// but does not turn a folded N x K compute fragment back into its physical
// (N/F) x (F*K) load shape.
template <int Bits_, int TN_, int F_>
struct FoldCase {
  static constexpr int bits=Bits_,tm=16,tn=TN_,tk=256,wm=16,fold=F_,artifact_tk=64;
  static constexpr int base_permk=cutlass::MixGemmMmaPermK<bits,tk,fold>::value;
  static_assert(fold>1 && base_permk==tk);
  using Element=std::conditional_t<bits==2,cutlass::uint2b_t,cutlass::uint1b_t>;
};
using I2Case=FoldCase<2,64,2>;
using Q3LowCase=FoldCase<2,128,2>;
using Q3HighCase=FoldCase<1,128,4>;

template <class C,int WN,int WK>
struct FoldPair {
  static_assert(C::tn%WN==0 && WN%16==0 && WK>0);
  static constexpr int wom=C::tm/C::wm,won=C::tn/WN;
#if defined(L123_BREAK_FOLDED_PERMK)
  using PermutationK=Int<C::base_permk*WK>;
#else
  using PermutationK=Int<C::base_permk>;
#endif
  static_assert(PermutationK::value<=C::tk && C::tk%PermutationK::value==0,
                "folded compute PermutationK must remain the tactic TileK");
  using Mma=TiledMMA<MMA_Atom<L123F16Atom>,Layout<Shape<Int<wom>,Int<won>,Int<WK>>>,
      Tile<Int<wom*16>,Int<won*16>,PermutationK>>;
  static_assert(size(Mma{})==32*wom*won*WK);
};

template <class C,int WN,int WK>
struct FoldShadow {
  using CTV=xplane::CubeTV<C::bits,C::tm,C::tn,C::tk,C::wm,WN,C::fold,C::artifact_tk>;
  static constexpr int wom=C::tm/C::wm,won=C::tn/WN;
  using PermutationK=Int<
#if defined(L123_BREAK_SHADOW_PERMK)
      32
#else
      32*WK
#endif
      >;
  using Mma=TiledMMA<MMA_Atom<PPU0015_16x16x32_S32S8S8S32_TN>,
      Layout<Shape<Int<wom>,Int<won>,Int<WK>>>,
      Tile<Int<CTV::ShadowM::ShadowPermutationM>,Int<won*16>,PermutationK>>;
  using Op=typename CTV::Op;
  static_assert(CTV::CopyRowB==32 && Copy_Traits<Op>::LogicalSlices==1,
                "L123 folded witnesses are exact one-32B artifact deliveries");
};

template <class C,int WN,int WK>
bool fold_topology(){
  using P=FoldPair<C,WN,WK>; constexpr int Threads=size(typename P::Mma{});
  std::vector<uint8_t> a[WK],b[WK],c[WK];
  for(int w=0;w<WK;++w){a[w].assign(C::tm*C::tk,0);b[w].assign(C::tn*C::tk,0);c[w].assign(C::tm*C::tn,0);}
  auto iA=make_identity_tensor(make_shape(Int<C::tm>{},Int<C::tk>{}));
  auto iB=make_identity_tensor(make_shape(Int<C::tn>{},Int<C::tk>{}));
  auto iC=make_identity_tensor(make_shape(Int<C::tm>{},Int<C::tn>{}));
  bool ok=true;
  for(int t=0;t<Threads;++t){auto mma=typename P::Mma{};auto tl=mma.get_thr_layout_vmnk();auto tc=tl.get_flat_coord(t);
    int wk=int(get<3>(tc));auto th=mma.get_thread_slice(t);
    auto ref=mma.get_thread_slice(int(tl(make_coord(get<0>(tc),get<1>(tc),get<2>(tc),Int<0>{}))));
    auto pA=th.partition_A(iA);auto pB=th.partition_B(iB);auto pC=th.partition_C(iC);auto rC=ref.partition_C(iC);
    ok&=size(pC)==size(rC);
    for(int i=0;i<int(size(pA));++i){auto x=pA(i);++a[wk][int(get<0>(x))*C::tk+int(get<1>(x))];}
    for(int i=0;i<int(size(pB));++i){auto x=pB(i);++b[wk][int(get<0>(x))*C::tk+int(get<1>(x))];}
    for(int i=0;i<int(size(pC));++i){auto x=pC(i),r=rC(i);++c[wk][int(get<0>(x))*C::tn+int(get<1>(x))];
      ok&=int(get<0>(x))==int(get<0>(r))&&int(get<1>(x))==int(get<1>(r));}
  }
  for(int w=0;w<WK;++w){
    for(int m=0;m<C::tm;++m)for(int k=0;k<C::tk;++k)
      ok&=a[w][m*C::tk+k]==(((k/16)%WK==w)?P::won:0);
    for(int n=0;n<C::tn;++n)for(int k=0;k<C::tk;++k)
      ok&=b[w][n*C::tk+k]==(((k/16)%WK==w)?P::wom:0);
    for(auto v:c[w])ok&=v==1;
  }
  return ok;
}

struct FoldDelivery {
  std::vector<int> map,cohort,vreg,code;
  int pair_owner_bad=0;
};

template <class C,int WN,int WK>
std::vector<int> plane_owner(int t,bool& valid){
  using S=FoldShadow<C,WN,WK>;using Tr=Copy_Traits<typename S::Op>;
  constexpr int CPB=8/C::bits,CPV=32/C::bits,WPR=Tr::LogicalWordsPerRow;
  auto s8=make_tensor(make_smem_ptr((int8_t*)nullptr),
      make_layout(Shape<Int<S::CTV::Ng>,Int<S::CTV::FullRowB>>{},Stride<Int<S::CTV::FullRowB>,_1>{}));
  auto sid=make_identity_tensor(make_shape(Int<S::CTV::Ng>{},Int<S::CTV::FullRowB>{}));
  auto load=typename S::Mma{}.get_thread_slice(t).partition_fragment_B(s8);
  auto cp=make_tiled_copy_B(Copy_Atom<typename S::Op,int8_t>{},typename S::Mma{})
              .get_thread_slice((t/32)*32);
  auto src=cp.partition_S(sid);auto view=cp.retile_D(load);
  std::vector<int> owner(CPB*cosize(load.layout()),-1);
  constexpr int CN=size<1>(decltype(view.layout()){}),CK=size<2>(decltype(view.layout()){});
  for(int ck=0;ck<CK;++ck)for(int cn=0;cn<CN;++cn)for(int v=0;v<4;++v)for(int c=0;c<CPV;++c){
    auto base=src(0,cn,ck);int word=int(typename Tr::LogicalTV{}(
        make_coord(make_coord(t%4,(t%32)/4),make_coord(v%2,v/2),_0{})));
    int byte=(int(get<0>(base))+word/WPR)*S::CTV::FullRowB+int(get<1>(base))+4*(word%WPR)+c/CPB;
    int dst=CPB*int(view.layout()(4*v+c/CPB,cn,ck))+c%CPB;
    valid&=dst>=0&&dst<int(owner.size())&&(owner[dst]<0||owner[dst]==CPB*byte+c%CPB);
    if(dst>=0&&dst<int(owner.size()))owner[dst]=CPB*byte+c%CPB;
  }
  return owner;
}

template <class C,int WN,int WK>
FoldDelivery single_plane_delivery(){
  using P=FoldPair<C,WN,WK>;using S=FoldShadow<C,WN,WK>;
  constexpr int Threads=size(typename P::Mma{}),CPV=32/C::bits,CPB=8/C::bits;
  FoldDelivery out;out.map.assign(C::tn*C::tk,-1);out.cohort.assign(C::tn*C::tk,-1);
  out.vreg.assign(C::tn*C::tk,-1);out.code.assign(C::tn*C::tk,-1);bool valid=true;
  auto s8=make_tensor(make_smem_ptr((int8_t*)nullptr),
      make_layout(Shape<Int<S::CTV::Ng>,Int<S::CTV::FullRowB>>{},Stride<Int<S::CTV::FullRowB>,_1>{}));
  auto s16=make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<C::tn>,Int<C::tk>>{},Stride<Int<C::tk>,_1>{}));
  auto bid=make_identity_tensor(make_shape(Int<C::tn>{},Int<C::tk>{}));
  for(int t=0;t<Threads;++t){auto mma=typename P::Mma{};auto tc=mma.get_thr_layout_vmnk().get_flat_coord(t);
    int wk=int(get<3>(tc));auto load=typename S::Mma{}.get_thread_slice(t).partition_fragment_B(s8);
    auto owner=plane_owner<C,WN,WK>(t,valid);auto frag=mma.get_thread_slice(t).partition_fragment_B(s16);
    auto part=mma.get_thread_slice(t).partition_B(bid);auto pi=right_inverse(frag.layout());
    constexpr int CK=size<2>(decltype(load.layout()){});using Scatter=cutlass::MixGemmArtifactScatter<C::bits,C::fold,CK>;
    for(int kb=0;kb<CK;++kb){auto in=recast<typename C::Element>(load(_,_,kb));
      constexpr int NI=size<1>(decltype(in.layout()){});static_assert(size<0>(decltype(in.layout()){})==4*CPV);
      for(int ii=0;ii<NI;++ii)for(int v=0;v<4;++v)for(int c=0;c<CPV;++c){
        int ri=CPB*int(load.layout()(0,0,kb))+int(in.layout()(0,ii))+v*CPV+c;
        int oi=Scatter::flat(cutlass::MixGemmEmit<C::bits>::index(c,v),kb,ii);
        valid&=ri>=0&&ri<int(owner.size())&&owner[ri]>=0&&oi>=0&&oi<int(size(frag));
        if(ri<0||ri>=int(owner.size())||owner[ri]<0||oi<0||oi>=int(size(frag)))continue;
        auto x=part(pi(oi));int logical=int(get<0>(x))*C::tk+int(get<1>(x)),slot=owner[ri];
        valid&=slot>=0&&slot<int(out.map.size())&&(out.map[slot]<0||out.map[slot]==logical)&&
               (out.cohort[slot]<0||out.cohort[slot]==wk)&&(out.vreg[slot]<0||out.vreg[slot]==v)&&
               (out.code[slot]<0||out.code[slot]==c);
        if(slot>=0&&slot<int(out.map.size())){out.map[slot]=logical;out.cohort[slot]=wk;out.vreg[slot]=v;out.code[slot]=c;}
      }
    }
  }
  if(!valid)++out.pair_owner_bad;
  for(int x:out.map)out.pair_owner_bad+=x<0;
  return out;
}

constexpr bool q3_low_emit_identity(){using Cvt=cutlass::MixGemm2Plane<2,1>;
  for(int v=0;v<4;++v)for(int lt=0;lt<Cvt::kPairs;++lt)for(int half=0;half<2;++half)
    if(2*Cvt::at_plain(lt,v)+half!=
       cutlass::MixGemmEmit<2>::index(Cvt::lo_code(lt,half),v))return false;
  return true;
}
static_assert(q3_low_emit_identity(),"two-plane low output must be the single-plane int2 emission");

template <int WN,int WK,bool StaleHigh=false>
std::array<FoldDelivery,2> q3_delivery(){
  using C=Q3LowCase;using H=Q3HighCase;using P=FoldPair<C,WN,WK>;
  using LS=FoldShadow<C,WN,WK>;using HSd=FoldShadow<H,WN,WK>;
  constexpr int Threads=size(typename P::Mma{}),LCPV=16,HCPV=32,LCPB=4,HCPB=8;
  std::array<FoldDelivery,2> out;out[0]=single_plane_delivery<C,WN,WK>();
  out[1].map.assign(C::tn*C::tk,-1);out[1].cohort.assign(C::tn*C::tk,-1);
  out[1].vreg.assign(C::tn*C::tk,-1);out[1].code.assign(C::tn*C::tk,-1);
  std::vector<int> low_seen(C::tn*C::tk),high_hits(C::tn*C::tk);bool valid=true;
  auto ls8=make_tensor(make_smem_ptr((int8_t*)nullptr),
      make_layout(Shape<Int<LS::CTV::Ng>,Int<LS::CTV::FullRowB>>{},Stride<Int<LS::CTV::FullRowB>,_1>{}));
  auto hs8=make_tensor(make_smem_ptr((int8_t*)nullptr),
      make_layout(Shape<Int<HSd::CTV::Ng>,Int<HSd::CTV::FullRowB>>{},Stride<Int<HSd::CTV::FullRowB>,_1>{}));
  using Cvt=cutlass::MixGemm2Plane<2,1>;constexpr int P2DIV=(C::fold*C::bits)/(H::fold*H::bits);
  static_assert(P2DIV==1);
  for(int t=0;t<Threads;++t){auto mma=typename P::Mma{};auto tc=mma.get_thr_layout_vmnk().get_flat_coord(t);
    int wk=int(get<3>(tc));auto ll=typename LS::Mma{}.get_thread_slice(t).partition_fragment_B(ls8);
    auto hl=typename HSd::Mma{}.get_thread_slice(t).partition_fragment_B(hs8);
    auto lo_owner=plane_owner<C,WN,WK>(t,valid),hi_owner=plane_owner<H,WN,WK>(t,valid);
    constexpr int LCK=size<2>(decltype(ll.layout()){});
    for(int kb=0;kb<LCK;++kb){auto li=recast<cutlass::uint2b_t>(ll(_,_,kb));auto hi=recast<cutlass::uint1b_t>(hl(_,_,kb/P2DIV));
      constexpr int NumIter=size<1>(decltype(li.layout()){}),N2=size<1>(decltype(hi.layout()){});
      using Src=HiPlaneSrc<N2,NumIter,P2DIV>;
      for(int ii=0;ii<NumIter;++ii)for(int v=0;v<4;++v)for(int lt=0;lt<Cvt::kPairs;++lt)for(int half=0;half<2;++half){
        int lc=Cvt::lo_code(lt,half),hc=Cvt::hi_code(lt,v,half);
        int lr=LCPB*int(ll.layout()(0,0,kb))+int(li.layout()(0,ii))+v*LCPV+lc;
        int hbase=StaleHigh ? kb%P2DIV : Src::base(ii,kb);
        int hr=HCPB*int(hl.layout()(0,0,kb/P2DIV))+int(hi.layout()(0,Src::slot(ii)))+
               (hbase+Cvt::hi_vreg(v))*HCPV+hc;
        bool inrange=lr>=0&&lr<int(lo_owner.size())&&lo_owner[lr]>=0&&
                     hr>=0&&hr<int(hi_owner.size())&&hi_owner[hr]>=0;
        valid&=inrange;if(!inrange)continue;int lslot=lo_owner[lr],hslot=hi_owner[hr],logical=out[0].map[lslot];
        ++low_seen[lslot];bool good=logical>=0&&hslot>=0&&hslot<int(out[1].map.size())&&
          (out[1].map[hslot]<0||out[1].map[hslot]==logical)&&
          (out[1].cohort[hslot]<0||out[1].cohort[hslot]==wk)&&
          (out[1].vreg[hslot]<0||out[1].vreg[hslot]==hbase+Cvt::hi_vreg(v))&&
          (out[1].code[hslot]<0||out[1].code[hslot]==hc);
        if(!good){++out[1].pair_owner_bad;valid=false;continue;}
        out[1].map[hslot]=logical;out[1].cohort[hslot]=wk;
        out[1].vreg[hslot]=hbase+Cvt::hi_vreg(v);out[1].code[hslot]=hc;++high_hits[logical];
      }
    }
  }
  if(!valid)++out[1].pair_owner_bad;
  for(int i=0;i<int(out[1].map.size());++i){out[0].pair_owner_bad+=low_seen[i]!=1;
    out[1].pair_owner_bad+=out[1].map[i]<0||high_hits[i]!=1;}
  return out;
}

template <class C,int WNA,int WNB>
std::size_t stored_byte_diff_cross(std::vector<int> const& a,std::vector<int> const& b){
  constexpr int Bytes=C::tn*C::tk*C::bits/8,Passes=(17+C::bits-1)/C::bits,Mask=(1<<C::bits)-1;
  std::vector<int8_t> pa(Bytes),pb(Bytes);std::vector<uint8_t> q(C::tn*C::tk);std::vector<uint8_t> bad(Bytes);
  for(int pass=0;pass<Passes;++pass){for(int k=0;k<C::tk;++k)for(int n=0;n<C::tn;++n)
      q[k*C::tn+n]=uint8_t(((k*C::tn+n+1)>>(pass*C::bits))&Mask);
    xplane::place_from_map<C::bits,C::tm,C::tn,C::tk,C::wm,WNA,C::fold,C::artifact_tk>(pa.data(),a,q,C::tn,C::tk);
    xplane::place_from_map<C::bits,C::tm,C::tn,C::tk,C::wm,WNB,C::fold,C::artifact_tk>(pb.data(),b,q,C::tn,C::tk);
    for(int i=0;i<Bytes;++i)bad[i]|=pa[i]!=pb[i];}
  std::size_t d=0;for(auto x:bad)d+=x;return d;
}
template <class C,int WN>
std::size_t stored_byte_diff(std::vector<int> const& a,std::vector<int> const& b){
  return stored_byte_diff_cross<C,WN,WN>(a,b);
}

struct FoldMetric {int cross_row=0,max_delta=0,max_vreg_delta=0;};
template <class C>
FoldMetric fold_metric(FoldDelivery const& x,std::vector<int> const& shipping,int wk_count){
  FoldMetric r;constexpr int RowCodes=C::fold*C::tk;std::vector<int> inv(shipping.size(),-1);
  for(int p=0;p<int(shipping.size());++p)if(shipping[p]>=0&&shipping[p]<int(inv.size()))inv[shipping[p]]=p;
  for(int wk=0;wk<wk_count;++wk){std::vector<uint8_t> ds(RowCodes);
    for(int p=0;p<int(x.map.size());++p)if(x.cohort[p]==wk){int q=x.map[p]>=0&&x.map[p]<int(inv.size())?inv[x.map[p]]:-1;
      if(q<0||q/RowCodes!=p/RowCodes){++r.cross_row;continue;}ds[(q-p+RowCodes)%RowCodes]=1;}
    int nd=0;for(auto z:ds)nd+=z;r.max_delta=std::max(r.max_delta,nd);
    for(int v=0;v<4;++v){std::vector<uint8_t> vd(RowCodes);
      for(int p=0;p<int(x.map.size());++p)if(x.cohort[p]==wk&&x.vreg[p]==v){int q=x.map[p]>=0&&x.map[p]<int(inv.size())?inv[x.map[p]]:-1;
        if(q>=0&&q/RowCodes==p/RowCodes)vd[(q-p+RowCodes)%RowCodes]=1;}
      int nv=0;for(auto z:vd)nv+=z;r.max_vreg_delta=std::max(r.max_vreg_delta,nv);}
  }
  return r;
}

bool fold_row_metric_controls(){
  using C=I2Case;constexpr int RowCodes=C::fold*C::tk,Total=C::tn*C::tk;
  std::vector<int> identity(Total);for(int i=0;i<Total;++i)identity[i]=i;
  FoldDelivery within,across;
  for(auto* x:{&within,&across}){x->map.resize(Total);x->cohort.assign(Total,0);x->vreg.assign(Total,0);}
  for(int p=0;p<Total;++p){int row=p/RowCodes,off=p%RowCodes;
    within.map[p]=row*RowCodes+(off+C::tk)%RowCodes;
    across.map[p]=((row+1)%(Total/RowCodes))*RowCodes+off;
  }
  auto a=fold_metric<C>(within,identity,1),b=fold_metric<C>(across,identity,1);
  return a.cross_row==0&&a.max_delta==1&&b.cross_row==Total;
}

template <class C,int WN,int WK>
bool fold_row(char const* name,FoldDelivery const& x,std::vector<int> const& base){
  auto sd=diff(x.map,base),bd=stored_byte_diff<C,WN>(x.map,base);auto m=fold_metric<C>(x,base,WK);
  bool ok=fold_topology<C,WN,WK>()&&bijective(x.map)&&x.pair_owner_bad==0;
  constexpr std::size_t Entries=C::tn*C::tk,Bytes=Entries*C::bits/8;
  if constexpr(WK==1)ok&=sd==0&&bd==0&&m.cross_row==0&&m.max_delta==1&&m.max_vreg_delta==1;
  else ok&=sd==3*Entries/4&&bd==Bytes&&m.cross_row==0&&m.max_delta==4&&m.max_vreg_delta==4;
  std::printf("L123 %s WN=%d WK=%d entries=%zu slot-diff=%zu stored-byte-diff=%zu "
              "cross-physical-row=%d max-delta=%d max-vreg=%d pair-owner-bad=%d %s\n",
              name,WN,WK,x.map.size(),sd,bd,m.cross_row,m.max_delta,m.max_vreg_delta,x.pair_owner_bad,ok?"PASS":"FAIL");
  return ok;
}
} // namespace
int main(){
  auto b=xplane::plane_map<Bits,TM,TN,TK,WM,16,Fold,ArtifactTK>();
  std::vector<int> c2,c4,v2,v4,code2,code4;
  auto w1=delivery_map<16,1>(),n1=delivery_map<32,1>(),k2=delivery_map<16,2>(&c2,&v2),n2=delivery_map<32,2>();
  k2=delivery_map<16,2>(&c2,&v2,&code2);
  auto k4=delivery_map<16,4>(&c4,&v4,&code4),n4=delivery_map<32,4>(); bool ok=true;
#if defined(L123_BREAK_SHADOW_PERMK)
  auto stale2=single_plane_delivery<I2Case,32,2>();
  auto stale4=single_plane_delivery<I2Case,32,4>();
  auto holes=[](FoldDelivery const& x){int n=0;for(int v:x.map)n+=v<0;return n;};
  int h2=holes(stale2),h4=holes(stale4);
  bool shadow_red=h2==8192&&h4==12288&&stale2.pair_owner_bad>0&&stale4.pair_owner_bad>0;
  std::printf("L123 negative: shadow AtomLayout.K changed with stale PermK=32 -> "
              "WK2-holes=%d/16384 WK4-holes=%d/16384 %s\n",h2,h4,
              shadow_red?"EXPECTED-RED":"UNEXPECTED-GREEN");
  return shadow_red?0:1;
#endif
  ok&=row<16,1>(w1,b); ok&=row<32,1>(n1,b); ok&=row<16,2>(k2,b);
  ok&=row<32,2>(n2,b); ok&=row<16,4>(k4,b); ok&=row<32,4>(n4,b);
  bool same_wn=diff(k2,n2)==0&&diff(k4,n4)==0&&diff(w1,n1)==0;
  bool anchored=diff(w1,b)==0&&diff(n1,b)==0;
  bool direct_remap=diff(k2,b)==6144&&diff(n2,b)==6144&&diff(k4,b)==6144&&diff(n4,b)==6144;
  auto fixed4_diff=diff(k2,n4); bool two_dim=fixed4_diff>0;
  auto kb2=k_base_oracle(k2,c2,v2,b,2),kb4=k_base_oracle(k4,c4,v4,b,4);
  auto ph2=phase_oracle(k2,c2,v2,code2,b,2),ph4=phase_oracle(k4,c4,v4,code4,b,4);
  bool phase_exact=ph2.exact&&ph4.exact;
  for(int v=0;v<4;++v)phase_exact&=ph2.total[v]==2048&&ph2.bad[v]==1536&&
                                      ph4.total[v]==2048&&ph4.bad[v]==1536;
  auto vr=fixed_vreg_remap_oracle(b);
  bool kbase_excluded=!kb2.works&&!kb4.works&&kb2.cross_row==0&&kb4.cross_row==0&&
                      kb2.max_deltas==4&&kb4.max_deltas==4&&
                      kb2.max_vreg_deltas==2&&kb4.max_vreg_deltas==2;
  auto k2_fixed=delivery_map<16,2,false>(),k4_fixed=delivery_map<16,4,false>();
  bool compute_permk_repeats=diff(k2,k2_fixed)==0&&diff(k4,k4_fixed)==0;
  bool hist_shape=vr.histogram==std::map<int,int>{{6144,1},{7168,6},{7680,8},{8192,9}};
  bool fixed_vreg_excluded=!vr.common_base&&vr.identity_unique_best&&vr.paired_hist&&
                           hist_shape;
  ok&=same_wn&&anchored&&direct_remap&&two_dim&&phase_exact&&kbase_excluded&&
      compute_permk_repeats&&fixed_vreg_excluded;
  std::printf("L123 current=2Nx1K candidate=2Nx4K fixed4={2Nx2K,1Nx4K} "
              "shipping-anchor=%zu WN-invariant=%s direct-shadow-artifact-invariance=REFUTED(d2=%zu,d4=%zu) "
              "fixed4-diff=%zu compute-PermK-fixed-vs-expanded=%s simple-K-base=%s "
              "phase-rule=%s vreg-bad=%d/%d/%d/%d fixed-vreg-plus-base=%s "
              "{wk2:cross-row=%d,max-delta-set=%d,max-vreg-set=%d;"
              "wk4:cross-row=%d,max-delta-set=%d,max-vreg-set=%d} result=%s\n",
              diff(w1,b),same_wn?"YES":"NO",diff(k2,b),diff(k4,b),
              fixed4_diff,compute_permk_repeats?"EQUAL":"DIFFERENT",kbase_excluded?"EXCLUDED":"NOT-EXCLUDED",
              phase_exact?"EXACT":"FAIL",ph4.bad[0],ph4.bad[1],ph4.bad[2],ph4.bad[3],
              fixed_vreg_excluded?"EXCLUDED":"NOT-EXCLUDED",
              kb2.cross_row,kb2.max_deltas,kb2.max_vreg_deltas,
              kb4.cross_row,kb4.max_deltas,kb4.max_vreg_deltas,ok?"PASS":"FAIL");
  std::printf("L123 int4 global-vreg-remap direct-diff-hist=");
  for(auto const& [d,n]:vr.histogram)std::printf("%s%d:%d",d==vr.histogram.begin()->first?"":",",d,n);
  std::puts("; identity is the unique best and no permutation + per-cohort base serves WK2/WK4");

  auto i2b32=xplane::plane_map<2,16,64,256,16,32,2,64>();
  auto i2b64=xplane::plane_map<2,16,64,256,16,64,2,64>();
  auto i2_32_1=single_plane_delivery<I2Case,32,1>(),i2_32_2=single_plane_delivery<I2Case,32,2>(),
       i2_32_4=single_plane_delivery<I2Case,32,4>();
  auto i2_64_1=single_plane_delivery<I2Case,64,1>(),i2_64_2=single_plane_delivery<I2Case,64,2>(),
       i2_64_4=single_plane_delivery<I2Case,64,4>();
  ok&=fold_row<I2Case,32,1>("int2-F2",i2_32_1,i2b32);
  ok&=fold_row<I2Case,32,2>("int2-F2",i2_32_2,i2b32);
  ok&=fold_row<I2Case,32,4>("int2-F2",i2_32_4,i2b32);
  ok&=fold_row<I2Case,64,1>("int2-F2",i2_64_1,i2b64);
  ok&=fold_row<I2Case,64,2>("int2-F2",i2_64_2,i2b64);
  ok&=fold_row<I2Case,64,4>("int2-F2",i2_64_4,i2b64);
  auto i2_wn_slot=diff(i2b32,i2b64),i2_wn_bytes=stored_byte_diff_cross<I2Case,32,64>(i2b32,i2b64);
  ok&=i2_wn_slot==8192&&i2_wn_bytes==2048;
  std::printf("L123 int2-F2 WN-baseline-slot-diff=%zu WN-baseline-stored-byte-diff=%zu\n",
              i2_wn_slot,i2_wn_bytes);

  auto q3lo64=xplane::plane_map<2,16,128,256,16,64,2,64>();
  auto q3lo128=xplane::plane_map<2,16,128,256,16,128,2,64>();
  auto q3hi64=xplane::tile_map_hi<2,1,16,128,256,16,64,4,2,64>();
  auto q3hi128=xplane::tile_map_hi<2,1,16,128,256,16,128,4,2,64>();
  auto q3_64_1=q3_delivery<64,1>(),q3_64_2=q3_delivery<64,2>(),q3_64_4=q3_delivery<64,4>();
  auto q3_128_1=q3_delivery<128,1>(),q3_128_2=q3_delivery<128,2>(),q3_128_4=q3_delivery<128,4>();
  ok&=fold_row<Q3LowCase,64,1>("Q3-low2-F2",q3_64_1[0],q3lo64);
  ok&=fold_row<Q3HighCase,64,1>("Q3-high1-F4",q3_64_1[1],q3hi64);
  ok&=fold_row<Q3LowCase,64,2>("Q3-low2-F2",q3_64_2[0],q3lo64);
  ok&=fold_row<Q3HighCase,64,2>("Q3-high1-F4",q3_64_2[1],q3hi64);
  ok&=fold_row<Q3LowCase,64,4>("Q3-low2-F2",q3_64_4[0],q3lo64);
  ok&=fold_row<Q3HighCase,64,4>("Q3-high1-F4",q3_64_4[1],q3hi64);
  ok&=fold_row<Q3LowCase,128,1>("Q3-low2-F2",q3_128_1[0],q3lo128);
  ok&=fold_row<Q3HighCase,128,1>("Q3-high1-F4",q3_128_1[1],q3hi128);
  ok&=fold_row<Q3LowCase,128,2>("Q3-low2-F2",q3_128_2[0],q3lo128);
  ok&=fold_row<Q3HighCase,128,2>("Q3-high1-F4",q3_128_2[1],q3hi128);
  ok&=fold_row<Q3LowCase,128,4>("Q3-low2-F2",q3_128_4[0],q3lo128);
  ok&=fold_row<Q3HighCase,128,4>("Q3-high1-F4",q3_128_4[1],q3hi128);
  auto stale=q3_delivery<64,1,true>();bool stale_red=stale[0].pair_owner_bad>0||stale[1].pair_owner_bad>0;
  ok&=stale_red;
  auto q3lo_wn_slot=diff(q3lo64,q3lo128),q3hi_wn_slot=diff(q3hi64,q3hi128);
  auto q3lo_wn_bytes=stored_byte_diff_cross<Q3LowCase,64,128>(q3lo64,q3lo128);
  auto q3hi_wn_bytes=stored_byte_diff_cross<Q3HighCase,64,128>(q3hi64,q3hi128);
  ok&=q3lo_wn_slot==16384&&q3hi_wn_slot==16384&&q3lo_wn_bytes==4096&&q3hi_wn_bytes==2048;
  std::printf("L123 Q3 pair-owner=PASS stale-HiPlaneSrc=%s "
              "WN-low={slot:%zu,bytes:%zu} WN-high={slot:%zu,bytes:%zu}\n",
              stale_red?"EXPECTED-RED":"UNEXPECTED-GREEN",q3lo_wn_slot,q3lo_wn_bytes,
              q3hi_wn_slot,q3hi_wn_bytes);
  bool row_metric_ok=fold_row_metric_controls();ok&=row_metric_ok;
  std::printf("L123 fold-row-metric +TK=same-row +(F*TK)=next-row %s\n",row_metric_ok?"PASS":"FAIL");
  std::printf("L123 conclusion: WK is an offline-packer/artifact-descriptor parameter carried with TileK/fold (#37), "
              "not a new quantization format; WK1=0/8192 remains permanent. WN is byte-map invariant for the F1-int4 "
              "anchor; folded int2/Q3 report their distinct WN artifact classes above.\n");
  return ok?0:1;
}
#endif
