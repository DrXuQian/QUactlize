// L213 -- classify the Q4/A32 device mismatch by constructive output shape.
//
// Reuse L123's exact production source ownership, but deliberately change only
// the destination scatter.  The real M64/N1024/K5120 fixture is then evaluated
// through each planted map.  Matching raw_bad and first_bad would identify the
// failed seam without treating an aggregate mismatch count as a diagnosis.

#define main l123_embedded_main
#include "l123_warp_nk_topology.cu"
#undef main

#include <array>
#include <cmath>

namespace {

enum class ScatterPlant { Correct, DeliveryMajor };

FoldDelivery q4_map(ScatterPlant plant) {
  using C = Q4A32Case;
  using P = FoldPair<C,32,1>;
  using S = FoldShadow<C,32,1>;
  constexpr int Threads=size(typename P::Mma{}),CPV=8,CPB=2;
  constexpr int CK=4;
  using Scatter=cutlass::MixGemmArtifactScatter<4,2,CK>;
  FoldDelivery out;
  out.map.assign(C::tn*C::tk,-1);
  out.cohort.assign(C::tn*C::tk,-1);
  out.vreg.assign(C::tn*C::tk,-1);
  out.code.assign(C::tn*C::tk,-1);
  bool valid=true;
  auto s8=make_tensor(make_smem_ptr((int8_t*)nullptr),
      make_layout(Shape<Int<S::CTV::Ng>,Int<S::CTV::FullRowB>>{},
                  Stride<Int<S::CTV::FullRowB>,_1>{}));
  auto s16=make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<C::tn>,Int<C::tk>>{},Stride<Int<C::tk>,_1>{}));
  auto bid=make_identity_tensor(make_shape(Int<C::tn>{},Int<C::tk>{}));
  for(int t=0;t<Threads;++t) {
    auto mma=typename P::Mma{};
    auto load=typename S::Mma{}.get_thread_slice(t).partition_fragment_B(s8);
    auto owner=plane_owner<C,32,1>(t,valid);
    auto frag=mma.get_thread_slice(t).partition_fragment_B(s16);
    auto part=mma.get_thread_slice(t).partition_B(bid);
    auto pi=right_inverse(frag.layout());
    std::array<int,128> raw_logical{};
    for(int raw=0;raw<128;++raw) {
      auto x=part(pi(raw));
      raw_logical[raw]=int(get<0>(x))*C::tk+int(get<1>(x));
    }
    for(int kb=0;kb<CK;++kb) {
      auto in=recast<typename C::Element>(load(_,_,kb));
      for(int v=0;v<4;++v)for(int c=0;c<CPV;++c) {
        int const ri=CPB*int(load.layout()(0,0,kb))+
                     int(in.layout()(0,0))+v*CPV+c;
        int const e=cutlass::MixGemmEmit<4>::index(c,v);
        int const raw=plant==ScatterPlant::Correct
            ? Scatter::flat(e,kb,0)
            : kb*32+e;
        int const slot=(ri>=0&&ri<int(owner.size()))?owner[ri]:-1;
        valid&=slot>=0&&slot<int(out.map.size())&&raw>=0&&raw<128;
        if(slot<0||slot>=int(out.map.size())||raw<0||raw>=128)continue;
        int const logical=raw_logical[raw];
        valid&=out.map[slot]<0||out.map[slot]==logical;
        out.map[slot]=logical;
      }
    }
  }
  if(!valid)++out.pair_owner_bad;
  for(int x:out.map)out.pair_owner_bad+=x<0;
  return out;
}

constexpr int M=64,N=1024,K=5120,GS=32;
int qvalue(int n,int k) { return ((13*n+7*k+3)%15)-7; }
int scale(int n,int k) { return 1<<((17*(k/GS)+29*n+1)%3); }
int zero(int n,int k) { return ((11*(k/GS)+7*n)%3-1)*3; }

std::pair<int,float> signature(FoldDelivery const& consumer,
                               std::vector<int> const& producer) {
  std::vector<int> inverse(Q4A32Case::tn*Q4A32Case::tk,-1);
  for(int p=0;p<int(consumer.map.size());++p)
    if(consumer.map[p]>=0)inverse[consumer.map[p]]=p;
  std::array<int,8> active{};
  for(int s=0;s<8;++s)active[s]=s*K/8+(37*s+11)%(K/8);
  int bad=0;float first=0;
  for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
    float want=0,got=0;
    for(int s=0;s<8;++s) {
      int const k=active[s],ln=n%64,lk=k%128;
      int const p=inverse[ln*128+lk];
      int sn=n,sk=k;
      if(p>=0) {
        int const src=producer[p];
        sn=(n/64)*64+src/128;
        sk=(k/128)*128+src%128;
      }
      float const a=((m+s)&1)?-.5f:.5f;
      want+=a*(scale(n,k)*qvalue(n,k)+zero(n,k));
      got +=a*(scale(n,k)*(p>=0?qvalue(sn,sk):-8)+zero(n,k));
    }
    if(got!=want) { if(bad==0)first=got; ++bad; }
  }
  return {bad,first};
}

} // namespace

int main() {
  auto producer=xplane::plane_map<4,64,64,128,16,32,2,32>();
  auto good=q4_map(ScatterPlant::Correct);
  auto transposed=q4_map(ScatterPlant::DeliveryMajor);
  auto a=signature(good,producer),b=signature(transposed,producer);
  bool ok=good.pair_owner_bad==0&&a.first==0&&a.second==0&&
          transposed.pair_owner_bad==0&&b.first!=0;
  std::printf("L213 correct raw_bad=%d first=%g; delivery-major raw_bad=%d first=%g %s\n",
              a.first,a.second,b.first,b.second,ok?"PASS":"FAIL");
  return ok?0:1;
}
