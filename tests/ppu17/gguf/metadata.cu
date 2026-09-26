#include <cstdio>
#include <cstdlib>
#include <vector>
#include "quactlize/backends/ppu17/gguf_prepacked.cuh"
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "GGUF metadata failed line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

template<class T, int N, int M, int Stages> void proof() {
  using Types = quactlize::ppu17::GgufQ4Sf<T,N,M,1,Stages>;
  using Loop = typename Types::Mainloop;
  using Meta = typename Loop::Metadata;
  using namespace cute;
  CHECK(Loop::MmaK == 16 && Loop::KBlocksPerGroup == 2);
  CHECK(cosize(typename Meta::Physical{}) == N*2*Stages);
  CHECK(cosize(typename Meta::Logical{}) == N*2*Stages);
  std::vector<int> tags(N*2*Stages);
  for (int i=0; i<int(tags.size()); ++i) tags[i] = i;
  auto metadata = make_tensor(tags.data(), typename Meta::Logical{});
  auto coordinates = make_identity_tensor(make_shape(Int<N>{}, _64{}, Int<Stages>{}));
  typename Loop::TiledMma mma;
  unsigned checked = 0, stale_group = 0, stale_stage = 0;
  for (int tid=0; tid<size(mma); ++tid) {
    auto slice = mma.get_thread_slice(tid);
    auto actual = slice.partition_A(metadata);
    auto coords = slice.partition_A(coordinates);
    CHECK(size<2>(actual) == 4);
    for (int stage=0; stage<Stages; ++stage)
      for (int block=0; block<4; ++block)
        for (int row=0; row<size<1>(actual); ++row)
          for (int value=0; value<size<0>(actual); ++value) {
            auto c = coords(value,row,block,stage);
            int n = int(get<0>(c)), k = int(get<1>(c));
            int expected = stage*N*2 + (k/32)*N + n;
            CHECK(actual(value,row,block,stage) == expected);
            CHECK(actual(value,row,block-block%2,stage) == expected);
            stale_group += actual(value,row,0,stage) != expected;
            stale_stage += actual(value,row,block,(stage+1)%Stages) != expected;
            ++checked;
          }
  }
  CHECK(stale_group == checked/2 && stale_stage == checked);
  // Every metadata byte is physical; logical K-broadcast adds no TMA bytes.
  CHECK(Loop::TmaTransactionBytes == N*64/2 + M*64*2 + N*2*2*2);
  using Map = typename Types::Packing;
  auto inner = Map::ppblock_ilvd_NK_to_offset();
  std::vector<int> owners(8192);
  for (int n=0; n<128; ++n) for (int k=0; k<64; ++k) {
    int t=(n/64)*128+((n%64)/16)*32+(n%8)*4+(k%8)/2;
    int v=(k%2)+((n%16)/8)*2+((k%16)/8)*4+(k/16)*8;
    int ilv=(v/8)*8+(v%2)*4+(v%8)/2;
    int offset=inner(make_coord(n,k));
    CHECK(offset==t*32+ilv && offset>=0 && offset<8192);
    ++owners[offset];
  }
  for (int count : owners) CHECK(count==1);
  char const* dtype = std::is_same<T,cutlass::half_t>::value ? "FP16" : "BF16";
  std::printf("GGUF_Q4_METADATA PASS dtype=%s N=%d M=%d stages=%d coordinates=%u group_negative=%u stage_negative=%u tma_bytes=%u\n",
      dtype,N,M,Stages,checked,stale_group,stale_stage,Loop::TmaTransactionBytes);
}
template<class T, int Stages> void packed_proof() {
  using namespace cute;
  using Types=quactlize::ppu17::GgufQ4Fq<T,128,16,Stages>;
  using Loop=typename Types::Mainloop;
  using Unit=quactlize::ppu17::GgufQ4Unit;
  CHECK(Loop::TmaTransactionBytes==128*64/2+16*64*2+128*16);
  CHECK(cosize(typename Loop::SmemLayoutScale{})==2*128*Stages);
  CHECK(size<0>(typename Loop::SmemLayoutScale{})<=256);
  unsigned checked=0,wrong_phase=0,wrong_field=0;
  for(int example=0;example<64;++example) {
    Unit unit{};
    auto d=cutlass::half_t(.000231f*(example+1));
    auto dm=cutlass::half_t(.000117f*(example+1));
    unit.words[0]=uint32_t(d.raw())|(uint32_t(dm.raw())<<16);
    int scales[8],mins[8];
    for(int g=0;g<8;++g) {
      scales[g]=(example+g*7)&63;mins[g]=(example*3+g*11)&63;
      for(int minimum=0;minimum<2;++minimum) {
        int begin=32+(g/4)*48+minimum*24+(g%4)*6;
        unsigned value=minimum?mins[g]:scales[g];
        for(int b=0;b<6;++b)unit.words[(begin+b)/32]|=((value>>b)&1)<<((begin+b)%32);
      }
    }
    for(int g=0;g<8;++g) {
      CHECK(unit.field(g,false)==scales[g] && unit.field(g,true)==mins[g]);
      T scale(float(d)*float(scales[g])),zero(-float(dm)*float(mins[g]));
      float correction=8*float(scale);
      if constexpr(std::is_same_v<T,cutlass::bfloat16_t>)correction=float(T(correction));
      zero=T(float(zero)+correction);
      auto got=unit.template affine<T>(g);
      CHECK(got.scale.raw()==scale.raw()&&got.zero.raw()==zero.raw());
      wrong_field+=unit.field(g,false)!=unit.field(g,true);
    }
  }
  typename Loop::TiledMma mma;
  auto coordinates=make_identity_tensor(make_shape(_128{},_64{},Int<Stages>{}));
  typename Loop::SmemLayoutScale raw;
  for(int tid=0;tid<size(mma);++tid) {
    auto mapped=mma.get_thread_slice(tid).partition_A(coordinates);
    for(int start=0;start<4;++start)for(int count=0;count<Stages*3;++count) {
      int stage=count%Stages,group_begin=((start+count)%4)*2;
      for(int k=0;k<4;k+=2)for(int n=0;n<size<1>(mapped);++n)
        for(int v=0;v<size<0>(mapped);++v) {
          auto c=mapped(v,n,k,stage);
          CHECK(raw(2*get<0>(c),0,stage)==2*128*stage+2*get<0>(c));
          int group=group_begin+get<1>(c)/32;
          CHECK(group==((start+count)*64+get<1>(c))/32%8);
          wrong_phase+=(get<1>(c)/32)!=group;
          ++checked;
        }
    }
  }
  CHECK(wrong_field>0&&wrong_phase==checked*3/4);
  std::printf("GGUF_Q4_PACKED_METADATA PASS dtype=%s stages=%d coordinates=%u wrong_phase=%u fields=512\n",
      (std::is_same_v<T,cutlass::half_t> ? "FP16" : "BF16"),Stages,checked,wrong_phase);
}
int main() {
  proof<cutlass::half_t,128,16,4>();
  proof<cutlass::bfloat16_t,128,16,4>();
  proof<cutlass::half_t,128,128,8>();
  proof<cutlass::bfloat16_t,256,64,4>();
  packed_proof<cutlass::half_t,4>();
  packed_proof<cutlass::bfloat16_t,8>();
}
