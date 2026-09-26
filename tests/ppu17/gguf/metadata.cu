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
int main() {
  proof<cutlass::half_t,128,16,4>();
  proof<cutlass::bfloat16_t,128,16,4>();
  proof<cutlass::half_t,128,128,8>();
  proof<cutlass::bfloat16_t,256,64,4>();
}
