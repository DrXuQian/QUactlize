// Execute the actual router headers with synchronized host warp collectives.
// This checks selection/arithmetic and alias semantics, not PPU lowering.
#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

struct ThreadIndex { int x; };
thread_local ThreadIndex threadIdx;
constexpr ThreadIndex blockDim{32};
static std::barrier warp_barrier(32);
static std::array<uint32_t,32> exchange_values;
using std::isnan;
#define CUTLASS_DEVICE inline

static uint32_t exchange(uint32_t value,int source,int reduction=0) {
  const int lane=threadIdx.x%32;
  exchange_values[lane]=value;
  warp_barrier.arrive_and_wait();
  uint32_t result=exchange_values[source];
  if(reduction==1) result=*std::max_element(exchange_values.begin(),exchange_values.end());
  if(reduction==2) result=*std::min_element(exchange_values.begin(),exchange_values.end());
  warp_barrier.arrive_and_wait();
  return result;
}
template<class T> T __shfl_xor_sync(unsigned,T value,int mask,int) {
  return std::bit_cast<T>(exchange(std::bit_cast<uint32_t>(value),(threadIdx.x%32)^mask));
}
template<class T> T __shfl_sync(unsigned,T value,int source,int) {
  return std::bit_cast<T>(exchange(std::bit_cast<uint32_t>(value),source));
}
inline unsigned __reduce_max_sync(unsigned,unsigned value) { return exchange(value,0,1); }
inline int __reduce_min_sync(unsigned,int value) { return int(exchange(unsigned(value),0,2)); }
inline unsigned __float_as_uint(float value) { return std::bit_cast<unsigned>(value); }
inline float __uint_as_float(unsigned value) { return std::bit_cast<float>(value); }

#ifndef QK_ROUTER_CANDIDATE
#define QK_ROUTER_CANDIDATE "quactlize/execution/moe_router_warp.cuh"
#endif
#include QK_ROUTER_CANDIDATE

struct Result {
  std::array<int,13> ids;
  std::array<float,8> weights;
};

template<bool Coalesced>
Result run(std::array<float,256> input,bool candidate,int alias) {
  Result result;
  result.ids.fill(-123);
  result.weights.fill(-999.f);
  std::array<int,8> shared_ids;
  shared_ids.fill(-456);
  std::array<int,32> local_ids{};
  auto original=input;
  qk_llama_router_v1 r{1,sizeof(r),0,1,0,0,6.103515625e-5f,1.25f,
                       input.data(),nullptr,alias<0?result.weights.data():input.data()+alias};
  qk_llama_indexed_v1 io{};
  io.ids=result.ids.data();io.ids_stride=13;
  std::vector<std::thread> threads;
  for(int lane=0;lane<32;++lane) threads.emplace_back([&,lane] {
    // Exercise the modulo-32 lane mapping of a non-first token warp too.
    threadIdx.x=candidate&&!Coalesced ? 32+lane : lane;
    if(candidate) local_ids[lane]=quactlize::llama::router_256_top8_warp<false,Coalesced>(r,io,shared_ids.data());
    else quactlize::llama::router_256_top8<false>(r,io,shared_ids.data());
  });
  for(auto& t:threads) t.join();
  for(int s=0;s<8;++s) {
    assert(result.ids[s]>=0 && result.ids[s]<256);
    for(int j=0;j<s;++j) assert(result.ids[s]!=result.ids[j]);
    if(candidate) assert(local_ids[s]==result.ids[s]);
    assert(shared_ids[s]==(candidate&&Coalesced ? -456 : result.ids[s]));
  }
  for(int s=8;s<13;++s) assert(result.ids[s]==-123);
  if(alias>=0) {
    std::copy_n(input.begin()+alias,8,result.weights.begin());
    for(int i=0;i<256;++i) if(i<alias || i>=alias+8)
      assert(std::bit_cast<uint32_t>(input[i])==std::bit_cast<uint32_t>(original[i]));
  } else assert(!std::memcmp(input.data(),original.data(),sizeof(input)));
  return result;
}

int main() {
  int cases=0;
  for(int fixture=0;fixture<14;++fixture) {
    std::array<float,256> logits{};
    for(int i=0;i<256;++i) {
      if(fixture==1) logits[i]=(i%2)?-0.f:0.f;
      if(fixture==2) logits[i]=float(i%7);
      if(fixture==3) logits[i]=i%32==0 ? float(100-i/32) : -100.f;
      if(fixture==4) logits[i]=i%13==0 ? NAN : float(i%11);
      if(fixture==5) logits[i]=i==0 ? INFINITY : float(i%11);
      if(fixture==6) logits[i]=-INFINITY;
      if(fixture==7) logits[i]=i==127 ? 1000.f : -1000.f;
      if(fixture>=8) logits[i]=float(((i*17+fixture*29)%263)-131)*.0625f;
    }
    for(int coalesced=0;coalesced<2;++coalesced) {
      const int aliases[]={-1,0,8,248};
      // All fixtures have a disjoint-output proof; the tie-heavy and random
      // fixtures also cover each admitted in-place logits slice.
      for(int a=0;a<(fixture==2 || fixture==9 ? 4:1);++a) {
        Result reference=run<false>(logits,false,aliases[a]);
        Result candidate=coalesced?run<true>(logits,true,aliases[a]):run<false>(logits,true,aliases[a]);
        assert(reference.ids==candidate.ids);
        assert(!std::memcmp(reference.weights.data(),candidate.weights.data(),sizeof(reference.weights)));
        if(fixture<2) for(int s=0;s<8;++s) assert(candidate.ids[s]==s);
        ++cases;
      }
    }
  }
  std::printf("MOE_ROUTER_HOST PASS cases=%d scope=ORDINARY_SOFTMAX_NO_BIAS_NOT_DEVICE_ADMISSION\n",cases);
}
