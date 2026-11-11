#include <hggc_runtime.h>
dim3 threadIdx{0,0,0};
#include "cutlass/numeric_types.h"
// Declarations only for uninstantiated device templates in vendor headers.
// No device intrinsic is executed or emulated by this layout test.
__half2 __hfma2(__half2,__half2,__half2);
int __dp4a(int,int,int);
unsigned __byte_perm(unsigned,unsigned,unsigned);
void __syncthreads();
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_decode_input.hpp"
#include <cassert>
#include <cstdio>
#include <vector>

namespace detail=cutlass::gemm::collective::detail;
template<class Source,int Height,int TileK> struct Input {
  using value_type=Source;
  Source const* data;
  Source const& operator()(int row,int k,int tile) const {
    return data[tile*Height*TileK+row*TileK+k];
  }
};

template<class Source,class Compute,int Height,int TileK,int Stages,int Threads>
int check(int valid_rows) {
  constexpr int Count=Height*TileK*Stages;
  using Operand=cutlass::gemm::config::DefaultGemm_TensorOpPPU_Operand<
      Compute,false,8,cute::Int<64>,Threads,cute::UniversalCopy<cute::uint128_t>>;
  // Use the vendor's actual 128B b16 XOR layout, not the publisher expression.
  auto layout=cute::tile_to_shape(typename Operand::SmemLayoutAtom{},
      cute::Shape<cute::Int<Height>,cute::Int<64>>{});
  std::vector<Source> input(Count);
  std::vector<Compute> got(Count+16),old(Count+16);
  for(int i=0;i<Count;++i) input[i]=Source(float((i*37)%4093-2046)/32);
  for(auto* v:{&got,&old}) for(auto& x:*v) x=Compute::bitcast(0x7fc1);
  for(int stage=0;stage<Stages;++stage) for(int tid=0;tid<Threads;++tid) {
    detail::copy_decode_a<Height,TileK,Threads,false,0,0,0,cutlass::arch::PPU0015>(
        Input<Source,Height,TileK>{input.data()},got.data(),stage,stage,tid,valid_rows);
    detail::copy_decode_a<Height,TileK,Threads,false,0,0,0>(
        Input<Source,Height,TileK>{input.data()},old.data(),stage,stage,tid,valid_rows);
  }
  int wrong=0;
  std::vector<int> owners(Count);
  for(int stage=0;stage<Stages;++stage) for(int cube=0;cube<TileK/64;++cube)
    for(int row=0;row<Height;++row) for(int k=0;k<64;++k) {
      int address=stage*Height*TileK+cube*Height*64+layout(row,k);
      auto expected=Compute(row<valid_rows ? float(input[stage*Height*TileK+row*TileK+cube*64+k]) : 0.f);
      assert(got[address].raw()==expected.raw());
      assert(++owners[address]==1);
      wrong+=old[address].raw()!=expected.raw();
    }
  for(int n:owners) assert(n==1);
  for(int i=Count;i<Count+16;++i) assert(got[i].raw()==0x7fc1 && old[i].raw()==0x7fc1);
  assert(wrong>Count/20); // Exact legacy publisher must remain observably wrong.
  return Count;
}

int main() {
  using F=cutlass::half_t;using B=cutlass::bfloat16_t;
  int cells=0;
  for(int rows:{1,9,16}) {
    cells+=check<float,B,16,64,2,32>(rows);
    cells+=check<float,B,16,128,3,128>(rows);
    cells+=check<float,F,16,256,2,128>(rows);
    cells+=check<B,B,16,256,3,64>(rows);
  }
  cells+=check<float,B,64,256,2,256>(64);
  std::printf("ZW890_TYPED_INPUT_HOST PASS cells=%d guards=PASS legacy_negative=PASS\n",cells);
}
