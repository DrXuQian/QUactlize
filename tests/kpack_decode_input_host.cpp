#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_decode_input.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <stdexcept>
#include <vector>

using Half=cutlass::half_t;
namespace detail=cutlass::gemm::collective::detail;

template<class Scalar> struct Input {
  using value_type=Scalar;
  Scalar const* data;
  int rows,k,tile;
  Scalar const& operator()(int r,int c,int t) const {
    if (r<0 || r>=rows || c<0 || c>=tile || t<0 || (t+1)*tile>k)
      throw std::runtime_error("source load escaped real A, including padding rows");
    return data[r*k+t*tile+c];
  }
};

// Independent calibrated PPU TSM read side (not the writer's address helper).
static int read_half(int h,int row,int col) {
  int slice=col/16, vreg=(row%16/8)*2+col%16/8;
  int lane=(row%8)*4+col%8/2, coord_h=row/16*16;
  int line=((vreg/2)*8+lane/4+coord_h)/4;
  int vector=((((vreg/2)*8+lane/4+coord_h)%4)*2+vreg%2);
  int start=(((slice&1)<<1)+((slice&2)>>1))*2;
  int swizzled=((vector+start)%8)^(line&1);
  return 2*(h*8*slice+line*32+swizzled*4+lane%4)+col%2;
}

template<int Height,int TK,int Threads,bool Packed,class Scalar>
static void run(int fault=0) {
  constexpr int pitch=Packed?detail::aPackPitchForRows(1):Height*64;
  constexpr int stage_pitch=Packed?detail::aPackStagePitchHalfs(pitch,TK/64,Height*64):Height*TK;
  for (int m:{1,2,3,4,5,6,7,8}) {
    if (Packed && m!=1) continue;
    constexpr int k=TK*4,guard=64;
    std::vector<Scalar> input(m*k);
    for (size_t i=0;i<input.size();++i) input[i]=Scalar(float(int(i%3001)-1500)*.017391f);
    Input<Scalar> tensor{input.data(),m,k,TK};
    std::vector<Half> memory(stage_pitch*3+2*guard,Half(-777.f));
    for (int stage=0;stage<3;++stage) for (int tid=0;tid<Threads;++tid) {
      if (fault==2 && tid==0) continue;
      detail::copy_decode_a<Height,TK,Threads,Packed,1,pitch,stage_pitch>(
          tensor,memory.data()+guard,stage+1-(fault==1),(stage+(fault==3))%3,tid,m);
    }
    for (int stage=0;stage<3;++stage) for (int cube=0;cube<TK/64;++cube)
      for (int row=0;row<(Packed?1:Height);++row) for (int col=0;col<64;++col) {
        int at=guard+stage*stage_pitch+cube*pitch+read_half(Height,row,col);
        Half want=row<m?Half(float(input[row*k+(stage+1)*TK+cube*64+col])):Half(0.f);
        if (memory.at(at).raw()!=want.raw()) {
          std::fprintf(stderr,"first: bits=%zu H=%d TK=%d threads=%d packed=%d M=%d stage=%d cube=%d row=%d col=%d at=%d want=%04x got=%04x\n",
              sizeof(Scalar)*8,Height,TK,Threads,int(Packed),m,stage,cube,row,col,at,want.raw(),memory.at(at).raw());
          throw std::runtime_error("physical TSM reader disagrees with typed A");
        }
      }
    for (int i=0;i<guard;++i)
      if (memory[i].raw()!=Half(-777.f).raw() || memory[memory.size()-1-i].raw()!=Half(-777.f).raw())
        throw std::runtime_error("decode writer changed a guard");
  }
}

template<class Scalar> static void matrix() {
  run<16,64,32,false,Scalar>();run<16,128,64,false,Scalar>();run<16,256,128,false,Scalar>();
  run<32,256,64,false,Scalar>();run<64,128,128,false,Scalar>();
  run<128,256,256,false,Scalar>();run<256,64,256,false,Scalar>();
  run<16,64,32,true,Scalar>();run<16,128,64,true,Scalar>();run<16,256,128,true,Scalar>();
  for (int fault:{1,2,3}) {
    bool red=false;
    try {run<16,128,64,false,Scalar>(fault);} catch (std::runtime_error const&) {red=true;}
    if (!red) throw std::runtime_error("coordinate/owner negative was not detected");
  }
}

static void range_boundary() {
  // Frozen first-token SwiGLU outlier, not a random small-value fixture.
  constexpr int k=25600, tile=128, index=5613, height=16, threads=128;
  std::vector<float> input(k,0.f);
  input[index]=243383.484375f;
  Input<float> tensor{input.data(),1,k,tile};
  std::vector<Half> narrow(height*tile);
  std::vector<cutlass::bfloat16_t> wide(height*tile);
  for (int thread=0;thread<threads;++thread) {
    detail::copy_decode_a<height,tile,threads,false,1,height*64,height*tile>(
        tensor,narrow.data(),index/tile,0,thread,1);
    detail::copy_decode_a<height,tile,threads,false,1,height*64,height*tile>(
        tensor,wide.data(),index/tile,0,thread,1);
  }
  int col=index%tile;
  int offset=col/64*height*64+read_half(height,0,col%64);
  // This proves the existing writer's range loss; it does not admit a BF16 MMA.
  if (narrow[offset].raw()!=0x7c00 || wide[offset].raw()!=0x486e)
    throw std::runtime_error("range-boundary historical/counterfactual signature differs");
  std::puts("KPACK_DECODE_RANGE LEGACY_F16_RED input=243383.484375 index=5613 narrow=0x7c00 bf16_writer=0x486e compute_fix=PENDING");
}

int main() {
  try {
    static_assert(!detail::DecodeInputTraits<cute::identity,Half>::enabled);
    matrix<float>();matrix<cutlass::bfloat16_t>();
    range_boundary();
    std::puts("KPACK_DECODE_INPUT PASS F32+BF16 M1..8 physical_M16..256 TK64/128/256 padded+packed multi-stage guards negatives=6");
    return 0;
  } catch (std::exception const& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
