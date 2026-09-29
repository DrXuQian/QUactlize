// Host-only contract check against the existing canonical packed-unit writer.
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "quactlize/backends/ppu17/gguf_q4_unit.cuh"
#include "quactlize/backends/ppu17/format.cuh"
#include "gguf_packed_unit.hpp"

int main() {
  using F=quactlize::ppu17::Format<12>;
  using Unit=gguf_scale::packed_unit::Unit<gguf_scale::KType::Q4_K>;
  using Candidate=quactlize::ppu17::GgufQ4Unit;
  static_assert(Unit::kUnitBytes==sizeof(Candidate));
  unsigned checks=0;
  for(int i=0;i<1024;++i) {
    alignas(16) uint8_t bytes[16]{};
    auto d=cutlass::half_t(.0000231f*(i+1));
    auto dm=cutlass::half_t(.0000713f*((i*7)%1024+1));
    bytes[0]=d.raw();bytes[1]=d.raw()>>8;
    bytes[2]=dm.raw();bytes[3]=dm.raw()>>8;
    for(int g=0;g<8;++g)for(int min=0;min<2;++min)
      gguf_scale::packed_unit::put_code<F::type>(bytes,g,min,(i*(min?3:1)+g*11)&63);
    Candidate candidate;std::memcpy(&candidate,bytes,sizeof(candidate));
    for(int g=0;g<8;++g) {
      for(int min=0;min<2;++min) {
        if(candidate.field(g,min)!=gguf_scale::packed_unit::code_of<F::type>(bytes,g,min))
          throw std::runtime_error("canonical field differs");
        ++checks;
      }
      auto f16=candidate.affine<cutlass::half_t>(g);
      auto old16=gguf_scale::packed_unit::unit_group<F::type,8>(bytes,g);
      auto bf16=candidate.affine<cutlass::bfloat16_t>(g);
      auto oldbf=F::affine<cutlass::bfloat16_t>(bytes,1,0,g);
      if(f16.scale.raw()!=old16.scale.raw()||f16.zero.raw()!=old16.zero.raw()||
         bf16.scale.raw()!=oldbf.scale.raw()||bf16.zero.raw()!=oldbf.zero.raw())
        throw std::runtime_error("canonical typed affine differs");
      checks+=4;
    }
  }
  std::printf("GGUF_Q4_PACKED_ABI PASS checks=%u canonical_writer=existing direct_bf16=1\n",checks);
}
