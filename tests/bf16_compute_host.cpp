#include <hggc_runtime.h>
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "quactlize/execution/simt_validation.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

using Half=cutlass::half_t;
using Bf16=cutlass::bfloat16_t;
namespace md=cutlass::gemm::collective::detail;

static void physical_contract() {
  using H=cute::MMA_Traits<cute::PPU0010_8x16x16_F32F16F16F32_TN>;
  using B=cute::MMA_Traits<cute::PPU0010_8x16x16_F32BF16BF16F32_TN>;
  static_assert(std::is_same_v<H::ALayout,B::ALayout> && std::is_same_v<H::BLayout,B::BLayout>);
  static_assert(std::is_same_v<H::CLayout,B::CLayout> && std::is_same_v<B::ValTypeA,Bf16>);
}

static void arithmetic_contract() {
  float outlier=243383.484375f;
  assert(std::isinf(float(Half(outlier))));
  assert(Bf16(outlier).raw()==0x486e && std::isfinite(float(Bf16(outlier))));
  int unequal_reinterpret=0;
  for(int code=-128;code<=127;++code) {
    assert(float(Half(float(code)))==float(Bf16(float(code))));
    unequal_reinterpret+=Half(float(code)).raw()!=Bf16(float(code)).raw();
  }
  assert(unequal_reinterpret>240); // A reinterpretation must not look like value conversion.
  auto b=cute::make_tensor<Bf16>(cute::make_shape(cute::_8{}));
  auto s=cute::make_tensor<Half>(cute::make_shape(cute::_8{}));
  auto z=cute::make_tensor<Half>(cute::make_shape(cute::_8{}));
  for(int i=0;i<8;++i) {b(i)=Bf16(float(i-4));s(i)=Half(.0127f*(i+1));z(i)=Half(.0073f*i);}
  float expected[8];
  for(int i=0;i<8;++i) expected[i]=float(Bf16(float(Bf16(float(b(i))*float(s(i))))+float(z(i))));
  md::transform_metadata(b,s,b,cute::multiplies{});
  md::transform_metadata(b,z,b,cute::plus{});
  for(int i=0;i<8;++i) assert(float(b(i))==expected[i]);
}

static void query_contract() {
  namespace simt=quactlize::execution::simt;
  auto arrangement=ppu_arrangements::kquant_kpack_transpose_v1(14);
  qkg_call_v1 c{};
  c.version=1;c.size=sizeof(c);c.qtype=14;c.n=256;c.k=512;c.experts=1;
  c.rows=8;c.mode=QKG_DENSE;c.input_type=QKG_F32;c.channels=1;c.topk=1;
  c.a_row_stride=512;c.a_token_stride=512;c.ids_stride=1;c.out_row_stride=256;
  qkg_simt_call_v2 d{2,sizeof(d),c,QKG_COMPUTE_BF16};
  qkg_simt_config_v1 f{1,sizeof(f),3,4,4,4,8};
  qkg_sizes_v1 sizes{};
  assert(simt::query_v2(d,f,&arrangement,sizes)==QKG_OK);
  assert(sizes.workspace_bytes==8*256*8*4);
  d.call.input_type=QKG_SIMT_BF16;
  assert(simt::query_v2(d,f,&arrangement,sizes)==QKG_OK);
  assert(simt::query(d.call,f,&arrangement,sizes)==QKG_INVALID);
  d.compute_type=QKG_COMPUTE_F16;
  assert(simt::query_v2(d,f,&arrangement,sizes)==QKG_INVALID);
  d.compute_type=7;assert(simt::query_v2(d,f,&arrangement,sizes)==QKG_INVALID);
  d.compute_type=QKG_COMPUTE_BF16;--d.size;
  assert(simt::query_v2(d,f,&arrangement,sizes)==QKG_INVALID);++d.size;
  d.call.rows=9;assert(simt::query_v2(d,f,&arrangement,sizes)==QKG_SHAPE);
}

int main() {
  physical_contract();arithmetic_contract();query_contract();
  std::puts("BF16_COMPUTE_HOST PASS b16_mapping, bounded_codes, metadata_value_cast, range, version_negatives");
}
