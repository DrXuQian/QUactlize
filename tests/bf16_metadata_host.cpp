#include <hggc_runtime.h>
#include "actlize_extensions/cutlass/gguf_bfloat_scale.h"
#include "actlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "quactlize/dequant/reader.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

using B = cutlass::bfloat16_t;
using H = cutlass::half_t;
namespace md = cutlass::gguf_packed;

static uint16_t bits(float v) {
  uint32_t u; std::memcpy(&u, &v, 4);
  return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float value(uint16_t u) {
  uint32_t b = uint32_t(u) << 16; float v; std::memcpy(&v, &b, 4); return v;
}
static float rnd(float v) { return value(bits(v)); }

template<int Width, int Bias> void code_pairs() {
  for (int a=0;a<(1<<Width);++a) for (int b=0;b<(1<<Width);++b) {
    auto got = cutlass::direct_bfloat_code_pair<Width,Bias>(uint32_t(a)|(uint32_t(b)<<16));
    assert(uint16_t(got)==bits(float(a-Bias)));
    assert(uint16_t(got>>16)==bits(float(b-Bias)));
  }
}

template<int N> void uint2_mapping() {
  using Convert = cutlass::MixGemmNumericArrayConverter<B,cutlass::uint2b_t,N>;
  typename Convert::source_type input;
  auto* word = reinterpret_cast<uint32_t*>(&input);
  for(int seed=0;seed<128;++seed) {
    for(int w=0;w<N/16;++w) word[w]=0x13579bdu*(seed+1)+0x9e3779b9u*w;
    auto got=Convert::convert(input);
    for(int v=0;v<N/16;++v) for(int t=0;t<8;++t) for(int h=0;h<2;++h) {
      // Literal established b16 delivery map, independent of converter at().
      int index=N==16 ? 2*t+h : 2*(16*(v&1)+2*(v/2)+4*(t/2)+(t&1))+h;
      int shift=N==16 ? 4*t+2*h : 2*t+16*h;
      assert(B(got[index]).raw()==bits(float((word[v]>>shift)&3)));
    }
  }
}

template<int Lo,int Hi,int Bias> void two_plane_mapping() {
  using C = cutlass::MixGemm2Plane<Lo,Hi,-1,1,true,cutlass::MixGemm2PlaneDefaultFrag<Lo>,Bias,B>;
  uint32_t low[4], high[4], result[C::kOut/2];
  for(int seed=0;seed<256;++seed) {
    for(int v=0;v<4;++v) { low[v]=0x13579bdu*(seed+v+1); high[v]=0x9e3779b9u*(seed+3*v+1); }
    C::convert(low,high,result);
    for(int v=0;v<4;++v) for(int t=0;t<16/Lo;++t) for(int h=0;h<2;++h) {
      int const lc=int((low[v]>>(Lo*t+16*h))&((1<<Lo)-1));
      int const hv=(v/(Lo/Hi))*(Lo/Hi), hc=t+(16/Lo)*(v%(Lo/Hi));
      int const code=lc | (((high[hv]>>(Hi*hc+16*h))&((1<<Hi)-1))<<Lo);
      int const index=C::at(t,v);
      assert(uint16_t(result[index]>>(16*h))==bits(float(code-Bias)));
    }
  }
}

template<md::Fmt F,int ZMul> void metadata() {
  using U=md::Unit<F>;
  int half_intermediate_diff=0, header_round_diff=0;
  uint16_t const headers[]={0x0001,0x0203,0x1357,0x2a19,0x33ab,0x4107,0x7bff};
  for(auto hdr:headers) for(int sc=0;sc<(1<<U::kScaleBits);++sc) {
    uint32_t words[(U::kSbBytes+3)/4]{};
    words[0]=uint32_t(hdr)|(uint32_t(U::kHasMin?0x2357:0)<<16);
    int const mn=U::kHasMin ? (sc*7+11)&((1<<U::kMinBits)-1) : 0;
    // Independent sequential field writer. It never calls the decoder's
    // Unit::bit_of or code_from_words to construct the test input.
    auto put=[&](int bit,int width,int x) {
      for(int b=0;b<width;++b) if(x&(1<<b)) words[(bit+b)/32]|=1u<<((bit+b)%32);
    };
    for(int g=0;g<U::kGroups;++g) {
      int run=U::kHasMin ? g/(U::kGroups/2) : 0;
      int local=U::kHasMin ? g%(U::kGroups/2) : g;
      int base=U::kHeaderBytes*8+run*(U::kGroups/2)*(U::kScaleBits+U::kMinBits);
      put(base+local*U::kScaleBits,U::kScaleBits,sc);
      if constexpr(U::kHasMin) put(base+(U::kGroups/2)*U::kScaleBits+local*U::kMinBits,U::kMinBits,mn);
    }
    int signed_sc=U::kSigned && sc>=128 ? sc-256 : sc;
    signed_sc-=U::kScaleBias;
    float d=float(H::bitcast(hdr)), dm=U::kHasMin ? float(H::bitcast(0x2357)) : 0.f;
    float scale=rnd(d*signed_sc), zero=U::kHasMin ? rnd(-dm*mn) : 0.f;
    if constexpr(ZMul!=0) zero=rnd(zero+rnd(float(ZMul)*scale));
    half_intermediate_diff+=bits(d*signed_sc)!=bits(float(H(d*signed_sc)));
    header_round_diff+=bits(d*signed_sc)!=bits(rnd(d)*signed_sc);
    auto head=md::bfloat_head_of_words(words);
    cute::for_each(cute::make_int_sequence<U::kGroups>{},[&](auto g) {
      auto reg=md::bfloat_group_of_words<decltype(g)::value,ZMul,F>(words,head);
      auto ptr=md::bfloat_group_of<F,ZMul>(reinterpret_cast<uint8_t const*>(words),int(g));
      assert(reg.scale.raw()==bits(scale) && reg.zero.raw()==bits(zero));
      assert(ptr.scale.raw()==reg.scale.raw() && ptr.zero.raw()==reg.zero.raw());
      assert(std::isfinite(float(reg.scale)) && std::isfinite(float(reg.zero)));
    });
  }
  assert(half_intermediate_diff>0 && header_round_diff>0);
}

template<gguf_scale::KType T,int Width,int Bias> void full_dequant() {
  using R=quactlize::dequant::Reader<T>;
  int negative=0;
  for(float scale:{.00103759765625f,.061279296875f,152.125f,65504.f})
    for(float mn:{0.f,.0311737060546875f}) for(int code=0;code<(1<<Width);++code) {
      volatile float product=float(code-Bias)*scale;
      float want=product-mn;
      assert(R::weight(code,{scale,mn})==bits(want));
      negative+=bits(float(H(want)))!=bits(want);
    }
  assert(negative>0);
}

int main() {
  code_pairs<2,0>();code_pairs<3,0>();code_pairs<3,4>();code_pairs<4,8>();
  code_pairs<5,8>();code_pairs<6,8>();code_pairs<6,32>();
  uint2_mapping<16>();uint2_mapping<64>();
  two_plane_mapping<2,1,0>();two_plane_mapping<2,1,4>();
  two_plane_mapping<4,1,8>();two_plane_mapping<4,2,8>();two_plane_mapping<4,2,32>();
  metadata<md::Fmt::Q2K,0>();metadata<md::Fmt::Q3K,-4>();metadata<md::Fmt::Q4K,8>();
  metadata<md::Fmt::Q5K,8>();metadata<md::Fmt::Q6K,-24>();
  using gguf_scale::KType;
  full_dequant<KType::Q2_K,2,0>();full_dequant<KType::Q3_K,3,4>();
  full_dequant<KType::Q4_K,4,0>();full_dequant<KType::Q5_K,5,0>();full_dequant<KType::Q6_K,6,32>();
  std::puts("BF16_METADATA_HOST PASS codes, paired_planes, direct_metadata, full_weight_single_rounding, negatives");
}
