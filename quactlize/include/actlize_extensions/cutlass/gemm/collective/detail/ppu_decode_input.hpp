#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "ppu_a_pack.hpp"

namespace cutlass::gemm::collective::detail {

// A storage conversion, not an MMA transform. Existing identity transforms
// retain the original AIU load and the original argument/parameter layout.
template<class Source> struct DecodeInput {
  using SourceElement = Source;
  template<class T> CUTE_HOST_DEVICE T operator()(T const& value) const { return value; }
};

template<class Transform,class Compute> struct DecodeInputTraits {
  static constexpr bool enabled = false;
  using SourceElement = Compute;
};
template<class Source,class Compute> struct DecodeInputTraits<DecodeInput<Source>,Compute> {
  static constexpr bool enabled = true;
  using SourceElement = Source;
};

// Publish the SAME physical AIU A cube consumed by the existing swizzle atom.
// Each owner loads eight contiguous source elements and writes one uint128.
// Real M is at most eight; every physical padding row is explicitly zeroed.
// The packed-row variant retains its disjoint cube/stage pitches unchanged.
template<int Height,int TileK,int Threads,bool Packed,int PackedRows,
         int PackedPitch,int PackedStagePitch,class Tensor,class Compute>
CUTLASS_DEVICE void copy_decode_a(Tensor const& source,Compute* destination,
    int k_tile,int stage,int thread,int valid_rows) {
  static_assert(Height>=16 && Height%16==0 && TileK%64==0);
  constexpr int Rows=Packed ? PackedRows : Height;
  constexpr int Cubes=TileK/64;
  constexpr int Pitch=Packed ? PackedPitch : Height*64;
  constexpr int StagePitch=Packed ? PackedStagePitch : Height*TileK;
  for (int owner=thread;owner<Cubes*Rows*8;owner+=Threads) {
    int cube=owner/(Rows*8), row=(owner/8)%Rows, half_run=owner%8;
    int run=half_run/2, half=half_run%2;
    using Scalar=typename Tensor::value_type;
    static_assert(sizeof(Scalar)==2 || sizeof(Scalar)==4);
    cutlass::AlignedArray<Compute,8,16> values;
    if (row<valid_rows) {
      cutlass::AlignedArray<Scalar,8,16> raw;
      auto ptr=&source(row,cube*64+run*16+half*8,k_tile);
      CUTE_UNROLL
      for (int v=0;v<int(sizeof(Scalar)*8/16);++v)
        // Fixed-size copies preserve vector transport without violating the
        // BF16/uint128 strict-aliasing rules in the host or device compiler.
        __builtin_memcpy(reinterpret_cast<char*>(&raw)+v*16,
            reinterpret_cast<char const*>(ptr)+v*16,16);
      CUTE_UNROLL
      for (int i=0;i<8;++i) values[i]=Compute(float(raw[i]));
    } else {
      CUTE_UNROLL
      for (int i=0;i<8;++i) values[i]=Compute(0.f);
    }
    auto offset=stage*StagePitch+cube*Pitch+aPackRunOffsetHalfs(Height,row,run)
        +(half^((row/4)&1))*8;
    __builtin_memcpy(destination+offset,&values,16);
  }
}
} // namespace cutlass::gemm::collective::detail
