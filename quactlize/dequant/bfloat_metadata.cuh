#pragma once
#include "actlize_extensions/cutlass/gguf_bfloat_scale.h"
#include "cute/tensor.hpp"

namespace quactlize::dequant {

// Same canonical [paired-unit,N,bytes] input and [K-group,N] output as FQ.
// A warp owns 32 adjacent columns, hence each publication is a contiguous
// 64-byte span. Q4/Q5 fetch one uint4 per column; paired Q3/Q6 headers remain
// independent and may be only two-byte aligned. No FP16 expanded plane.
template <cutlass::gguf_packed::Fmt F, int ZMul, int Threads>
__global__ __launch_bounds__(Threads) void sf_bfloat_units(
    uint8_t const* __restrict__ units, cutlass::bfloat16_t* __restrict__ scale,
    cutlass::bfloat16_t* __restrict__ zero, int n, int k) {
  using namespace cutlass::gguf_packed;
  using U = Unit<F>;
  int const col = blockIdx.x * Threads + threadIdx.x;
  if (col >= n) return;
  int const e = blockIdx.z, sb = blockIdx.y, superblocks = k / 256;
  auto const* p = units + ((int64_t(e) * (superblocks / U::kSbPerUnit) + sb / U::kSbPerUnit) * n + col)
      * U::kUnitTotal + (sb % U::kSbPerUnit) * U::kSbBytes;
  uint32_t words[(U::kSbBytes + 3) / 4]{};
  if constexpr (U::kSbBytes == 16) {
    uint4 const v = *reinterpret_cast<uint4 const*>(p);
    words[0] = v.x; words[1] = v.y; words[2] = v.z; words[3] = v.w;
  } else {
    CUTLASS_PRAGMA_UNROLL
    for (int w = 0; w < (U::kSbBytes + 3) / 4; ++w) {
      CUTLASS_PRAGMA_UNROLL
      for (int b = 0; b < 4; ++b) if (4*w+b < U::kSbBytes) words[w] |= uint32_t(p[4*w+b]) << (8*b);
    }
  }
  auto const h = bfloat_head_of_words(words);
  cute::for_each(cute::make_int_sequence<U::kGroups>{}, [&](auto g) {
    auto const sz = bfloat_group_of_words<decltype(g)::value, ZMul, F>(words, h);
    int64_t const o = ((int64_t(e) * superblocks + sb) * U::kGroups + int(g)) * n + col;
    scale[o] = sz.scale;
    zero[o] = sz.zero;
  });
}

template <cutlass::gguf_packed::Fmt F, int ZMul>
void launch_sf_bfloat(uint8_t const* units, uint16_t* scale, uint16_t* zero,
    int n, int k, int experts, int threads, hggcStream_t stream) {
  dim3 grid((n + threads - 1) / threads, k / 256, experts);
  auto s = reinterpret_cast<cutlass::bfloat16_t*>(scale);
  auto z = reinterpret_cast<cutlass::bfloat16_t*>(zero);
  if (threads == 128) sf_bfloat_units<F, ZMul, 128><<<grid,128,0,stream>>>(units,s,z,n,k);
  else sf_bfloat_units<F, ZMul, 256><<<grid,256,0,stream>>>(units,s,z,n,k);
}

}  // namespace quactlize::dequant
