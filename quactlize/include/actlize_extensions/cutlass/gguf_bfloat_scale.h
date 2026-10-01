#pragma once

#include "cutlass/bfloat16.h"
#include "gguf_packed_scale.h"

namespace cutlass::gguf_packed {

struct BfloatGroupScale {
  bfloat16_t scale;
  bfloat16_t zero;
};

// GGUF stores d/dmin as FP16. Read their values exactly into FP32; never
// form a decoded FP16 scale before rounding to BF16. In particular, rounding
// the header to BF16 before multiplication is NOT the same contract.
struct BfloatUnitHead {
  float d;
  float dmin;
};

template <int NWords>
CUTLASS_HOST_DEVICE BfloatUnitHead bfloat_head_of_words(uint32_t const (&u)[NWords]) {
  return {float(half_t::bitcast(uint16_t(u[0]))),
          float(half_t::bitcast(uint16_t(u[0] >> 16)))};
}

// One rounding per published BF16 value. The reader-centre correction is
// multiply-then-add in BF16, matching the collective's affine arithmetic.
// The same operation serves FQ register decode and the SF workspace producer.
template <int ZMul, bool HasMin>
CUTLASS_HOST_DEVICE BfloatGroupScale bfloat_group(int sc, int mn, BfloatUnitHead h) {
  BfloatGroupScale out{bfloat16_t(h.d * float(sc)), bfloat16_t(0.f)};
  if constexpr (HasMin) out.zero = bfloat16_t(-h.dmin * float(mn));
  if constexpr (ZMul != 0) {
    auto correction = bfloat16_t(float(ZMul)) * out.scale;
    out.zero = out.zero + correction;
  }
  return out;
}

template <int G, int ZMul, Fmt F, int NWords>
CUTLASS_HOST_DEVICE BfloatGroupScale bfloat_group_of_words(
    uint32_t const (&u)[NWords], BfloatUnitHead h) {
  using U = Unit<F>;
  int sc = code_from_words<U::bit_of(G, 0), NWords, U::kScaleBits>(u);
  if constexpr (U::kSigned) sc = (sc ^ (1 << (U::kScaleBits - 1))) - (1 << (U::kScaleBits - 1));
  int mn = 0;
  if constexpr (U::kHasMin) mn = code_from_words<U::bit_of(G, 1), NWords, U::kMinBits>(u);
  return bfloat_group<ZMul, U::kHasMin>(sc - U::kScaleBias, mn, h);
}

template <Fmt F, int ZMul>
CUTLASS_HOST_DEVICE BfloatGroupScale bfloat_group_of(uint8_t const* unit, int g) {
  using U = Unit<F>;
  BfloatUnitHead h{float(half_t::bitcast(uint16_t(unit[0]) | uint16_t(unit[1]) << 8)), 0.f};
  int sc = code_of_fmt<F>(unit, g, 0);
  if constexpr (U::kSigned) sc = (sc ^ (1 << (U::kScaleBits - 1))) - (1 << (U::kScaleBits - 1));
  int mn = 0;
  if constexpr (U::kHasMin) {
    h.dmin = float(half_t::bitcast(uint16_t(unit[2]) | uint16_t(unit[3]) << 8));
    mn = code_of_fmt<F>(unit, g, 1);
  }
  return bfloat_group<ZMul, U::kHasMin>(sc - U::kScaleBias, mn, h);
}

}  // namespace cutlass::gguf_packed
