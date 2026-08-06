#pragma once
// GGUF K-quant SCALE BLOCK, described as cute Layouts.  NEW FILE -- nothing here touches the fp16 scale path.
// fp16 is the native form for GPTQ and AWQ, so ElementScale = half_t stays the default specialization and this
// header is an addition beside it (plan #20's own constraint).
//
// Every K-quant's per-group scale is a 4-bit LOW field plus a (ScaleBits-4)-bit HIGH field, and each field's byte
// index and bit shift is AFFINE in a nested group coordinate. That is the same shape as MixGemm2Plane's
// LoCodeL / HiCodeL, so the machinery is not new -- only the constants are.
//
// PROVENANCE. Every map below was read off real_weight/dump_real_weights.py (itself regressed against
// marlin_gguf_ppu.cuh:gguf_q4k_scales) and verified line by line, not written from memory. The Q4_K h=0 case is a
// single 6-bit field in one byte; expressing it as low4@0 + high2@4 OF THE SAME BYTE reconstructs it exactly, which
// is what lets one affine form cover both halves.
#include <cstdint>
#include "cute/tensor.hpp"

namespace gguf_scale {

enum class KType { Q2_K, Q3_K, Q4_K, Q5_K, Q6_K };

// A field is (byte_base + ByteL(g), shift_base + ShiftL(g), width).
template <class ByteL_, int ByteBase_, class ShiftL_, int ShiftBase_, int Width_>
struct Field {
  using ByteL  = ByteL_;
  using ShiftL = ShiftL_;
  static constexpr int kByteBase  = ByteBase_;
  static constexpr int kShiftBase = ShiftBase_;
  static constexpr int kWidth     = Width_;
  template <class Coord>
  CUTE_HOST_DEVICE static constexpr int byte_of (Coord const& c) { return kByteBase  + ByteL{}(c); }
  template <class Coord>
  CUTE_HOST_DEVICE static constexpr int shift_of(Coord const& c) { return kShiftBase + ShiftL{}(c); }
  template <class Coord>
  CUTE_HOST_DEVICE static constexpr int extract(uint8_t const* p, Coord const& c) {
    return (p[byte_of(c)] >> shift_of(c)) & ((1 << kWidth) - 1);
  }
};

// Four adjacent bytes carry four codes at the same within-byte shift. This is the code-plane counterpart of Field:
// ByteL maps a (word-in-run, group-bits...) coordinate to the first byte of a 32-bit word, ShiftL maps it to the
// shared shift, and extraction keeps the four fields byte-packed. The scalar accessor is the explicit tail path.
template <class ByteL_, int ByteBase_, class ShiftL_, int ShiftBase_, int Width_>
struct PackedField4 : Field<ByteL_, ByteBase_, ShiftL_, ShiftBase_, Width_> {
  using Base = Field<ByteL_, ByteBase_, ShiftL_, ShiftBase_, Width_>;
  template <class Coord>
  CUTE_HOST_DEVICE static constexpr uint32_t extract_word(uint32_t word, Coord const& c) {
    constexpr uint32_t kMask = uint32_t((1 << Width_) - 1) * 0x01010101u;
    return (word >> Base::shift_of(c)) & kMask;
  }
  template <class Coord>
  CUTE_HOST_DEVICE static constexpr int extract_byte(uint8_t const* p, Coord const& c, int byte_in_word) {
    return (p[Base::byte_of(c) + byte_in_word] >> Base::shift_of(c)) & ((1 << Width_) - 1);
  }
};

template <KType> struct Traits;

// ---------------------------------------------------------------------------------------------------------------
// Q4_K / Q5_K: 12 bytes -> 8 sc + 8 mn of 6 bits, gs = 32.  Group coord (t, h) = (g%4, g/4).
//   dump_real_weights.py:get_scale_min_k4
//     h=0:  sc[g] = q[g] & 63                                  mn[g] = q[g+4] & 63
//     h=1:  sc[g] = (q[g+4] & 0xF) | ((q[g-4] >> 6) << 4)      mn[g] = (q[g+4] >> 4) | ((q[g] >> 6) << 4)
template <> struct Traits<KType::Q4_K> {
  static constexpr int  kScaleBits = 6, kMinBits = 6, kGroupSize = 32, kGroups = 8, kBlockBytes = 12;
  static constexpr bool kHasMin = true, kSigned = false;
  // THE CODE'S OWN CENTRE, per dump_real_weights.py: Q4_K is scale = d*sc6 with sc6 unsigned, so nothing to subtract.
  static constexpr int  kScaleBias = 0;
  using GroupShape = cute::Shape<cute::_4, cute::_2>;                               // (t, h)
  using ScLo = Field<cute::Layout<GroupShape, cute::Stride<cute::_1, cute::_8>>, 0,  // byte t + 8h
                     cute::Layout<GroupShape, cute::Stride<cute::_0, cute::_0>>, 0, 4>;   // shift 0
  using ScHi = Field<cute::Layout<GroupShape, cute::Stride<cute::_1, cute::_0>>, 0,  // byte t
                     cute::Layout<GroupShape, cute::Stride<cute::_0, cute::_2>>, 4, 2>;   // shift 4 + 2h
  using MnLo = Field<cute::Layout<GroupShape, cute::Stride<cute::_1, cute::_4>>, 4,  // byte 4 + t + 4h
                     cute::Layout<GroupShape, cute::Stride<cute::_0, cute::_4>>, 0, 4>;   // shift 4h
  using MnHi = Field<cute::Layout<GroupShape, cute::Stride<cute::_1, cute::_0>>, 4,  // byte 4 + t
                     cute::Layout<GroupShape, cute::Stride<cute::_0, cute::_2>>, 4, 2>;   // shift 4 + 2h
};
template <> struct Traits<KType::Q5_K> : Traits<KType::Q4_K> {};                    // same scale block

// ---------------------------------------------------------------------------------------------------------------
// Q3_K: 12 bytes -> 16 sc of 6 bits, gs = 16, no min.  Group coord (t, q0, q1) with g = 4q + t, q = q0 + 2q1.
//   dump_real_weights.py:unpack_q3k_expert
//     sc6[ 0+t] = ( s[0+t]      & 0xF) | (((s[8+t] >> 0) & 0x3) << 4)
//     sc6[ 4+t] = ( s[4+t]      & 0xF) | (((s[8+t] >> 2) & 0x3) << 4)
//     sc6[ 8+t] = ((s[0+t] >> 4) & 0xF) | (((s[8+t] >> 4) & 0x3) << 4)
//     sc6[12+t] = ((s[4+t] >> 4) & 0xF) | (((s[8+t] >> 6) & 0x3) << 4)
template <> struct Traits<KType::Q3_K> {
  static constexpr int  kScaleBits = 6, kMinBits = 0, kGroupSize = 16, kGroups = 16, kBlockBytes = 12;
  static constexpr bool kHasMin = false, kSigned = false;
  // Q3_K is scale = d*(sc6 - 32) (dump_real_weights.py:unpack_q3k_expert, `dl = d * (sc6[is_full] - 32)`). This is
  // the ONE per-format constant that is not zero and not derivable from the width, so it is read off that file.
  static constexpr int  kScaleBias = 32;
  using GroupShape = cute::Shape<cute::_4, cute::_2, cute::_2>;                     // (t, q0, q1)
  using ScLo = Field<cute::Layout<GroupShape, cute::Stride<cute::_1, cute::_4, cute::_0>>, 0,  // t + 4q0
                     cute::Layout<GroupShape, cute::Stride<cute::_0, cute::_0, cute::_4>>, 0, 4>;  // 4q1
  using ScHi = Field<cute::Layout<GroupShape, cute::Stride<cute::_1, cute::_0, cute::_0>>, 8,  // 8 + t
                     cute::Layout<GroupShape, cute::Stride<cute::_0, cute::_2, cute::_4>>, 0, 2>;  // 2q0 + 4q1
  using MnLo = void; using MnHi = void;
};

// ---------------------------------------------------------------------------------------------------------------
// Q2_K: 16 bytes, gs = 16.  byte = g, sc at shift 0 (4 bits), min at shift 4 (4 bits). One plane each.
template <> struct Traits<KType::Q2_K> {
  static constexpr int  kScaleBits = 4, kMinBits = 4, kGroupSize = 16, kGroups = 16, kBlockBytes = 16;
  static constexpr bool kHasMin = true, kSigned = false;
  // Q2_K: dl = d*(sc & 0xF), ml = dmin*(sc >> 4), zero = -ml -- both codes unsigned.
  static constexpr int  kScaleBias = 0;
  using GroupShape = cute::Shape<cute::_16>;
  using ScLo = Field<cute::Layout<GroupShape, cute::Stride<cute::_1>>, 0,
                     cute::Layout<GroupShape, cute::Stride<cute::_0>>, 0, 4>;
  using ScHi = void;
  using MnLo = Field<cute::Layout<GroupShape, cute::Stride<cute::_1>>, 0,
                     cute::Layout<GroupShape, cute::Stride<cute::_0>>, 4, 4>;
  using MnHi = void;
};

// ---------------------------------------------------------------------------------------------------------------
// Q6_K: int8 scales[16], gs = 16, no min, SIGNED. byte = g, shift 0, one 8-bit plane.
template <> struct Traits<KType::Q6_K> {
  static constexpr int  kScaleBits = 8, kMinBits = 0, kGroupSize = 16, kGroups = 16, kBlockBytes = 16;
  static constexpr bool kHasMin = false, kSigned = true;
  // Q6_K's scales are SIGNED int8 and scale_of() already sign-extends, so the centre is carried by the code itself.
  static constexpr int  kScaleBias = 0;
  using GroupShape = cute::Shape<cute::_16>;
  using ScLo = Field<cute::Layout<GroupShape, cute::Stride<cute::_1>>, 0,
                     cute::Layout<GroupShape, cute::Stride<cute::_0>>, 0, 8>;
  using ScHi = void; using MnLo = void; using MnHi = void;
};

// ---------------------------------------------------------------------------------------------------------------
// THE BITS MUST TILE THE BLOCK EXACTLY. Every K-quant's scale block is fully used:
//   Q4_K/Q5_K  8*6 + 8*6 = 96 = 12 bytes      Q3_K  16*6 = 96 = 12 bytes
//   Q2_K      16*4 + 16*4 = 128 = 16 bytes    Q6_K  16*8 = 128 = 16 bytes
// so "every bit claimed exactly once" is both necessary and sufficient, and it is strictly stronger than the
// injectivity check the plan asked for: it catches a miss as well as a collision. A silently non-injective map is
// the failure mode that cost rung 5, so it is asserted, not tested.
template <class F, KType T>
CUTE_HOST_DEVICE constexpr void claim_bits(int* used) {
  if constexpr (!cute::is_void_v<F>) {
    for (int g = 0; g < Traits<T>::kGroups; ++g)
      for (int b = 0; b < F::kWidth; ++b) ++used[F::byte_of(g) * 8 + F::shift_of(g) + b];
  }
}
template <KType T>
CUTE_HOST_DEVICE constexpr bool bits_tile_exactly() {
  using Tr = Traits<T>;
  int used[Tr::kBlockBytes * 8] = {};
  claim_bits<typename Tr::ScLo, T>(used);
  claim_bits<typename Tr::ScHi, T>(used);
  claim_bits<typename Tr::MnLo, T>(used);
  claim_bits<typename Tr::MnHi, T>(used);
  for (int i = 0; i < Tr::kBlockBytes * 8; ++i) if (used[i] != 1) return false;
  return true;
}
static_assert(bits_tile_exactly<KType::Q4_K>(), "Q4_K scale bit map does not tile its 12 bytes exactly");
static_assert(bits_tile_exactly<KType::Q3_K>(), "Q3_K scale bit map does not tile its 12 bytes exactly");
static_assert(bits_tile_exactly<KType::Q2_K>(), "Q2_K scale bit map does not tile its 16 bytes exactly");
static_assert(bits_tile_exactly<KType::Q6_K>(), "Q6_K scale bit map does not tile its 16 bytes exactly");

// The decode itself: low field, then the high field shifted up by 4.
template <KType T>
CUTE_HOST_DEVICE constexpr int scale_of(uint8_t const* p, int g) {
  using Tr = Traits<T>;
  int v = Tr::ScLo::extract(p, g);
  if constexpr (!cute::is_void_v<typename Tr::ScHi>) v |= Tr::ScHi::extract(p, g) << 4;
  if constexpr (Tr::kScaleBits == 8) v = int(int8_t(p[Tr::ScLo::byte_of(g)]));   // Q6_K is signed
  return v;
}
template <KType T>
CUTE_HOST_DEVICE constexpr int min_of(uint8_t const* p, int g) {
  using Tr = Traits<T>;
  static_assert(Tr::kHasMin, "this format has no min");
  int v = Tr::MnLo::extract(p, g);
  if constexpr (!cute::is_void_v<typename Tr::MnHi>) v |= Tr::MnHi::extract(p, g) << 4;
  return v;
}

// ---------------------------------------------------------------------------------------------------------------
// K-quant CODE PLANES, using the same Field byte/shift abstraction as scale/min. The coordinate is word-fast:
// `word_coord(g,w) = w + kWordsPerGroup*g`, and each format factors g into the bit coordinates its GGUF map uses.
// Low/high are separate physical planes; Q3's inverted high-bit meaning is value semantics applied by its decoder,
// not part of the address map.
template <KType> struct CodeTraits;

template <> struct CodeTraits<KType::Q2_K> {
  static constexpr int kGroups = 16, kWordsPerGroup = 4, kLoOffset = 16, kLoBytes = 64;
  static constexpr int kHiOffset = 0, kHiBytes = 0;
  using CodeShape = cute::Shape<cute::_4, cute::_2, cute::_4, cute::_2>;  // (word, g%2, (g/2)%4, g/8)
  using Lo = PackedField4<cute::Layout<CodeShape, cute::Stride<cute::_4, cute::_16, cute::_0, cute::_32>>, 0,
                          cute::Layout<CodeShape, cute::Stride<cute::_0, cute::_0, cute::_2, cute::_0>>, 0, 2>;
  using Hi = void;
  CUTE_HOST_DEVICE static constexpr int word_coord(int g, int w) { return w + kWordsPerGroup * g; }
};

template <> struct CodeTraits<KType::Q3_K> {
  static constexpr int kGroups = 16, kWordsPerGroup = 4, kLoOffset = 32, kLoBytes = 64;
  static constexpr int kHiOffset = 0, kHiBytes = 32;
  using CodeShape = cute::Shape<cute::_4, cute::_2, cute::_4, cute::_2>;  // (word, g%2, (g/2)%4, g/8)
  using Lo = PackedField4<cute::Layout<CodeShape, cute::Stride<cute::_4, cute::_16, cute::_0, cute::_32>>, 0,
                          cute::Layout<CodeShape, cute::Stride<cute::_0, cute::_0, cute::_2, cute::_0>>, 0, 2>;
  using Hi = PackedField4<cute::Layout<CodeShape, cute::Stride<cute::_4, cute::_16, cute::_0, cute::_0>>, 0,
                          cute::Layout<CodeShape, cute::Stride<cute::_0, cute::_0, cute::_1, cute::_4>>, 0, 1>;
  CUTE_HOST_DEVICE static constexpr int word_coord(int g, int w) { return w + kWordsPerGroup * g; }
};

template <> struct CodeTraits<KType::Q4_K> {
  static constexpr int kGroups = 8, kWordsPerGroup = 8, kLoOffset = 16, kLoBytes = 128;
  static constexpr int kHiOffset = 0, kHiBytes = 0;
  using CodeShape = cute::Shape<cute::_8, cute::_2, cute::_4>;  // (word, g%2, g/2)
  using Lo = PackedField4<cute::Layout<CodeShape, cute::Stride<cute::_4, cute::_0, cute::_32>>, 0,
                          cute::Layout<CodeShape, cute::Stride<cute::_0, cute::_4, cute::_0>>, 0, 4>;
  using Hi = void;
  CUTE_HOST_DEVICE static constexpr int word_coord(int g, int w) { return w + kWordsPerGroup * g; }
};

template <> struct CodeTraits<KType::Q5_K> {
  static constexpr int kGroups = 8, kWordsPerGroup = 8, kLoOffset = 48, kLoBytes = 128;
  static constexpr int kHiOffset = 16, kHiBytes = 32;
  using CodeShape = cute::Shape<cute::_8, cute::_2, cute::_4>;  // (word, g%2, g/2)
  using Lo = PackedField4<cute::Layout<CodeShape, cute::Stride<cute::_4, cute::_0, cute::_32>>, 0,
                          cute::Layout<CodeShape, cute::Stride<cute::_0, cute::_4, cute::_0>>, 0, 4>;
  using Hi = PackedField4<cute::Layout<CodeShape, cute::Stride<cute::_4, cute::_0, cute::_0>>, 0,
                          cute::Layout<CodeShape, cute::Stride<cute::_0, cute::_1, cute::_2>>, 0, 1>;
  CUTE_HOST_DEVICE static constexpr int word_coord(int g, int w) { return w + kWordsPerGroup * g; }
};

template <> struct CodeTraits<KType::Q6_K> {
  static constexpr int kGroups = 16, kWordsPerGroup = 4, kLoOffset = 0, kLoBytes = 128;
  static constexpr int kHiOffset = 128, kHiBytes = 64;
  using CodeShape = cute::Shape<cute::_4, cute::_2, cute::_2, cute::_2, cute::_2>;  // (word,p,q,n,h), g=p+2q+4n+8h
  using Lo = PackedField4<cute::Layout<CodeShape, cute::Stride<cute::_4, cute::_16, cute::_32, cute::_0, cute::_64>>, 0,
                          cute::Layout<CodeShape, cute::Stride<cute::_0, cute::_0, cute::_0, cute::_4, cute::_0>>, 0, 4>;
  using Hi = PackedField4<cute::Layout<CodeShape, cute::Stride<cute::_4, cute::_16, cute::_0, cute::_0, cute::_32>>, 0,
                          cute::Layout<CodeShape, cute::Stride<cute::_0, cute::_0, cute::_2, cute::_4, cute::_0>>, 0, 2>;
  CUTE_HOST_DEVICE static constexpr int word_coord(int g, int w) { return w + kWordsPerGroup * g; }
};

// Each field claims its width in every one of the word's four bytes. Exact tiling separately per physical plane is
// stronger than injectivity: it catches both overlaps and holes in the actual words the GEMV loads.
template <class F, int Bytes, class Tr>
CUTE_HOST_DEVICE constexpr bool code_field_tiles_exactly() {
  if constexpr (cute::is_void_v<F>) {
    return Bytes == 0;
  } else {
    int used[Bytes * 8] = {};
    for (int c = 0; c < Tr::kGroups * Tr::kWordsPerGroup; ++c) {
      int const byte0 = F::byte_of(c);
      int const shift = F::shift_of(c);
      if (byte0 < 0 || byte0 + 3 >= Bytes || shift < 0 || shift + F::kWidth > 8) return false;
      for (int lane = 0; lane < 4; ++lane)
        for (int bit = 0; bit < F::kWidth; ++bit) ++used[(byte0 + lane) * 8 + shift + bit];
    }
    for (int bit = 0; bit < Bytes * 8; ++bit) if (used[bit] != 1) return false;
    return true;
  }
}

template <KType T>
CUTE_HOST_DEVICE constexpr bool code_bits_tile_exactly() {
  using Tr = CodeTraits<T>;
  return code_field_tiles_exactly<typename Tr::Lo, Tr::kLoBytes, Tr>() &&
         code_field_tiles_exactly<typename Tr::Hi, Tr::kHiBytes, Tr>();
}

static_assert(code_bits_tile_exactly<KType::Q2_K>(), "Q2_K code-plane map does not tile its words exactly");
static_assert(code_bits_tile_exactly<KType::Q3_K>(), "Q3_K code-plane map does not tile its words exactly");
static_assert(code_bits_tile_exactly<KType::Q4_K>(), "Q4_K code-plane map does not tile its words exactly");
static_assert(code_bits_tile_exactly<KType::Q5_K>(), "Q5_K code-plane map does not tile its words exactly");
static_assert(code_bits_tile_exactly<KType::Q6_K>(), "Q6_K code-plane map does not tile its words exactly");

}  // namespace gguf_scale
