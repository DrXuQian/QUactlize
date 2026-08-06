#pragma once
// THE PACKED SCALE UNIT, GENERALISED PAST Q4_K. One trait per format; the staging skeleton reads the trait.
//
// WHAT WAS HARDCODED. quactlize_extensions/cutlass/gguf_packed_scale.h describes exactly one unit -- 16 bytes,
// d and dmin in the first
// four, eight groups of (6-bit scale, 6-bit min) split into two self-contained halves -- and the collective is built
// on those numbers: kPackedScaleUnit = 16, one 128-bit cp.async, Scale_TileK == 8, four 32-bit words, 6-bit
// extraction, kPackedHasMin, kPackedZMul = 8. schemes.py records the consequence: Q2_K is single-plane so the SHAPE
// fits but none of the constants do, and Q3/Q5/Q6 reach a different collective entirely.
//
// WHY A REORDERED UNIT AT ALL, since this is the question that decides whether the generalisation is even allowed.
// GGUF's own scale packing is not half-separable -- Q4_K's get_scale_min_k4 takes groups 4..7 from bytes 8-11 AND
// the top two bits of bytes 0-3 -- so a k-tile covering half a superblock could not read half a block. The unit
// fixes that by making each run of groups self-contained, AT NO COST IN BYTES. That last clause is the whole
// licence for this path, so it is asserted per format below rather than believed.
//
// STORED BYTES, GGUF against this unit:
//     Q4_K/Q5_K   d,dmin 4 + 12 code = 16   ->  4 + 8*(6+6)/8 = 16     neutral
//     Q2_K        d,dmin 4 + 16 code = 20   ->  4 + 16*(4+4)/8 = 20    neutral
//     Q3_K        d     2 + 12 code = 14    ->  2 + 16*6/8 = 14        neutral
//     Q6_K        d     2 + 16 code = 18    ->  2 + 16*8/8 = 18        neutral
//
// The unit SIZE therefore differs per format, which is precisely what the collective's single 128-bit cp.async
// assumes away. Generalising the collective means the staging tile and its copy come from this trait; nothing here
// commits to a transfer width, deliberately.
#include <cstdint>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "gguf_scale_layout.hpp"
#include "gguf_scale_decode.hpp"

namespace gguf_scale {
namespace packed_unit {

using cutlass::half_t;

// ONE TRAIT, and every number the staging skeleton needs comes from it. Adding a format is filling this in, not
// finding the places that said 16.
template <KType T>
struct Unit {
  using Tr = Traits<T>;
  static constexpr int  kGroups     = Tr::kGroups;
  static constexpr int  kScaleBits  = Tr::kScaleBits;
  static constexpr int  kMinBits    = Tr::kMinBits;
  static constexpr bool kHasMin     = Tr::kHasMin;
  static constexpr bool kSigned     = Tr::kSigned;
  static constexpr int  kFields     = kHasMin ? 2 : 1;
  // The header is d, plus dmin only when there is a min channel to scale.
  static constexpr int  kHeaderBytes = kHasMin ? 4 : 2;
  static constexpr int  kCodeBits    = kGroups * (kScaleBits + kMinBits);
  static constexpr int  kUnitBytes   = kHeaderBytes + kCodeBits / 8;
  // BYTE NEUTRALITY IS THE LICENCE FOR THE WHOLE PATH, so it is a static_assert and not a comment. GGUF's stored
  // scale metadata is the header plus kBlockBytes; the unit must not exceed it.
  static_assert(kCodeBits % 8 == 0, "the code field must fill whole bytes or the unit is not byte-addressable");
  static_assert(kUnitBytes <= kHeaderBytes + Tr::kBlockBytes,
                "the reordered unit must not be larger than GGUF's own scale metadata -- an offline reorder is "
                "permitted, an increase in stored bytes is not");

  // SUPERBLOCKS PER UNIT, which is what makes Q3_K and Q6_K movable at all. ppu.cp.async takes 4, 8 or 16 bytes;
  // one superblock of Q3_K is 14 and of Q6_K 18, both 2 mod 4, so neither can be bulk-copied. TWO superblocks OF THE
  // SAME COLUMN are 28 and 36, both divisible by 4, with no padding and no change to which column a thread owns.
  //
  // The axis matters and I had it wrong first: pairing along n -- two COLUMNS -- needs the staged tensor recast to
  // (TN/2, 2*unit) and changes ownership, which is why that attempt was withdrawn. Pairing along k needs neither,
  // because a thread still owns exactly its own column and simply covers two of its superblocks.
  static constexpr int kSbPerUnit = ((kHeaderBytes + kCodeBits / 8) % 4 == 0) ? 1 : 2;
  static constexpr int kSbBytes   = kHeaderBytes + kCodeBits / 8;      // one superblock's own share
  static constexpr int kUnitTotal = kSbPerUnit * kSbBytes;             // what the copy actually moves

  // THE BIT POSITION, one rule, no per-project shift arithmetic written anywhere else. Fields are laid out
  // group-major within a RUN of kRunGroups, and each run is self-contained so a k-tile covering part of a superblock
  // reads a contiguous byte range -- which is the entire reason the unit is reordered.
  //
  // The run is half the groups when a format has two fields (matching the shipped Q4_K unit's two halves) and the
  // whole superblock otherwise; both give whole-byte runs for every format above, which is checked.
  static constexpr int kRunGroups = kHasMin ? (kGroups / 2) : kGroups;
  static constexpr int kRunBits   = kRunGroups * (kScaleBits + kMinBits);
  static_assert(kRunBits % 8 == 0, "a run must be whole bytes or it is not self-contained in memory");

  // CUTE OWNS THE NEW ADDRESSING RULE. A scale and a min have different bit strides within a run, so forcing them
  // into one affine layout would merely hide a conditional stride in arithmetic. Keep one layout per field and pass
  // the selected layout to field_bit_of, exactly as the two-plane converter passes its logical layouts to both the
  // offline and device sides. This replaces the old hand-expanded g/run expression; code_of/put_code retain only the
  // unavoidable byte extraction after the layout has named the bit.
  using RunShape = cute::Shape<cute::Int<kRunGroups>, cute::Int<kGroups / kRunGroups>>;
  using ScaleBitLayout = cute::Layout<RunShape, cute::Stride<cute::Int<kScaleBits>, cute::Int<kRunBits>>>;
  // MinBitLayout is never selected for scale-only formats. A zero stride is nonetheless the truthful type there and
  // keeps the trait fully generic without inventing a dummy bit width.
  using MinBitLayout = cute::Layout<RunShape, cute::Stride<cute::Int<kMinBits>, cute::Int<kRunBits>>>;

  template <class BitLayout>
  CUTLASS_HOST_DEVICE static constexpr int field_bit_of(int g, BitLayout const& layout) {
    return int(layout(g % kRunGroups, g / kRunGroups));
  }

  CUTLASS_HOST_DEVICE static constexpr int bit_of(int g, int which) {
    return kHeaderBytes * 8 + (which
        ? kRunGroups * kScaleBits + field_bit_of(g, MinBitLayout{})
        : field_bit_of(g, ScaleBitLayout{}));
  }
  // The same position inside superblock `sb` of a multi-superblock unit. Each superblock keeps its OWN header, so a
  // consumer that wants only one of them reads a contiguous kSbBytes run and needs nothing from the other -- which
  // is the property that makes the pairing free rather than a coupling.
  CUTLASS_HOST_DEVICE static constexpr int bit_of_sb(int sb, int g, int which) {
    return sb * kSbBytes * 8 + bit_of(g, which);
  }
};

// A field may straddle a byte boundary (6-bit fields always do), so a 16-bit read and one shift. Stated once.
// THE SECOND BYTE IS READ ONLY WHEN THE FIELD STRADDLES, and that is a bounds requirement rather than an
// optimisation. Every field here is at most 8 bits, so two bytes always suffice -- but reading the second
// unconditionally runs off the END of the unit for the last field of the last run: Q4_K's last min field ends at
// bit 122 of a 16-byte unit, and Q6_K's byte-aligned 8-bit scales put the last one at byte 17 of 18. The first
// version read three bytes unconditionally and was out of bounds for both.
template <KType T>
CUTLASS_HOST_DEVICE int code_of(uint8_t const* unit, int g, int which) {
  int const bit = Unit<T>::bit_of(g, which);
  int const bits = which ? Unit<T>::kMinBits : Unit<T>::kScaleBits;
  int const off = bit & 7;
  uint32_t word = uint32_t(unit[bit >> 3]);
  if (off + bits > 8) word |= uint32_t(unit[(bit >> 3) + 1]) << 8;
  return int((word >> off) & ((1u << bits) - 1u));
}

template <KType T>
CUTLASS_HOST_DEVICE void put_code(uint8_t* unit, int g, int which, int v) {
  int const bit = Unit<T>::bit_of(g, which);
  int const bits = which ? Unit<T>::kMinBits : Unit<T>::kScaleBits;
  int const off = bit & 7;
  uint32_t const shifted = uint32_t(v) << off;
  unit[bit >> 3] |= uint8_t(shifted & 0xFFu);
  if (off + bits > 8) unit[(bit >> 3) + 1] |= uint8_t((shifted >> 8) & 0xFFu);
}

// THE OFFLINE SIDE. Takes the codes a format's own layout yields -- scale_of/min_of on the GGUF scale block, which
// is already checked against llama.cpp -- and writes the unit. Header first, verbatim.
template <KType T>
CUTLASS_HOST_DEVICE void pack_unit_sb(uint8_t const* gguf_scale_block, half_t d, half_t dmin, int sb, uint8_t* unit) {
  uint8_t* p = unit + sb * Unit<T>::kSbBytes;
  for (int i = 0; i < Unit<T>::kSbBytes; ++i) p[i] = 0;
  uint16_t const db = d.raw();
  p[0] = uint8_t(db & 0xFF); p[1] = uint8_t(db >> 8);
  if constexpr (Unit<T>::kHasMin) { uint16_t const mb = dmin.raw(); p[2] = uint8_t(mb & 0xFF); p[3] = uint8_t(mb >> 8); }
  for (int g = 0; g < Unit<T>::kGroups; ++g) {
    int const sc = scale_of<T>(gguf_scale_block, g) & ((1 << Unit<T>::kScaleBits) - 1);
    put_code<T>(p, g, 0, sc);
    if constexpr (Unit<T>::kHasMin) put_code<T>(p, g, 1, min_of<T>(gguf_scale_block, g));
  }
}

template <KType T>
CUTLASS_HOST_DEVICE void pack_unit(uint8_t const* gguf_scale_block, half_t d, half_t dmin, uint8_t* unit) {
  for (int i = 0; i < Unit<T>::kUnitBytes; ++i) unit[i] = 0;
  uint16_t const db = d.raw();
  unit[0] = uint8_t(db & 0xFF);
  unit[1] = uint8_t(db >> 8);
  if constexpr (Unit<T>::kHasMin) {
    uint16_t const mb = dmin.raw();
    unit[2] = uint8_t(mb & 0xFF);
    unit[3] = uint8_t(mb >> 8);
  }
  for (int g = 0; g < Unit<T>::kGroups; ++g) {
    // SIGNED CODES ARE STORED AS THEIR BIT PATTERN. Q6_K's scales are int8 and scale_of sign-extends them, so the
    // mask on the way back in is what makes the round trip exact rather than clamping negatives to zero.
    int const sc = scale_of<T>(gguf_scale_block, g) & ((1 << Unit<T>::kScaleBits) - 1);
    put_code<T>(unit, g, 0, sc);
    if constexpr (Unit<T>::kHasMin) put_code<T>(unit, g, 1, min_of<T>(gguf_scale_block, g));
  }
}

// THE DEVICE SIDE: one group's (scale, zero) straight out of the unit, no fp16 plane in between. ZMul is the
// CONSUMER's converter shift, exactly as in gguf_scale_prepass.hpp -- a property of the weight width, not the
// format -- and it is applied after the split for the same reason.
template <KType T, int ZMul>
CUTLASS_HOST_DEVICE GroupScale unit_group(uint8_t const* unit, int g) {
  half_t const d = half_t::bitcast(uint16_t(unit[0] | (uint16_t(unit[1]) << 8)));
  half_t const dmin = Unit<T>::kHasMin
                    ? half_t::bitcast(uint16_t(unit[2] | (uint16_t(unit[3]) << 8))) : half_t(0.f);
  int sc = code_of<T>(unit, g, 0);
  if constexpr (Unit<T>::kSigned) {                       // Q6_K: restore the sign the bit pattern carries
    constexpr int kSignBit = 1 << (Unit<T>::kScaleBits - 1);
    if (sc & kSignBit) sc -= (1 << Unit<T>::kScaleBits);
  }
  GroupScale out;
  if constexpr (Unit<T>::kHasMin) {
    out = make_group_scale<T>(sc, code_of<T>(unit, g, 1), d, dmin);
  } else {
    out.scale = make_group_scale_only<T>(sc, d);
    out.zero = half_t(0.f);
  }
  // A PERSISTENT WORKSPACE MUST BE BIT-IDENTICAL on host and device. Native fp16 arithmetic rounds the product
  // before the add on CUDA, while the host operators promote through float; that made the packed-unit prepass differ
  // by one fp16 ulp from its portable definition. Match group_scale_zero's single float expression explicitly.
  if constexpr (ZMul != 0) out.zero = half_t(float(out.zero) + float(ZMul) * float(out.scale));
  return out;
}

// ONE SUPERBLOCK OF A MULTI-SUPERBLOCK UNIT. Each superblock carries its own header, so this is just an offset --
// a consumer wanting only superblock sb reads a contiguous kSbBytes run and needs nothing from the other, which is
// the property that makes pairing along k free rather than a coupling between two k-tiles.
template <KType T, int ZMul>
CUTLASS_HOST_DEVICE GroupScale unit_group_sb(uint8_t const* unit, int sb, int g) {
  return unit_group<T, ZMul>(unit + sb * Unit<T>::kSbBytes, g);
}

}  // namespace packed_unit
}  // namespace gguf_scale
