/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Whole-word Q4 reader for the resident xplane artifact shared by prefill and decode.
 *
 * This file does not define a weight format.  The producer has already placed each logical 32-code
 * Q4 group in four consecutive uint32 words.  Physical nibble p in those words owns logical K
 *
 *     k = (p & 3) * 8 + (p >> 2),       p in [0, 32).
 *
 * Q4WordPlan names that fixed register permutation.  The exhaustive producer/consumer oracle binds
 * it to place_derived for every supported ArtifactTileK; changing the artifact to suit this reader is
 * explicitly out of scope.
 **************************************************************************************************/
#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

#if defined(__HGGCCC__)
#include <hggc_fp16.h>
#else
#include <cuda_fp16.h>
#endif

namespace gguf_scale::bc_vecdot::q4_reader {

// actlize's vendor CUTLASS_DEVICE macro intentionally keys only on the PPU
// compiler.  This reader is also an NVIDIA decode target, so its annotations
// must name both device compilers explicitly instead of inheriting that
// PPU-only policy.
#if defined(__CUDACC__) || defined(__HGGCCC__)
#define QUACTLIZE_BC_Q4_HD __host__ __device__ __forceinline__
#define QUACTLIZE_BC_Q4_DEVICE __device__ __forceinline__
#else
#define QUACTLIZE_BC_Q4_HD inline
#define QUACTLIZE_BC_Q4_DEVICE inline
#endif

// The shipping Q4 kernels intentionally use 16-byte vector loads for the
// activation, resident code, and packed-unit planes. cudaMalloc/DevBuf bases
// satisfy this, but the public device-pointer ABI also accepts caller-owned
// suballocations. Keep the actual predicate beside the loads so the ABI entry
// can reject a sliced pointer instead of relying on allocator folklore.
static constexpr uintptr_t kVectorLoadAlignment = 16;

QUACTLIZE_BC_Q4_HD bool vector_load_pointer_is_aligned(void const* pointer) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (kVectorLoadAlignment - 1)) == 0;
}

QUACTLIZE_BC_Q4_HD bool vector_load_contract(void const* activation,
                                             void const* low,
                                             void const* units) {
  return vector_load_pointer_is_aligned(activation) &&
         vector_load_pointer_is_aligned(low) &&
         vector_load_pointer_is_aligned(units);
}

static_assert((kVectorLoadAlignment & (kVectorLoadAlignment - 1)) == 0,
              "Q4 vector-load alignment must remain a power of two");

template <int ArtifactTileK>
struct Q4WordPlan {
  static_assert(ArtifactTileK == 32 || ArtifactTileK == 64 ||
                ArtifactTileK == 128 || ArtifactTileK == 256,
                "Q4 whole-word reader requires a proved resident ArtifactTileK");

  static constexpr int kCodes = 32;
  static constexpr int kWords = 4;
  static constexpr int kCodesPerWord = 8;

  QUACTLIZE_BC_Q4_HD
  static constexpr int logical_k_from_physical_nibble(int physical) {
    return (physical & 3) * 8 + (physical >> 2);
  }

  QUACTLIZE_BC_Q4_HD
  static constexpr int physical_nibble_from_logical_k(int logical_k) {
    return (logical_k & 7) * 4 + (logical_k >> 3);
  }

  QUACTLIZE_BC_Q4_HD
  static constexpr int physical_nibble_from_pair_lane(int pair, int lane) {
    return pair + 4 * lane;
  }

  QUACTLIZE_BC_Q4_HD
  static constexpr unsigned code_from_pair_lane(uint32_t packed, int pair, int lane) {
    return (packed >> (4 * physical_nibble_from_pair_lane(pair, lane))) & 0xfu;
  }

  QUACTLIZE_BC_Q4_HD
  static constexpr int word_index(int physical) { return physical >> 3; }

  QUACTLIZE_BC_Q4_HD
  static constexpr int nibble_in_word(int physical) { return physical & 7; }

 private:
  static constexpr bool permutation_is_bijective() {
    bool seen[kCodes] = {};
    for (int p = 0; p < kCodes; ++p) {
      int const k = logical_k_from_physical_nibble(p);
      if (k < 0 || k >= kCodes || seen[k] || physical_nibble_from_logical_k(k) != p) return false;
      seen[k] = true;
    }
    return true;
  }

 public:
  static_assert(permutation_is_bijective(), "Q4 P4x32 register permutation is not bijective");
};

// Four half2 values cover the eight nibbles of one uint32.  Pair t owns physical
// nibbles (t, t+4).  This is the natural output of the two mantissa-insertion
// levels and lets the dot consumer gather the matching activation pair without
// a register shuffle.
struct Q4WordHalf2 {
  half2 pair[4];
};

// Native Q4_K metadata view.  Keeping this object in CUDA/PPU scalar and half
// types is intentional: routing the same bytes through cutlass::half_t makes
// NVIDIA instantiate its portable fp16 operators in the hottest decode body.
// The 16-byte unit is byte-neutral with the producer; this is only a reader.
struct Q4PackedMetadata {
  uint32_t header;
  uint32_t run0_lo;
  uint32_t run_bridge;
  uint32_t run1_hi;

  QUACTLIZE_BC_Q4_HD
  static constexpr uint64_t run_bits(uint32_t lo, uint32_t bridge, uint32_t hi,
                                     unsigned group) {
    return (group & 4u)
        ? uint64_t(bridge >> 16) | (uint64_t(hi) << 16)
        : uint64_t(lo) | (uint64_t(bridge & 0xffffu) << 32);
  }

  QUACTLIZE_BC_Q4_HD
  constexpr unsigned scale_code(unsigned group) const {
    return unsigned((run_bits(run0_lo, run_bridge, run1_hi, group) >>
                     (6u * (group & 3u))) & 63u);
  }

  QUACTLIZE_BC_Q4_HD
  constexpr unsigned min_code(unsigned group) const {
    return unsigned((run_bits(run0_lo, run_bridge, run1_hi, group) >>
                     (24u + 6u * (group & 3u))) & 63u);
  }

  QUACTLIZE_BC_Q4_HD
  constexpr uint16_t d_bits() const { return uint16_t(header); }

  QUACTLIZE_BC_Q4_HD
  constexpr uint16_t dmin_bits() const { return uint16_t(header >> 16); }
};

struct Q4NativeScaleZero {
  half scale;
  half zero;  // signed affine term: -dmin * min_code
};

QUACTLIZE_BC_Q4_DEVICE Q4PackedMetadata load_metadata(uint8_t const* unit) {
  uint4 const words = *reinterpret_cast<uint4 const*>(unit);
  return {words.x, words.y, words.z, words.w};
}

QUACTLIZE_BC_Q4_HD half native_half_from_bits(uint16_t bits) {
  static_assert(sizeof(half) == sizeof(uint16_t), "native fp16 width changed");
  return reinterpret_cast<half const&>(bits);
}

QUACTLIZE_BC_Q4_HD half2 native_half2_from_bits(uint32_t bits) {
  static_assert(sizeof(half2) == sizeof(uint32_t), "native fp16x2 width changed");
  return reinterpret_cast<half2 const&>(bits);
}

QUACTLIZE_BC_Q4_HD Q4NativeScaleZero decode_scale_zero(
    Q4PackedMetadata const& metadata, unsigned group) {
  unsigned const sc = metadata.scale_code(group);
  unsigned const mn = metadata.min_code(group);
  uint32_t const magic_bits = uint32_t(0x6400u | sc) |
                              (uint32_t(0x6400u | mn) << 16);
  half2 const codes = __hsub2(native_half2_from_bits(magic_bits),
                              native_half2_from_bits(0x64006400u));
  half const d = native_half_from_bits(metadata.d_bits());
  half const dmin = native_half_from_bits(metadata.dmin_bits());
  return {__hmul(d, codes.x), __hneg(__hmul(dmin, codes.y))};
}

namespace detail {

template <int Pair>
struct Q4PairConstants {
  static_assert(Pair >= 0 && Pair < 4, "Q4 half2 pair index out of range");
  static constexpr bool kShiftByte = Pair >= 2;
  static constexpr int kBitPosition = (Pair & 1) * 4;
  static constexpr uint32_t kMask16 = uint32_t(0x000fu << kBitPosition);
  static constexpr uint32_t kMask = kMask16 | (kMask16 << 16);
#if defined(Q4K_BC_PLANT_WRONG_MAGIC)
  // Negative-control-only: corrupt exactly one exponent bit in the high half.
  // The shipping build must never define this macro.
  static constexpr uint32_t kMagic = 0x60006400u;
#else
  static constexpr uint32_t kMagic = 0x64006400u; // half2(1024, 1024)
#endif
  static constexpr uint32_t kMul16 = uint32_t((15 - kBitPosition) << 10);
  static constexpr uint32_t kMul = kMul16 | (kMul16 << 16);
  // Convert unsigned Q4: (1024 + q*2^bpos) * 2^-bpos - 2^(10-bpos) = q.
  static constexpr uint32_t kAdd16 = uint32_t(0x8000u | ((25 - kBitPosition) << 10));
  static constexpr uint32_t kAdd = kAdd16 | (kAdd16 << 16);
};

static_assert(Q4PairConstants<0>::kMask == 0x000f000fu &&
              Q4PairConstants<1>::kMask == 0x00f000f0u &&
#if !defined(Q4K_BC_PLANT_WRONG_MAGIC)
              Q4PairConstants<0>::kMagic == 0x64006400u &&
#endif
              Q4PairConstants<0>::kMul == 0x3c003c00u &&
              Q4PairConstants<1>::kMul == 0x2c002c00u &&
              Q4PairConstants<0>::kAdd == 0xe400e400u &&
              Q4PairConstants<1>::kAdd == 0xd400d400u,
              "Q4 fast-dequant constants drifted from the unsigned code contract");

template <int Pair>
QUACTLIZE_BC_Q4_DEVICE uint32_t dequantize_pair_bits(uint32_t packed) {
  using C = Q4PairConstants<Pair>;
  uint32_t const source = C::kShiftByte ? packed >> 8 : packed;
  uint32_t converted;

#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  asm volatile("ppu.lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(converted)
               : "r"(source), "n"(C::kMask), "n"(C::kMagic), "n"(0xeau));
  // PPU and NVIDIA accept the arithmetic constants through registers.  In
  // particular, NVIDIA f16x2 arithmetic does not accept the immediate operand
  // spelling used by the PPU assembler.
  uint32_t const mul = C::kMul, add = C::kAdd;
  asm volatile("ppu.fma.rtte.f16x2 %0, %1, %2, %3;\n"
               : "=r"(converted) : "r"(converted), "r"(mul), "r"(add));
#elif defined(__CUDA_ARCH__)
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(converted)
               : "r"(source), "n"(C::kMask), "n"(C::kMagic), "n"(0xeau));
  uint32_t const mul = C::kMul, add = C::kAdd;
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(converted) : "r"(converted), "r"(mul), "r"(add));
#else
  unsigned const lo = (source >> C::kBitPosition) & 0xfu;
  unsigned const hi = (source >> (16 + C::kBitPosition)) & 0xfu;
  half2 const value = __floats2half2_rn(float(lo), float(hi));
  converted = reinterpret_cast<uint32_t const&>(value);
#endif
  return converted;
}

template <int Pair>
QUACTLIZE_BC_Q4_DEVICE half2 dequantize_pair(uint32_t packed) {
  uint32_t const bits = dequantize_pair_bits<Pair>(packed);
  return reinterpret_cast<half2 const&>(bits);
}

} // namespace detail

QUACTLIZE_BC_Q4_DEVICE Q4WordHalf2 dequantize_word(uint32_t packed) {
  return {{detail::dequantize_pair<0>(packed), detail::dequantize_pair<1>(packed),
           detail::dequantize_pair<2>(packed), detail::dequantize_pair<3>(packed)}};
}

} // namespace gguf_scale::bc_vecdot::q4_reader

#undef QUACTLIZE_BC_Q4_HD
#undef QUACTLIZE_BC_Q4_DEVICE
