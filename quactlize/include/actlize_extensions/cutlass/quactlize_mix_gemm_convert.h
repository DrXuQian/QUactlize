/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*!
    \file
    \brief quactlize's additions to actlize's mixed-input numeric conversion: unsigned bytes, sub-byte widths
           (uint2b_t, uint1b_t), the shared four-byte primitive, and the two-plane bit-plane converters.

    WHAT IS HERE AND WHAT IS NOT. actlize's fast_numeric_conversion_for_mix_gemm.h defines seven converters --
    the primary template plus int8 and int4b_t for half_t and bfloat16_t. Not one of them is redefined here; that
    header is included below and its converters are used as the vendor wrote them. This file adds the widths
    GGUF k-quants need and the machinery around them.

    THE ONE THING THAT USED TO BE AN EDIT. Until 2026-08-06 quactlize's copy of that header rewrote upstream's
    half_t<-int8_t body to call MixGemmByte4ToHalf<128>. The rewrite was ours, not a correctness fix: upstream's
    sequence constructs 1024+byte and subtracts 1152, giving byte-128 in the expected interleaved order, and
    MixGemmByte4ToHalf<128> reproduces those exact PPU operations and constants. It existed only so the unsigned
    path below could share the primitive. Since the two are behaviourally identical, upstream's converter serves
    the int8 case and the shared primitive stays here for ours -- no vendor file is edited to get it.
*/

#pragma once

// actlize's converters, unmodified. This must come first: the specialisations below extend its primary template.
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"
// MixGemmEmit is a cute Layout now (it was a closed-form loop), so the algebra has to be available here.
#include "cute/layout.hpp"
#include <utility>

namespace cutlass
{

// Two unsigned codes at bit offsets 0 and 16. BF16 represents 128+code
// exactly for up to seven code bits; subtract the construction and quant bias.
template<int Bits, int Bias = 0>
CUTLASS_HOST_DEVICE uint32_t direct_bfloat_code_pair(uint32_t codes) {
  static_assert(Bits > 0 && Bits <= 7 && Bias >= 0 && Bias <= 128);
  constexpr uint32_t mask = ((1u << Bits) - 1u) * 0x00010001u;
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  uint32_t result;
  asm volatile("ppu.lop3.b32 %0,%1,%2,%3,%4;\n" : "=r"(result)
               : "r"(codes), "n"(mask), "n"(0x43004300u), "n"(0xeau));
  constexpr uint32_t bias = (0xc300u + Bias) * 0x00010001u;
  asm volatile("ppu.fma.rtte.bf16x2 %0,%1,%2,%3;\n" : "=r"(result)
               : "r"(result), "r"(0x3f803f80u), "r"(bias));
  return result;
#else
  auto a = bfloat16_t(float(int(codes & ((1u << Bits) - 1u)) - Bias));
  auto b = bfloat16_t(float(int((codes >> 16) & ((1u << Bits) - 1u)) - Bias));
  return uint32_t(a.raw()) | (uint32_t(b.raw()) << 16);
#endif
}

// This converter is meant to be used with data interleaved in a 32-bit register where the even elements are in the low
// bits and the odd elemeents are in the high bits of the register. In addition, it assumes elements were originally
// signed and had a bias of 2**(b-1) added (where b is the number of bits in the type) to make all numbers unsigned.
// This converter will uninterleave the data and subtract the bias while converting to the result type.

template <int Bits>
struct MixGemmEmit {
  static constexpr int kCodesPerVreg = 32 / Bits;
  static constexpr int kNumCodeBits  =
      (Bits == 1) ? 5 : (Bits == 2) ? 4 : (Bits == 4) ? 3 : (Bits == 8) ? 2 : 0;
  static_assert(kNumCodeBits != 0, "MixGemmEmit: only 1/2/4/8-bit codes");

  template <int I>
  using CodeStride = cute::Int<(I == kNumCodeBits - 1) ? 1 : (I == 0 ? 2 : (4 << I))>;

  template <std::size_t... I>
  static constexpr auto make_emit_layout(std::index_sequence<I...>) {
    return cute::make_layout(
        cute::make_shape (cute::make_shape((void(I), cute::_2{})...), cute::make_shape(cute::_2{}, cute::_2{})),
        cute::make_stride(cute::make_stride(CodeStride<int(I)>{}...),
                          cute::make_stride(cute::Int<2 * kCodesPerVreg>{}, cute::_4{})));
  }
  using EmitLayout = decltype(make_emit_layout(std::make_index_sequence<kNumCodeBits>{}));

  // (code within a vreg, vreg) -> fp16 element index within the converter's 4*kCodesPerVreg outputs
  static constexpr int index(int code, int vreg) {
    return int(EmitLayout{}(code + kCodesPerVreg * vreg));
  }
  static constexpr int kNumOutputs = 4 * kCodesPerVreg;

  // a permutation or the model is wrong; checked at compile time so it cannot rot
  static constexpr bool bijective() {
    bool seen[4 * 32] = {};
    for (int v = 0; v < 4; ++v)
      for (int c = 0; c < kCodesPerVreg; ++c) {
        const int e = index(c, v);
        if (e < 0 || e >= kNumOutputs || seen[e]) return false;
        seen[e] = true;
      }
    return true;
  }
  static_assert(bijective(), "MixGemmEmit: emission map is not a permutation");
};

template <int Bits, int Fold, int Deliveries>
struct MixGemmArtifactScatter {
  static constexpr int VEC = MixGemmEmit<Bits>::kNumOutputs;
  static_assert(Fold >= 1 && Deliveries >= 1 && VEC % Fold == 0,
                "artifact scatter requires whole converter fold groups and deliveries");
  static constexpr int Group = VEC / Fold;
  static_assert(Group % 2 == 0, "artifact scatter must preserve complete half2 writes");

  using EmitLayout = cute::Layout<
      cute::Shape<cute::Int<Group>, cute::_1, cute::Int<Fold>>,
      cute::Stride<cute::_1, cute::Int<Group>, cute::Int<Group>>>;

  static constexpr int flat(int e, int delivery, int n_instance = 0) {
    return n_instance * Deliveries * VEC
         + (e / Group) * Deliveries * Group
         + delivery * Group
         + e % Group;
  }
  static constexpr int group_base(int fold_group, int delivery, int n_instance = 0) {
    return n_instance * Deliveries * VEC
         + fold_group * Deliveries * Group
         + delivery * Group;
  }

  static constexpr bool bijective_and_half2_safe() {
    bool seen[VEC * Deliveries] = {};
    for (int d = 0; d < Deliveries; ++d)
      for (int e = 0; e < VEC; ++e) {
        int const dst = flat(e, d);
        if (dst < 0 || dst >= VEC * Deliveries || seen[dst]) return false;
        seen[dst] = true;
        if ((e % 2) == 0 && (dst % 2 != 0 || flat(e + 1, d) != dst + 1)) return false;
      }
    return true;
  }
  static_assert(bijective_and_half2_safe(),
                "artifact delivery scatter must be bijective and preserve half2 pairs");
};

namespace detail {
constexpr int mixgemm_int4_composed(int c, int v) {
  const int l = 2 * (c & 3) + (c >= 4 ? 1 : 0);        // base-8: h[c&3], low lane c<4, high lane c>=4
  const int p[4] = {0, 2, 1, 3};
  const int g0 = 8 * p[v] + l;
  int b = g0 / 4;
  b = (b == 1) ? 2 : (b == 2) ? 1 : (b == 5) ? 6 : (b == 6) ? 5 : b;
  return 4 * b + (g0 % 4);
}
constexpr bool mixgemm_int4_agrees() {
  for (int v = 0; v < 4; ++v)
    for (int c = 0; c < 8; ++c)
      if (mixgemm_int4_composed(c, v) != MixGemmEmit<4>::index(c, v)) return false;
  return true;
}
static_assert(mixgemm_int4_agrees(),
              "int4's composed emission (base8 o vreg-perm o block-swap) no longer matches MixGemmEmit<4>");
}  // namespace detail

template <int Bits, int BlockK, int FoldF>
struct MixGemmMmaPermK {
  static constexpr int value = (FoldF > 1) ? BlockK : (32 * 8 / Bits);
};

// WHERE A CHUNKED OUTPUT GOES, as two compositions over the fragment layout. Extracted so the 2-plane converter can
// share it: the two converters have DIFFERENT emission orders (MixGemmEmit for one plane, the _E2 lines' AtLayout for
// two), but the PLACEMENT rule is the same, and re-deriving it for the second one is exactly how this session lost a
// day. Both of the 2-plane defects found were hand-written indices, one of them a second copy of a rule already fixed
// elsewhere.
//
//   ka(e)     the k-atom e belongs to  -- mode2 stride 1, everything else 0, so only the K coordinate survives
//   at_h2(e)  the compact destination  -- mode0 keeps its stride, mode1 (N) gets size<0>, mode2 (K) gets 0, which is
//             exactly the layout of a one-k-atom fragment. THIS IS WHY NAPC > 1 NEEDS NO CALLER ARITHMETIC: which
//             n-atom an output belongs to is something the fragment layout already knows.
template <class FragLayout>
struct ChunkPlace {
  using AtL = decltype(cute::composition(
      cute::make_layout(cute::shape(FragLayout{}),
                        cute::make_stride(cute::stride<0>(FragLayout{}),
                                          cute::Int<cute::size<0>(FragLayout{})>{}, cute::_0{})),
      cute::right_inverse(FragLayout{})));
  using KaL = decltype(cute::composition(
      cute::make_layout(cute::shape(FragLayout{}),
                        cute::make_stride(cute::repeat_like(cute::stride<0>(FragLayout{}), cute::_0{}),
                                          cute::_0{}, cute::_1{})),
      cute::right_inverse(FragLayout{})));
  static constexpr int ka(int e)    { return int(KaL{}(e)); }
  static constexpr int at_h2(int e) { return int(AtL{}(e)) / 2; }
};

// Bias: the integer subtracted from every code, folded into the converter's additive constant instead of costing a
// separate pass. int4's -8 (add_bias_and_interleave_int4s pre-adds 8 offline) has always used this; it is now a
// PARAMETER so a symmetric format can name its own centre and become ScaleOnly. Default reproduces the old value
// exactly, so nothing changes unless a caller asks.
//
// Q3_K's int2 plane is the case that motivates it: its dequant is dl*(low2 - 4) + 4*dl*high1, so with Bias=4 the
// int2 plane's zero becomes 0 and the plane needs no zero channel at all. Plan #20 phase 1.
// ONE definition of the default fragment view, so the 2-plane path can name `Bias` (parameter 6) without repeating a
// layout literal that would then be free to drift from this one.
using MixGemmEmitDefaultFragLayout =
    cute::Layout<cute::Shape<cute::Shape<cute::_2,cute::_2,cute::_2>, cute::_4, cute::_4>,
                 cute::Stride<cute::Stride<cute::_1,cute::_2,cute::_4>, cute::_32, cute::_8>>;

template <int Bits, int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = MixGemmEmitDefaultFragLayout,
          int Bias = (Bits == 4) ? 8 : 0>
struct MixGemmChunkEmit {
  static constexpr int kCodesPerVreg = 32 / Bits;
  static constexpr int kOut          = 4 * kCodesPerVreg;      // 128 for int1, 64 for int2
  static constexpr int kPairsPerVreg = kCodesPerVreg / 2;
  static constexpr int kPerLevel     = 8 / Bits;
  static constexpr int kPer          = kOut / NChunk;
  static_assert(Bits == 1 || Bits == 2 || Bits == 4, "MixGemmChunkEmit covers int1, int2 and int4");
  static constexpr int kBias = Bias;
  static_assert(NChunk >= 1 && kOut % NChunk == 0, "MixGemmChunkEmit: NChunk must divide the output count");
  static_assert(Chunk < NChunk, "MixGemmChunkEmit: Chunk out of range");
  static_assert(Chunk < 0 || cute::size(FragLayout{}) == kOut,
                "MixGemmChunkEmit: FragLayout must cover all outputs when chunking");

  using Place = ChunkPlace<FragLayout>;
  using AtL = typename Place::AtL;                 // names kept: several harnesses print them
  using KaL = typename Place::KaL;

  static constexpr bool keep(int t, int v) {
    return Chunk < 0 || Place::ka(MixGemmEmit<Bits>::index(t, v)) == Chunk;
  }
  static constexpr int at(int t, int v) {
    const int e = MixGemmEmit<Bits>::index(t, v);
    return (Chunk < 0 || !Rebase) ? e / 2 : Place::at_h2(e);
  }
  static constexpr int kHalf2 = (Chunk < 0 || !Rebase ? kOut : kPer) / 2;

  static constexpr uint32_t dup(uint32_t h) { return h | (h << 16); }   // also used by MixGemm2Plane
  static  constexpr int bpos_of(int t)             { return (t % kPerLevel) * Bits; }
  template <int T> static constexpr int      bpos() { return bpos_of(T); }
  template <int T> static constexpr uint32_t mask() { return dup(uint32_t(((1u << Bits) - 1u) << bpos<T>())); }
  template <int T> static constexpr uint32_t mul()  { return dup(uint32_t((15 - bpos<T>()) << 10)); }
  // add = -(2^(10-bpos) + kBias), as an fp16 bit pattern: sign, exponent field 25-bpos, mantissa field kBias<<bpos.
  // The old form wrote the mantissa as 1<<(bpos+3), which is 8<<bpos -- the kBias=8 special case. Exactness needs
  // the mantissa to fit its 10 bits, i.e. kBias << bpos_max < 1024, which is asserted rather than assumed.
  //   kBias=8, Bits=4  bpos 0,4     -> 0x6408 / 0x5480          (the shipped int4 constants, unchanged)
  //   kBias=4, Bits=2  bpos 0,2,4,6 -> 0x6404 / 0x5C10 / 0x5440 / 0x4D00
  static_assert(kBias == 0 || (uint32_t(kBias) << (8 - Bits)) < 1024u,
                "Bias is not exactly representable in the fp16 additive constant at the top bpos");
  template <int T> static constexpr uint32_t add()  {
    return dup(uint32_t(0x8000u | ((25 - bpos<T>()) << 10) | (uint32_t(kBias) << bpos<T>())));
  }

  template <int T, int V>
  CUTLASS_DEVICE static void emit_one(uint32_t reg, uint32_t* h2) {
    const uint32_t src = (T / kPerLevel) ? (reg >> 8) : reg;
    uint32_t x;
    asm volatile("ppu.lop3.b32 %0,%1,%2,%3,%4;\n"
                 : "=r"(x) : "r"(src), "n"(mask<T>()), "n"(0x64006400u), "n"(0xEAu));
    asm volatile("ppu.fma.rtte.f16x2 %0,%1,%2,%3;\n"
                 : "=r"(x) : "r"(x), "r"(mul<T>()), "r"(add<T>()));
    h2[at(T, V)] = x;
  }

  template <int V>
  CUTLASS_DEVICE static void emit_v(uint32_t reg, uint32_t* h2) {
    cute::for_each(cute::make_int_sequence<kPairsPerVreg>{}, [&] (auto t) {
      constexpr int T = decltype(t)::value;
      if constexpr (keep(T, V)) emit_one<T, V>(reg, h2);
    });
  }

  CUTLASS_DEVICE static void emit(uint32_t const* s, uint32_t* h2) {
    emit_v<0>(s[0], h2); emit_v<1>(s[1], h2); emit_v<2>(s[2], h2); emit_v<3>(s[3], h2);
  }
};



// Select native pair arithmetic only in the PPU device pass.
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
#  define CUTLASS_MIX_GEMM_BYTE4_TO_HALF_PPU_ASM 1
#else
#  define CUTLASS_MIX_GEMM_BYTE4_TO_HALF_PPU_ASM 0
#endif

// Four byte lanes -> four fp16 values. The byte selectors and exponent bias live here because both mixed-input
// GEMM and native-GGUF SIMT decode need the same operation; keeping a GEMV-private prmt copy made the two paths
// independently mutable. Bias is a VALUE convention: 0 converts ordinary unsigned bytes, while 128 preserves the
// historical biased-int8 converter below. Interleave preserves that converter's (0,2,1,3) emission; false emits
// ordinary adjacent pairs for callers whose packed word already follows logical element order.
template <int Bias, bool Interleave = true>
struct MixGemmByte4ToHalf {
    using result_type = Array<half_t, 4>;
    static_assert(Bias >= 0 && Bias <= 255, "byte4 bias must fit the source lane");

    CUTLASS_DEVICE
    static result_type convert(uint32_t bytes)
    {
        result_type result;
        uint32_t* h = reinterpret_cast<uint32_t*>(&result);
        static constexpr uint32_t kMaskForPair0 = Interleave ? 0x5250u : 0x5150u;
        static constexpr uint32_t kMaskForPair1 = Interleave ? 0x5351u : 0x5352u;

#if CUTLASS_MIX_GEMM_BYTE4_TO_HALF_PPU_ASM
        static constexpr uint32_t kStartByteForFp16 = 0x64646464u;
        asm volatile("ppu.prmt.b32 %0,%1,%2,%3;\n"
                     : "=r"(h[0]) : "r"(bytes), "n"(kStartByteForFp16), "n"(kMaskForPair0));
        asm volatile("ppu.prmt.b32 %0,%1,%2,%3;\n"
                     : "=r"(h[1]) : "r"(bytes), "n"(kStartByteForFp16), "n"(kMaskForPair1));

        static constexpr uint32_t kHalfOffset = 0x6400u + uint32_t(Bias); // fp16(1024 + Bias), exact
        static constexpr uint32_t kOffset = kHalfOffset | (kHalfOffset << 16);
        asm volatile("ppu.sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(kOffset));
        asm volatile("ppu.sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(kOffset));
#else
        // Keep the reference arm in the selected emission order so it witnesses the object the device returns.
        result[0] = half_t(float(int((bytes      ) & 0xffu) - Bias));
        result[1] = half_t(float(int((bytes >> (Interleave ? 16 : 8)) & 0xffu) - Bias));
        result[2] = half_t(float(int((bytes >> (Interleave ? 8 : 16)) & 0xffu) - Bias));
        result[3] = half_t(float(int((bytes >> 24) & 0xffu) - Bias));
#endif
        return result;
    }
};


template <>
struct MixGemmNumericArrayConverter<half_t, uint8_t, 4>
{
    using result_type = Array<half_t, 4>;
    using source_type = Array<uint8_t, 4>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        return MixGemmByte4ToHalf<0, false>::convert(reinterpret_cast<uint32_t const&>(source));
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

// ============================ W2A16 : uint2b_t -> fp16 ============================
// Q2 base plane (also the high plane of Q6, low plane of Q3). q2 in [0,3] UNSIGNED (no +bias:
// the per-group affine 'zero' term absorbs the offset). Correctness-first: plain shift-extract,
// NO lop3 magic-trick (perf later). Numeric validated in low_bit/w2a16_smoke.cu (bad=0 on ppu001).
template <>
struct MixGemmNumericArrayConverter<half_t, uint2b_t, 16>   // 16 x uint2 = 32 bits -> 16 half
{
    using result_type = Array<half_t, 16>;
    using source_type = Array<uint2b_t, 16>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;
        uint32_t const bits = reinterpret_cast<uint32_t const&>(source);   // 16 x 2-bit, LSB-first
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 16; ++i)
            result[i] = half_t(float((bits >> (2 * i)) & 0x3u));           // elt i = bits[2i:2i+2]
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s) { return convert(s); }
};

// Wide converter used by convert_tensor (CPY_VEC = 4*32/2 = 64 for uint2). Composes base-16 x4.
template <>
struct MixGemmNumericArrayConverter<half_t, uint2b_t, 64>
{
    using result_type = Array<half_t, 64>;
    using source_type = Array<uint2b_t, 64>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;
        MixGemmChunkEmit<2, -1, 1>::emit(reinterpret_cast<uint32_t const*>(&source),
                                         reinterpret_cast<uint32_t*>(&result));
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s) { return convert(s); }
};



template <int N>
struct MixGemmNumericArrayConverter<bfloat16_t, uint2b_t, N> {
    static_assert(N == 16 || N == 64, "uint2 BF16 retains the established converter cohorts");
    using source_type = Array<uint2b_t, N>;
    using result_type = Array<bfloat16_t, N>;
    CUTLASS_DEVICE static result_type convert(source_type const& source) {
        result_type result;
        auto const* input = reinterpret_cast<uint32_t const*>(&source);
        auto* output = reinterpret_cast<uint32_t*>(&result);
        if constexpr (N == 16) {
            CUTLASS_PRAGMA_UNROLL
            for (int pair = 0; pair < 8; ++pair) {
                uint32_t bits = input[0] >> (4 * pair);
                uint32_t codes = (bits & 3u) | (((bits >> 2) & 3u) << 16);
                output[pair] = direct_bfloat_code_pair<2>(codes);
            }
        } else {
            cute::for_each(cute::make_int_sequence<4>{}, [&](auto v) {
                cute::for_each(cute::make_int_sequence<8>{}, [&](auto t) {
                    constexpr int V = decltype(v)::value, T = decltype(t)::value;
                    output[MixGemmChunkEmit<2>::at(T, V)] = direct_bfloat_code_pair<2>(input[V] >> (2 * T));
                });
            });
        }
        return result;
    }
    CUTLASS_DEVICE result_type operator()(source_type const& source) { return convert(source); }
};

template <>
struct MixGemmNumericArrayConverter<half_t, uint1b_t, 128>
{
    using result_type = Array<half_t, 128>;
    using source_type = Array<uint1b_t, 128>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        // STAGE-2: pure lop3 magic (mirror of int2's, scaled to 1-bit / 32-bit-register). DEPENDS on the OFFLINE
        // split-at-16 relayout in add_bias_and_interleave_int1s (unfused_weight_dequantize.hpp): after it, lop3
        // h[t]=(bit 2t, bit 2t+1) = the validated (magic-OR) adjacent pairs. Per vreg: 8 masks (1<<b)|(1<<(16+b))
        // b=0..7, x2 levels (reg / reg>>8); bit lands at fp16 mantissa bit b -> value=1024+2^b*bit, undone by
        // fma(2^-b, -2^(10-b)). N-half placement = validated Stage-1: base = 32*(v&1) + 2*(v>=2); pair t -> h2[base
        // + {0,1,4,5,8,9,12,13}[t%8] + 16*(t>=8)].
        // Delegates to MixGemmInt1Emit<-1,1> so the chunked and unchunked paths cannot drift apart. The
        // emission comment lives on that struct.
        result_type result;
        MixGemmChunkEmit<1, -1, 1>::emit(reinterpret_cast<uint32_t const*>(&source),
                                         reinterpret_cast<uint32_t*>(&result));
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s) { return convert(s); }
};

template <int LowBits>
using MixGemm2PlaneDefaultFrag = cute::Layout<cute::Shape<cute::_8, cute::_1, cute::Int<4 * (16 / LowBits) / 4>>>;

template <int LowBits, int HiBits, int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = MixGemm2PlaneDefaultFrag<LowBits>,
          int Bias = (LowBits == 4) ? 8 : 0, class Compute = half_t>
struct MixGemm2Plane
{
    static_assert(LowBits == 2 || LowBits == 4, "low plane is int2 (Q3) or int4 (Q5/Q6)");
    static_assert(HiBits == 1 || HiBits == 2, "high plane is int1 (Q3/Q5) or int2 (Q6)");
    static_assert(LowBits > HiBits && LowBits % HiBits == 0, "the high plane must be the sparser one");

    static constexpr int kCodesPerVreg = 32 / LowBits;        // low codes in one vreg    (int2 16, int4 8)
    static constexpr int kPairs        = kCodesPerVreg / 2;   // half2 pairs per vreg     (int2 8,  int4 4)
    static constexpr int kOut          = 4 * kCodesPerVreg;   // fp16 outputs per delivery(int2 64, int4 32)
    static constexpr int kPerAtom      = 4;                   // half2 per mma B k-atom (4 half2 == 8 fp16)
    static constexpr int kAtoms        = (kOut / 2) / kPerAtom;// mma atoms one delivery covers (int2 8, int4 4)
    // THE VREG RATIO INSIDE ONE DELIVERY -- deliberately NOT called P2_DIV, which in the collective means the COPY STEP
    // ratio DL1/DL2. The two coincide only when neither plane folds, and conflating them broke the first version of the
    // offline generalisation while leaving Block_K=256 (where they agree) passing.
    static constexpr int kVregRatio    = LowBits / HiBits;
    static constexpr int kHiStride     = 16 / HiBits;          // high codes between the two half2 lanes
    static constexpr int kHiVregs      = 4 * HiBits / LowBits; // high vregs one low delivery consumes
    static constexpr int P2_DIV        = kVregRatio;           // old name, kept for the harnesses that print it
    static constexpr int kPerLevel     = 8 / LowBits;          // low codes per byte level
    static constexpr int kLowBitsPub   = LowBits;              // so host-side emulation can read it back
    static_assert(Chunk < NChunk, "MixGemm2Plane: Chunk out of range");
    static_assert(Chunk < 0 || cute::size(FragLayout{}) == kOut,
                  "FragLayout must cover exactly one delivery's outputs");

    // (mask, mul, add, bpos) are the LOW plane's, unchanged. ONE rule, shared with the single-plane converter.
    using E = MixGemmChunkEmit<LowBits, -1, 1, true, MixGemmEmitDefaultFragLayout, Bias>;
    static constexpr int kBias = Bias;                       // so a harness can read back what the kernel subtracted
    // Placement is the low plane's own emission, composed with the fragment layout by the SAME ChunkPlace the
    // single-plane path uses. The emission ORDERS differ between formats; the placement rule does not.
    using Place = ChunkPlace<FragLayout>;
    static constexpr int  at_plain(int t, int v) { return MixGemmEmit<LowBits>::index(t, v) / 2; }
    static constexpr bool keep(int t, int v)     { return Chunk < 0 || Place::ka(2 * at_plain(t, v)) == Chunk; }
    static constexpr int  at(int t, int v)       { return Chunk < 0 ? at_plain(t, v) : Place::at_h2(2 * at_plain(t, v)); }
    static constexpr bool vreg_used(int v) {
      for (int t = 0; t < kPairs; ++t) if (keep(t, v)) return true;
      return false;
    }
    using LoCodeL = cute::Layout<cute::Shape <cute::Int<kPairs>, cute::_2>,
                                 cute::Stride<cute::_1,          cute::Int<kPairs>>>;
    using HiCodeL = cute::Layout<cute::Shape <cute::Int<kPairs>, cute::Shape<cute::Int<kVregRatio>, cute::Int<4 / kVregRatio>>, cute::_2>,
                                 cute::Stride<cute::_1,          cute::Stride<cute::Int<kPairs>,    cute::_0>,                 cute::Int<16 / HiBits>>>;
    using HVregL  = cute::Layout<cute::Shape <cute::Shape<cute::Int<kVregRatio>, cute::Int<4 / kVregRatio>>>,
                                 cute::Stride<cute::Stride<cute::_0,             cute::Int<kVregRatio>>>>;
    static constexpr int  lo_code(int t, int half)        { return int(LoCodeL{}(t, half)); }
    static constexpr int  hi_code(int t, int v, int half) { return int(HiCodeL{}(t, v, half)); }
    static constexpr int  hi_vreg(int v)                  { return int(HVregL{}(v)); }
    // the converter needs the same thing in BITS, for lane 0; the dup'd mask reaches lane 1 sixteen bits up, which is
    // hstride codes and therefore HiCodeL's `half` mode -- so this is one layout, not two.
    static constexpr int  hshift(int t, int v) { return HiBits * hi_code(t, v, 0); }
    static constexpr uint32_t himask() { return E::dup(uint32_t((1u << HiBits) - 1u)); }

    // Shift high-plane fields above the low-plane mantissa bits, then share
    // the final conversion. Packed storage remains separate for each plane.
    template <int T, int V>
    CUTLASS_DEVICE static void emit_one(uint32_t reg, uint32_t r8, uint32_t hreg, uint32_t* h2) {
      if constexpr (cute::is_same_v<Compute, bfloat16_t>) {
        uint32_t lo = (reg >> (T * LowBits)) & E::dup((1u << LowBits) - 1u);
        uint32_t hi = ((hreg >> hshift(T, V)) & himask()) << LowBits;
        h2[at(T, V)] = direct_bfloat_code_pair<LowBits + HiBits, Bias>(lo | hi);
      } else {
      static_assert(cute::is_same_v<Compute, half_t>);
      const uint32_t src = (T / kPerLevel) ? r8 : reg;        // the upper byte level of the low vreg
      uint32_t x;
      asm volatile("ppu.lop3.b32 %0,%1,%2,%3,%4;\n" : "=r"(x)
                   : "r"(src), "n"(E::template mask<T>()), "n"(0x64006400u), "n"(0xEAu));
      x |= ((hreg >> hshift(T, V)) & himask()) << (E::template bpos<T>() + LowBits);
      asm volatile("ppu.fma.rtte.f16x2 %0,%1,%2,%3;\n" : "=r"(x)
                   : "r"(x), "r"(E::template mul<T>()), "r"(E::template add<T>()));
      h2[at(T, V)] = x;
      }
    }

    template <int V>
    CUTLASS_DEVICE static void emit_v(uint32_t const* lo, uint32_t const* hi, uint32_t* h2) {
      const uint32_t reg = lo[V], r8 = reg >> 8;
      // hi already points at the high fragment offset by this k_block's parity (HiPlaneSrc in the collective); the
      // vregs one k_block owns are P2_DIV apart, so this picks the one serving low vreg V.
      const uint32_t hreg = hi[hi_vreg(V)];                   // (g) HVregL, shared with the offline
      cute::for_each(cute::make_int_sequence<kPairs>{}, [&] (auto t) {
        constexpr int T = decltype(t)::value;
        if constexpr (keep(T, V)) emit_one<T, V>(reg, r8, hreg, h2);
      });
    }

    // lo: the delivery's 4 low vregs. hi: base already offset by the low k_block parity.
    // out: kOut/2 half2 for Chunk < 0, else kPerAtom half2 for k-atom `Chunk`.
    CUTLASS_DEVICE
    static void convert(uint32_t const* lo, uint32_t const* hi, uint32_t* h2)
    {
      cute::for_each(cute::make_int_sequence<4>{}, [&] (auto v) {
        constexpr int V = decltype(v)::value;
        if constexpr (vreg_used(V)) emit_v<V>(lo, hi, h2);
      });
    }
};

// THE SYMMETRIC k-QUANT BIAS, in one place. A 2-plane operand is a GGUF k-quant by construction -- Q3/Q5/Q6 are the only
// formats that split one code across two planes -- and the k-quants with no min channel dequant as d*sc*(c - 2^(W-1)) over
// the concatenated code, W = LowBits+HiBits. Q3_K: W=3, bias 4. Q6_K: W=6, bias 32. (Q5_K has a min, so it keeps the zero
// channel and never asks for this.) Naming it here means the collective and any harness quote ONE constant.
template <int LowBits, int HiBits>
static constexpr int kSymBias2Plane = 1 << (LowBits + HiBits - 1);

// Q3's name, kept so nothing downstream changes. Q6 and Q5 are the same object at different widths.
template <int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = cute::Layout<cute::Shape<cute::_8, cute::_1, cute::_8>>, int Bias = 0>
using MixGemm2Plane_uint2_uint1 = MixGemm2Plane<2, 1, Chunk, NChunk, Rebase, FragLayout, Bias>;
template <int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = cute::Layout<cute::Shape<cute::_8, cute::_1, cute::_4>>, int Bias = 8>
using MixGemm2Plane_int4_uint2 = MixGemm2Plane<4, 2, Chunk, NChunk, Rebase, FragLayout, Bias>;   // Q6
template <int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = cute::Layout<cute::Shape<cute::_8, cute::_1, cute::_4>>, int Bias = 8>
using MixGemm2Plane_int4_uint1 = MixGemm2Plane<4, 1, Chunk, NChunk, Rebase, FragLayout, Bias>;   // Q5

// The collective picks the pair from the two plane ELEMENTS, so a new format needs no dispatch table.
template <class LowElem, class HiElem, int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = MixGemm2PlaneDefaultFrag<cutlass::sizeof_bits<LowElem>::value>,
          int Bias = (cutlass::sizeof_bits<LowElem>::value == 4) ? 8 : 0>
struct MixGemm2PlaneFor {
  using type = MixGemm2Plane<cutlass::sizeof_bits<LowElem>::value, cutlass::sizeof_bits<HiElem>::value,
                             Chunk, NChunk, Rebase, FragLayout, Bias>;
};

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
