/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone Marlin PPU W4 dequantization.
 *
 * This is the same biased-int4 algorithm and the same constants used by both the local PPU-classic
 * Marlin port and CalebDu/Awesome-Cute's marlin_gemm: two LOP3 constructions, exact fp16 bias
 * removal, then the independent per-group scale multiply.  A target disassembler may choose a
 * different final mnemonic for part of this expression; that is code generation, not permission to
 * silently change this arithmetic contract.
 **************************************************************************************************/
#pragma once

#include <cuda_fp16.h>

#include "cutlass/cutlass.h"

namespace cutlass::gemm::collective::marlin_ppu_detail {

struct FragmentB {
  __half2 value[2];
};

struct FragmentScale {
  __half2 value[1];
};

template <int Lut>
CUTLASS_DEVICE int lop3(int a, int b, int c) {
  int result;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(result)
               : "r"(a), "r"(b), "r"(c), "n"(Lut));
  return result;
}

CUTLASS_DEVICE FragmentB dequantize_biased_int4(int q) {
  constexpr int kLo = 0x000f000f;
  constexpr int kHi = 0x00f000f0;
  constexpr int kExponent = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, kLo, kExponent);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, kHi, kExponent);
  constexpr int kSubtract = 0x64086408;
  constexpr int kMultiply = 0x2c002c00;
  constexpr int kAdd = 0xd480d480;
  FragmentB result;
  result.value[0] = __hsub2(
      *reinterpret_cast<__half2*>(&lo),
      *reinterpret_cast<__half2 const*>(&kSubtract));
  result.value[1] = __hfma2(
      *reinterpret_cast<__half2*>(&hi),
      *reinterpret_cast<__half2 const*>(&kMultiply),
      *reinterpret_cast<__half2 const*>(&kAdd));
  return result;
}

CUTLASS_DEVICE void scale(
    FragmentB& fragment, FragmentScale const& scales, int half) {
  __half const scalar = reinterpret_cast<__half const*>(&scales)[half];
  __half2 const pair = __half2half2(scalar);
  fragment.value[0] = __hmul2(fragment.value[0], pair);
  fragment.value[1] = __hmul2(fragment.value[1], pair);
}

}  // namespace cutlass::gemm::collective::marlin_ppu_detail
