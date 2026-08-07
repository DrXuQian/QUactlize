#pragma once
#include <cstdint>
#include <cstring>
struct __ppu_bfloat16 { uint16_t __x; };
struct __ppu_bfloat16_raw { uint16_t x; __ppu_bfloat16_raw() : x(0) {} __ppu_bfloat16_raw(__ppu_bfloat16 v){ std::memcpy(&x,&v.__x,2); } };
using bfloat16 = __ppu_bfloat16;

// ---------------------------------------------------------------------------------------------------------
// THE bf16 INTRINSICS actlize's cutlass/bfloat16.h CALLS, so a local build can reach the DEVICE spelling.
//
// These did not exist because nothing local ever compiled the device branch: cute/config.hpp:63 keys
// CUTE_INLINE_CONSTANT on __HGGC_ARCH__, which only hgcc defines, so a local nvcc build took the host branch and
// died with thousands of "cute::_ is undefined in device code" instead of ever reaching bfloat16.h. Once
// __HGGC_ARCH__ is supplied for the DEVICE PASS ONLY -- which is what the box does, hgcc for device and g++ for
// host -- those vanish and this is what is left.
//
// SEMANTICS ARE NOT MODELLED, and must not be. Every one of these is bit-manipulation on a 16-bit field whose
// real behaviour lives in the PPU; a stub that computed plausible answers would invite someone to read a local
// result as a numerical one. They exist so the FRONT END can type-check the device branch. The bodies are the
// minimum that keeps the types honest: comparisons return bool, arithmetic returns __ppu_bfloat16.
#define QZ_BF16_CMP(NAME, OP)                                                                     \
  inline bool NAME(__ppu_bfloat16 a, __ppu_bfloat16 b) { return a.__x OP b.__x; }
QZ_BF16_CMP(__heq, ==) QZ_BF16_CMP(__hne, !=) QZ_BF16_CMP(__hlt, <)
QZ_BF16_CMP(__hle, <=) QZ_BF16_CMP(__hgt, >)  QZ_BF16_CMP(__hge, >=)
#undef QZ_BF16_CMP

#define QZ_BF16_ARITH(NAME)                                                                       \
  inline __ppu_bfloat16 NAME(__ppu_bfloat16 a, __ppu_bfloat16 b) { (void)b; return a; }
QZ_BF16_ARITH(__hadd) QZ_BF16_ARITH(__hsub) QZ_BF16_ARITH(__hmul) QZ_BF16_ARITH(__hdiv)
#undef QZ_BF16_ARITH
inline __ppu_bfloat16 __hneg(__ppu_bfloat16 a) { a.__x ^= 0x8000u; return a; }

// The PACKED pair. cutlass/numeric_conversion.h reinterpret_casts 32-bit words to it, so only the size and the
// element type have to be right.
struct __ppu_bfloat162 { __ppu_bfloat16 x, y; };
static_assert(sizeof(__ppu_bfloat162) == 4, "__ppu_bfloat162 must be two packed bf16");
