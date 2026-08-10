// Compile-only proof for #114's dormant ppu001 plain-LDSM defect.
//
// nvcc is used only as a C++/PTX front end.  __HGGC_ARCH__ is defined in the
// device pass, never globally, so actlize's host pass keeps its ordinary
// compatibility path.  The CI driver builds four cases:
//   ppu001, calls off  -> include/unused must compile;
//   ppu001, calls on   -> exactly 12 deleted-function diagnostics;
//   ppu001, helpers on -> two dependent static-assert reasons;
//   ppu0015, calls on  -> compile and retain 12 tc02 opcodes in PTX.

// This does not establish a tc01 opcode spelling.  It proves the opposite:
// until one has an SDK + numerical gate, every ppu001 plain call fails before
// an assembler can see it.

#if defined(__CUDA_ARCH__)
#  if !defined(PPU_PLAIN_LDSM_PROBE_ARCH)
#    error "PPU_PLAIN_LDSM_PROBE_ARCH must be 100 or 150"
#  endif
#  define __HGGC_ARCH__ PPU_PLAIN_LDSM_PROBE_ARCH
#endif

#include <cstdint>
#include <type_traits>

#include "cute/arch/copy_ppu.hpp"
#include "cutlass/arch/memory_ppu.h"

#if PPU_PLAIN_LDSM_PROBE_ARCH == 150
// The fail-close must not silently turn ppu0015's six ordinary member
// functions into templates or otherwise change their exact pointer types.
using U32x1Fn = void (*)(cute::uint128_t const&, std::uint32_t&);
using U32x2Fn = void (*)(cute::uint128_t const&, std::uint32_t&, std::uint32_t&);
using U32x4Fn = void (*)(cute::uint128_t const&, std::uint32_t&, std::uint32_t&,
                         std::uint32_t&, std::uint32_t&);
static_assert(std::is_same_v<decltype(&cute::PPU_U32x1_LDSM_N::copy), U32x1Fn>);
static_assert(std::is_same_v<decltype(&cute::PPU_U32x2_LDSM_N::copy), U32x2Fn>);
static_assert(std::is_same_v<decltype(&cute::PPU_U32x4_LDSM_N::copy), U32x4Fn>);
static_assert(std::is_same_v<decltype(&cute::PPU_U16x2_LDSM_T::copy), U32x1Fn>);
static_assert(std::is_same_v<decltype(&cute::PPU_U16x4_LDSM_T::copy), U32x2Fn>);
static_assert(std::is_same_v<decltype(&cute::PPU_U16x8_LDSM_T::copy), U32x4Fn>);
#endif

#if PPU_PLAIN_LDSM_PROBE_HELPERS
__global__ void plain_ldsm_probe() {
  __shared__ cute::uint128_t src[1];
  std::uint32_t dst[4]{};
  cute::copy_ldsm(src, dst);
  cute::copy_ldsm_trans(src, dst);
}
#elif PPU_PLAIN_LDSM_PROBE_CALLS
__global__ void plain_ldsm_probe() {
  __shared__ cute::uint128_t src[1];
  std::uint32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

  cute::PPU_U32x1_LDSM_N::copy(src[0], d0);
  cute::PPU_U32x2_LDSM_N::copy(src[0], d0, d1);
  cute::PPU_U32x4_LDSM_N::copy(src[0], d0, d1, d2, d3);
  cute::PPU_U16x2_LDSM_T::copy(src[0], d0);
  cute::PPU_U16x4_LDSM_T::copy(src[0], d0, d1);
  cute::PPU_U16x8_LDSM_T::copy(src[0], d0, d1, d2, d3);

  cutlass::Array<unsigned, 1> r1{};
  cutlass::Array<unsigned, 2> r2{};
  cutlass::Array<unsigned, 4> r4{};
  cutlass::arch::ldsm<cutlass::layout::RowMajor, 1>(r1, src);
  cutlass::arch::ldsm<cutlass::layout::RowMajor, 2>(r2, src);
  cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(r4, src);
  cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 1>(r1, src);
  cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 2>(r2, src);
  cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 4>(r4, src);
}
#else
__global__ void plain_ldsm_probe() {}
#endif
