// THE DEVICE-PASS MARKER, so a local nvcc build takes the same branch the box does.
//
//   nvcc ... -include dev/fold_derivation/stub_inc/ppu_arch_shim.h ...
//
// WHAT IT FIXES, measured on benchmarks/test_lowbit_dense_bench.cu:
//
//     without           5954 errors, all "cute::_ / cute::product is undefined in device code"
//     with               164 errors, ALL of one class from one vendor header (see below)
//
// cute/config.hpp:63 keys CUTE_INLINE_CONSTANT on __HGGC_ARCH__: `static const __device__` when defined,
// `static constexpr` otherwise. On the box that macro is hgcc's, and hgcc compiles ONLY the device half -- the
// host half is g++ and never sees it. nvcc -x cu compiles both halves of one TU, so:
//
//   define it nowhere        device code odr-uses namespace-scope `static constexpr` -> "undefined in device code"
//   define it unconditionally the HOST half also takes device spellings -> "calling a __device__ function from a
//                            __host__ __device__ function", 135 of them
//   define it under __CUDA_ARCH__   each half sees what it sees on the box.       <- this file
//
// dev/fold_derivation/syntax_check.sh does not use it YET: turning it on changes every recorded baseline at once
// (they were recorded against the 5954-error world), so that is a re-baselining commit of its own. See task #39.
//
// WHAT REMAINS, and it is not ours to fix: 164 instances of "asm operand type size(4) does not match type/size
// implied by constraint", every one from third_party/actlize/include/cute/atom/mma_ppu0015.hpp. The constraints
// are written in hgcc's asm dialect and nvcc type-checks them anyway -- `nvcc -cuda` does NOT treat inline asm as
// an opaque string, which syntax_check.sh's own header comment assumes it does. actlize is a vendor submodule and
// ci/check_actlize_pristine.py exists to keep us out of it, so this is a floor, not a bug to chase.
#pragma once
#if defined(__CUDA_ARCH__) && !defined(__HGGC_ARCH__)
#define __HGGC_ARCH__ __CUDA_ARCH__
#endif
