#pragma once

// These feature macros must come from the compiler, never the build caller.
#if defined(__CUDA_ARCH__)
# if __CUDA_ARCH__ < 900 || !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#  error "PPU1.7 requires compiler-provided SM90A device features"
# endif
#endif

#if defined(Q17_NATIVE)
# if !defined(__HGGCCC__) && !defined(__HGGC__)
#  error "Native PPU1.7 admission requires the PPU compiler"
# endif
# if !defined(ACOMPUTE_VERSION) || ACOMPUTE_VERSION != 10700
#  error "Native PPU1.7 requires ACOMPUTE_VERSION=10700"
# endif
#endif
