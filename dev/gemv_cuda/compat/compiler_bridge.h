#pragma once
#if !defined(__NVCC__) || defined(__HGGCCC_VER_MAJOR__)
#error NVIDIA-only diagnostic bridge; never use with the PPU SDK.
#endif
#define __HGGCCC__ 1
#ifdef __CUDA_ARCH__
#define __HGGC_ARCH__ __CUDA_ARCH__
#endif
