#pragma once

// Local nvcc syntax probes need the same host/device split as hgcc, but must
// also select the exact PPU generation named by the probe.  The generic
// ppu_arch_shim.h mirrors __CUDA_ARCH__ (sm_80 -> 800), which reaches the
// PPU0015 fallback and therefore cannot validate a PPU0010 production body.
// Keep the host pass untouched; only the nvcc device pass impersonates the
// ppu0010 hgcc target.
#if defined(__CUDA_ARCH__) && !defined(__HGGC_ARCH__)
#define __HGGC_ARCH__ 100
#endif
