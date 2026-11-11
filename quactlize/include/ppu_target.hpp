#pragma once
#include "cutlass/arch/arch.h"

// Explicit in both host and device compilation. Packages default to PPU1.0;
// the public builder selects PPU1.5 only with --arch ppu_15.
#ifndef QUACTLIZE_PPU_ARCH
#define QUACTLIZE_PPU_ARCH 100
#endif
#if QUACTLIZE_PPU_ARCH != 100 && QUACTLIZE_PPU_ARCH != 150
#error "QUACTLIZE_PPU_ARCH must be 100 or 150"
#endif
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ != QUACTLIZE_PPU_ARCH
#error "Quactlize TC target and HGGC device architecture differ"
#endif

namespace quactlize::target {
#if QUACTLIZE_PPU_ARCH == 150
using Arch = cutlass::arch::PPU0015;
#else
using Arch = cutlass::arch::PPU0010;
#endif
}
