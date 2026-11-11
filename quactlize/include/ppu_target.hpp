#pragma once
#include "cutlass/arch/arch.h"

// Explicit in both host and device compilation. The installed product still
// defaults to PPU1.0; only the bounded PPU1.5 TC runner opts in for now.
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
