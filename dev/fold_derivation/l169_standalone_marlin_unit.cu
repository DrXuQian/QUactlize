// Compile-only production-unit oracle for the standalone classic-aligned
// Marlin PPU stack.  This is deliberately the same include shape CMake emits:
// a generated per-config TU includes lowbit_dense_unit.inc, which in turn
// instantiates the benchmark wrapper, adapter and device kernel.
#ifdef PPU_B_CHUNK
#undef PPU_B_CHUNK
#endif
#define PPU_B_CHUNK 0
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(l169_standalone_marlin,16,128,128,16,64,4,0)
#include "lowbit_dense_unit.inc"
