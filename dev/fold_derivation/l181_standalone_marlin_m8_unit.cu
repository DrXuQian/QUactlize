// Compile-only generated-unit oracle for standalone Marlin's packed-M1 m8
// row.  Its one-row A stage and plain-x2 load are production properties; the
// B/scale artifact and 2N x 4K topology remain identical to m16.
#ifdef PPU_B_CHUNK
#undef PPU_B_CHUNK
#endif
#define PPU_B_CHUNK 0
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(l181_standalone_marlin_m8,8,128,128,8,64,4,0)
#include "lowbit_dense_unit.inc"
