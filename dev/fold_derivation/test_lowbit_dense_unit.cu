// One real generated-unit instantiation for the local front-end gate. Compiling only the dense main TU proves
// declarations and the registry parse; it does not compile lowbit_dense_unit.inc, where each exported tag and
// wrapper is actually defined. CMake emits this same preamble around batches of rows.
#ifndef PPU_B_CHUNK
#define PPU_B_CHUNK 0
#endif
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(lowbit_dense_measurement_probe,64,64,64,32,32,4,PPU_B_CHUNK)
#include "lowbit_dense_unit.inc"
