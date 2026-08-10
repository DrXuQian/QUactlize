// One real generated-unit instantiation for the local front-end gate. Compiling only the dense main TU proves
// declarations and the registry parse; it does not compile lowbit_dense_unit.inc, where each exported tag and
// wrapper is actually defined. CMake emits this same preamble around batches of rows.
#ifndef PPU_B_CHUNK
#define PPU_B_CHUNK 0
#endif
#if defined(DENSE_STREAMK_AB)
// 107b's isolated four-warp row.  Keep it separate from 107a: Stream-K
// fixup() currently requires exactly one 128-thread barrier cohort.
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(lowbit_dense_streamk_probe,64,128,64,64,32,2,0)
#elif defined(DENSE_PERSISTENT_AB)
// 107a's two causal anchors: BACKTEST A0 and the exact #10/ACU rung whose wave
// geometry implies the 11.1% tail.  Instantiate both scheduler arms for both rows.
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(lowbit_dense_persistent_a0_probe,64,64,64,64,32,3,0) \
  X(lowbit_dense_persistent_rung3_probe,64,128,64,32,32,2,0)
#else
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(lowbit_dense_measurement_probe,64,64,64,32,32,4,PPU_B_CHUNK)
#endif
#include "lowbit_dense_unit.inc"
