// One real generated-unit instantiation for the local front-end gate. Compiling only the dense main TU proves
// declarations and the registry parse; it does not compile lowbit_dense_unit.inc, where each exported tag and
// wrapper is actually defined. CMake emits this same preamble around batches of rows.
#ifndef PPU_B_CHUNK
#define PPU_B_CHUNK 0
#endif
#if defined(DENSE_MARLIN_STANDALONE_SWEEP)
// One exact row from the independent eight-field standalone authority.  WarpK
// and the load token must reach the generated unit; the retired generic
// Marlin wrapper below is not evidence for this branch.
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(lowbit_dense_marlin_standalone_probe,8,128,128,8,64,32,4,CP_ASYNC)
#elif defined(DENSE_MARLIN_SWEEP)
// One row that is present in the committed int4 table and survives the exact
// two/four-warp Marlin filter.  Unlike the A/B arm below, the generated sweep
// wrapper has no runtime scheduler switch: this branch compiles that distinct
// unconditional unit path locally.
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(lowbit_dense_marlin_sweep_probe,16,128,128,16,32,3,0)
#elif defined(DENSE_MARLIN_AB)
// Marlin's scheduler is exercised only when Q<CU.  This legal 128-thread
// decode row has Q=32 for M=1,N=4096 on the 72-CU box; artifact TK remains
// BENCH_TSK=64 while the tactic deliberately consumes it at TK=128.
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(lowbit_dense_marlin_probe,16,128,128,16,32,3,0)
#elif defined(DENSE_STREAMK_AB)
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
