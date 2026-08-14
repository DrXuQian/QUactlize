// Representative production generated-unit compile for the standalone
// Marlin TN/TK axis.  These are not parallel model types: each row takes the
// exact lowbit_dense_unit.inc route emitted by the production CMake target.
#ifdef PPU_B_CHUNK
#undef PPU_B_CHUNK
#endif
#define PPU_B_CHUNK 0
#ifndef L185_UNIT_CASE
#define L185_UNIT_CASE 0
#endif
#if L185_UNIT_CASE == 0
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(l185_tm16_tn64_tk128,16,64,128,16,64,32,4,CP_ASYNC)
#elif L185_UNIT_CASE == 1
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(l185_tm8_tn128_tk64,8,128,64,8,64,32,4,CP_ASYNC)
#elif L185_UNIT_CASE == 2
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(l185_tm16_tn128_tk64,16,128,64,16,64,32,4,CP_ASYNC)
#elif L185_UNIT_CASE == 3
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(l185_tm8_tn256_tk64,8,256,64,8,64,32,4,CP_ASYNC)
#elif L185_UNIT_CASE == 4
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(l185_tm16_tn256_tk64,16,256,64,16,64,32,4,CP_ASYNC)
#elif L185_UNIT_CASE == 5
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(l185_tm8_tn128_tk64_wn128,8,128,64,8,128,16,4,CP_ASYNC)
#elif L185_UNIT_CASE == 6
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(l185_tm16_tn128_tk128_wn128,16,128,128,16,128,16,4,CP_ASYNC)
#elif L185_UNIT_CASE == 7
#define LOWBIT_DENSE_UNIT_CONFIGS(X) \
  X(l185_tm16_tn256_tk64_wn128,16,256,64,16,128,16,4,CP_ASYNC)
#else
#error "L185_UNIT_CASE must be in [0,7]"
#endif
#include "lowbit_dense_unit.inc"
