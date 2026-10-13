// Compile-only matrix: QK_QTYPE=8,10..14; QKD_TEST_TYPED=0/1;
// QKD_USE_BF16_COMPUTE=0/1 for typed endpoints. No device test is implied.
#define QK_TM 16
#define QK_TN 64
#define QK_TK 256
#define QK_WM 16
#define QK_WN 16
#define QK_STAGES 2
#define QK_AP 0
#define QK_DN 16
#define QK_PERSISTENT_PARENT 0
#define QK_PARENT "dynamic-compile-probe"
#define QK_BUILD_KEY "compile-only"
#if QKD_TEST_TYPED
#include "quactlize/decode/dense.cuh"
#else
#include "quactlize/runtime/module.cuh"
#endif
