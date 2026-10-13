// Compile-only proof: the JIT fallback uses the old device backend's exact
// Shipping type, including DN0 and its default metadata publication. Kernel
// definitions are instantiated, but no device/numeric/timing gate is implied.
#define QK_ROUTE 0
#if QK_QTYPE == 12
#if QZ_LEGACY_DECODE
#define QK_TM 8
#define QK_TN 64
#define QK_WM 8
#else
#define QK_TM 64
#define QK_TN 128
#define QK_WM 64
#endif
#define QK_TK 256
#define QK_WN 16
#define QK_STAGES 2
#else
#define QK_TM 64
#define QK_TN 64
#define QK_WM 32
#define QK_WN 32
#define QK_STAGES 3
#if QK_QTYPE == 14
#define QK_TK 128
#else
#define QK_TK 256
#endif
#endif
#define QK_AP 0
#define QK_DN 0
#define QK_PERSISTENT_PARENT 0
#define QK_PARENT "legacy-compile-probe"
#define QK_BUILD_KEY "compile-only"
#include "quactlize/runtime/module.cuh"

using Current=quactlize::runtime::DenseTypes<QK_QTYPE,QK_TM,QK_TN,QK_TK,QK_WM,QK_WN,QK_STAGES,0,0>;
#if QK_QTYPE == 12
using Old=fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,Current::Schedule,Current::Tile,
    Current::ScaleTile,Current::Warp,QK_STAGES,true,0,0>;
#else
using Old=fpa_intb_ppu::DenseKPackKernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,Current::Schedule,Current::Tile,
    Current::ScaleTile,Current::Warp,QK_STAGES,true,Current::Low,Current::High,0,0>;
#endif
static_assert(std::is_same_v<Old,Current::Shipping>);
static_assert(std::is_same_v<Old::GemmKernel,Current::Shipping::GemmKernel>);
