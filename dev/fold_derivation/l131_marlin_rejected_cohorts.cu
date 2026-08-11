// Compile witness for the dense Marlin rejection census.
//
// These are four REAL int4 rows from lowbit_dense_configs.inc, one for every
// CTA-warp cohort rejected by the Marlin-only CMake filter.  With no extra
// define this is the ordinary dense unit path and must match its syntax
// baseline.  run_l131_marlin_rejected_cohorts.sh adds exactly
// DENSE_MARLIN_SWEEP=1, which is equivalent to admitting the rows past the
// CMake cohort filter while leaving the tactic, format, group size, artifact
// TileK and all collective guards untouched.  The resulting compile must red
// on the Marlin cooperative's explicit current-implementation assertions.

#ifndef PPU_B_CHUNK
#define PPU_B_CHUNK 0
#endif

#if !defined(L131_ONLY_CTA_WARPS)
#define LOWBIT_DENSE_UNIT_CONFIGS(X)                                      \
  X(l131_cta_warp1,  8,  16, 64,  8, 16, 2, 0) /*  32 threads */         \
  X(l131_cta_warp8,  8, 128, 64,  8, 16, 2, 0) /* 256 threads */         \
  X(l131_cta_warp16, 8, 256, 64,  8, 16, 2, 0) /* 512 threads */         \
  X(l131_cta_warp32,32, 256, 64, 16, 16, 2, 0) /*1024 threads */
#elif L131_ONLY_CTA_WARPS == 1
#define LOWBIT_DENSE_UNIT_CONFIGS(X) X(l131_cta_warp1,8,16,64,8,16,2,0)
#elif L131_ONLY_CTA_WARPS == 8
#define LOWBIT_DENSE_UNIT_CONFIGS(X) X(l131_cta_warp8,8,128,64,8,16,2,0)
#elif L131_ONLY_CTA_WARPS == 16
#define LOWBIT_DENSE_UNIT_CONFIGS(X) X(l131_cta_warp16,8,256,64,8,16,2,0)
#elif L131_ONLY_CTA_WARPS == 32
#define LOWBIT_DENSE_UNIT_CONFIGS(X) X(l131_cta_warp32,32,256,64,16,16,2,0)
#else
#error "L131_ONLY_CTA_WARPS must be one of 1, 8, 16 or 32"
#endif

#include "lowbit_dense_unit.inc"

// Lower primitives have no 64/128 type-system restriction.  This is not a
// device-correctness claim, but it prevents the census from mislabelling our
// four authored Marlin assertions as an ISA/compiler limit.  Extending the
// cooperative still needs per-cohort lock, coverage and numerical gates.
template <int Cohort, int FragmentElements>
struct L131LowerPrimitiveWitness {
  using Fragment = cutlass::Array<float, FragmentElements>;
  using Striped = cutlass::BlockStripedReduce<Cohort, Fragment>;
  using Barrier = cutlass::NamedBarrierManager<
      Cohort, static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0), 1>;
  static_assert(Striped::kStripes == FragmentElements);
  static_assert(Barrier::ThreadCount == Cohort);
};
static_assert(sizeof(L131LowerPrimitiveWitness<32, 4>) > 0);
static_assert(sizeof(L131LowerPrimitiveWitness<256, 4>) > 0);
static_assert(sizeof(L131LowerPrimitiveWitness<512, 4>) > 0);
static_assert(sizeof(L131LowerPrimitiveWitness<1024, 8>) > 0);
