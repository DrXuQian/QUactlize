// Compile witness for the dense Marlin cohort capability.
//
// These are four REAL int4 rows from lowbit_dense_configs.inc, one for every
// CTA-warp cohort that the old 64/128-thread whitelist rejected.  With no
// extra define this remains the ordinary dense syntax witness.  With
// DENSE_MARLIN_SWEEP=1 each exact same row must instantiate its real named
// Marlin wrapper.  L131_WRONG_EXPLICIT_COHORT then supplies a different but
// otherwise structurally-capable cohort to the production scheduler; that
// compile must red on the exact accumulator-derived binding, not a whitelist.

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
#define L131_TM 8
#define L131_TN 16
#define L131_TK 64
#define L131_WM 8
#define L131_WN 16
#define L131_ST 2
#elif L131_ONLY_CTA_WARPS == 8
#define LOWBIT_DENSE_UNIT_CONFIGS(X) X(l131_cta_warp8,8,128,64,8,16,2,0)
#define L131_TM 8
#define L131_TN 128
#define L131_TK 64
#define L131_WM 8
#define L131_WN 16
#define L131_ST 2
#elif L131_ONLY_CTA_WARPS == 16
#define LOWBIT_DENSE_UNIT_CONFIGS(X) X(l131_cta_warp16,8,256,64,8,16,2,0)
#define L131_TM 8
#define L131_TN 256
#define L131_TK 64
#define L131_WM 8
#define L131_WN 16
#define L131_ST 2
#elif L131_ONLY_CTA_WARPS == 32
#define LOWBIT_DENSE_UNIT_CONFIGS(X) X(l131_cta_warp32,32,256,64,16,16,2,0)
#define L131_TM 32
#define L131_TN 256
#define L131_TK 64
#define L131_WM 16
#define L131_WN 16
#define L131_ST 2
#else
#error "L131_ONLY_CTA_WARPS must be one of 1, 8, 16 or 32"
#endif

#include "lowbit_dense_unit.inc"

// Lower primitives consume the exact same cohort value.  This is not a device
// progress claim; it is a compile-time witness that neither primitive hides a
// private 128-thread default after the named wrapper has selected its cohort.
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

#if defined(L131_WRONG_EXPLICIT_COHORT)
#if !defined(DENSE_MARLIN_SWEEP) || !defined(L131_ONLY_CTA_WARPS)
#error "L131 wrong-cohort control requires one selected named Marlin wrapper"
#endif

using L131Cfg = Cfg<32, L131_TM, L131_TN, L131_TK,
                    L131_WM, L131_WN, L131_ST>;
using L131RealGemm = typename L131Cfg::MarlinGemm;
using L131RealKernel = typename L131RealGemm::GemmKernel;
using L131WrongScheduler =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPUMarlin<
        typename L131RealKernel::TileShape,
        typename L131RealKernel::ClusterShape,
        L131_WRONG_EXPLICIT_COHORT>;
using L131Fragment = decltype(cute::make_fragment_like<float>(
    cute::partition_fragment_C(
        typename L131RealKernel::TiledMma{},
        cute::take<0, 2>(typename L131RealKernel::TileShape{}))));

// A real device entry forces instantiation of fixup_predicated(), where the
// explicit cohort is compared with the real accumulator fragment.  Merely
// naming L131WrongScheduler would only prove that the broad structural
// capability accepts it; it would not prove the exact binding is enforced.
extern "C" __global__ void l131_wrong_explicit_cohort_probe() {
  typename L131WrongScheduler::Params params{};
  typename L131WrongScheduler::WorkTileInfo work{};
  L131Fragment accumulators;
  L131WrongScheduler::fixup(params, work, accumulators, 1, 0);
}
#endif
