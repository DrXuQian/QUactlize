// Generated dense units define LOWBIT_DENSE_UNIT_CONFIGS(X), then include this file. The benchmark source is
// reused in declaration/unit mode so wrappers and main share the exact Cfg, Options, Result, and run<> types.
#define LOWBIT_DENSE_UNIT_BUILD 1
#include "test_lowbit_dense_bench.cu"

template <int GroupSize, int TM, int TN, int TK, int WM, int WN, int ST>
static Result lowbit_dense_run_config(Options& options, TileCfg const& cfg) {
#if defined(DENSE_MARLIN_SWEEP)
  // This executable is a Marlin table, not a DP table plus a runtime switch.
  // CMake filters the committed rows before generating these wrappers; retain
  // the production scheduler's structural capability here so a future
  // generator mistake fails compilation instead of growing a second cohort
  // whitelist in the benchmark.
  using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::MarlinGemm;
  using Kernel = typename G::GemmKernel;
  static_assert(Kernel::TileScheduler::fixup_thread_count_capable(
                    Kernel::MaxThreadsPerBlock),
                "the generic dense Marlin sweep requires a structurally capable CTA cohort");
  static_assert(Kernel::TileScheduler::FixupThreadCount ==
                    Kernel::MaxThreadsPerBlock,
                "the generic dense Marlin sweep must preserve its exact CTA/fixup cohort");
  return run<G>(options, dense_tactic(cfg), "marlin");
#else
#if defined(DENSE_MARLIN_WK4_AB)
  // The classic-aligned target is a Marlin-only kernel, not a fourth runtime
  // arm on the historical one-row A/B binary.  In particular, a TiledMma
  // with four K cohorts is not a valid DP/Stream-K result: those kernels do
  // not perform the CTA-local K reduction.  main() independently requires
  // --marlin; keeping the wrapper compile-time Marlin-only prevents a future
  // CLI regression from silently launching one of those numerically invalid
  // alternatives.
  static_assert(GroupSize == 128 &&
                    ((TM == 8 && WM == 8) || (TM == 16 && WM == 16)) &&
                    TN == 128 && TK == 128 && WN == 64 && ST == 4,
                "the first standalone family changes only the real m8/m16 M atom");
  static_assert(DENSE_AB_WARP_K == 32,
                "classic-aligned WarpK is the per-warp K extent 32");
  using StandaloneCfg = StandaloneMarlinCfg<
      GroupSize, TM, TN, TK, WM, WN, ST, DENSE_AB_WARP_K>;
  using G = typename StandaloneCfg::MarlinGemm;
  using Kernel = typename G::GemmKernel;
  static_assert(Kernel::WarpKCohorts == 4 &&
                    Kernel::MaxThreadsPerBlock == 256 &&
                    Kernel::OutputThreads == 64,
                "standalone m8/m16 must be 1M x 2N x 4K = 8 warps with a 64-thread output cohort");
  static_assert(Kernel::InstructionM == TM,
                "the compiled PPU instruction M must match the tactic M atom");
  static_assert(int(Kernel::HandoffMode) == PPU_MARLIN_HANDOFF_HACK,
                "the generated standalone unit must carry the requested compile-time handoff diagnostic mode");
  static_assert(Kernel::IsStandaloneMarlin,
                "the classic-aligned row must instantiate the standalone Marlin kernel");
  static_assert(std::is_same_v<typename Kernel::CollectiveMainloop,
                               typename StandaloneCfg::MarlinMain>,
                "the standalone kernel must consume the standalone Marlin collective");
  return run<G>(options, dense_tactic(cfg), "marlin");
#else
#if defined(DENSE_MARLIN_AB)
  if (options.marlin) {
    using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::MarlinGemm;
    return run<G>(options, dense_tactic(cfg), "marlin");
  }
#endif
#if defined(DENSE_STREAMK_AB)
  if (options.streamk) {
    using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::StreamKGemm;
    return run<G>(options, dense_tactic(cfg), "streamk");
  }
#endif
#if defined(DENSE_SCHEDULER_AB)
  if (options.persistent) {
    using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::PersistentGemm;
    return run<G>(options, dense_tactic(cfg), "persistent");
  }
#endif
  using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST>::Gemm;
  return run<G>(options, dense_tactic(cfg), "non-persistent");
#endif
#endif
}

#if DENSE_GS_ARM(16)
#define LOWBIT_DENSE_CASE_16(TM,TN,TK,WM,WN,ST) \
  case 16: return lowbit_dense_run_config<16,TM,TN,TK,WM,WN,ST>(options, cfg);
#else
#define LOWBIT_DENSE_CASE_16(TM,TN,TK,WM,WN,ST)
#endif
#if DENSE_GS_ARM(32)
#define LOWBIT_DENSE_CASE_32(TM,TN,TK,WM,WN,ST) \
  case 32: return lowbit_dense_run_config<32,TM,TN,TK,WM,WN,ST>(options, cfg);
#else
#define LOWBIT_DENSE_CASE_32(TM,TN,TK,WM,WN,ST)
#endif
#if DENSE_GS_ARM(64)
#define LOWBIT_DENSE_CASE_64(TM,TN,TK,WM,WN,ST) \
  case 64: return lowbit_dense_run_config<64,TM,TN,TK,WM,WN,ST>(options, cfg);
#else
#define LOWBIT_DENSE_CASE_64(TM,TN,TK,WM,WN,ST)
#endif
#if DENSE_GS_ARM(128)
#define LOWBIT_DENSE_CASE_128(TM,TN,TK,WM,WN,ST) \
  case 128: return lowbit_dense_run_config<128,TM,TN,TK,WM,WN,ST>(options, cfg);
#else
#define LOWBIT_DENSE_CASE_128(TM,TN,TK,WM,WN,ST)
#endif

#if defined(BENCH_GS)
#define LOWBIT_DENSE_TAG_GS BENCH_GS
#else
#define LOWBIT_DENSE_TAG_GS 16
#endif
template <int TM, int TN, int TK, int WM, int WN, int ST>
constexpr int lowbit_dense_effective() {
#if defined(DENSE_MARLIN_WK4_AB)
  // The standalone target owns neither the generic mixed-input policy nor its
  // B-chunk descriptor.  Its one admitted classic format has no B-chunk axis.
  return 0;
#else
  using TagCfg = Cfg<LOWBIT_DENSE_TAG_GS,TM,TN,TK,WM,WN,ST>;
  return TagCfg::Policy::Descriptor::atom_at_a_time ? 1 : 0;
#endif
}
// The tag reads the instantiated policy descriptor.  The request macro is not evidence that the collective took
// the path: ordinary/folded/two-plane builders expose one unified atom_at_a_time witness through Descriptor.
#define LOWBIT_DENSE_DEFINE_WRAPPER(FN,TM,TN,TK,WM,WN,ST,BC) \
  int FN##_effective() { \
    return lowbit_dense_effective<TM,TN,TK,WM,WN,ST>(); \
  } \
  char const* FN##_tag() { \
    static char tag[bench_measure::kTagBytes] = {}; \
    if (!tag[0]) bench_measure::format_tag(tag, sizeof tag, \
        bench_measure::Tactic{nullptr,TM,TN,TK,WM,WN,ST,BC,FN##_effective(),false}); \
    return tag; \
  } \
  Result FN(Options& options, TileCfg const& cfg) { \
    switch (options.g) { \
      LOWBIT_DENSE_CASE_16(TM,TN,TK,WM,WN,ST) \
      LOWBIT_DENSE_CASE_32(TM,TN,TK,WM,WN,ST) \
      LOWBIT_DENSE_CASE_64(TM,TN,TK,WM,WN,ST) \
      LOWBIT_DENSE_CASE_128(TM,TN,TK,WM,WN,ST) \
      default: \
        std::fprintf(stderr, "unsupported dense group size %d (supported: 16, 32, 64, 128)\n", options.g); \
        return {}; \
    } \
  }
LOWBIT_DENSE_UNIT_CONFIGS(LOWBIT_DENSE_DEFINE_WRAPPER)

#undef LOWBIT_DENSE_DEFINE_WRAPPER
#undef LOWBIT_DENSE_CASE_128
#undef LOWBIT_DENSE_CASE_64
#undef LOWBIT_DENSE_CASE_32
#undef LOWBIT_DENSE_CASE_16
#undef LOWBIT_DENSE_TAG_GS
