// Generated dense units define LOWBIT_DENSE_UNIT_CONFIGS(X), then include this file. The benchmark source is
// reused in declaration/unit mode so wrappers and main share the exact Cfg, Options, Result, and run<> types.
#define LOWBIT_DENSE_UNIT_BUILD 1
#include "test_lowbit_dense_bench.cu"

template <int GroupSize, int TM, int TN, int TK, int WM, int WN, int ST>
static Result lowbit_dense_run_config(Options& options, TileCfg const& cfg) {
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
// The tag reads the instantiated policy descriptor.  The request macro is not evidence that the collective took
// the path: ordinary/folded/two-plane builders expose one unified atom_at_a_time witness through Descriptor.
#define LOWBIT_DENSE_DEFINE_WRAPPER(FN,TM,TN,TK,WM,WN,ST,BC) \
  int FN##_effective() { \
    using TagCfg = Cfg<LOWBIT_DENSE_TAG_GS,TM,TN,TK,WM,WN,ST>; \
    return TagCfg::Policy::Descriptor::atom_at_a_time ? 1 : 0; \
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
