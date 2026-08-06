// Generated dense units define LOWBIT_DENSE_UNIT_CONFIGS(X), then include this file. The benchmark source is
// reused in declaration/unit mode so wrappers and main share the exact Cfg, Options, Result, and run<> types.
#define LOWBIT_DENSE_UNIT_BUILD 1
#include "test_lowbit_dense_bench.cu"

template <int GroupSize, int TM, int TN, int TK, int WM, int WN, int ST, int ACR>
static Result lowbit_dense_run_config(Options& options, TileCfg const& cfg) {
  using G = typename Cfg<GroupSize, TM, TN, TK, WM, WN, ST, ACR>::Gemm;
  return run<G>(options, cfg.name, cfg.acr);
}

#if DENSE_GS_ARM(16)
#define LOWBIT_DENSE_CASE_16(TM,TN,TK,WM,WN,ST,ACR) \
  case 16: return lowbit_dense_run_config<16,TM,TN,TK,WM,WN,ST,ACR>(options, cfg);
#else
#define LOWBIT_DENSE_CASE_16(TM,TN,TK,WM,WN,ST,ACR)
#endif
#if DENSE_GS_ARM(32)
#define LOWBIT_DENSE_CASE_32(TM,TN,TK,WM,WN,ST,ACR) \
  case 32: return lowbit_dense_run_config<32,TM,TN,TK,WM,WN,ST,ACR>(options, cfg);
#else
#define LOWBIT_DENSE_CASE_32(TM,TN,TK,WM,WN,ST,ACR)
#endif
#if DENSE_GS_ARM(64)
#define LOWBIT_DENSE_CASE_64(TM,TN,TK,WM,WN,ST,ACR) \
  case 64: return lowbit_dense_run_config<64,TM,TN,TK,WM,WN,ST,ACR>(options, cfg);
#else
#define LOWBIT_DENSE_CASE_64(TM,TN,TK,WM,WN,ST,ACR)
#endif
#if DENSE_GS_ARM(128)
#define LOWBIT_DENSE_CASE_128(TM,TN,TK,WM,WN,ST,ACR) \
  case 128: return lowbit_dense_run_config<128,TM,TN,TK,WM,WN,ST,ACR>(options, cfg);
#else
#define LOWBIT_DENSE_CASE_128(TM,TN,TK,WM,WN,ST,ACR)
#endif

#define LOWBIT_DENSE_DEFINE_WRAPPER(FN,TM,TN,TK,WM,WN,ST,ACR) \
  Result FN(Options& options, TileCfg const& cfg) { \
    switch (options.g) { \
      LOWBIT_DENSE_CASE_16(TM,TN,TK,WM,WN,ST,ACR) \
      LOWBIT_DENSE_CASE_32(TM,TN,TK,WM,WN,ST,ACR) \
      LOWBIT_DENSE_CASE_64(TM,TN,TK,WM,WN,ST,ACR) \
      LOWBIT_DENSE_CASE_128(TM,TN,TK,WM,WN,ST,ACR) \
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
