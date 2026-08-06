// The dense tactics compiled into libquactlize_ppu.so. Keep the current shipping tactic first: it is the default
// used by ABI entries that do not select a config, and it remains the fallback for stale/unknown names. The backend
// identifies ID==Default at template instantiation and proves its exact SharedStorageSize and unrestricted-A path;
// renaming or replacing this row cannot silently weaken the fallback contract.
//
// These rows are a bootstrap SET drawn from configurations the dense benchmark already instantiated. The coverage
// sweep can replace the non-default rows without changing the dispatcher, enumeration ABI, or GEMM callers.
#define QUACTLIZE_PPU_DENSE_CONFIGS(X) \
  X(Default,     "64x64:32x32:s3",    64,  64, 32, 32, 3) \
  X(SmallSquare, "32x32:16x16:s3",    32,  32, 16, 16, 3) \
  X(ShortWide,   "16x128:16x32:s3",   16, 128, 16, 32, 3) \
  X(MidWide,     "32x128:32x32:s3",   32, 128, 32, 32, 3) \
  X(Tall,        "128x64:64x32:s2",  128,  64, 64, 32, 2)

// One CUDA-core family competes with the tensor-core tiles above. Its fields are deliberately meaningless: callers
// compare enable_cuda_kernel first and ask quactlize_ppu_gemv_lowbit_config_valid_v1 about the actual problem.
#define QUACTLIZE_PPU_DENSE_CUDA_CONFIG_NAME "gemv_lowbit"
