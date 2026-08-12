// The dense tactics compiled into libquactlize_ppu.so. Keep the current shipping tactic first: it remains the
// M>=8 default and the fallback for stale/unknown explicit names. Empty/null config names use the shape policy in
// ppu_dense_shipping_policy.hpp: M<8 selects ShortWideM8S3. The backend identifies both shipping defaults at
// template instantiation and proves their exact SharedStorageSize and unrestricted-A path; renaming or replacing
// either row cannot silently weaken that fallback contract.
//
// ShortWideM8 is one generated family, not one hand-picked row. Its six stages are the same finite stage axis used by
// the tactic-space emitter. The broad host arithmetic is executable in l147_dense_shipping_tm8.cpp; production and
// l148 share the compiled GemmKernel type, whose exact SharedStorageSize (including packed staging and C++ object
// padding) decides which format/mode cells list_valid_* admits under one block's 256KiB limit.
#define QUACTLIZE_PPU_DENSE_M8_SHORTWIDE_CONFIGS(X) \
  X(ShortWideM8S2,  "8x128:8x32:s2",    8, 128,  8, 32,  2) \
  X(ShortWideM8S3,  "8x128:8x32:s3",    8, 128,  8, 32,  3) \
  X(ShortWideM8S4,  "8x128:8x32:s4",    8, 128,  8, 32,  4) \
  X(ShortWideM8S6,  "8x128:8x32:s6",    8, 128,  8, 32,  6) \
  X(ShortWideM8S8,  "8x128:8x32:s8",    8, 128,  8, 32,  8) \
  X(ShortWideM8S12, "8x128:8x32:s12",   8, 128,  8, 32, 12)

// These rows are a bootstrap SET drawn from configurations the dense benchmark already instantiated. The coverage
// sweep can replace the non-default rows without changing the dispatcher, enumeration ABI, or GEMM callers.
#define QUACTLIZE_PPU_DENSE_CONFIGS(X) \
  X(Default,     "64x64:32x32:s3",    64,  64, 32, 32, 3) \
  X(SmallSquare, "32x32:16x16:s3",    32,  32, 16, 16, 3) \
  X(ShortWide,   "16x128:16x32:s3",   16, 128, 16, 32, 3) \
  QUACTLIZE_PPU_DENSE_M8_SHORTWIDE_CONFIGS(X) \
  X(MidWide,     "32x128:32x32:s3",   32, 128, 32, 32, 3) \
  X(Tall,        "128x64:64x32:s2",  128,  64, 64, 32, 2)

// One CUDA-core family competes with the tensor-core tiles above. Its fields are deliberately meaningless: callers
// compare enable_cuda_kernel first and ask quactlize_ppu_gemv_lowbit_config_valid_v1 about the actual problem.
#define QUACTLIZE_PPU_DENSE_CUDA_CONFIG_NAME "gemv_lowbit"
