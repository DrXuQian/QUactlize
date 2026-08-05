// The grouped tactics compiled into libquactlize_ppu.so. Keep the current shipping tactic first: it is the default
// used by ABI entries that do not select a config, and it remains the fallback for stale/unknown names. The backend
// identifies ID==Default at template instantiation and proves its exact SharedStorageSize and unrestricted-A path;
// renaming or replacing this row cannot silently weaken the fallback contract.
//
// These rows are a bootstrap SET drawn from configurations the grouped benchmark already instantiates. The coverage
// sweep can replace the non-default rows without changing the dispatcher, enumeration ABI, or GEMM callers.
#define QUACTLIZE_PPU_GROUPED_CONFIGS(X) \
  X(Default,     "16x128:16x16:s2",    16, 128, 16, 16, 2) \
  X(SmallSquare, "32x32:16x16:s3",     32,  32, 16, 16, 3) \
  X(ShortWide,   "16x128:16x32:s3",    16, 128, 16, 32, 3) \
  X(MidWide,     "32x128:32x32:s3",    32, 128, 32, 32, 3) \
  X(Tall,        "128x64:64x32:s2",   128,  64, 64, 32, 2)

// The grouped CUDA-core family has one tactic. Keep its opaque identity next to the tensor-core inventory so the
// enumeration and the config-selecting vecdot entry cannot acquire two spellings.
#define QUACTLIZE_PPU_GROUPED_CUDA_CONFIG_NAME "vecdot_moe"
