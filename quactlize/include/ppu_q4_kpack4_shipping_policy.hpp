#pragma once

// Production K-pack4 tactics are a separate inventory from Xplane.  The byte map is one canonical v2 descriptor;
// these rows change only compute geometry and fixed Split-K S.  Encoding them in the old dense table would compile
// every K-pack-only row for all five Xplane formats and, worse, would let one config name appear to mean the same
// physical reader on both sides.
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ppu_q4_kpack4_shipping {

inline constexpr char kScaleFirstPersistentName[] =
    "kpack4:scalefirst:64x64x64:64x32:s3:persistent";

#define QUACTLIZE_PPU_Q4_KPACK4_CONFIGS(X) \
  X(DecodeN32S4,  "kpack4:8x32x256:8x16:s3:S4",    8,  32, 256,  8, 16, 3, 4) \
  X(DecodeN64S4,  "kpack4:8x64x256:8x16:s2:S4",    8,  64, 256,  8, 16, 2, 4) \
  X(DecodeN128S4, "kpack4:8x128x256:8x16:s2:S4",   8, 128, 256,  8, 16, 2, 4) \
  X(DecodeN64S1,  "kpack4:8x64x256:8x16:s2:S1",    8,  64, 256,  8, 16, 2, 1) \
  X(PrefillS1,    "kpack4:64x128x256:64x16:s2:S1", 64, 128, 256, 64, 16, 2, 1)

enum class ConfigId : int32_t {
#define QZ_KPACK4_ID(ID, NAME, TM, TN, TK, WM, WN, STAGES, SPLIT) ID,
  QUACTLIZE_PPU_Q4_KPACK4_CONFIGS(QZ_KPACK4_ID)
#undef QZ_KPACK4_ID
  Count,
};

struct Config {
  ConfigId id;
  char const* name;
  int32_t tile_m;
  int32_t tile_n;
  int32_t tile_k;
  int32_t warp_m;
  int32_t warp_n;
  int32_t stages;
  int32_t split;
};

inline constexpr Config kConfigs[] = {
#define QZ_KPACK4_ROW(ID, NAME, TM, TN, TK, WM, WN, STAGES, SPLIT) \
  {ConfigId::ID, NAME, TM, TN, TK, WM, WN, STAGES, SPLIT},
  QUACTLIZE_PPU_Q4_KPACK4_CONFIGS(QZ_KPACK4_ROW)
#undef QZ_KPACK4_ROW
};

static_assert(sizeof(kConfigs) / sizeof(kConfigs[0]) == size_t(ConfigId::Count));

constexpr ConfigId default_config(int m, int n, int k) {
  // Decode policy is the compact form of the 20-shape closure.  It is deliberately shape-only: the one resident
  // K-pack4 byte class never changes with M.  Explicit config names remain available for a deployment registry.
  if (m <= 8) {
    if (n <= 2048) return ConfigId::DecodeN32S4;
    if (n >= 16384 || (m == 8 && k >= 16384)) return ConfigId::DecodeN64S1;
    if (n >= 7168) return ConfigId::DecodeN128S4;
    return ConfigId::DecodeN64S4;
  }
  return ConfigId::PrefillS1;
}

inline bool find_config(char const* name, int m, int n, int k, ConfigId& out) {
  if (!name || !name[0]) {
    out = default_config(m, n, k);
    return true;
  }
  for (auto const& row : kConfigs) {
    if (std::strcmp(name, row.name) == 0) {
      out = row.id;
      return true;
    }
  }
  return false;
}

constexpr Config const& row(ConfigId id) {
  return kConfigs[static_cast<int32_t>(id)];
}

}  // namespace ppu_q4_kpack4_shipping
