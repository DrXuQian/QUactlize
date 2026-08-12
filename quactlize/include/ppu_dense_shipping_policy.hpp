// Host-readable identity and default policy for the finite dense shipping inventory.
//
// This header deliberately depends only on the X-macro inventory. The device backend and l147 consume this exact
// type/function; the local oracle therefore cannot prove one shape policy while production silently uses another.
#pragma once

#include <array>

#include "ppu_dense_configs.inc"

namespace ppu_dense_shipping {

enum class ConfigId {
#define QUACTLIZE_PPU_DENSE_SHIPPING_ID(ID, NAME, TM, TN, WM, WN, STAGES) ID,
  QUACTLIZE_PPU_DENSE_CONFIGS(QUACTLIZE_PPU_DENSE_SHIPPING_ID)
#undef QUACTLIZE_PPU_DENSE_SHIPPING_ID
  Count,
};

struct Config {
  ConfigId id;
  char const* name;
  int tile_m, tile_n, warp_m, warp_n, stages;
};

inline constexpr std::array<Config, static_cast<int>(ConfigId::Count)> kConfigs{{
#define QUACTLIZE_PPU_DENSE_SHIPPING_ROW(ID, NAME, TM, TN, WM, WN, STAGES) \
  {ConfigId::ID, NAME, TM, TN, WM, WN, STAGES},
  QUACTLIZE_PPU_DENSE_CONFIGS(QUACTLIZE_PPU_DENSE_SHIPPING_ROW)
#undef QUACTLIZE_PPU_DENSE_SHIPPING_ROW
}};

inline constexpr ConfigId kLegacyDefault = ConfigId::Default;
inline constexpr ConfigId kDecodeDefault = ConfigId::ShortWideM8S3;
inline constexpr int kDecodeDefaultExclusiveM = 8;

constexpr ConfigId default_config_for_m(int m) {
  return m < kDecodeDefaultExclusiveM ? kDecodeDefault : kLegacyDefault;
}

constexpr bool same_name(char const* lhs, char const* rhs) {
  if (!lhs || !rhs) return lhs == rhs;
  for (; *lhs || *rhs; ++lhs, ++rhs) if (*lhs != *rhs) return false;
  return true;
}

// Null/empty is the one shape-selected case. A known non-empty name is always exact, and an unknown non-empty name
// is deliberately left unresolved so launch can retain its legacy stale-name warning/fallback while validity and
// arrangement queries fail closed. Backend and host oracle both consume this function.
constexpr bool find_config(char const* name, int m, ConfigId& result) {
  if (!name || !name[0]) {
    result = default_config_for_m(m);
    return true;
  }
  for (auto const& config : kConfigs) {
    if (same_name(name, config.name)) {
      result = config.id;
      return true;
    }
  }
  return false;
}

constexpr int minimum_tile_m() {
  int value = kConfigs[0].tile_m;
  for (auto const& config : kConfigs) value = config.tile_m < value ? config.tile_m : value;
  return value;
}

constexpr int minimum_tile_n() {
  int value = kConfigs[0].tile_n;
  for (auto const& config : kConfigs) value = config.tile_n < value ? config.tile_n : value;
  return value;
}

static_assert(default_config_for_m(1) == ConfigId::ShortWideM8S3 &&
              default_config_for_m(7) == ConfigId::ShortWideM8S3 &&
              default_config_for_m(8) == ConfigId::Default,
              "empty-config decode must use m8 while the M>=8 shipping default remains unchanged");
constexpr bool lookup_contract() {
  ConfigId id{};
  if (!find_config(nullptr, 1, id) || id != ConfigId::ShortWideM8S3) return false;
  if (!find_config("", 7, id) || id != ConfigId::ShortWideM8S3) return false;
  if (!find_config("8x128:8x32:s4", 1, id) || id != ConfigId::ShortWideM8S4) return false;
  if (!find_config("64x64:32x32:s3", 1, id) || id != ConfigId::Default) return false;
  return !find_config("stale-config", 1, id);
}
static_assert(lookup_contract(),
              "null/empty, known explicit and stale dense config names must remain distinct states");
static_assert(minimum_tile_m() == 8 && minimum_tile_n() == 32,
              "workspace bounds must include the complete m8 shipping inventory");

}  // namespace ppu_dense_shipping
