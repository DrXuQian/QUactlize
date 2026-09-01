// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Host-readable identity and default policy for the finite grouped shipping
// inventory.  Keeping this layer free of device headers lets offline policy
// generators and host-only tests name exactly the tactics compiled by the
// backend.
#pragma once

#include <array>

#include "ppu_grouped_configs.inc"

namespace ppu_grouped_shipping {

enum class ConfigId {
#define QUACTLIZE_PPU_GROUPED_SHIPPING_ID(ID, NAME, TM, TN, WM, WN, STAGES) ID,
  QUACTLIZE_PPU_GROUPED_CONFIGS(QUACTLIZE_PPU_GROUPED_SHIPPING_ID)
#undef QUACTLIZE_PPU_GROUPED_SHIPPING_ID
  Count,
};

struct Config {
  ConfigId id;
  char const* name;
  int tile_m, tile_n, warp_m, warp_n, stages;
};

inline constexpr std::array<Config, static_cast<int>(ConfigId::Count)> kConfigs{{
#define QUACTLIZE_PPU_GROUPED_SHIPPING_ROW(ID, NAME, TM, TN, WM, WN, STAGES) \
  {ConfigId::ID, NAME, TM, TN, WM, WN, STAGES},
  QUACTLIZE_PPU_GROUPED_CONFIGS(QUACTLIZE_PPU_GROUPED_SHIPPING_ROW)
#undef QUACTLIZE_PPU_GROUPED_SHIPPING_ROW
}};

inline constexpr ConfigId kDefault = ConfigId::Default;

constexpr ConfigId default_config() { return kDefault; }

constexpr bool same_name(char const* lhs, char const* rhs) {
  if (!lhs || !rhs) return lhs == rhs;
  for (; *lhs || *rhs; ++lhs, ++rhs) {
    if (*lhs != *rhs) return false;
  }
  return true;
}

// Null and empty names are the only default-selected cases.  A non-empty name
// either names one compiled tensor-core tactic exactly or remains unresolved.
constexpr bool find_config(char const* name, ConfigId& result) {
  if (!name || !name[0]) {
    result = default_config();
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
  for (auto const& config : kConfigs) {
    value = config.tile_m < value ? config.tile_m : value;
  }
  return value;
}

constexpr int minimum_tile_n() {
  int value = kConfigs[0].tile_n;
  for (auto const& config : kConfigs) {
    value = config.tile_n < value ? config.tile_n : value;
  }
  return value;
}

constexpr bool lookup_contract() {
  ConfigId id{};
  if (!find_config(nullptr, id) || id != ConfigId::Default) return false;
  if (!find_config("", id) || id != ConfigId::Default) return false;
  if (!find_config("32x32:16x16:s3", id) ||
      id != ConfigId::SmallSquare) return false;
  if (!find_config("16x128:16x16:s2", id) ||
      id != ConfigId::Default) return false;
  return !find_config("stale-config", id);
}

static_assert(lookup_contract(),
              "null/empty, known explicit and stale grouped config names "
              "must remain distinct states");
static_assert(minimum_tile_m() == 16 && minimum_tile_n() == 32,
              "workspace bounds must include the complete grouped inventory");

}  // namespace ppu_grouped_shipping
