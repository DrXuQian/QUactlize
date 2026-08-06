#pragma once

#include <array>

#include "ppu_format_config.inc"

namespace ppu_formats {

enum class Id {
#define QUACTLIZE_PPU_FORMAT_ID(ID, NAME, QTYPE, LOW, HIGH, GROUP, SCALE_TK, FQ_TK, PACKED) ID,
  QUACTLIZE_PPU_FORMAT_CONFIGS(QUACTLIZE_PPU_FORMAT_ID)
#undef QUACTLIZE_PPU_FORMAT_ID
};

struct Config {
  Id id;
  char const* name;
  int qtype;
  int low_bits;
  int high_bits;
  int group_size;
  int scale_first_tile_k;
  int fully_quantized_tile_k;
  int packed_format;
};

inline constexpr std::array<Config, 5> kConfigs{{
#define QUACTLIZE_PPU_FORMAT_ROW(ID, NAME, QTYPE, LOW, HIGH, GROUP, SCALE_TK, FQ_TK, PACKED) \
  {Id::ID, NAME, QTYPE, LOW, HIGH, GROUP, SCALE_TK, FQ_TK, PACKED},
  QUACTLIZE_PPU_FORMAT_CONFIGS(QUACTLIZE_PPU_FORMAT_ROW)
#undef QUACTLIZE_PPU_FORMAT_ROW
}};

inline constexpr Config kInvalid{Id::Q2_K, "invalid", -1, 0, 0, 0, 0, 0, -1};

constexpr Config const& for_qtype(int qtype) {
  for (auto const& config : kConfigs)
    if (config.qtype == qtype) return config;
  return kInvalid;
}

constexpr Config const& for_packed_format(int packed_format) {
  for (auto const& config : kConfigs)
    if (config.packed_format == packed_format) return config;
  return kInvalid;
}

constexpr int minimum_delivery_tile_k(Config config) {
  int const narrowest_bits = config.high_bits ? config.high_bits : config.low_bits;
  return narrowest_bits > 0 && 256 % narrowest_bits == 0 ? 256 / narrowest_bits : 0;
}

constexpr bool registry_is_valid() {
  for (auto const& config : kConfigs) {
    if (config.qtype < 0 || config.group_size <= 0 || config.low_bits <= 0 ||
        config.scale_first_tile_k != minimum_delivery_tile_k(config) ||
        config.fully_quantized_tile_k < config.scale_first_tile_k ||
        256 % config.scale_first_tile_k || 256 % config.fully_quantized_tile_k)
      return false;
  }
  return true;
}
static_assert(registry_is_valid(),
              "PPU per-format TileK must satisfy the canonical scale-first delivery and superblock domains");

}  // namespace ppu_formats
