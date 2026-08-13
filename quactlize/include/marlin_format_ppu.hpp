#pragma once

// Host-only helpers for the classic Marlin W4A16 physical representation.
//
// Independent source anchors:
//   /root/marlin_ppu/test_marlin_classic_group.cu
//   /root/marlin_ppu/ref/awesome-cute/gemm/marlin_gemm/marlin.py
//
// Logical weights are biased int4 codes in row-major [K, N] order.  Packed
// bytes are explicitly little-endian; the uint32 API therefore describes
// words, not host-native byte serialization.  The production shape contract
// is deliberately stricter than the algebra alone: K must be a multiple of
// 128 and N a multiple of 256, matching classic Marlin rather than silently
// accepting a layout the kernel cannot consume.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace quactlize::marlin {

struct ClassicFormatExtent {
  std::size_t logical_codes;
  std::size_t packed_words;
  std::size_t packed_bytes;
  std::size_t scale_elements;
};

namespace detail {

inline bool checked_product(std::size_t lhs, std::size_t rhs, std::size_t& result) noexcept {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

inline bool ranges_overlap(
    void const* lhs, std::size_t lhs_bytes, void const* rhs, std::size_t rhs_bytes) noexcept {
  auto const lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
  auto const rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
  if (lhs_begin > std::numeric_limits<std::uintptr_t>::max() - lhs_bytes ||
      rhs_begin > std::numeric_limits<std::uintptr_t>::max() - rhs_bytes) {
    return true;
  }
  auto const lhs_end = lhs_begin + lhs_bytes;
  auto const rhs_end = rhs_begin + rhs_bytes;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

inline std::size_t logical_offset(
    std::size_t k, std::size_t n, std::size_t n_size) noexcept {
  return k * n_size + n;
}

inline std::uint32_t load_classic_word(
    std::uint8_t const* logical, std::size_t n_size, std::size_t ktile,
    std::size_t nblock, std::size_t lane) noexcept {
  auto const n = nblock * 16 + lane / 4;
  auto const kb = ktile * 16 + (lane % 4) * 2;
  auto const code = [&](std::size_t k, std::size_t column) {
    return static_cast<std::uint32_t>(logical[logical_offset(k, column, n_size)]);
  };

  std::uint32_t word = 0;
  word |= code(kb, n) << 0;
  word |= code(kb + 1, n) << 16;
  word |= code(kb + 8, n) << 4;
  word |= code(kb + 9, n) << 20;
  word |= code(kb, n + 8) << 8;
  word |= code(kb + 1, n + 8) << 24;
  word |= code(kb + 8, n + 8) << 12;
  word |= code(kb + 9, n + 8) << 28;
  return word;
}

inline std::size_t classic_word_offset(
    std::size_t n_size, std::size_t ktile, std::size_t nblock,
    std::size_t lane) noexcept {
  auto const idx = (n_size / 2) * ktile + (nblock / 4) * 32 + lane;
  return idx * 4 + nblock % 4;
}

inline void store_le_u32(std::uint8_t* destination, std::uint32_t word) noexcept {
  destination[0] = static_cast<std::uint8_t>(word >> 0);
  destination[1] = static_cast<std::uint8_t>(word >> 8);
  destination[2] = static_cast<std::uint8_t>(word >> 16);
  destination[3] = static_cast<std::uint8_t>(word >> 24);
}

inline std::uint32_t load_le_u32(std::uint8_t const* source) noexcept {
  return static_cast<std::uint32_t>(source[0]) << 0 |
         static_cast<std::uint32_t>(source[1]) << 8 |
         static_cast<std::uint32_t>(source[2]) << 16 |
         static_cast<std::uint32_t>(source[3]) << 24;
}

inline void unpack_classic_word(
    std::uint32_t word, std::uint8_t* logical, std::size_t n_size,
    std::size_t ktile, std::size_t nblock, std::size_t lane) noexcept {
  auto const n = nblock * 16 + lane / 4;
  auto const kb = ktile * 16 + (lane % 4) * 2;
  auto const store = [&](std::size_t k, std::size_t column, unsigned shift) {
    logical[logical_offset(k, column, n_size)] =
        static_cast<std::uint8_t>((word >> shift) & 0xF);
  };

  store(kb, n, 0);
  store(kb + 1, n, 16);
  store(kb + 8, n, 4);
  store(kb + 9, n, 20);
  store(kb, n + 8, 8);
  store(kb + 1, n + 8, 24);
  store(kb + 8, n + 8, 12);
  store(kb + 9, n + 8, 28);
}

}  // namespace detail

inline bool classic_format_extent(
    std::size_t k_size, std::size_t n_size, ClassicFormatExtent& extent) noexcept {
  if (k_size == 0 || n_size == 0 || k_size % 128 != 0 || n_size % 256 != 0) {
    return false;
  }

  std::size_t logical_codes = 0;
  std::size_t scale_elements = 0;
  if (!detail::checked_product(k_size, n_size, logical_codes) ||
      !detail::checked_product(k_size / 128, n_size, scale_elements)) {
    return false;
  }

  extent = ClassicFormatExtent{
      logical_codes,
      logical_codes / 8,
      logical_codes / 2,
      scale_elements,
  };
  return true;
}

inline bool pack_biased_int4_u32(
    std::uint8_t const* logical, std::size_t logical_count,
    std::uint32_t* packed, std::size_t packed_word_count,
    std::size_t k_size, std::size_t n_size) noexcept {
  ClassicFormatExtent extent{};
  if (!classic_format_extent(k_size, n_size, extent) || logical == nullptr ||
      packed == nullptr || logical_count != extent.logical_codes ||
      packed_word_count != extent.packed_words ||
      detail::ranges_overlap(
          logical, logical_count, packed, packed_word_count * sizeof(std::uint32_t))) {
    return false;
  }

  // Validate before the first write, so an out-of-range code cannot leave a
  // plausible partial artifact behind.
  for (std::size_t index = 0; index < logical_count; ++index) {
    if (logical[index] > 0xF) {
      return false;
    }
  }

  for (std::size_t ktile = 0; ktile < k_size / 16; ++ktile) {
    for (std::size_t nblock = 0; nblock < n_size / 16; ++nblock) {
      for (std::size_t lane = 0; lane < 32; ++lane) {
        auto const destination =
            detail::classic_word_offset(n_size, ktile, nblock, lane);
        packed[destination] =
            detail::load_classic_word(logical, n_size, ktile, nblock, lane);
      }
    }
  }
  return true;
}

inline bool unpack_biased_int4_u32(
    std::uint32_t const* packed, std::size_t packed_word_count,
    std::uint8_t* logical, std::size_t logical_count,
    std::size_t k_size, std::size_t n_size) noexcept {
  ClassicFormatExtent extent{};
  if (!classic_format_extent(k_size, n_size, extent) || packed == nullptr ||
      logical == nullptr || packed_word_count != extent.packed_words ||
      logical_count != extent.logical_codes ||
      detail::ranges_overlap(
          packed, packed_word_count * sizeof(std::uint32_t), logical, logical_count)) {
    return false;
  }

  for (std::size_t ktile = 0; ktile < k_size / 16; ++ktile) {
    for (std::size_t nblock = 0; nblock < n_size / 16; ++nblock) {
      for (std::size_t lane = 0; lane < 32; ++lane) {
        auto const source = detail::classic_word_offset(n_size, ktile, nblock, lane);
        detail::unpack_classic_word(
            packed[source], logical, n_size, ktile, nblock, lane);
      }
    }
  }
  return true;
}

inline bool pack_biased_int4_bytes(
    std::uint8_t const* logical, std::size_t logical_count,
    std::uint8_t* packed, std::size_t packed_byte_count,
    std::size_t k_size, std::size_t n_size) noexcept {
  ClassicFormatExtent extent{};
  if (!classic_format_extent(k_size, n_size, extent) || logical == nullptr ||
      packed == nullptr || logical_count != extent.logical_codes ||
      packed_byte_count != extent.packed_bytes ||
      detail::ranges_overlap(logical, logical_count, packed, packed_byte_count)) {
    return false;
  }
  for (std::size_t index = 0; index < logical_count; ++index) {
    if (logical[index] > 0xF) {
      return false;
    }
  }

  for (std::size_t ktile = 0; ktile < k_size / 16; ++ktile) {
    for (std::size_t nblock = 0; nblock < n_size / 16; ++nblock) {
      for (std::size_t lane = 0; lane < 32; ++lane) {
        auto const destination =
            detail::classic_word_offset(n_size, ktile, nblock, lane) * 4;
        detail::store_le_u32(
            packed + destination,
            detail::load_classic_word(logical, n_size, ktile, nblock, lane));
      }
    }
  }
  return true;
}

inline bool unpack_biased_int4_bytes(
    std::uint8_t const* packed, std::size_t packed_byte_count,
    std::uint8_t* logical, std::size_t logical_count,
    std::size_t k_size, std::size_t n_size) noexcept {
  ClassicFormatExtent extent{};
  if (!classic_format_extent(k_size, n_size, extent) || packed == nullptr ||
      logical == nullptr || packed_byte_count != extent.packed_bytes ||
      logical_count != extent.logical_codes ||
      detail::ranges_overlap(packed, packed_byte_count, logical, logical_count)) {
    return false;
  }

  for (std::size_t ktile = 0; ktile < k_size / 16; ++ktile) {
    for (std::size_t nblock = 0; nblock < n_size / 16; ++nblock) {
      for (std::size_t lane = 0; lane < 32; ++lane) {
        auto const source =
            detail::classic_word_offset(n_size, ktile, nblock, lane) * 4;
        detail::unpack_classic_word(
            detail::load_le_u32(packed + source), logical, n_size, ktile, nblock, lane);
      }
    }
  }
  return true;
}

template <class Scale>
inline bool permute_gs128_scales(
    Scale const* plain, std::size_t plain_count,
    Scale* packed, std::size_t packed_count,
    std::size_t k_size, std::size_t n_size) noexcept {
  static_assert(
      std::is_trivially_copyable<Scale>::value,
      "classic Marlin scale permutation preserves representation bits");
  ClassicFormatExtent extent{};
  std::size_t scale_bytes = 0;
  if (!classic_format_extent(k_size, n_size, extent) || plain == nullptr ||
      packed == nullptr || plain_count != extent.scale_elements ||
      packed_count != extent.scale_elements ||
      !detail::checked_product(extent.scale_elements, sizeof(Scale), scale_bytes) ||
      detail::ranges_overlap(plain, scale_bytes, packed, scale_bytes)) {
    return false;
  }

  auto const groups = k_size / 128;
  for (std::size_t group = 0; group < groups; ++group) {
    for (std::size_t c0 = 0; c0 < n_size; c0 += 64) {
      for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
          packed[group * n_size + c0 + 8 * i + j] =
              plain[group * n_size + c0 + i + 8 * j];
        }
      }
    }
  }
  return true;
}

template <class Scale>
inline bool unpermute_gs128_scales(
    Scale const* packed, std::size_t packed_count,
    Scale* plain, std::size_t plain_count,
    std::size_t k_size, std::size_t n_size) noexcept {
  static_assert(
      std::is_trivially_copyable<Scale>::value,
      "classic Marlin scale permutation preserves representation bits");
  ClassicFormatExtent extent{};
  std::size_t scale_bytes = 0;
  if (!classic_format_extent(k_size, n_size, extent) || packed == nullptr ||
      plain == nullptr || packed_count != extent.scale_elements ||
      plain_count != extent.scale_elements ||
      !detail::checked_product(extent.scale_elements, sizeof(Scale), scale_bytes) ||
      detail::ranges_overlap(packed, scale_bytes, plain, scale_bytes)) {
    return false;
  }

  auto const groups = k_size / 128;
  for (std::size_t group = 0; group < groups; ++group) {
    for (std::size_t c0 = 0; c0 < n_size; c0 += 64) {
      for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
          plain[group * n_size + c0 + i + 8 * j] =
              packed[group * n_size + c0 + 8 * i + j];
        }
      }
    }
  }
  return true;
}

}  // namespace quactlize::marlin
