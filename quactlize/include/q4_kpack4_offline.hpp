#pragma once

#include <cstddef>
#include <cstdint>

// Canonical offline byte map for Q4_K's converter-native transposed K-pack4
// transport.  This header is deliberately host-only and CUTLASS-free: the
// checkpoint producer, recovery tool and CuTe/device type proof all consume
// the same address functions without pulling a PPU SDK into preprocessing.
//
// Native Q4 input is the repository's column-major nibble stream:
//   native[(n * K + k) / 2].nibble[(n * K + k) % 2]
//
// Placed storage is a compact little-endian b16 tensor [K/4, N], with N the
// contiguous dimension.  One word contains four codes from the same Q4_K
// gs32 group:
//   {q[n][32*g+r], q[n][32*g+r+8],
//    q[n][32*g+r+16], q[n][32*g+r+24]}.
//
// Neither a tactic TileK nor an ArtifactTileK occurs in this map.
namespace q4_kpack4 {

inline constexpr int kBits = 4;
inline constexpr int kPack = 4;
inline constexpr int kGroupK = 32;
inline constexpr int kResiduesPerGroup = kGroupK / kPack;
inline constexpr int kTransportK = 64;
inline constexpr int kTransportN = 16;

// Stable ABI identity.  This is an identifier for the mapping above, not a
// hash of a source file whose comments or formatting could change it.
inline constexpr std::uint32_t kLayoutId = 1;
inline constexpr std::uint64_t kMappingId = UINT64_C(0x51344b5034540001);

constexpr bool shape_supported(int n, int k) {
  return n > 0 && k > 0 && k % kGroupK == 0;
}

constexpr std::size_t placed_bytes(int n, int k) {
  return shape_supported(n, k)
      ? std::size_t(n) * std::size_t(k) * kBits / 8
      : std::size_t(0);
}

constexpr int physical_kgroup(int k) {
  return (k / kGroupK) * kResiduesPerGroup + (k % kResiduesPerGroup);
}

constexpr int nibble(int k) {
  return (k % kGroupK) / kResiduesPerGroup;
}

constexpr int logical_k(int physical_kg, int word_nibble) {
  return (physical_kg / kResiduesPerGroup) * kGroupK +
         (physical_kg % kResiduesPerGroup) +
         word_nibble * kResiduesPerGroup;
}

constexpr std::size_t word_index(int n, int k, int logical_n) {
  return std::size_t(physical_kgroup(k)) * std::size_t(logical_n) +
         std::size_t(n);
}

constexpr std::size_t native_code_index(int n, int k, int logical_k_extent) {
  return std::size_t(n) * std::size_t(logical_k_extent) + std::size_t(k);
}

inline std::uint8_t native_get(
    std::uint8_t const* native, int n, int k, int logical_k_extent) {
  std::size_t const i = native_code_index(n, k, logical_k_extent);
  return std::uint8_t((native[i >> 1] >> (4 * (i & 1))) & 0xf);
}

inline void native_put(
    std::uint8_t* native, int n, int k, int logical_k_extent,
    std::uint8_t code) {
  std::size_t const i = native_code_index(n, k, logical_k_extent);
  std::uint8_t const shift = std::uint8_t(4 * (i & 1));
  std::uint8_t const mask = std::uint8_t(0xfu << shift);
  native[i >> 1] = std::uint8_t(
      (native[i >> 1] & std::uint8_t(~mask)) |
      ((code & 0xfu) << shift));
}

inline std::uint16_t placed_word(std::uint8_t const* placed,
                                 std::size_t index) {
  // The format is explicitly little-endian; do not make host alignment or
  // native uint16 endianness part of the checkpoint ABI.
  std::size_t const byte = 2 * index;
  return std::uint16_t(placed[byte]) |
         (std::uint16_t(placed[byte + 1]) << 8);
}

inline void placed_word_put(std::uint8_t* placed, std::size_t index,
                            std::uint16_t word) {
  std::size_t const byte = 2 * index;
  placed[byte] = std::uint8_t(word & 0xffu);
  placed[byte + 1] = std::uint8_t(word >> 8);
}

inline std::uint8_t placed_get(
    std::uint8_t const* placed, int n, int k, int logical_n) {
  std::uint16_t const word = placed_word(placed, word_index(n, k, logical_n));
  return std::uint8_t((word >> (4 * nibble(k))) & 0xf);
}

inline void placed_put(std::uint8_t* placed, int n, int k, int logical_n,
                       std::uint8_t code) {
  std::size_t const index = word_index(n, k, logical_n);
  std::uint16_t word = placed_word(placed, index);
  int const shift = 4 * nibble(k);
  word = std::uint16_t((word & ~(std::uint16_t(0xf) << shift)) |
                       ((std::uint16_t(code) & 0xf) << shift));
  placed_word_put(placed, index, word);
}

// Returns zero on success.  Error 20 is the existing dense-layout malformed
// argument code; 24 identifies a layout-specific unsupported shape.
inline int prepare(std::uint8_t const* native, std::uint8_t* placed,
                   int n, int k) {
  if (!native || !placed || n <= 0 || k <= 0) return 20;
  if (!shape_supported(n, k)) return 24;
  std::size_t const bytes = placed_bytes(n, k);
  for (std::size_t i = 0; i < bytes; ++i) placed[i] = 0;
  for (int col = 0; col < n; ++col)
    for (int kk = 0; kk < k; ++kk)
      placed_put(placed, col, kk, n, native_get(native, col, kk, k));
  return 0;
}

inline int recover(std::uint8_t const* placed, std::uint8_t* native,
                   int n, int k) {
  if (!placed || !native || n <= 0 || k <= 0) return 20;
  if (!shape_supported(n, k)) return 24;
  std::size_t const bytes = placed_bytes(n, k);
  for (std::size_t i = 0; i < bytes; ++i) native[i] = 0;
  for (int col = 0; col < n; ++col)
    for (int kk = 0; kk < k; ++kk)
      native_put(native, col, kk, k, placed_get(placed, col, kk, n));
  return 0;
}

static_assert(physical_kgroup(0) == 0 && physical_kgroup(7) == 7);
static_assert(physical_kgroup(8) == 0 && physical_kgroup(31) == 7);
static_assert(physical_kgroup(32) == 8 && physical_kgroup(63) == 15);
static_assert(logical_k(0, 0) == 0 && logical_k(0, 3) == 24);
static_assert(logical_k(15, 0) == 39 && logical_k(15, 3) == 63);
static_assert(kTransportN * (kTransportK / kPack) * 2 == 512);

}  // namespace q4_kpack4
