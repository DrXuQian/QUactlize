#pragma once

#include <cstddef>
#include <cstdint>

// Canonical host-only byte map for the Q4 N16 x K64 direct delivery.
//
// One logical N16 x K64 atom contains 1024 four-bit codes (512 bytes).  Its
// plain physical representation is [K/16=4][2*N=32] little-endian u32 words.
// The low-to-high bits of the atom's physical nibble index are:
//
//   p[0..9] = {k3,k4,k0,k5,n3,k1,k2,n0,n1,n2}.
//
// Larger tensors repeat that atom in the same compact physical tensor:
//
//   [K/16][2*N] u32, stride [2*N,1].
//
// The mapping was derived by L240 from the complete production CuTe chain:
// UniversalCopy<uint128_t>, register recast, MixGemmEmit<4>, and the actual
// m8 partition_fragment_B destination.  This header deliberately remains
// CUTLASS-free so checkpoint preparation and recovery do not depend on a PPU
// SDK.  The serialized byte operations below define little endian explicitly;
// neither host alignment nor native uint32_t endianness is part of the ABI.
namespace q4_n16k64_direct {

inline constexpr int kBits = 4;
inline constexpr int kNAtom = 16;
inline constexpr int kKAtom = 64;
inline constexpr int kCodesPerWord = 8;
inline constexpr int kPhysicalKPerRow = 16;
inline constexpr int kWordsPerLogicalN = 2;
inline constexpr int kPhysicalKRowsPerAtom = kKAtom / kPhysicalKPerRow;
inline constexpr int kPhysicalNWordsPerAtom =
    kWordsPerLogicalN * kNAtom;

// Stable semantic identities.  kMappingId names the complete outer byte map;
// kAtomMappingFingerprint is the FNV witness printed by L240 for the 1024-entry
// logical-to-physical atom permutation.  The fingerprint is evidence, not a
// substitute for the semantic mapping id carried by the artifact descriptor.
inline constexpr std::uint32_t kLayoutId = 3;
inline constexpr std::uint64_t kMappingId =
    UINT64_C(0x51344e3136440001);
inline constexpr std::uint64_t kAtomMappingFingerprint =
    UINT64_C(0x74443aed0cce4083);

struct LogicalCoordinate {
  int n;
  int k;
};

constexpr bool shape_supported(int n, int k) {
  return n > 0 && n % kNAtom == 0 &&
         k > 0 && k % kKAtom == 0;
}

constexpr std::size_t placed_bytes(int n, int k) {
  return shape_supported(n, k)
      ? std::size_t(n) * std::size_t(k) * kBits / 8
      : std::size_t(0);
}

// Physical nibble within one N16 x K64 atom.
constexpr int atom_physical_nibble(int n, int k) {
  return ((k >> 3) & 1) * 1 +
         ((k >> 4) & 1) * 2 +
         ((k >> 0) & 1) * 4 +
         ((k >> 5) & 1) * 8 +
         ((n >> 3) & 1) * 16 +
         ((k >> 1) & 1) * 32 +
         ((k >> 2) & 1) * 64 +
         ((n >> 0) & 1) * 128 +
         ((n >> 1) & 1) * 256 +
         ((n >> 2) & 1) * 512;
}

constexpr LogicalCoordinate atom_logical_coordinate(int physical_nibble) {
  return LogicalCoordinate{
      ((physical_nibble >> 7) & 1) * 1 +
          ((physical_nibble >> 8) & 1) * 2 +
          ((physical_nibble >> 9) & 1) * 4 +
          ((physical_nibble >> 4) & 1) * 8,
      ((physical_nibble >> 2) & 1) * 1 +
          ((physical_nibble >> 5) & 1) * 2 +
          ((physical_nibble >> 6) & 1) * 4 +
          ((physical_nibble >> 0) & 1) * 8 +
          ((physical_nibble >> 1) & 1) * 16 +
          ((physical_nibble >> 3) & 1) * 32};
}

// Complete logical (n,k) -> serialized physical nibble map.  physical_n_word
// is the contiguous coordinate of the [K/16][2*N] u32 tensor.
constexpr std::size_t physical_nibble(int logical_n, int logical_k,
                                      int logical_n_extent) {
  int const local_n = logical_n % kNAtom;
  int const local_k = logical_k % kKAtom;
  int const local_p = atom_physical_nibble(local_n, local_k);
  int const local_word = local_p / kCodesPerWord;
  int const word_nibble = local_p % kCodesPerWord;
  int const local_n_word = local_word % kPhysicalNWordsPerAtom;
  int const local_k_row = local_word / kPhysicalNWordsPerAtom;
  int const physical_n_word =
      (logical_n / kNAtom) * kPhysicalNWordsPerAtom + local_n_word;
  int const physical_k_row =
      (logical_k / kKAtom) * kPhysicalKRowsPerAtom + local_k_row;
  std::size_t const word =
      std::size_t(physical_k_row) *
          std::size_t(kWordsPerLogicalN * logical_n_extent) +
      std::size_t(physical_n_word);
  return std::size_t(kCodesPerWord) * word +
         std::size_t(word_nibble);
}

// Inverse of physical_nibble for the same logical-N extent.  Callers validate
// the returned coordinate against their logical K extent.
constexpr LogicalCoordinate logical_coordinate(std::size_t physical_p,
                                               int logical_n_extent) {
  std::size_t const word = physical_p / kCodesPerWord;
  int const word_nibble = int(physical_p % kCodesPerWord);
  int const physical_n_words = kWordsPerLogicalN * logical_n_extent;
  int const physical_k_row = int(word / std::size_t(physical_n_words));
  int const physical_n_word = int(word % std::size_t(physical_n_words));
  int const atom_n = physical_n_word / kPhysicalNWordsPerAtom;
  int const atom_k = physical_k_row / kPhysicalKRowsPerAtom;
  int const local_n_word = physical_n_word % kPhysicalNWordsPerAtom;
  int const local_k_row = physical_k_row % kPhysicalKRowsPerAtom;
  int const local_p =
      kCodesPerWord *
          (local_k_row * kPhysicalNWordsPerAtom + local_n_word) +
      word_nibble;
  LogicalCoordinate const local = atom_logical_coordinate(local_p);
  return LogicalCoordinate{atom_n * kNAtom + local.n,
                           atom_k * kKAtom + local.k};
}

constexpr std::size_t native_code_index(int n, int k,
                                        int logical_k_extent) {
  return std::size_t(n) * std::size_t(logical_k_extent) +
         std::size_t(k);
}

inline std::uint8_t native_get(std::uint8_t const* native, int n, int k,
                               int logical_k_extent) {
  std::size_t const i = native_code_index(n, k, logical_k_extent);
  return std::uint8_t((native[i >> 1] >> (4 * (i & 1))) & 0xfu);
}

inline void native_put(std::uint8_t* native, int n, int k,
                       int logical_k_extent, std::uint8_t code) {
  std::size_t const i = native_code_index(n, k, logical_k_extent);
  unsigned const shift = unsigned(4 * (i & 1));
  std::uint8_t const mask = std::uint8_t(0xfu << shift);
  native[i >> 1] = std::uint8_t(
      (native[i >> 1] & std::uint8_t(~mask)) |
      ((code & 0xfu) << shift));
}

inline std::uint8_t placed_get(std::uint8_t const* placed, int n, int k,
                               int logical_n_extent) {
  std::size_t const p = physical_nibble(n, k, logical_n_extent);
  return std::uint8_t((placed[p >> 1] >> (4 * (p & 1))) & 0xfu);
}

inline void placed_put(std::uint8_t* placed, int n, int k,
                       int logical_n_extent, std::uint8_t code) {
  std::size_t const p = physical_nibble(n, k, logical_n_extent);
  unsigned const shift = unsigned(4 * (p & 1));
  std::uint8_t const mask = std::uint8_t(0xfu << shift);
  placed[p >> 1] = std::uint8_t(
      (placed[p >> 1] & std::uint8_t(~mask)) |
      ((code & 0xfu) << shift));
}

// Returns zero on success.  Error 20 is the established malformed-argument
// code and 24 identifies a layout-specific unsupported shape.
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

constexpr std::uint64_t atom_mapping_fingerprint() {
  std::uint64_t h = UINT64_C(1469598103934665603);
  for (int n = 0; n < kNAtom; ++n) {
    for (int k = 0; k < kKAtom; ++k) {
      std::uint16_t const value =
          std::uint16_t(atom_physical_nibble(n, k));
      h ^= std::uint8_t(value & 0xffu);
      h *= UINT64_C(1099511628211);
      h ^= std::uint8_t(value >> 8);
      h *= UINT64_C(1099511628211);
    }
  }
  return h;
}

constexpr bool atom_formula_is_bijection() {
  bool seen[kNAtom * kKAtom] = {};
  for (int n = 0; n < kNAtom; ++n) {
    for (int k = 0; k < kKAtom; ++k) {
      int const p = atom_physical_nibble(n, k);
      if (p < 0 || p >= kNAtom * kKAtom || seen[p]) return false;
      seen[p] = true;
      LogicalCoordinate const inverse = atom_logical_coordinate(p);
      if (inverse.n != n || inverse.k != k) return false;
    }
  }
  return true;
}

static_assert(kPhysicalKRowsPerAtom == 4 &&
              kPhysicalNWordsPerAtom == 32);
static_assert(kNAtom * kKAtom * kBits / 8 == 512);
static_assert(atom_formula_is_bijection());
static_assert(atom_mapping_fingerprint() == kAtomMappingFingerprint);
static_assert(atom_physical_nibble(0, 0) == 0 &&
              atom_physical_nibble(15, 63) == 1023);
static_assert(physical_nibble(0, 0, 32) == 0);

}  // namespace q4_n16k64_direct
