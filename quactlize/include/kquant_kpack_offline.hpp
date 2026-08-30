#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Converter-native physical byte map shared by the non-Q4 GGUF k-quant code
// planes.  Every logical plane is transported as an opaque little-endian b16
// tensor [K/Pack,N], where Pack = 16/Bits.  One word gathers the converter's
// K8-spaced cohort {r, r+8, ...}; eight adjacent words therefore cover one
// logical K=(8*Pack) converter delivery.  This is intentionally independent
// of the metadata group boundary: scale/zero are selected after conversion,
// per logical MMA atom, so Q2/Q3/Q6 words may span more than one group.
//
// For example Q3_K is two independent planes:
//   low2  -> Pack8  -> [K/8,N] b16
//   high1 -> Pack16 -> [K/16,N] b16
// The collective recasts the delivered b16 registers back to uint2/uint1 and
// reuses the existing two-plane converter.  The map is tactic-independent.
namespace kquant_kpack {

inline constexpr std::uint32_t kLayoutId = 2;
inline constexpr std::uint64_t kMappingId = UINT64_C(0x514b504b54000001);
inline constexpr int kWordBits = 16;
inline constexpr int kReaderPhysicalK = 16;
inline constexpr int kTransportN = 16;

constexpr int pack_for_bits(int bits) {
  return bits == 1 || bits == 2 || bits == 4 ? kWordBits / bits : 0;
}

constexpr int transport_k(int low_bits, int high_bits) {
  int const low = pack_for_bits(low_bits);
  int const high = high_bits ? pack_for_bits(high_bits) : 0;
  int const pack = low > high ? low : high;
  return pack ? kReaderPhysicalK * pack : 0;
}

template <int Bits, int GroupK>
struct PlaneMap {
  static_assert(Bits == 1 || Bits == 2 || Bits == 4,
                "K-pack b16 supports one-, two- and four-bit code planes");
  static constexpr int kBits = Bits;
  static constexpr int kPack = kWordBits / Bits;
  static_assert(GroupK > 0,
                "the format metadata group remains part of the descriptor");
  static constexpr int kGroupK = GroupK;
  static constexpr int kWordsPerDelivery = 8;
  static constexpr int kLogicalKPerDelivery = kWordsPerDelivery * kPack;
  static constexpr int kTransportK = kReaderPhysicalK * kPack;

  static constexpr bool shape_supported(int n, int k) {
    return n > 0 && k > 0 && k % kLogicalKPerDelivery == 0;
  }

  static constexpr std::size_t placed_bytes(int n, int k) {
    return shape_supported(n, k)
        ? std::size_t(n) * std::size_t(k) * Bits / 8
        : std::size_t(0);
  }

  static constexpr int physical_kgroup(int k) {
    return (k / kLogicalKPerDelivery) * kWordsPerDelivery +
           (k % kWordsPerDelivery);
  }

  static constexpr int word_slot(int k) {
    return (k % kLogicalKPerDelivery) / kWordsPerDelivery;
  }

  static constexpr int logical_k(int physical_kg, int slot) {
    return (physical_kg / kWordsPerDelivery) * kLogicalKPerDelivery +
           (physical_kg % kWordsPerDelivery) + slot * kWordsPerDelivery;
  }

  static constexpr int logical_n(int physical_n, int, int) {
    return physical_n;
  }

  static constexpr std::size_t word_index(int logical_n, int logical_k,
                                          int n_extent) {
    return std::size_t(physical_kgroup(logical_k)) * std::size_t(n_extent) +
           std::size_t(logical_n);
  }

  static constexpr std::size_t native_bit_index(int logical_n, int logical_k,
                                                 int k_extent) {
    return (std::size_t(logical_n) * std::size_t(k_extent) +
            std::size_t(logical_k)) * Bits;
  }

  static std::uint8_t native_get(std::uint8_t const* native, int logical_n,
                                 int logical_k, int k_extent) {
    std::size_t const bit = native_bit_index(logical_n, logical_k, k_extent);
    return std::uint8_t((native[bit >> 3] >> (bit & 7)) & ((1u << Bits) - 1));
  }

  static void native_put(std::uint8_t* native, int logical_n, int logical_k,
                         int k_extent, std::uint8_t code) {
    std::size_t const bit = native_bit_index(logical_n, logical_k, k_extent);
    std::uint8_t const shift = std::uint8_t(bit & 7);
    std::uint8_t const mask = std::uint8_t(((1u << Bits) - 1) << shift);
    native[bit >> 3] = std::uint8_t(
        (native[bit >> 3] & std::uint8_t(~mask)) |
        ((code << shift) & mask));
  }

  static std::uint16_t placed_word(std::uint8_t const* placed,
                                   std::size_t index) {
    std::size_t const byte = 2 * index;
    return std::uint16_t(placed[byte]) |
           (std::uint16_t(placed[byte + 1]) << 8);
  }

  static void placed_word_put(std::uint8_t* placed, std::size_t index,
                              std::uint16_t word) {
    std::size_t const byte = 2 * index;
    placed[byte] = std::uint8_t(word & 0xffu);
    placed[byte + 1] = std::uint8_t(word >> 8);
  }

  static std::uint8_t placed_get(std::uint8_t const* placed, int logical_n,
                                 int logical_k, int n_extent) {
    std::uint16_t const word = placed_word(
        placed, word_index(logical_n, logical_k, n_extent));
    return std::uint8_t((word >> (Bits * word_slot(logical_k))) &
                        ((1u << Bits) - 1));
  }

  static void placed_put(std::uint8_t* placed, int logical_n, int logical_k,
                         int n_extent, std::uint8_t code) {
    std::size_t const index = word_index(logical_n, logical_k, n_extent);
    std::uint16_t word = placed_word(placed, index);
    int const shift = Bits * word_slot(logical_k);
    std::uint16_t const mask = std::uint16_t(((1u << Bits) - 1) << shift);
    word = std::uint16_t((word & ~mask) |
                         ((std::uint16_t(code) << shift) & mask));
    placed_word_put(placed, index, word);
  }

  static int prepare(std::uint8_t const* native, std::uint8_t* placed,
                     int n, int k) {
    if (!native || !placed || n <= 0 || k <= 0) return 20;
    if (!shape_supported(n, k)) return 24;
    std::fill(placed, placed + placed_bytes(n, k), std::uint8_t(0));
    for (int col = 0; col < n; ++col)
      for (int kk = 0; kk < k; ++kk)
        placed_put(placed, col, kk, n, native_get(native, col, kk, k));
    return 0;
  }

  static int recover(std::uint8_t const* placed, std::uint8_t* native,
                     int n, int k) {
    if (!placed || !native || n <= 0 || k <= 0) return 20;
    if (!shape_supported(n, k)) return 24;
    std::fill(native, native + placed_bytes(n, k), std::uint8_t(0));
    for (int col = 0; col < n; ++col)
      for (int kk = 0; kk < k; ++kk)
        native_put(native, col, kk, k, placed_get(placed, col, kk, n));
    return 0;
  }
};

// Q5's 4+1 converter consumes one 1-bit delivery through four low-plane
// k-blocks.  The m16 transposed loader places those four high vregs in a
// fixed (N8,K64,K128) topology; unlike the 2:1 Q3/Q6 plane ratios, that
// topology is not the standalone int1 MixGemmEmit order.  Encoding the
// following bit transpose offline preserves a zero-instruction hot path:
//
//   physical n bit 3  <- logical k bit 7
//   physical kg bit 3 <- logical k bit 6
//   physical slot bit 3 <- logical n bit 3
//
// The lower three n/kg/slot bits retain logical n, k%8 and (k/8)%8.
// l236 composes this map with the exact loader, HiPlaneSrc and
// MixGemm2Plane<4,1> for every admitted TN/WN geometry.
struct Q5HighPlaneMap {
  static constexpr int kBits = 1;
  static constexpr int kPack = 16;
  static constexpr int kGroupK = 32;
  static constexpr int kTransportK = kReaderPhysicalK * kPack;

  static constexpr bool shape_supported(int n, int k) {
    return n > 0 && n % 16 == 0 && k > 0 && k % 256 == 0;
  }

  static constexpr std::size_t placed_bytes(int n, int k) {
    return shape_supported(n, k)
        ? std::size_t(n) * std::size_t(k) / 8
        : std::size_t(0);
  }

  static constexpr int physical_n(int logical_n, int logical_k) {
    return (logical_n & ~15) | (logical_n & 7) |
           (((logical_k >> 7) & 1) << 3);
  }

  static constexpr int physical_kgroup(int logical_k) {
    return (logical_k / 256) * 16 |
           (((logical_k >> 6) & 1) << 3) | (logical_k & 7);
  }

  static constexpr int word_slot(int logical_n, int logical_k) {
    return (((logical_n >> 3) & 1) << 3) |
           ((logical_k >> 3) & 7);
  }

  static constexpr int logical_n(int physical_n, int, int slot) {
    return (physical_n & ~15) | (physical_n & 7) |
           (((slot >> 3) & 1) << 3);
  }

  static constexpr int logical_k(int physical_n, int physical_kg,
                                 int slot) {
    return (physical_kg / 16) * 256 |
           (((physical_n >> 3) & 1) << 7) |
           (((physical_kg >> 3) & 1) << 6) |
           ((slot & 7) << 3) | (physical_kg & 7);
  }

  static constexpr std::size_t word_index(int logical_n, int logical_k,
                                          int n_extent) {
    return std::size_t(physical_kgroup(logical_k)) * n_extent +
           std::size_t(physical_n(logical_n, logical_k));
  }

  static std::uint8_t placed_get(std::uint8_t const* placed, int logical_n,
                                 int logical_k, int n_extent) {
    std::uint16_t const word = PlaneMap<1, 32>::placed_word(
        placed, word_index(logical_n, logical_k, n_extent));
    return std::uint8_t(
        (word >> word_slot(logical_n, logical_k)) & 1u);
  }

  static void placed_put(std::uint8_t* placed, int logical_n, int logical_k,
                         int n_extent, std::uint8_t code) {
    std::size_t const index = word_index(logical_n, logical_k, n_extent);
    std::uint16_t word = PlaneMap<1, 32>::placed_word(placed, index);
    int const shift = word_slot(logical_n, logical_k);
    std::uint16_t const mask = std::uint16_t(1u << shift);
    word = std::uint16_t((word & ~mask) |
                         ((std::uint16_t(code) << shift) & mask));
    PlaneMap<1, 32>::placed_word_put(placed, index, word);
  }

  static int prepare(std::uint8_t const* native, std::uint8_t* placed,
                     int n, int k) {
    if (!native || !placed || n <= 0 || k <= 0) return 20;
    if (!shape_supported(n, k)) return 24;
    std::fill(placed, placed + placed_bytes(n, k), std::uint8_t(0));
    for (int col = 0; col < n; ++col)
      for (int kk = 0; kk < k; ++kk)
        placed_put(placed, col, kk, n,
                   PlaneMap<1, 32>::native_get(native, col, kk, k));
    return 0;
  }

  static int recover(std::uint8_t const* placed, std::uint8_t* native,
                     int n, int k) {
    if (!placed || !native || n <= 0 || k <= 0) return 20;
    if (!shape_supported(n, k)) return 24;
    std::fill(native, native + placed_bytes(n, k), std::uint8_t(0));
    for (int col = 0; col < n; ++col)
      for (int kk = 0; kk < k; ++kk)
        PlaneMap<1, 32>::native_put(
            native, col, kk, k, placed_get(placed, col, kk, n));
    return 0;
  }
};

template <int LowBits, int HighBits, int GroupK>
struct HighPlaneMap : PlaneMap<HighBits, GroupK> {};

template <>
struct HighPlaneMap<4, 1, 32> : Q5HighPlaneMap {};

template <int LowBits, int HighBits, int GroupK, bool Recover>
int transform(std::uint8_t const* low_in, std::uint8_t const* high_in,
              std::uint8_t* low_out, std::uint8_t* high_out, int n, int k) {
  if (!low_in || !low_out) return 20;
  int rc = 0;
  if constexpr (Recover)
    rc = PlaneMap<LowBits, GroupK>::recover(low_in, low_out, n, k);
  else
    rc = PlaneMap<LowBits, GroupK>::prepare(low_in, low_out, n, k);
  if (rc) return rc;
  if constexpr (HighBits != 0) {
    if (!high_in || !high_out) return 21;
    if constexpr (Recover)
      return HighPlaneMap<LowBits, HighBits, GroupK>::recover(
          high_in, high_out, n, k);
    else
      return HighPlaneMap<LowBits, HighBits, GroupK>::prepare(
          high_in, high_out, n, k);
  } else {
    return high_in || high_out ? 21 : 0;
  }
}

static_assert(PlaneMap<2, 16>::kPack == 8 &&
              PlaneMap<2, 16>::logical_k(0, 7) == 56);
static_assert(PlaneMap<1, 16>::kPack == 16 &&
              PlaneMap<1, 16>::logical_k(0, 15) == 120);
static_assert(PlaneMap<4, 32>::kPack == 4 &&
              PlaneMap<4, 32>::logical_k(7, 3) == 31);
static_assert(transport_k(2, 0) == 128 && transport_k(2, 1) == 256 &&
              transport_k(4, 1) == 256 && transport_k(4, 2) == 128);
static_assert(Q5HighPlaneMap::physical_n(8, 0) == 0 &&
              Q5HighPlaneMap::word_slot(8, 0) == 8 &&
              Q5HighPlaneMap::physical_n(0, 128) == 8 &&
              Q5HighPlaneMap::logical_n(8, 0, 0) == 0 &&
              Q5HighPlaneMap::logical_k(8, 0, 0) == 128);

}  // namespace kquant_kpack
