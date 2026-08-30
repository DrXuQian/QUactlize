// L235 -- host ABI closure for the canonical Q2/Q3/Q5/Q6 per-plane K-pack
// checkpoint layout.
//
// The expected byte stream below is deliberately derived without calling
// kquant_kpack::{PlaneMap,prepare,placed_get}.  It therefore catches an
// equally-wrong producer/recover pair, and links the public arrangement-v2 C
// ABI rather than proving only a private helper.

#include "quactlize_ppu_config.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

std::uint8_t native_get(std::vector<std::uint8_t> const& bytes,
                        std::size_t logical, int bits) {
  std::size_t const bit = logical * std::size_t(bits);
  return std::uint8_t((bytes[bit >> 3] >> (bit & 7)) & ((1u << bits) - 1));
}

void native_put(std::vector<std::uint8_t>& bytes, std::size_t logical,
                int bits, std::uint8_t code) {
  std::size_t const bit = logical * std::size_t(bits);
  unsigned const shift = unsigned(bit & 7);
  std::uint8_t const mask = std::uint8_t(((1u << bits) - 1) << shift);
  bytes[bit >> 3] = std::uint8_t(
      (bytes[bit >> 3] & std::uint8_t(~mask)) |
      ((code << shift) & mask));
}

void expected_plane(std::vector<std::uint8_t> const& native,
                    std::vector<std::uint8_t>& expected,
                    int n, int k, int bits, int group) {
  int const pack = 16 / bits;
  int const logical_k_per_delivery = 8 * pack;
  expected.assign(native.size(), std::uint8_t(0));
  for (int col = 0; col < n; ++col) {
    for (int kk = 0; kk < k; ++kk) {
      int const physical_kg =
          (kk / logical_k_per_delivery) * 8 + kk % 8;
      int const slot = (kk % logical_k_per_delivery) / 8;
      std::size_t const word = std::size_t(physical_kg) * n + col;
      std::uint16_t current = std::uint16_t(expected[2 * word]) |
                              (std::uint16_t(expected[2 * word + 1]) << 8);
      int const shift = bits * slot;
      std::uint16_t const mask =
          std::uint16_t(((1u << bits) - 1) << shift);
      current = std::uint16_t(
          (current & ~mask) |
          ((std::uint16_t(native_get(
                native, std::size_t(col) * k + kk, bits)) << shift) & mask));
      expected[2 * word] = std::uint8_t(current & 0xffu);
      expected[2 * word + 1] = std::uint8_t(current >> 8);
    }
  }
}

void expected_q5_high(std::vector<std::uint8_t> const& native,
                      std::vector<std::uint8_t>& expected,
                      int n, int k) {
  expected.assign(native.size(), std::uint8_t(0));
  for (int col = 0; col < n; ++col) {
    for (int kk = 0; kk < k; ++kk) {
      int const physical_n =
          (col & ~15) | (col & 7) | (((kk >> 7) & 1) << 3);
      int const physical_kg =
          (kk / 256) * 16 | (((kk >> 6) & 1) << 3) | (kk & 7);
      int const slot = (((col >> 3) & 1) << 3) | ((kk >> 3) & 7);
      std::size_t const word = std::size_t(physical_kg) * n + physical_n;
      std::uint16_t current = std::uint16_t(expected[2 * word]) |
                              (std::uint16_t(expected[2 * word + 1]) << 8);
      current = std::uint16_t(
          (current & ~(std::uint16_t(1) << slot)) |
          (std::uint16_t(native_get(
               native, std::size_t(col) * k + kk, 1)) << slot));
      expected[2 * word] = std::uint8_t(current & 0xffu);
      expected[2 * word + 1] = std::uint8_t(current >> 8);
    }
  }
}

template <int QType, int LowBits, int HighBits, int Group, int TransportK>
int prove_format() {
  constexpr int N = 32;
  constexpr int K = 256;
  std::size_t const low_bytes = std::size_t(N) * K * LowBits / 8;
  std::size_t const high_bytes = std::size_t(N) * K * HighBits / 8;
  std::vector<std::uint8_t> low_native(low_bytes, 0);
  std::vector<std::uint8_t> high_native(high_bytes, 0);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      native_put(low_native, std::size_t(n) * K + k, LowBits,
                 std::uint8_t((13 * n + 7 * k + k / Group + 1) &
                              ((1 << LowBits) - 1)));
      if constexpr (HighBits != 0) {
        native_put(high_native, std::size_t(n) * K + k, HighBits,
                   std::uint8_t((5 * n + 11 * k + k / Group + 1) &
                                ((1 << HighBits) - 1)));
      }
    }
  }

  std::vector<std::uint8_t> low_expected, high_expected;
  expected_plane(low_native, low_expected, N, K, LowBits, Group);
  if constexpr (HighBits != 0) {
    if constexpr (LowBits == 4 && HighBits == 1 && Group == 32)
      expected_q5_high(high_native, high_expected, N, K);
    else
      expected_plane(high_native, high_expected, N, K, HighBits, Group);
  }

  quactlize_ppu_placed_arrangement_v2 const exact{
      QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2,
      QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1,
      LowBits, HighBits, 0, TransportK, Group, 0,
      QUACTLIZE_PPU_KQUANT_KPACK_MAPPING_ID};
  std::vector<std::uint8_t> low_placed(low_bytes, 0xcd);
  std::vector<std::uint8_t> high_placed(high_bytes, 0xcd);
  std::vector<std::uint8_t> low_recovered(low_bytes, 0xab);
  std::vector<std::uint8_t> high_recovered(high_bytes, 0xab);
  auto const* high_in = HighBits ? high_native.data() : nullptr;
  auto* high_out = HighBits ? high_placed.data() : nullptr;
  auto* high_back = HighBits ? high_recovered.data() : nullptr;

  int bad = 0;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType, &exact) != 0;
  bad += low_placed != low_expected;
  if constexpr (HighBits != 0) bad += high_placed != high_expected;
  bad += quactlize_ppu_recover_dense_for_arrangement_v2(
      low_placed.data(), HighBits ? high_placed.data() : nullptr,
      low_recovered.data(), high_back, N, K, QType, &exact) != 0;
  bad += low_recovered != low_native;
  if constexpr (HighBits != 0) bad += high_recovered != high_native;

  // Every identity field, qtype, and plane topology is fail-closed.
  auto wrong = exact;
  wrong.mapping_id ^= 1;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType, &wrong) != 25;
  wrong = exact; wrong.transport_tile_k /= 2;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType, &wrong) != 25;
  wrong = exact; wrong.group_size *= 2;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType, &wrong) != 25;
  wrong = exact; wrong.artifact_tile_k = 64;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType, &wrong) != 25;
  wrong = exact; wrong.version = QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType, &wrong) != 25;
  wrong = exact; wrong.layout = QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType, &wrong) != 25;
  wrong = exact; ++wrong.bits;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType, &wrong) != 25;
  wrong = exact; ++wrong.high_bits;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType, &wrong) != 25;
  wrong = exact; wrong.reserved = 1;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType, &wrong) != 25;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      low_native.data(), high_in, low_placed.data(), high_out,
      N, K, QType == 10 ? 11 : 10, &exact) != 25;
  if constexpr (HighBits == 0) {
    std::uint8_t high = 0;
    bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
        low_native.data(), &high, low_placed.data(), nullptr,
        N, K, QType, &exact) != 21;
  } else {
    bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
        low_native.data(), nullptr, low_placed.data(), high_out,
        N, K, QType, &exact) != 21;
  }

  std::printf(
      "L235 KQUANT_KPACK qtype=%d low=%d/pack%d high=%d/pack%d "
      "group=%d transport=%d bytes=%zu+%zu result=%s\n",
      QType, LowBits, 16 / LowBits, HighBits,
      HighBits ? 16 / HighBits : 0, Group, TransportK,
      low_bytes, high_bytes, bad ? "FAIL" : "PASS");
  return bad;
}

}  // namespace

int main() {
  int bad = 0;
  bad += prove_format<10, 2, 0, 16, 128>();
  bad += prove_format<11, 2, 1, 16, 256>();
  bad += prove_format<13, 4, 1, 32, 256>();
  bad += prove_format<14, 4, 2, 16, 128>();
  std::printf("L235 KQUANT_KPACK_OFFLINE_ABI %s formats=4 reds=44\n",
              bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
