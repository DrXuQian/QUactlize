// L243 -- explicit layout-3 descriptor and host prepare/recover closure.
//
// The expected bytes are built from the committed atom bit permutation here,
// without calling q4_n16k64_direct::{prepare,physical_nibble}.  The exported
// C ABI comes from ppu_dense_layout.cu, which this gate links separately.

#include "ppu_placed_arrangement.hpp"
#include "quactlize_ppu_config.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

std::uint8_t get_nibble(std::vector<std::uint8_t> const& bytes,
                        std::size_t index) {
  return std::uint8_t((bytes[index >> 1] >> (4 * (index & 1))) & 0xfu);
}

void put_nibble(std::vector<std::uint8_t>& bytes, std::size_t index,
                std::uint8_t value) {
  unsigned const shift = unsigned(4 * (index & 1));
  std::uint8_t const mask = std::uint8_t(0xfu << shift);
  bytes[index >> 1] = std::uint8_t(
      (bytes[index >> 1] & std::uint8_t(~mask)) |
      ((value & 0xfu) << shift));
}

// Independent transcription of p[0..9] =
// {k3,k4,k0,k5,n3,k1,k2,n0,n1,n2}, plus the outer atom tiling.
std::size_t expected_physical_nibble(int n, int k, int n_extent) {
  int const ln = n & 15;
  int const lk = k & 63;
  int const local =
      ((lk >> 3) & 1) * 1 + ((lk >> 4) & 1) * 2 +
      ((lk >> 0) & 1) * 4 + ((lk >> 5) & 1) * 8 +
      ((ln >> 3) & 1) * 16 + ((lk >> 1) & 1) * 32 +
      ((lk >> 2) & 1) * 64 + ((ln >> 0) & 1) * 128 +
      ((ln >> 1) & 1) * 256 + ((ln >> 2) & 1) * 512;
  int const local_word = local / 8;
  int const physical_n_word = (n / 16) * 32 + local_word % 32;
  int const physical_k_row = (k / 64) * 4 + local_word / 32;
  std::size_t const word =
      std::size_t(physical_k_row) * std::size_t(2 * n_extent) +
      std::size_t(physical_n_word);
  return 8 * word + std::size_t(local & 7);
}

}  // namespace

int main() {
  constexpr int N = 32;
  constexpr int K = 128;
  constexpr std::size_t Bytes = std::size_t(N) * K / 2;
  int bad = 0;

  auto const exact = ppu_arrangements::q4_n16k64_direct_v1();
  bad += exact.version != QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2;
  bad += exact.layout != QUACTLIZE_PPU_LAYOUT_Q4_N16K64_DIRECT_V1;
  bad += exact.bits != 4 || exact.high_bits != 0 ||
         exact.artifact_tile_k != 0 || exact.transport_tile_k != 64 ||
         exact.group_size != 32 || exact.reserved != 0;
  bad += exact.mapping_id != QUACTLIZE_PPU_Q4_N16K64_DIRECT_MAPPING_ID;
  bad += !ppu_arrangements::q4_n16k64_direct_descriptor_compatible(
      &exact, 12, K, 64);
  bad += !ppu_arrangements::q4_n16k64_direct_descriptor_compatible(
      &exact, 12, K, 256);
  bad += ppu_arrangements::matches_compiled_tactic(&exact, 12, K, 256);
  bad += ppu_arrangements::packed_tensor_reader_supported(
      &exact, 12, K, 256);

  std::vector<std::uint8_t> native(Bytes, 0);
  std::vector<std::uint8_t> expected(Bytes, 0);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      std::uint8_t const code = std::uint8_t(
          (n * 11 + k * 5 + (n ^ (k >> 2)) * 3) & 0xf);
      put_nibble(native, std::size_t(n) * K + k, code);
      put_nibble(expected, expected_physical_nibble(n, k, N), code);
    }
  }

  std::vector<std::uint8_t> placed(Bytes, 0xcd);
  std::vector<std::uint8_t> recovered(Bytes, 0xab);
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 12, &exact) != 0;
  bad += placed != expected;
  bad += quactlize_ppu_recover_dense_for_arrangement_v2(
      placed.data(), nullptr, recovered.data(), nullptr,
      N, K, 12, &exact) != 0;
  bad += recovered != native;

  // Fail-closed controls: descriptor identity, plane topology and atom shape
  // are all semantic ABI fields, not reader knobs.
  auto wrong = exact;
  wrong.mapping_id ^= 1;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 12, &wrong) != 25;
  bad += ppu_arrangements::q4_n16k64_direct_descriptor_compatible(
      &wrong, 12, K, 256);

  wrong = exact;
  wrong.transport_tile_k = 128;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 12, &wrong) != 25;
  wrong = exact;
  wrong.group_size = 64;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 12, &wrong) != 25;
  wrong = exact;
  wrong.artifact_tile_k = 64;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 12, &wrong) != 25;
  wrong = exact;
  wrong.reserved = 1;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 12, &wrong) != 25;

  std::uint8_t high = 1;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), &high, placed.data(), nullptr,
      N, K, 12, &exact) != 25;
  bad += quactlize_ppu_recover_dense_for_arrangement_v2(
      placed.data(), nullptr, recovered.data(), &high,
      N, K, 12, &exact) != 25;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      24, K, 12, &exact) != 24;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, 96, 12, &exact) != 24;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 13, &exact) != 25;

  std::printf(
      "L243 Q4_N16K64_LAYOUT3_ABI %s N=%d K=%d bytes=%zu "
      "layout=%d mapping=0x%016llx coverage=prepare+recover reds=11\n",
      bad ? "FAIL" : "PASS", N, K, Bytes, exact.layout,
      static_cast<unsigned long long>(exact.mapping_id));
  return bad ? 1 : 0;
}
