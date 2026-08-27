// Host ABI closure for the canonical Q4_K K-pack4 checkpoint layout.
//
// The expected byte stream below is intentionally derived without calling
// q4_kpack4::{prepare,placed_get}.  That keeps the oracle independent from the
// implementation exported by ppu_dense_layout.cu and makes an equally wrong
// producer/consumer pair fail here.
#include "quactlize_ppu_config.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

void put_nibble(std::vector<std::uint8_t>& bytes, std::size_t code_index,
                std::uint8_t value) {
  unsigned const shift = unsigned(code_index & 1u) * 4u;
  bytes[code_index >> 1] = std::uint8_t(
      (bytes[code_index >> 1] & ~(std::uint8_t(0xfu << shift))) |
      ((value & 0xfu) << shift));
}

std::uint8_t get_nibble(std::vector<std::uint8_t> const& bytes,
                        std::size_t code_index) {
  return std::uint8_t((bytes[code_index >> 1] >>
                       (unsigned(code_index & 1u) * 4u)) & 0xfu);
}

}  // namespace

int main() {
  constexpr int N = 18;
  constexpr int K = 96;
  constexpr std::size_t Bytes = std::size_t(N) * K / 2;
  int bad = 0;

  std::vector<std::uint8_t> native(Bytes, 0);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      std::uint8_t const code = std::uint8_t(
          (n * 5 + k * 3 + (k / 8) * 7 + (n ^ k)) & 0xf);
      put_nibble(native, std::size_t(n) * K + k, code);
    }
  }

  // Independent definition of [K/4,N] little-endian b16 packing:
  // word(8*g+r,n) = q(32*g+r+{0,8,16,24},n).
  std::vector<std::uint8_t> expected(Bytes, 0);
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < K / 32; ++g) {
      for (int r = 0; r < 8; ++r) {
        std::uint16_t word = 0;
        for (int p = 0; p < 4; ++p) {
          int const k = 32 * g + r + 8 * p;
          word |= std::uint16_t(
              get_nibble(native, std::size_t(n) * K + k)) << (4 * p);
        }
        std::size_t const word_index = std::size_t(8 * g + r) * N + n;
        expected[2 * word_index] = std::uint8_t(word & 0xffu);
        expected[2 * word_index + 1] = std::uint8_t(word >> 8);
      }
    }
  }

  quactlize_ppu_placed_arrangement_v2 const exact{
      QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2,
      QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1,
      4, 0, 0, 64, 32, 0,
      QUACTLIZE_PPU_Q4_KPACK4_MAPPING_ID};
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

  // RED controls: no field may be inferred or silently defaulted.
  auto wrong = exact;
  wrong.mapping_id ^= 1;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 12, &wrong) != 25;
  wrong = exact;
  wrong.artifact_tile_k = 32;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 12, &wrong) != 25;
  wrong = exact;
  wrong.transport_tile_k = 32;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 12, &wrong) != 25;
  std::uint8_t high = 0;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), &high, placed.data(), nullptr,
      N, K, 12, &exact) != 25;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 13, &exact) != 25;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, 80, 12, &exact) != 24;
  bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
      native.data(), nullptr, placed.data(), nullptr,
      N, K, 12, nullptr) != 25;

  std::printf(
      "L230 Q4_K KPACK4 offline-abi %s N=%d K=%d bytes=%zu "
      "layout=%d mapping=0x%016llx reds=7\n",
      bad ? "FAIL" : "PASS", N, K, Bytes, exact.layout,
      static_cast<unsigned long long>(exact.mapping_id));
  return bad ? 1 : 0;
}
