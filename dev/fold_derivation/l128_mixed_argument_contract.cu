// L128 -- CALLER STRIDES AND LOGICAL RESIDUES MUST SURVIVE OUTER ADDRESSING.
//
// This oracle covers two silent substitutions found while enumerating the
// mixed-input ABI.  It calls the exact lightweight helpers used by all three
// shipping collectives, but anchors them to independent int64 formulae:
//
//   * a rank-3 dA owns both its row pitch and its expert/batch pitch;
//   * metadata N residues are measured in logical TileN columns, even when B
//     is physically folded to TileN/F columns.
//
// Unique source offsets are the A anchor.  The negative arms replay the old
// compact `row*K` base and physical-B residue formula and must turn red.

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_argument_contract.hpp"

namespace arg = cutlass::gemm::collective::detail;
using namespace cute;

using AStride = Stride<int64_t, _1, int64_t>;

int main() {
  constexpr int kM = 7;
  constexpr int kK = 256;
  constexpr int kL = 5;
  constexpr int64_t kRowPitch = kK + 16;
  constexpr int64_t kExpertPitch = kM * kRowPitch + 32;
  AStride const dA = make_stride(
      int64_t(kRowPitch), _1{}, int64_t(kExpertPitch));
  std::vector<uint32_t> source(
      static_cast<std::size_t>(kL * kExpertPitch + 128), uint32_t{0});
  for (std::size_t i = 0; i < source.size(); ++i) {
    source[i] = uint32_t(i) * 2654435761u ^ 0x8d31a7c5u;
  }

  int uniform_bad = 0;
  int uniform_legacy_red = 0;
  for (int e = 0; e < kL; ++e) {
    auto const* got = arg::mixed_a_expert_base(
        source.data(), dA, nullptr, e);
    int64_t const expected = int64_t(e) * kExpertPitch;
    int64_t const legacy = int64_t(e) * kM * kK;
    uniform_bad += got - source.data() != expected;
    uniform_bad += *got != source[static_cast<std::size_t>(expected)];
    uniform_legacy_red += legacy != expected;
  }

  std::array<int, kL> const rows{0, 1, 4, 4, 9};
  int ragged_bad = 0;
  int ragged_legacy_red = 0;
  for (int e = 0; e < kL; ++e) {
    auto const* got = arg::mixed_a_expert_base(
        source.data(), dA, rows.data(), e);
    int64_t const expected = int64_t(rows[e]) * kRowPitch;
    int64_t const legacy = int64_t(rows[e]) * kK;
    ragged_bad += got - source.data() != expected;
    ragged_bad += *got != source[static_cast<std::size_t>(expected)];
    ragged_legacy_red += legacy != expected;
  }

  constexpr int kTileN = 64;
  int residue_bad = 0;
  int physical_formula_red = 0;
  int residue_cases = 0;
  for (int fold : {1, 2, 4}) {
    int const physical_tile_n = kTileN / fold;
    for (int N = 1; N <= 2 * kTileN + 1; ++N) {
      int const ntiles = (N + kTileN - 1) / kTileN;
      for (int n = 0; n < ntiles; ++n) {
        int64_t const expected = int64_t(N) - int64_t(n) * kTileN;
        int64_t const got = arg::mixed_logical_n_residue(N, kTileN, n);
        int64_t const legacy = int64_t(N) - int64_t(n) * physical_tile_n;
        residue_bad += got != expected;
        physical_formula_red += legacy != expected;
        ++residue_cases;
      }
    }
  }

  bool const positive = uniform_bad == 0 && ragged_bad == 0 && residue_bad == 0;
  bool const negatives = uniform_legacy_red == kL - 1 &&
                         ragged_legacy_red == kL - 1 &&
                         physical_formula_red > 0;
  std::printf(
      "L128 A uniform_bad=%d ragged_bad=%d explicit-int64-stride-anchor=%s\n",
      uniform_bad, ragged_bad, positive ? "PASS" : "FAIL");
  std::printf(
      "L128 A old-row-times-K uniform_red=%d/%d ragged_red=%d/%d -> %s\n",
      uniform_legacy_red, kL, ragged_legacy_red, kL,
      negatives ? "EXPECTED-RED" : "FAIL");
  std::printf(
      "L128 N-residue cases=%d bad=%d physical-TileN-over-F-red=%d -> %s\n",
      residue_cases, residue_bad, physical_formula_red,
      positive && physical_formula_red > 0 ? "PASS/EXPECTED-RED" : "FAIL");
  std::printf(
      "L128 result=%s scope=dA-outer-base+logical-N-residue\n",
      positive && negatives ? "PASS" : "FAIL");
  return positive && negatives ? 0 : 1;
}
