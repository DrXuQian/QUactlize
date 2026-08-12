// Exhaustive BROAD host oracle for INBOX 127's shipped TileM=8 ShortWide family.
//
// This is not a second tactic model: both the production inventory/default policy and every legality reason come
// from the headers consumed by ppu_dense_backend.cu. The 60 cells are the Cartesian product of five registered
// k-quant formats, scale-first/fully-quantized TileK, and the six shipping stages. Every cell is printed. This
// approximate gs16 model emits a finite candidate set; l148 reads production's compiled GemmKernel type and owns
// the final list_valid decision, including raw packed staging and C++ object padding.
#include <cstdio>
#include <cstring>

#include "ppu_dense_shipping_policy.hpp"
#include "ppu_format_config.hpp"
#include "ppu_tactic_space.hpp"

namespace {

using ppu_dense_shipping::Config;
using ppu_tactics::Candidate;
using ppu_tactics::Exclusion;
using ppu_tactics::Format;
using ppu_tactics::FormatSpec;

constexpr Format format_id(ppu_formats::Config const& f) {
  if (f.low_bits == 2 && f.high_bits == 1) return Format::Q3_K;
  if (f.low_bits == 4 && f.high_bits == 1) return Format::Q5_K;
  if (f.low_bits == 4 && f.high_bits == 2) return Format::Q6_K;
  if (f.low_bits == 2 && f.high_bits == 0) return Format::I2;
  return Format::I4;
}

constexpr bool is_tm8_shortwide(Config const& c) {
  return c.tile_m == 8 && c.tile_n == 128 && c.warp_m == 8 && c.warp_n == 32;
}

constexpr Exclusion exclusion(Candidate const& c, int stages) {
  Exclusion const topology = ppu_tactics::DenseSpace::topology_exclusion(c, stages);
  return topology != Exclusion::None ? topology : ppu_tactics::common_producer_exclusion(c);
}

constexpr int family_rows() {
  int count = 0;
  for (auto const& c : ppu_dense_shipping::kConfigs) count += is_tm8_shortwide(c);
  return count;
}

constexpr int count_cells(Exclusion wanted) {
  int count = 0;
  for (auto const& f : ppu_formats::kConfigs) {
    FormatSpec const spec{format_id(f), f.name, f.low_bits, f.high_bits};
    for (int mode = 0; mode < 2; ++mode) {
      int const tk = mode == 0 ? f.scale_first_tile_k : f.fully_quantized_tile_k;
      for (auto const& cfg : ppu_dense_shipping::kConfigs) {
        if (!is_tm8_shortwide(cfg)) continue;
        Candidate const c{spec, cfg.tile_m, cfg.tile_n, tk, cfg.warp_m, cfg.warp_n, tk};
        if (exclusion(c, cfg.stages) == wanted) ++count;
      }
    }
  }
  return count;
}

static_assert(family_rows() == 6, "INBOX 127 requires the complete six-stage TM8 ShortWide family");
static_assert(count_cells(Exclusion::None) == 52,
              "the broad TM8 candidate model must retain 52 format/mode/stage cells");
static_assert(count_cells(Exclusion::MinimumStageSmem) == 8,
              "the eight illegal TM8 cells must be rejected only by their physical shared-memory footprint");
static_assert(ppu_tactics::physical_a_rows(Candidate{
                  {Format::I4, "m8-physical-A", 4, 0}, 8, 128, 64, 8, 32, 64}) == 16,
              "logical TM8 must continue paying for the AIU's physical 16-row A cube");
static_assert(ppu_dense_shipping::default_config_for_m(1) ==
                  ppu_dense_shipping::ConfigId::ShortWideM8S3 &&
              ppu_dense_shipping::default_config_for_m(7) ==
                  ppu_dense_shipping::ConfigId::ShortWideM8S3 &&
              ppu_dense_shipping::default_config_for_m(8) ==
                  ppu_dense_shipping::ConfigId::Default,
              "empty-config routing must change at M=8 and nowhere else");

}  // namespace

int main() {
  int legal = 0, illegal = 0;
  for (auto const& f : ppu_formats::kConfigs) {
    FormatSpec const spec{format_id(f), f.name, f.low_bits, f.high_bits};
    for (int mode = 0; mode < 2; ++mode) {
      char const* mode_name = mode == 0 ? "scale-first" : "fully-quantized";
      int const tk = mode == 0 ? f.scale_first_tile_k : f.fully_quantized_tile_k;
      for (auto const& cfg : ppu_dense_shipping::kConfigs) {
        if (!is_tm8_shortwide(cfg)) continue;
        Candidate const c{spec, cfg.tile_m, cfg.tile_n, tk, cfg.warp_m, cfg.warp_n, tk};
        Exclusion const why = exclusion(c, cfg.stages);
        bool const ok = why == Exclusion::None;
        legal += ok;
        illegal += !ok;
        std::printf("broad-cell format=%s mode=%s tk=%d config=%s physical_a_rows=%d verdict=%s",
                    f.name, mode_name, tk, cfg.name, ppu_tactics::physical_a_rows(c),
                    ok ? "LEGAL" : "ILLEGAL");
        if (!ok) std::printf(" reason=%s", ppu_tactics::exclusion_clause(why));
        std::printf("\n");
      }
    }
  }
  std::printf("broad-summary family_rows=%d cells=%d legal=%d illegal=%d default_m1=%s default_m8=%s\n",
              family_rows(), legal + illegal, legal, illegal,
              ppu_dense_shipping::kConfigs[static_cast<int>(
                  ppu_dense_shipping::default_config_for_m(1))].name,
              ppu_dense_shipping::kConfigs[static_cast<int>(
                  ppu_dense_shipping::default_config_for_m(8))].name);
  return legal == 52 && illegal == 8 ? 0 : 1;
}
