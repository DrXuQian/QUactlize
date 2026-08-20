// Emit the complete ScaleFirst dense tactic superset for one resident layout.
//
// This deliberately does not reuse kFormats' historical single-plane aliases:
// qtype is a semantic identity, not a bit width.  In particular Q4_K/Q5_K
// own gs32 metadata while Q3_K/Q6_K have no zero plane.  Those facts change
// shared-memory admission and therefore the denominator.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ppu_tactic_space.hpp"

namespace {

using namespace ppu_tactics;

struct ScaleFirstFormat {
  int qtype;
  char const* name;
  Format format;
  int low_bits;
  int high_bits;
  int group_size;
  int metadata_planes;
  char const* quant_mode;
};

constexpr ScaleFirstFormat kFormats[] = {
    {8,  "Q8_0", Format::I8,   8, 0, 32, 1, "ScaleOnly"},
    {10, "Q2_K", Format::I2,   2, 0, 16, 2, "ScaleZero"},
    {11, "Q3_K", Format::Q3_K, 2, 1, 16, 1, "ScaleOnly"},
    {12, "Q4_K", Format::I4,   4, 0, 32, 2, "ScaleZero"},
    {13, "Q5_K", Format::Q5_K, 4, 1, 32, 2, "ScaleZero"},
    {14, "Q6_K", Format::Q6_K, 4, 2, 16, 1, "ScaleOnly"},
};

ScaleFirstFormat const* format_for_qtype(int qtype) {
  for (auto const& format : kFormats)
    if (format.qtype == qtype) return &format;
  return nullptr;
}

char const* exclusion_name(Exclusion exclusion) {
  switch (exclusion) {
    case Exclusion::None: return "NONE";
    case Exclusion::AtomAlignment: return "ATOM_ALIGNMENT";
    case Exclusion::WarpDoesNotDivideTile: return "WARP_DOES_NOT_DIVIDE_TILE";
    case Exclusion::TooManyWarps: return "TOO_MANY_WARPS";
    case Exclusion::AccumulatorRegisters: return "ACCUMULATOR_REGISTERS";
    case Exclusion::ArtifactTileKDoesNotTileTacticK:
      return "ARTIFACT_TILEK_DOES_NOT_TILE_TACTIC_K";
    case Exclusion::ArtifactLowRun: return "ARTIFACT_LOW_RUN";
    case Exclusion::ArtifactHighRun: return "ARTIFACT_HIGH_RUN";
    case Exclusion::LowFoldDoesNotDivideTileN:
      return "LOW_FOLD_DOES_NOT_DIVIDE_TILE_N";
    case Exclusion::HighFoldDoesNotDivideTileN:
      return "HIGH_FOLD_DOES_NOT_DIVIDE_TILE_N";
    case Exclusion::LowDelivery: return "LOW_DELIVERY";
    case Exclusion::HighDelivery: return "HIGH_DELIVERY";
    case Exclusion::MinimumStageSmem: return "MINIMUM_STAGE_SMEM";
    case Exclusion::ProducerWarpN: return "PRODUCER_WARP_N";
    case Exclusion::ProducerMap: return "PRODUCER_MAP";
    case Exclusion::ProducerConsumerLayout:
      return "PRODUCER_CONSUMER_LAYOUT";
    case Exclusion::BChunkUnsupportedBits: return "BCHUNK_UNSUPPORTED_BITS";
  }
  return "UNKNOWN";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
        "usage: emit_scalefirst_internal_superset "
        "<qtype:8|10..14> <artifact-tk> <tactic-tk|0=all> "
        "[--plant-q4-legacy-gs16] [stage ...]\n");
    return 2;
  }
  int const qtype = std::atoi(argv[1]);
  int const artifact_tk = std::atoi(argv[2]);
  int const requested_tactic_tk = std::atoi(argv[3]);
  auto const* format = format_for_qtype(qtype);
  if (!format || artifact_tk <= 0 || requested_tactic_tk < 0) {
    std::fprintf(stderr, "invalid qtype/artifact/tactic tuple\n");
    return 2;
  }

  bool plant_q4_legacy = false;
  int stages[32];
  int stage_count = 0;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--plant-q4-legacy-gs16")) {
      plant_q4_legacy = true;
      continue;
    }
    if (stage_count == 32) {
      std::fprintf(stderr, "too many stage values\n");
      return 2;
    }
    stages[stage_count++] = std::atoi(argv[i]);
  }
  if (stage_count == 0) {
    stages[0] = 2; stages[1] = 3; stages[2] = 4;
    stages[3] = 6; stages[4] = 8; stages[5] = 12;
    stage_count = 6;
  }
  if (plant_q4_legacy && qtype != 12) {
    std::fprintf(stderr,
        "--plant-q4-legacy-gs16 is a checker-only Q4_K negative\n");
    return 2;
  }

  int const group_size = plant_q4_legacy ? 16 : format->group_size;
  FormatSpec const spec{format->format, format->name, format->low_bits,
                        format->high_bits, group_size,
                        format->metadata_planes};
  int raw = 0, eligible = 0, rejected = 0;
  for (int tactic_tk : kTileK) {
    if (requested_tactic_tk && tactic_tk != requested_tactic_tk) continue;
    for (int tm : kTileM)
      for (int tn : kTileN)
        for (int wm : kWarpM)
          for (int wn : kWarpN)
            for (int bchunk : kBChunkModes)
              for (int si = 0; si < stage_count; ++si) {
                int const stage = stages[si];
                Candidate const candidate{
                    spec, tm, tn, tactic_tk, wm, wn, artifact_tk, bchunk};
                Exclusion exclusion = common_topology_exclusion(candidate, stage);
                if (exclusion == Exclusion::None)
                  exclusion = common_producer_exclusion(candidate);
                ++raw;
                if (exclusion == Exclusion::None) ++eligible;
                else ++rejected;
                std::printf(
                    "SF_SUPERSET_ROW q=%d format=%s mode=%s gs=%d planes=%d "
                    "A=%d fold_low=%d fold_high=%d tm=%d tn=%d tk=%d "
                    "wm=%d wn=%d stages=%d bchunk=%d status=%s reason=%s\n",
                    qtype, format->name, format->quant_mode, group_size,
                    format->metadata_planes, artifact_tk,
                    artifact_low_fold(candidate), artifact_high_fold(candidate),
                    tm, tn, tactic_tk, wm, wn, stage, bchunk,
                    exclusion == Exclusion::None ? "TYPE_ADMISSION_REQUIRED" :
                                                   "STATIC_REJECT",
                    exclusion_name(exclusion));
              }
  }
  std::printf(
      "SF_SUPERSET_SUMMARY q=%d format=%s mode=%s gs=%d planes=%d A=%d "
      "plant_q4_legacy_gs16=%d raw=%d eligible=%d rejected=%d\n",
      qtype, format->name, format->quant_mode, group_size,
      format->metadata_planes, artifact_tk, int(plant_q4_legacy), raw,
      eligible, rejected);
  return 0;
}
