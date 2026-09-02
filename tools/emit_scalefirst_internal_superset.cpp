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
    {11, "Q3_K", Format::Q3_K, 2, 1, 16, 2, "ScaleZero"},
    {12, "Q4_K", Format::I4,   4, 0, 32, 2, "ScaleZero"},
    {13, "Q5_K", Format::Q5_K, 4, 1, 32, 2, "ScaleZero"},
    {14, "Q6_K", Format::Q6_K, 4, 2, 16, 2, "ScaleZero"},
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
        "[--weight-layout=0|1|2] [--plant-q4-legacy-gs16] "
        "[--plant-q3q6-scale-only] [stage ...]\n");
    return 2;
  }
  int const qtype = std::atoi(argv[1]);
  int const artifact_tk = std::atoi(argv[2]);
  int const requested_tactic_tk = std::atoi(argv[3]);
  auto const* format = format_for_qtype(qtype);
  if (!format || artifact_tk < 0 || requested_tactic_tk < 0) {
    std::fprintf(stderr, "invalid qtype/artifact/tactic tuple\n");
    return 2;
  }

  bool plant_q4_legacy = false;
  bool plant_q3q6_scale_only = false;
  int weight_layout = 0;
  int stages[32];
  int stage_count = 0;
  for (int i = 4; i < argc; ++i) {
    if (!std::strncmp(argv[i], "--weight-layout=", 16)) {
      weight_layout = std::atoi(argv[i] + 16);
      continue;
    }
    if (!std::strcmp(argv[i], "--plant-q4-legacy-gs16")) {
      plant_q4_legacy = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--plant-q3q6-scale-only")) {
      plant_q3q6_scale_only = true;
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
  if (plant_q3q6_scale_only && qtype != 11 && qtype != 14) {
    std::fprintf(stderr,
        "--plant-q3q6-scale-only is a checker-only Q3_K/Q6_K negative\n");
    return 2;
  }
  bool const q4_kpack = weight_layout == 1;
  bool const generic_kpack = weight_layout == 2;
  bool const canonical_generic_qtype =
      qtype == 10 || qtype == 11 || qtype == 13 || qtype == 14;
  if ((weight_layout == 0 && artifact_tk <= 0) ||
      (q4_kpack && (qtype != 12 || artifact_tk != 0)) ||
      (generic_kpack && (!canonical_generic_qtype || artifact_tk != 0)) ||
      weight_layout < 0 || weight_layout > 2 ||
      (plant_q4_legacy && weight_layout != 0)) {
    std::fprintf(stderr,
        "weight layout is Xplane(0,A>0), Q4 K-pack(1,q12/A0), or "
        "generic K-pack(2,q10/q11/q13/q14/A0)\n");
    return 2;
  }
  bool const use_kpack = q4_kpack || generic_kpack;
  int const low_pack = 16 / format->low_bits;
  int const high_pack = format->high_bits ? 16 / format->high_bits : 0;
  int const kpack_transport_k = 16 * (low_pack > high_pack ? low_pack : high_pack);

  int const group_size = plant_q4_legacy ? 16 : format->group_size;
  int const metadata_planes = plant_q3q6_scale_only
      ? 1 : format->metadata_planes;
  char const* quant_mode = plant_q3q6_scale_only
      ? "ScaleOnly" : format->quant_mode;
  FormatSpec const spec{format->format, format->name, format->low_bits,
                        format->high_bits, group_size,
                        metadata_planes};
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
                Exclusion exclusion = Exclusion::None;
                char const* forced_reason = nullptr;
                if (use_kpack && bchunk != 0)
                  forced_reason = "KPACK_BCHUNK_REQUIRES_BC0";
                else if (use_kpack &&
                         (tactic_tk < kpack_transport_k ||
                          tactic_tk % kpack_transport_k))
                  forced_reason = "KPACK_TRANSPORT_DOES_NOT_TILE_TACTIC_K";
                else
                  exclusion = common_topology_exclusion(candidate, stage);
                // Xplane's offline producer owns additional WN/map gates.
                // Canonical K-pack is tactic independent and therefore must
                // not inherit those artifact-producer restrictions.
                if (!use_kpack && exclusion == Exclusion::None)
                  exclusion = common_producer_exclusion(candidate);
                ++raw;
                if (!forced_reason && exclusion == Exclusion::None) ++eligible;
                else ++rejected;
                std::printf(
                    "SF_SUPERSET_ROW q=%d format=%s mode=%s gs=%d planes=%d "
                    "A=%d weight_layout=%d fold_low=%d fold_high=%d "
                    "tm=%d tn=%d tk=%d "
                    "wm=%d wn=%d stages=%d bchunk=%d status=%s reason=%s\n",
                    qtype, format->name, quant_mode, group_size,
                    metadata_planes, artifact_tk, weight_layout,
                    artifact_low_fold(candidate), artifact_high_fold(candidate),
                    tm, tn, tactic_tk, wm, wn, stage, bchunk,
                    !forced_reason && exclusion == Exclusion::None
                        ? "TYPE_ADMISSION_REQUIRED" : "STATIC_REJECT",
                    forced_reason ? forced_reason : exclusion_name(exclusion));
              }
  }
  std::printf(
      "SF_SUPERSET_SUMMARY q=%d format=%s mode=%s gs=%d planes=%d A=%d "
      "weight_layout=%d plant_q4_legacy_gs16=%d "
      "plant_q3q6_scale_only=%d raw=%d eligible=%d rejected=%d\n",
      qtype, format->name, quant_mode, group_size,
      metadata_planes, artifact_tk, weight_layout,
      int(plant_q4_legacy), int(plant_q3q6_scale_only), raw, eligible,
      rejected);
  return 0;
}
