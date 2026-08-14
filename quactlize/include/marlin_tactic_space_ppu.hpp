// Host-readable tactic authority for the standalone Marlin PPU stack.
//
// This file intentionally does not include CUTLASS, CuTe, HGGC, or the generic
// ppu_tactic_space.hpp.  A standalone Marlin tactic has a different format,
// loader, warp-K topology, reduction, and scheduler contract.  Sharing the
// generic candidate type would make it possible for a generic pruning rule to
// silently remove a Marlin row (or vice versa).
#pragma once

#include <array>
#include <cstdint>

namespace marlin_tactics_ppu {

inline constexpr int kMarlinGroupSize = 128;
inline constexpr int64_t kMarlinBlockSmemBytes = 262144;
inline constexpr int kMarlinWarpThreads = 32;
inline constexpr int kMarlinMaxCtaWarps = 32;

enum class MarlinLoadKindPPU : uint8_t {
  CpAsync,
  Aiu,
};

constexpr char const* load_kind_name(MarlinLoadKindPPU load) {
  switch (load) {
    case MarlinLoadKindPPU::CpAsync: return "cp_async";
    case MarlinLoadKindPPU::Aiu: return "aiu";
  }
  return "unknown";
}

// These are declared axes, not a claim that every Cartesian row is currently
// implemented.  The census emits every row and the classifier supplies one
// explicit first-failure reason.  An axis becomes an active production search
// axis only after at least two of its values survive admission.
inline constexpr std::array<int, 5> kMarlinTileM{{8, 16, 32, 48, 64}};
inline constexpr std::array<int, 3> kMarlinTileN{{64, 128, 256}};
inline constexpr std::array<int, 4> kMarlinTileK{{32, 64, 128, 256}};
inline constexpr std::array<int, 5> kMarlinWarpM{{8, 16, 32, 48, 64}};
inline constexpr std::array<int, 4> kMarlinWarpN{{16, 32, 64, 128}};
inline constexpr std::array<int, 5> kMarlinWarpK{{16, 32, 64, 128, 256}};
inline constexpr std::array<int, 5> kMarlinStages{{2, 3, 4, 5, 6}};
inline constexpr std::array<MarlinLoadKindPPU, 2> kMarlinLoadKinds{{
    MarlinLoadKindPPU::CpAsync,
    MarlinLoadKindPPU::Aiu,
}};

struct MarlinTacticPPU {
  int tm;
  int tn;
  int tk;
  int wm;
  int wn;
  int warp_k;
  int stages;
  MarlinLoadKindPPU load;
};

constexpr bool operator==(MarlinTacticPPU a, MarlinTacticPPU b) {
  return a.tm == b.tm && a.tn == b.tn && a.tk == b.tk &&
         a.wm == b.wm && a.wn == b.wn && a.warp_k == b.warp_k &&
         a.stages == b.stages && a.load == b.load;
}

constexpr bool operator!=(MarlinTacticPPU a, MarlinTacticPPU b) {
  return !(a == b);
}

inline constexpr MarlinTacticPPU kMarlinClassicReferencePPU{
    16, 128, 128, 16, 64, 32, 4, MarlinLoadKindPPU::CpAsync};

enum class MarlinExclusionKindPPU : uint8_t {
  Admitted,
  HardwareOrIsa,
  ResourceLimit,
  CurrentImplementation,
};

constexpr char const* exclusion_kind_name(MarlinExclusionKindPPU kind) {
  switch (kind) {
    case MarlinExclusionKindPPU::Admitted: return "ADMITTED";
    case MarlinExclusionKindPPU::HardwareOrIsa: return "HARDWARE_OR_ISA";
    case MarlinExclusionKindPPU::ResourceLimit: return "RESOURCE_LIMIT";
    case MarlinExclusionKindPPU::CurrentImplementation:
      return "CURRENT_IMPLEMENTATION";
  }
  return "UNKNOWN";
}

// First-failure order is part of the emitted census ABI.  It prevents one row
// from being counted in several buckets and makes a changed pruning reason
// visible rather than silently changing the denominator of a sweep.
enum class MarlinTacticExclusionPPU : uint8_t {
  None,
  AtomAlignment,
  WarpDoesNotDivideTile,
  WarpKDoesNotDivideTile,
  CtaWarpCount,
  WarpKCohortNotPowerOfTwo,
  TileNNotMarlinBlock,
  SharedMemoryCapacity,
  ACopyVectorCoverage,
  BCopyVectorCoverage,
  BInnerIterations,
  AccumulatorRegisterCeiling,
  GroupScaleTileMismatch,
  AiuLoadUnproved,
  PipelineDepthUnproved,
  ClassicOutputMapUnproved,
  ClassicMainloopGeometryUnproved,
  Count,
};

constexpr char const* exclusion_name(MarlinTacticExclusionPPU exclusion) {
  switch (exclusion) {
    case MarlinTacticExclusionPPU::None: return "NONE";
    case MarlinTacticExclusionPPU::AtomAlignment: return "ATOM_ALIGNMENT";
    case MarlinTacticExclusionPPU::WarpDoesNotDivideTile:
      return "WARP_DOES_NOT_DIVIDE_TILE";
    case MarlinTacticExclusionPPU::WarpKDoesNotDivideTile:
      return "WARP_K_DOES_NOT_DIVIDE_TILE";
    case MarlinTacticExclusionPPU::CtaWarpCount: return "CTA_WARP_COUNT";
    case MarlinTacticExclusionPPU::WarpKCohortNotPowerOfTwo:
      return "WARP_K_COHORT_NOT_POWER_OF_TWO";
    case MarlinTacticExclusionPPU::TileNNotMarlinBlock:
      return "TILE_N_NOT_MARLIN_BLOCK";
    case MarlinTacticExclusionPPU::SharedMemoryCapacity:
      return "SHARED_MEMORY_CAPACITY";
    case MarlinTacticExclusionPPU::ACopyVectorCoverage:
      return "A_COPY_VECTOR_COVERAGE";
    case MarlinTacticExclusionPPU::BCopyVectorCoverage:
      return "B_COPY_VECTOR_COVERAGE";
    case MarlinTacticExclusionPPU::BInnerIterations:
      return "B_INNER_ITERATIONS";
    case MarlinTacticExclusionPPU::AccumulatorRegisterCeiling:
      return "ACCUMULATOR_REGISTER_CEILING";
    case MarlinTacticExclusionPPU::GroupScaleTileMismatch:
      return "GROUP_SCALE_TILE_MISMATCH";
    case MarlinTacticExclusionPPU::AiuLoadUnproved:
      return "AIU_LOAD_UNPROVED";
    case MarlinTacticExclusionPPU::PipelineDepthUnproved:
      return "PIPELINE_DEPTH_UNPROVED";
    case MarlinTacticExclusionPPU::ClassicOutputMapUnproved:
      return "CLASSIC_OUTPUT_MAP_UNPROVED";
    case MarlinTacticExclusionPPU::ClassicMainloopGeometryUnproved:
      return "CLASSIC_MAINLOOP_GEOMETRY_UNPROVED";
    case MarlinTacticExclusionPPU::Count: return "COUNT";
  }
  return "UNKNOWN";
}

constexpr MarlinExclusionKindPPU exclusion_kind(
    MarlinTacticExclusionPPU exclusion) {
  switch (exclusion) {
    case MarlinTacticExclusionPPU::None:
      return MarlinExclusionKindPPU::Admitted;
    case MarlinTacticExclusionPPU::AtomAlignment:
    case MarlinTacticExclusionPPU::WarpDoesNotDivideTile:
    case MarlinTacticExclusionPPU::WarpKDoesNotDivideTile:
    case MarlinTacticExclusionPPU::CtaWarpCount:
    case MarlinTacticExclusionPPU::WarpKCohortNotPowerOfTwo:
    case MarlinTacticExclusionPPU::TileNNotMarlinBlock:
      return MarlinExclusionKindPPU::HardwareOrIsa;
    case MarlinTacticExclusionPPU::SharedMemoryCapacity:
    case MarlinTacticExclusionPPU::AccumulatorRegisterCeiling:
      return MarlinExclusionKindPPU::ResourceLimit;
    case MarlinTacticExclusionPPU::ACopyVectorCoverage:
    case MarlinTacticExclusionPPU::BCopyVectorCoverage:
    case MarlinTacticExclusionPPU::BInnerIterations:
    case MarlinTacticExclusionPPU::GroupScaleTileMismatch:
    case MarlinTacticExclusionPPU::AiuLoadUnproved:
    case MarlinTacticExclusionPPU::PipelineDepthUnproved:
    case MarlinTacticExclusionPPU::ClassicOutputMapUnproved:
    case MarlinTacticExclusionPPU::ClassicMainloopGeometryUnproved:
      return MarlinExclusionKindPPU::CurrentImplementation;
    case MarlinTacticExclusionPPU::Count:
      break;
  }
  return MarlinExclusionKindPPU::CurrentImplementation;
}

constexpr char const* exclusion_clause(MarlinTacticExclusionPPU exclusion) {
  switch (exclusion) {
    case MarlinTacticExclusionPPU::None:
      return "";
    case MarlinTacticExclusionPPU::AtomAlignment:
      return "tile and warp extents must align to a real m8n16k16 or m16n16k16 PPU atom";
    case MarlinTacticExclusionPPU::WarpDoesNotDivideTile:
      return "WarpM and WarpN must divide TileM and TileN exactly";
    case MarlinTacticExclusionPPU::WarpKDoesNotDivideTile:
      return "WarpK must be no larger than and divide TileK exactly";
    case MarlinTacticExclusionPPU::CtaWarpCount:
      return "the (M,N,K) warp product must fit the 1..32 warp CTA range";
    case MarlinTacticExclusionPPU::WarpKCohortNotPowerOfTwo:
      return "the CTA-local Marlin reduction requires a power-of-two K cohort";
    case MarlinTacticExclusionPPU::TileNNotMarlinBlock:
      return "Marlin's N-warp geometry consumes whole 64-column blocks";
    case MarlinTacticExclusionPPU::SharedMemoryCapacity:
      return "A+B+scale stages exceed the PPU 256 KiB block shared-memory limit";
    case MarlinTacticExclusionPPU::ACopyVectorCoverage:
      return "classic 16-byte A copies do not partition exactly across the CTA threads";
    case MarlinTacticExclusionPPU::BCopyVectorCoverage:
      return "classic 16-byte B copies do not partition exactly across the CTA threads";
    case MarlinTacticExclusionPPU::BInnerIterations:
      return "the classic overlap cadence requires at least two B inner iterations";
    case MarlinTacticExclusionPPU::AccumulatorRegisterCeiling:
      return "the FP32 output fragment exceeds the 192-value per-thread admission ceiling";
    case MarlinTacticExclusionPPU::GroupScaleTileMismatch:
      return "the first gs128 loader proves one scale group per K tile only";
    case MarlinTacticExclusionPPU::AiuLoadUnproved:
      return "AIU load is declared but has no standalone Marlin byte/cadence proof yet";
    case MarlinTacticExclusionPPU::PipelineDepthUnproved:
      return "the standalone ring proof admits pipeline depths 2 through 6";
    case MarlinTacticExclusionPPU::ClassicOutputMapUnproved:
      return "the standalone final-output and reduction maps are proved for matched m8/m16 M atoms with WarpN64";
    case MarlinTacticExclusionPPU::ClassicMainloopGeometryUnproved:
      return "the standalone copy/dequant/MMA cadence is proved only for TileN128/TileK128/WarpK32";
    case MarlinTacticExclusionPPU::Count:
      return "sentinel";
  }
  return "unknown exclusion";
}

constexpr bool is_power_of_two(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

constexpr int instruction_m(MarlinTacticPPU tactic) {
  return tactic.tm == 8 && tactic.wm == 8 ? 8 : 16;
}

constexpr int cta_warps(MarlinTacticPPU tactic) {
  if (tactic.wm <= 0 || tactic.wn <= 0 || tactic.warp_k <= 0 ||
      tactic.tm % tactic.wm || tactic.tn % tactic.wn ||
      tactic.tk % tactic.warp_k) {
    return 0;
  }
  return (tactic.tm / tactic.wm) * (tactic.tn / tactic.wn) *
         (tactic.tk / tactic.warp_k);
}

constexpr int cta_threads(MarlinTacticPPU tactic) {
  return kMarlinWarpThreads * cta_warps(tactic);
}

constexpr int warp_k_cohorts(MarlinTacticPPU tactic) {
  return tactic.warp_k > 0 && tactic.tk % tactic.warp_k == 0
             ? tactic.tk / tactic.warp_k
             : 0;
}

constexpr int stored_a_rows(MarlinTacticPPU tactic) {
  // The exact m8 target is dense decode M=1.  Its plain-x2 providers alias
  // masked rows back to the one packed resident row; the m16 reference keeps
  // its classic 16-row stage.
  return tactic.tm == 8 ? 1 : tactic.tm;
}

// Bytes, not elements.  This is the standalone classic layout: A is fp16, B
// is packed int4, and one fp16 scale is staged per output column.
constexpr int64_t shared_bytes_per_stage(MarlinTacticPPU tactic) {
  return int64_t(2) * stored_a_rows(tactic) * tactic.tk +
         int64_t(tactic.tn) * tactic.tk / 2 +
         int64_t(2) * tactic.tn;
}

constexpr int64_t shared_bytes(MarlinTacticPPU tactic) {
  return shared_bytes_per_stage(tactic) * tactic.stages;
}

constexpr int a_stage_vectors(MarlinTacticPPU tactic) {
  return stored_a_rows(tactic) * tactic.tk / 8;
}

constexpr int b_stage_vectors(MarlinTacticPPU tactic) {
  return tactic.tn * tactic.tk / 32;
}

constexpr int b_inner_iterations(MarlinTacticPPU tactic) {
  int const threads = cta_threads(tactic);
  return threads > 0 && b_stage_vectors(tactic) % threads == 0
             ? b_stage_vectors(tactic) / threads
             : 0;
}

constexpr int accumulator_values_per_thread(MarlinTacticPPU tactic) {
  return tactic.wm * tactic.wn / kMarlinWarpThreads;
}

// This is a relationship, not an admission result.  It names the Awesome-CuTe
// / classic Marlin subspace inside our deliberately larger Cartesian domain.
// WN remains a normal axis outside this predicate; preserving that larger
// search space is one advantage of the quactlize implementation.
constexpr bool is_classic_subspace(MarlinTacticPPU tactic) {
  return tactic.wm == tactic.tm && tactic.wn == 64 &&
         tactic.warp_k == 32 && tactic.stages == 4 &&
         tactic.load == MarlinLoadKindPPU::CpAsync;
}

constexpr bool is_classic_reference(MarlinTacticPPU tactic) {
  return tactic == kMarlinClassicReferencePPU;
}

constexpr MarlinTacticExclusionPPU classify(MarlinTacticPPU tactic) {
  int const inst_m = instruction_m(tactic);
  if (tactic.tm % inst_m || tactic.wm % inst_m || tactic.tn % 16 ||
      tactic.wn % 16 || tactic.tk % 16 || tactic.warp_k % 16) {
    return MarlinTacticExclusionPPU::AtomAlignment;
  }
  if (tactic.wm > tactic.tm || tactic.wn > tactic.tn ||
      tactic.tm % tactic.wm || tactic.tn % tactic.wn) {
    return MarlinTacticExclusionPPU::WarpDoesNotDivideTile;
  }
  if (tactic.warp_k > tactic.tk || tactic.tk % tactic.warp_k) {
    return MarlinTacticExclusionPPU::WarpKDoesNotDivideTile;
  }
  int const warps = cta_warps(tactic);
  if (warps < 1 || warps > kMarlinMaxCtaWarps) {
    return MarlinTacticExclusionPPU::CtaWarpCount;
  }
  if (!is_power_of_two(warp_k_cohorts(tactic))) {
    return MarlinTacticExclusionPPU::WarpKCohortNotPowerOfTwo;
  }
  if (tactic.tn % 64) {
    return MarlinTacticExclusionPPU::TileNNotMarlinBlock;
  }
  if (shared_bytes(tactic) > kMarlinBlockSmemBytes) {
    return MarlinTacticExclusionPPU::SharedMemoryCapacity;
  }
  int const threads = cta_threads(tactic);
  int const a_quantum = tactic.tk / 8;
  int const a_vectors = a_stage_vectors(tactic);
  // A is copied in 16-byte vectors.  Full-m16 happens to assign one vector
  // to every CTA thread, but that is not an ownership requirement: the M=1
  // packed-m8 stage has only 16 distinct vectors and deliberately uses one
  // half-warp.  Requiring `a_vectors % threads == 0` would either reject that
  // exact cover or force 16 redundant copies of every A byte.  The current
  // one-iteration producer instead requires a non-empty whole-row vector
  // domain that fits within the CTA; source binding and L181 prove the active
  // prefix covers it exactly once.  Preserve the original full-CTA/multi-pass
  // classification for the rest of the declared space; only the one-row
  // representation has a smaller-than-CTA producer domain.
  bool const full_cta_cover = threads > 0 && a_vectors % threads == 0;
  bool const packed_row_prefix_cover =
      stored_a_rows(tactic) == 1 && a_vectors == a_quantum &&
      a_vectors <= threads;
  if (a_quantum <= 0 || threads % a_quantum || a_vectors <= 0 ||
      a_vectors % a_quantum ||
      (!full_cta_cover && !packed_row_prefix_cover)) {
    return MarlinTacticExclusionPPU::ACopyVectorCoverage;
  }
  if (b_stage_vectors(tactic) % threads) {
    return MarlinTacticExclusionPPU::BCopyVectorCoverage;
  }
  if (b_inner_iterations(tactic) < 2) {
    return MarlinTacticExclusionPPU::BInnerIterations;
  }
  if (accumulator_values_per_thread(tactic) > 192) {
    return MarlinTacticExclusionPPU::AccumulatorRegisterCeiling;
  }
  if (tactic.tk > kMarlinGroupSize || kMarlinGroupSize % tactic.tk) {
    return MarlinTacticExclusionPPU::GroupScaleTileMismatch;
  }
  if (tactic.load != MarlinLoadKindPPU::CpAsync) {
    return MarlinTacticExclusionPPU::AiuLoadUnproved;
  }
  if (tactic.stages < 2 || tactic.stages > 6) {
    return MarlinTacticExclusionPPU::PipelineDepthUnproved;
  }
  bool const proved_m_atom =
      (tactic.tm == 8 && tactic.wm == 8) ||
      (tactic.tm == 16 && tactic.wm == 16);
  if (!proved_m_atom || tactic.wn != 64) {
    return MarlinTacticExclusionPPU::ClassicOutputMapUnproved;
  }
  if (tactic.tn != 128 || tactic.tk != 128 || tactic.warp_k != 32) {
    return MarlinTacticExclusionPPU::ClassicMainloopGeometryUnproved;
  }
  return MarlinTacticExclusionPPU::None;
}

constexpr bool admitted(MarlinTacticPPU tactic) {
  return classify(tactic) == MarlinTacticExclusionPPU::None;
}

constexpr uint64_t cartesian_size() {
  return uint64_t(kMarlinTileM.size()) * kMarlinTileN.size() *
         kMarlinTileK.size() * kMarlinWarpM.size() *
         kMarlinWarpN.size() * kMarlinWarpK.size() *
         kMarlinStages.size() * kMarlinLoadKinds.size();
}

template <class Visitor>
void for_each_declared(Visitor&& visitor) {
  for (int tm : kMarlinTileM)
    for (int tn : kMarlinTileN)
      for (int tk : kMarlinTileK)
        for (int wm : kMarlinWarpM)
          for (int wn : kMarlinWarpN)
            for (int warp_k : kMarlinWarpK)
              for (int stages : kMarlinStages)
                for (MarlinLoadKindPPU load : kMarlinLoadKinds)
                  visitor(MarlinTacticPPU{
                      tm, tn, tk, wm, wn, warp_k, stages, load});
}

static_assert(cartesian_size() == 60000,
              "standalone Marlin declared Cartesian domain drifted");
static_assert(is_classic_subspace(kMarlinClassicReferencePPU));
static_assert(admitted(kMarlinClassicReferencePPU));
static_assert(admitted(MarlinTacticPPU{
    8, 128, 128, 8, 64, 32, 4, MarlinLoadKindPPU::CpAsync}));
static_assert(admitted(MarlinTacticPPU{
    8, 128, 128, 8, 64, 32, 2, MarlinLoadKindPPU::CpAsync}));
static_assert(admitted(MarlinTacticPPU{
    8, 128, 128, 8, 64, 32, 6, MarlinLoadKindPPU::CpAsync}));
static_assert(admitted(MarlinTacticPPU{
    16, 128, 128, 16, 64, 32, 3, MarlinLoadKindPPU::CpAsync}));
static_assert(admitted(MarlinTacticPPU{
    16, 128, 128, 16, 64, 32, 5, MarlinLoadKindPPU::CpAsync}));
static_assert(shared_bytes(kMarlinClassicReferencePPU) == 50176);
static_assert(shared_bytes(MarlinTacticPPU{
                  8, 128, 128, 8, 64, 32, 4,
                  MarlinLoadKindPPU::CpAsync}) == 34816,
              "m8 decode must retain exactly one packed A row");
static_assert(cta_warps(kMarlinClassicReferencePPU) == 8 &&
              cta_threads(kMarlinClassicReferencePPU) == 256 &&
              warp_k_cohorts(kMarlinClassicReferencePPU) == 4);
static_assert(classify(MarlinTacticPPU{
                  16, 128, 128, 16, 64, 32, 4,
                  MarlinLoadKindPPU::Aiu}) ==
              MarlinTacticExclusionPPU::AiuLoadUnproved);
static_assert(classify(MarlinTacticPPU{
                  16, 128, 128, 16, 64, 32, 1,
                  MarlinLoadKindPPU::CpAsync}) ==
              MarlinTacticExclusionPPU::PipelineDepthUnproved);
static_assert(classify(MarlinTacticPPU{
                  16, 128, 128, 16, 64, 64, 4,
                  MarlinLoadKindPPU::CpAsync}) ==
              MarlinTacticExclusionPPU::ClassicMainloopGeometryUnproved);
static_assert(classify(MarlinTacticPPU{
                  16, 128, 128, 16, 128, 32, 4,
                  MarlinLoadKindPPU::CpAsync}) ==
              MarlinTacticExclusionPPU::ClassicOutputMapUnproved);

}  // namespace marlin_tactics_ppu
