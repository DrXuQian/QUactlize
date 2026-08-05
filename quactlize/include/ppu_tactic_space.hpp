// Host-readable description of the finite arrangement/tactic search domain shared by fpA dense and grouped GEMM.
//
// Keep this header free of CUTLASS/HGGC includes: dev/fold_derivation/emit_tactic_space.cpp compiles it with the host
// compiler and prints EVERY candidate, including exclusions.  The two launchers also name their own rule wrapper and
// static-assert the kernel part, so the emitter is not a transcription of constraints hidden in device-only code.
#pragma once

#include <array>
#include <cstdint>

#include "ppu_format_config.hpp"

namespace ppu_tactics {

// ppu001 exposes 256 KiB of shared storage to one block. Runtime tactic validation and the host sweep use the
// same named limit; a literal in either place is exactly how an emitted "legal" tactic can become an unlaunchable
// compiled tactic.
inline constexpr int64_t kBlockSmemBytes = 262144;
// Every emitted kernel must serve every runtime group size in the shared dispatch ladder. Sixteen is the smallest
// supported group and therefore the worst case for both metadata footprint and scale-copy thread coverage.
inline constexpr int kMinimumRuntimeGroupSize = 16;

enum class Format { I1, I2, I4, Q3_K, Q5_K, Q6_K };

struct FormatSpec {
  Format format;
  char const* name;
  int low_bits;
  int high_bits;
};

// The finite domain for the 025 sweep. Artifact folds are derived from (bits, ArtifactTileK), never selected. The
// consumer's TacticTileK is independent: it changes kernel geometry without changing stored bytes. Stage and split-K
// are tactic axes too; Stage=2 is used below as the existence test because it is the shallowest supported pipeline.
inline constexpr std::array<FormatSpec, 6> kFormats{{
    {Format::I1, "i1", 1, 0},
    {Format::I2, "i2", 2, 0},
    {Format::I4, "i4", 4, 0},
    {Format::Q3_K, ppu_formats::for_qtype(11).name,
                   ppu_formats::for_qtype(11).low_bits, ppu_formats::for_qtype(11).high_bits},
    {Format::Q5_K, ppu_formats::for_qtype(13).name,
                   ppu_formats::for_qtype(13).low_bits, ppu_formats::for_qtype(13).high_bits},
    {Format::Q6_K, ppu_formats::for_qtype(14).name,
                   ppu_formats::for_qtype(14).low_bits, ppu_formats::for_qtype(14).high_bits},
}};
inline constexpr std::array<int, 4> kTileK{{32, 64, 128, 256}};
inline constexpr std::array<int, 5> kTileM{{16, 32, 64, 128, 256}};
inline constexpr std::array<int, 4> kTileN{{16, 32, 64, 128}};
inline constexpr std::array<int, 3> kWarpM{{16, 32, 64}};
// WN=128 is deliberate.  The current MoE generator omits it, but it is the delivery escape for an int1 plane at TK32;
// whether the remaining constraints reject a row must be printed, not encoded by leaving the axis out.
inline constexpr std::array<int, 4> kWarpN{{16, 32, 64, 128}};

constexpr int fold_for(int bits, int tile_k) {
  int const run = tile_k * bits / 8;
  if (run <= 0 || (run < 32 && 32 % run)) return 0;
  return run >= 32 ? 1 : 32 / run;
}

enum class Exclusion {
  None,
  AtomAlignment,
  WarpDoesNotDivideTile,
  TooManyWarps,
  AccumulatorRegisters,
  ArtifactTileKDoesNotTileTacticK,
  ArtifactLowRun,
  ArtifactHighRun,
  LowFoldDoesNotDivideTileN,
  HighFoldDoesNotDivideTileN,
  LowDelivery,
  HighDelivery,
  ScaleCopyCoverage,
  MinimumStageSmem,
  CompactAUnavailable,
  CompactARowExtent,
  ProducerWarpN,
  ProducerMap,
};

constexpr char const* exclusion_clause(Exclusion e) {
  switch (e) {
    case Exclusion::None: return "";
    case Exclusion::AtomAlignment: return "tile and warp extents must be multiples of the 16x16x16 MMA atom";
    case Exclusion::WarpDoesNotDivideTile: return "warp shape must divide tile shape";
    case Exclusion::TooManyWarps: return "tile needs more than the 32-warp block limit";
    case Exclusion::AccumulatorRegisters: return "the fp32 accumulator alone exceeds the 192-register sweep ceiling";
    case Exclusion::ArtifactTileKDoesNotTileTacticK: return "ArtifactTileK must be atom-aligned and completely tile TacticTileK";
    case Exclusion::ArtifactLowRun: return "ArtifactLowFold must form whole 32-byte AIU runs";
    case Exclusion::ArtifactHighRun: return "ArtifactHighFold must form whole 32-byte AIU runs";
    case Exclusion::LowFoldDoesNotDivideTileN: return "ArtifactLowFold does not divide TacticTileN";
    case Exclusion::HighFoldDoesNotDivideTileN: return "ArtifactHighFold does not divide TacticTileN";
    case Exclusion::LowDelivery: return "the low plane over-delivers the warp fragment";
    case Exclusion::HighDelivery: return "the high plane over-delivers the warp fragment";
    case Exclusion::ScaleCopyCoverage: return "the conservative gs16 scale copy needs more thread slots than the CTA has";
    case Exclusion::MinimumStageSmem: return "the conservative gs16 scale+zero footprint exceeds the 256KB block limit";
    case Exclusion::CompactAUnavailable: return "the selected folded or two-plane collective has no compact-A reader";
    case Exclusion::CompactARowExtent: return "compact-A rows must be one of the built small-M capacities 1, 2, or 4 and divide TileM";
    case Exclusion::ProducerWarpN: return "the offline producer exposes only consumer-validated WarpN values through 64";
    case Exclusion::ProducerMap: return "the Q6 two-plane inverse at TileK=256 is incomplete";
  }
  return "unknown exclusion";
}

struct Candidate {
  FormatSpec spec;
  int tm, tn;
  // `tk` is a temporary source-compatibility alias for the existing emitter. New code must spell the distinction:
  // TacticTileK is per row; ArtifactTileK identifies the one resident byte layout shared by those rows.
  union { int tactic_tile_k; int tk; };
  int wm, wn;
  int artifact_tile_k;

  constexpr Candidate(FormatSpec spec_, int tm_, int tn_, int tactic_tile_k_, int wm_, int wn_,
                      int artifact_tile_k_ = 0)
      : spec(spec_), tm(tm_), tn(tn_), tactic_tile_k(tactic_tile_k_), wm(wm_), wn(wn_),
        artifact_tile_k(artifact_tile_k_ > 0 ? artifact_tile_k_ : tactic_tile_k_) {}
};

constexpr int artifact_low_fold(Candidate c) {
  return fold_for(c.spec.low_bits, c.artifact_tile_k);
}

constexpr int artifact_high_fold(Candidate c) {
  return c.spec.high_bits ? fold_for(c.spec.high_bits, c.artifact_tile_k) : 1;
}

constexpr bool artifact_run_is_exact(int bits, int artifact_tile_k) {
  if (bits <= 0 || artifact_tile_k <= 0 || (int64_t(bits) * artifact_tile_k) % 8) return false;
  int64_t const bytes = int64_t(bits) * artifact_tile_k / 8;
  return bytes >= 32 || (bytes > 0 && 32 % bytes == 0);
}

// This is the actual CTA warp count for the current PPU0010 builder, not a performance proxy. get_tiled_mma tiles one
// 32-thread MMA atom by Layout<Shape<TileM/WarpM, TileN/WarpN, _1>>, and both dense and grouped kernels launch
// cute::size(TiledMma{}) threads. Each launcher static-asserts this expression against its instantiated TiledMma so a
// future builder change cannot turn the host predicate into another unchecked re-derivation.
constexpr int cta_warps(Candidate c) {
  return (c.tm / c.wm) * (c.tn / c.wn);
}

constexpr int scale_copy_thread_slots(Candidate c, int group_size = kMinimumRuntimeGroupSize) {
  return (c.tn / 8) * ((c.tactic_tile_k + group_size - 1) / group_size);
}

constexpr bool scale_copy_thread_coverage(Candidate c, int group_size = kMinimumRuntimeGroupSize) {
  return int64_t(scale_copy_thread_slots(c, group_size)) <= int64_t(32) * cta_warps(c);
}

// These are kernel constraints, shared by the two launcher static_asserts and the host emitter.  They are kept apart
// from artifact reachability: a template may be a legal consumer while the shipping *_for_tile producer cannot yet
// make bytes for it.
constexpr Exclusion common_kernel_exclusion(Candidate c) {
  if (c.tm % 16 || c.tn % 16 || c.tactic_tile_k % 16 || c.wm % 16 || c.wn % 16)
    return Exclusion::AtomAlignment;
  if (c.wm > c.tm || c.wn > c.tn || c.tm % c.wm || c.tn % c.wn)
    return Exclusion::WarpDoesNotDivideTile;
  if (cta_warps(c) > 32)
    return Exclusion::TooManyWarps;

  if (c.artifact_tile_k <= 0 || c.artifact_tile_k % 16 || c.artifact_tile_k > c.tactic_tile_k ||
      c.tactic_tile_k % c.artifact_tile_k)
    return Exclusion::ArtifactTileKDoesNotTileTacticK;
  if (!artifact_run_is_exact(c.spec.low_bits, c.artifact_tile_k)) return Exclusion::ArtifactLowRun;
  if (c.spec.high_bits && !artifact_run_is_exact(c.spec.high_bits, c.artifact_tile_k))
    return Exclusion::ArtifactHighRun;

  int const flo = artifact_low_fold(c);
  int const fhi = artifact_high_fold(c);
  if (c.tn % flo) return Exclusion::LowFoldDoesNotDivideTileN;
  if (c.spec.high_bits && c.tn % fhi) return Exclusion::HighFoldDoesNotDivideTileN;
  // CheckDelivery's measured predicate: one 16-byte swzl delivery must fit the B fragment slots.
  if (int64_t(c.wn) * c.tactic_tile_k * c.spec.low_bits < 4096) return Exclusion::LowDelivery;
  if (c.spec.high_bits && int64_t(c.wn) * c.tactic_tile_k * c.spec.high_bits < 4096)
    return Exclusion::HighDelivery;
  // Match MetadataPolicy::ScaleCopyCoverage conservatively over the whole runtime group-size ladder. A row that
  // only works at gs32/64/128 is not a legal compiled tactic because the same instantiation is reachable at gs16.
  if (!scale_copy_thread_coverage(c)) return Exclusion::ScaleCopyCoverage;
  return Exclusion::None;
}

// THE DENSE AND GROUPED KERNEL PREDICATES ARE THE SAME PREDICATE, and this function exists only so the two
// routes keep separate names at the call sites.
//
// It used to carry one extra clause -- `if (cta_warps(c) < 4) return DenseSubFourWarpDeviceAbort;` -- recording
// that on 2026-08-04 the tested dense sub-four-warp instantiations aborted on ppu001. That clause was a
// CATEGORY ERROR and it was removed on 2026-08-05: it stated a MEMORY OF AN OBSERVATION inside the file that
// defines what is LEGAL. A constraint the kernel can express belongs in the kernel as a static_assert, where the
// compiler can refute it; a host-side list of things that once failed is a second source of truth that outlives
// the code it was drawn from, and this one did. What it excluded was the measured optimum: (64,64,64) w64x32 is
// two warps and is the recorded int4 65.0% winner, against 60.6% for the best row the clause left reachable.
//
// The evidence that retired it: the dense route now has 61 assert( sites of which 61 are static_assert, so the
// observed `Assertion 'false' failed` is unreachable in today's code; ci/check_route_admits.py compiles the
// two-warp row with a planted device-body static_assert as its control and finds no error; and codex reran that
// probe independently and reported no source-level basis for keeping the clause. The likely mechanism of the
// original aborts was the then-unimplemented ordinary COARSE scale path, whose selection
// (ppu_mma_aiu_multistage_mixed_input.hpp: "COARSE is a relation between the scale tile and the ACTUAL retiled B
// copy view") turns on fragment geometry and so co-varies with warp shape -- a correlation, not a warp-count law.
//
// If a two-warp row does fail on hardware, express the real condition as a static_assert with a name. Do not
// reinstate a remembered observation.
constexpr Exclusion dense_kernel_exclusion(Candidate c) {
  return common_kernel_exclusion(c);
}

constexpr Exclusion common_non_smem_exclusion(Candidate c) {
  if (auto const e = common_kernel_exclusion(c); e != Exclusion::None) return e;
  if ((c.wm * c.wn) / 32 > 192)
    return Exclusion::AccumulatorRegisters;
  return Exclusion::None;
}

constexpr Exclusion dense_non_smem_exclusion(Candidate c) {
  if (auto const e = dense_kernel_exclusion(c); e != Exclusion::None) return e;
  if ((c.wm * c.wn) / 32 > 192)
    return Exclusion::AccumulatorRegisters;
  return Exclusion::None;
}

constexpr int64_t common_per_stage_smem(Candidate c, int a_rows) {
  return int64_t(a_rows) * c.tactic_tile_k * 2
       + int64_t(c.tn) * c.tactic_tile_k * (c.spec.low_bits + c.spec.high_bits) / 8
       + int64_t(c.tn) * (c.tactic_tile_k / kMinimumRuntimeGroupSize) * 2 * 2;
}

constexpr Exclusion common_topology_exclusion_with_a_rows(Candidate c, int stages, int a_rows) {
  if (auto const e = common_non_smem_exclusion(c); e != Exclusion::None) return e;

  // Match moe_ok's conservative stage-2 existence test exactly. Fold cancels from B bytes; scale+zero is sized for
  // the smallest runtime group (16), because a too-loose filter produces a fake winner when initialize fails. A is
  // parameterised separately: ordinary kernels pass TileM, compact small-M specialisations pass their row capacity.
  if (common_per_stage_smem(c, a_rows) * stages > kBlockSmemBytes) return Exclusion::MinimumStageSmem;
  return Exclusion::None;
}

constexpr Exclusion common_topology_exclusion(Candidate c, int stages = 2) {
  return common_topology_exclusion_with_a_rows(c, stages, c.tm);
}

constexpr Exclusion dense_topology_exclusion_with_a_rows(Candidate c, int stages, int a_rows) {
  if (auto const e = dense_non_smem_exclusion(c); e != Exclusion::None) return e;
  if (common_per_stage_smem(c, a_rows) * stages > kBlockSmemBytes) return Exclusion::MinimumStageSmem;
  return Exclusion::None;
}

constexpr Exclusion dense_topology_exclusion(Candidate c, int stages = 2) {
  return dense_topology_exclusion_with_a_rows(c, stages, c.tm);
}

// PPU_A_CPASYNC currently exists in the ordinary, unfolded, one-plane collective. Folded low planes and every
// two-plane format select different collective types; their type-level witness is zero and their launcher rejects a
// compact build. Keep that reachability fact next to the footprint equation so the host inventory cannot claim that
// m*TK*2 is available for a collective that still allocates TileM*TK*2.
constexpr bool common_compact_a_supported(Candidate c) {
  return c.spec.high_bits == 0 && artifact_low_fold(c) == 1;
}

constexpr Exclusion common_compact_a_topology_exclusion(Candidate c, int stages, int compact_rows) {
  if (auto const e = common_non_smem_exclusion(c); e != Exclusion::None) return e;
  if (!common_compact_a_supported(c)) return Exclusion::CompactAUnavailable;
  if (!((compact_rows == 1 || compact_rows == 2 || compact_rows == 4) &&
        compact_rows <= c.tm && c.tm % compact_rows == 0))
    return Exclusion::CompactARowExtent;
  return common_topology_exclusion_with_a_rows(c, stages, compact_rows);
}

constexpr Exclusion dense_compact_a_topology_exclusion(Candidate c, int stages, int compact_rows) {
  if (auto const e = dense_non_smem_exclusion(c); e != Exclusion::None) return e;
  if (!common_compact_a_supported(c)) return Exclusion::CompactAUnavailable;
  if (!((compact_rows == 1 || compact_rows == 2 || compact_rows == 4) &&
        compact_rows <= c.tm && c.tm % compact_rows == 0))
    return Exclusion::CompactARowExtent;
  return dense_topology_exclusion_with_a_rows(c, stages, compact_rows);
}

constexpr Exclusion common_producer_exclusion(Candidate c) {
  // Artifact reachability is about the producer's layout, not the tactic that later consumes it. WN=128 remains
  // outside the producer's validated domain, and Q6/ArtifactTileK=256 is the one known bad inverse map.
  if (c.wn > 64) return Exclusion::ProducerWarpN;
  if (c.spec.format == Format::Q6_K && c.artifact_tile_k == 256) return Exclusion::ProducerMap;
  return Exclusion::None;
}

// Compile-time controls for the distinction this header owns. A fixed TK64 artifact is legal under larger tactics,
// including Q3's independent (low,high)=(2,4) folds; a tactic that cannot be partitioned into whole artifact K-blocks
// is refused before it can instantiate a provider with a partial physical row.
inline constexpr FormatSpec kArtifactFoldControlI2{Format::I2, "artifact-fold-control-i2", 2, 0};
inline constexpr FormatSpec kScaleCoverageControlI4{Format::I4, "scale-coverage-control-i4", 4, 0};
inline constexpr FormatSpec kArtifactFoldControlQ3{Format::Q3_K, "artifact-fold-control-q3", 2, 1};
inline constexpr Candidate kArtifactFoldControlI2Large{kArtifactFoldControlI2, 64, 64, 256, 32, 32, 64};
inline constexpr Candidate kArtifactFoldControlQ3Large{kArtifactFoldControlQ3, 64, 128, 256, 32, 32, 64};
static_assert(artifact_low_fold(kArtifactFoldControlI2Large) == 2);
static_assert(artifact_low_fold(kArtifactFoldControlQ3Large) == 2 &&
              artifact_high_fold(kArtifactFoldControlQ3Large) == 4);
static_assert(common_kernel_exclusion(kArtifactFoldControlI2Large) == Exclusion::None);
static_assert(common_kernel_exclusion(kArtifactFoldControlQ3Large) == Exclusion::None);
static_assert(common_kernel_exclusion(
                  Candidate{kArtifactFoldControlI2, 64, 64, 96, 64, 32, 64}) ==
              Exclusion::ArtifactTileKDoesNotTileTacticK);
static_assert(common_kernel_exclusion(
                  Candidate{kArtifactFoldControlI2, 64, 64, 96, 64, 32, 48}) ==
              Exclusion::ArtifactLowRun);
inline constexpr Candidate kScaleCoverageBoundary{
    kScaleCoverageControlI4, 16, 128, 64, 16, 64, 64};
inline constexpr Candidate kScaleCoverageOverflow{
    kScaleCoverageControlI4, 16, 128, 256, 16, 32, 64};
static_assert(scale_copy_thread_slots(kScaleCoverageBoundary) == 64 && cta_warps(kScaleCoverageBoundary) == 2);
static_assert(common_kernel_exclusion(kScaleCoverageBoundary) == Exclusion::None);
static_assert(scale_copy_thread_slots(kScaleCoverageOverflow) == 256 && cta_warps(kScaleCoverageOverflow) == 4);
static_assert(scale_copy_thread_slots(kScaleCoverageOverflow, 32) == 128 &&
              scale_copy_thread_coverage(kScaleCoverageOverflow, 32));
static_assert(common_kernel_exclusion(kScaleCoverageOverflow) == Exclusion::ScaleCopyCoverage);

// Everything that determines whether some topology for the candidate may be built, except the M- and stage-dependent
// shared footprint. size_sweep.cpp uses this before asking both the ordinary and compact topology predicates.
constexpr Exclusion common_static_sweep_exclusion(Candidate c) {
  if (auto const e = common_non_smem_exclusion(c); e != Exclusion::None) return e;
  return common_producer_exclusion(c);
}

constexpr Exclusion dense_static_sweep_exclusion(Candidate c) {
  if (auto const e = dense_non_smem_exclusion(c); e != Exclusion::None) return e;
  return common_producer_exclusion(c);
}

constexpr Exclusion common_sweep_exclusion(Candidate c) {
  if (auto const e = common_topology_exclusion(c, 2); e != Exclusion::None) return e;
  return common_producer_exclusion(c);
}

constexpr Exclusion dense_sweep_exclusion(Candidate c) {
  if (auto const e = dense_topology_exclusion(c, 2); e != Exclusion::None) return e;
  return common_producer_exclusion(c);
}

// Separate wrappers are intentional. The emitter asks each launcher for its own answer and the comparator reports
// their declared dense-only abort boundary; every other rule remains shared so an additional drift stays visible.
struct DenseSpace {
  static constexpr Exclusion kernel_exclusion(Candidate c) { return dense_kernel_exclusion(c); }
  static constexpr Exclusion topology_exclusion(Candidate c, int stages = 2) {
    return dense_topology_exclusion(c, stages);
  }
  static constexpr bool compact_a_supported(Candidate c) { return common_compact_a_supported(c); }
  static constexpr Exclusion compact_a_topology_exclusion(Candidate c, int stages, int compact_rows) {
    return dense_compact_a_topology_exclusion(c, stages, compact_rows);
  }
  static constexpr Exclusion static_sweep_exclusion(Candidate c) { return dense_static_sweep_exclusion(c); }
  static constexpr Exclusion sweep_exclusion(Candidate c) { return dense_sweep_exclusion(c); }
};
struct GroupedSpace {
  static constexpr Exclusion kernel_exclusion(Candidate c) { return common_kernel_exclusion(c); }
  static constexpr Exclusion topology_exclusion(Candidate c, int stages = 2) {
    return common_topology_exclusion(c, stages);
  }
  static constexpr bool compact_a_supported(Candidate c) { return common_compact_a_supported(c); }
  static constexpr Exclusion compact_a_topology_exclusion(Candidate c, int stages, int compact_rows) {
    return common_compact_a_topology_exclusion(c, stages, compact_rows);
  }
  static constexpr Exclusion static_sweep_exclusion(Candidate c) { return common_static_sweep_exclusion(c); }
  static constexpr Exclusion sweep_exclusion(Candidate c) { return common_sweep_exclusion(c); }
};

}  // namespace ppu_tactics
