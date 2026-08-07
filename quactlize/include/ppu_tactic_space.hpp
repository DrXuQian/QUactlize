// Host-readable description of the finite arrangement/tactic search domain shared by fpA dense and grouped GEMM.
//
// Keep this header free of CUTLASS/HGGC includes: dev/fold_derivation/emit_tactic_space.cpp compiles it with the host
// compiler and prints EVERY candidate, including exclusions. The two launchers retain public route aliases of the
// same TacticSpace and static-assert the kernel part, so the emitter is not a transcription of device-only constraints.
#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include "ppu_format_config.hpp"

namespace ppu_tactics {

// ppu001 exposes 256 KiB of shared storage to one block. Runtime tactic validation and the host sweep use the
// same named limit; a literal in either place is exactly how an emitted "legal" tactic can become an unlaunchable
// compiled tactic.
inline constexpr int64_t kBlockSmemBytes = 262144;
// Every emitted kernel must serve every runtime group size in the shared dispatch ladder. Sixteen is the smallest
// supported group and therefore the worst case for metadata footprint.
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
// PPU_B_CHUNK is a per-tactic compile-time axis.  The collective remains authoritative about whether a requested
// mode is effective for one concrete TiledMma; this host-readable domain only avoids duplicating formats for which
// no 1- or 2-bit plane can use the chunk emitter at all.
inline constexpr std::array<int, 2> kBChunkModes{{0, 1}};

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
  MinimumStageSmem,
  ProducerWarpN,
  ProducerMap,
  BChunkUnsupportedBits,
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
    case Exclusion::MinimumStageSmem: return "the conservative gs16 scale+zero footprint exceeds the 256KB block limit";
    case Exclusion::ProducerWarpN: return "the offline producer exposes only consumer-validated WarpN values through 64";
    case Exclusion::ProducerMap: return "the Q6 two-plane inverse at TileK=256 is incomplete";
    case Exclusion::BChunkUnsupportedBits: return "single-plane PPU_B_CHUNK requires a 1- or 2-bit format";
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
  int b_chunk;

  constexpr Candidate(FormatSpec spec_, int tm_, int tn_, int tactic_tile_k_, int wm_, int wn_,
                      int artifact_tile_k_ = 0, int b_chunk_ = 0)
      : spec(spec_), tm(tm_), tn(tn_), tactic_tile_k(tactic_tile_k_), wm(wm_), wn(wn_),
        artifact_tile_k(artifact_tile_k_ > 0 ? artifact_tile_k_ : tactic_tile_k_), b_chunk(b_chunk_) {}
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

// These are kernel constraints, shared by the two launcher static_asserts and the host emitter.  They are kept apart
// from artifact reachability: a template may be a legal consumer while the shipping *_for_tile producer cannot yet
// make bytes for it.
constexpr Exclusion common_kernel_exclusion(Candidate c) {
  // The two-plane collective owns its own mode gate and supports every registered pair.  For a single plane,
  // only int1/int2 have a chunk emitter; keeping this coarse avoids doubling dense/grouped int4 without copying
  // the collective's TiledMma-dependent effectiveness predicate into this host-only header.
  if (c.b_chunk != 0 && c.spec.high_bits == 0 && c.spec.low_bits != 1 && c.spec.low_bits != 2)
    return Exclusion::BChunkUnsupportedBits;
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
  return Exclusion::None;
}

constexpr Exclusion common_non_smem_exclusion(Candidate c) {
  if (auto const e = common_kernel_exclusion(c); e != Exclusion::None) return e;
  if ((c.wm * c.wn) / 32 > 192)
    return Exclusion::AccumulatorRegisters;
  return Exclusion::None;
}

constexpr int64_t common_per_stage_smem(Candidate c, int a_rows) {
  return int64_t(a_rows) * c.tactic_tile_k * 2
       + int64_t(c.tn) * c.tactic_tile_k * (c.spec.low_bits + c.spec.high_bits) / 8
       + int64_t(c.tn) * (c.tactic_tile_k / kMinimumRuntimeGroupSize) * 2 * 2;
}

constexpr Exclusion common_topology_exclusion(Candidate c, int stages = 2) {
  if (auto const e = common_non_smem_exclusion(c); e != Exclusion::None) return e;

  // Match moe_ok's conservative stage-2 existence test exactly. Fold cancels from B bytes; scale+zero is sized for
  // the smallest runtime group (16), because a too-loose filter produces a fake winner when initialize fails.
  if (common_per_stage_smem(c, c.tm) * stages > kBlockSmemBytes) return Exclusion::MinimumStageSmem;
  return Exclusion::None;
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
// Everything that determines whether some topology for the candidate may be built, except the M- and stage-dependent
// shared footprint. size_sweep.cpp uses this before asking the topology predicate.
constexpr Exclusion common_static_sweep_exclusion(Candidate c) {
  if (auto const e = common_non_smem_exclusion(c); e != Exclusion::None) return e;
  return common_producer_exclusion(c);
}

constexpr Exclusion common_sweep_exclusion(Candidate c) {
  if (auto const e = common_topology_exclusion(c, 2); e != Exclusion::None) return e;
  return common_producer_exclusion(c);
}

// ONE GENERATOR, TWO PUBLIC ROUTE NAMES. Dense and grouped currently have no legality asymmetry: the old dense_*
// chain was a byte-for-byte copy of this common chain, and dense_kernel_exclusion was a pure forwarder. Keeping two
// wrapper structs made future drift easy to express and then asked a comparator to notice it after the fact. Aliases
// make the invariant structural while preserving every launcher/emitter call site.
struct TacticSpace {
  static constexpr Exclusion kernel_exclusion(Candidate c) { return common_kernel_exclusion(c); }
  static constexpr Exclusion topology_exclusion(Candidate c, int stages = 2) {
    return common_topology_exclusion(c, stages);
  }
  static constexpr Exclusion static_sweep_exclusion(Candidate c) { return common_static_sweep_exclusion(c); }
  static constexpr Exclusion sweep_exclusion(Candidate c) { return common_sweep_exclusion(c); }
};
using DenseSpace = TacticSpace;
using GroupedSpace = TacticSpace;
static_assert(std::is_same_v<DenseSpace, GroupedSpace>,
              "dense and grouped must remain aliases of one tactic-space generator");

}  // namespace ppu_tactics
