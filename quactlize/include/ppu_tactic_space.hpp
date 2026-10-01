// Shared host-readable legality and resource limits for canonical K-pack kernels.
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
// The runtime rejects a dynamic-shared attribute request equal to the nominal
// 256-KiB block capacity.  Treat the boundary as exclusive everywhere so a
// host-emitted row cannot disagree with its compiled launch guard.
constexpr bool fits_block_smem(int64_t bytes) {
  return bytes >= 0 && bytes < kBlockSmemBytes;
}
static_assert(fits_block_smem(kBlockSmemBytes - 1));
static_assert(!fits_block_smem(kBlockSmemBytes));
// Every emitted kernel must serve every runtime group size in the shared dispatch ladder. Sixteen is the smallest
// supported group and therefore the worst case for metadata footprint.
inline constexpr int kMinimumRuntimeGroupSize = 16;

enum class Format { I1, I2, I4, I8, Q3_K, Q5_K, Q6_K };

struct FormatSpec {
  Format format;
  char const* name;
  int low_bits;
  int high_bits;
  // Metadata residency is part of format legality, not a heuristic.  The
  // historical low-bit families keep the conservative runtime ladder
  // (gs16, scale+zero); Q8_0 is fixed gs32 ScaleOnly and therefore owns one
  // fp16 metadata plane.
  int minimum_group_size;
  int metadata_planes;

  constexpr FormatSpec(Format format_, char const* name_, int low_bits_,
                       int high_bits_,
                       int minimum_group_size_ = kMinimumRuntimeGroupSize,
                       int metadata_planes_ = 2)
      : format(format_), name(name_), low_bits(low_bits_),
        high_bits(high_bits_), minimum_group_size(minimum_group_size_),
        metadata_planes(metadata_planes_) {}
};

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
  EpilogueSharedStorage,
  ProducerWarpN,
  ProducerMap,
  ProducerConsumerLayout,
  BChunkUnsupportedBits,
};

constexpr char const* exclusion_clause(Exclusion e) {
  switch (e) {
    case Exclusion::None: return "";
    case Exclusion::AtomAlignment:
      return "tile and warp extents must align to the selected m8n16k16 or m16n16k16 MMA atom";
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
    case Exclusion::EpilogueSharedStorage:
      return "the fp32 epilogue shared tile reaches or exceeds the usable 256KB dynamic-shared limit";
    case Exclusion::ProducerWarpN: return "the offline producer exposes only consumer-validated WarpN values through 64";
    case Exclusion::ProducerMap: return "the Q6 offline producer inverse at ArtifactTileK=256 is incomplete";
    case Exclusion::ProducerConsumerLayout:
      return "the tactic's folded B reader does not decode the canonical resident artifact byte map";
    case Exclusion::BChunkUnsupportedBits: return "single-plane BChunk requires a 1- or 2-bit format";
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

// Q6 is the only registered 4+2-bit format. KernelPolicyGuard deliberately carries the physical plane tuple rather
// than a GGUF qtype, so kernel exclusions must recognize this pair instead of trusting Candidate::spec.format.
constexpr bool has_q6_plane_pair(Candidate c) {
  return c.spec.low_bits == 4 && c.spec.high_bits == 2;
}

// Mirror GetMmaInstForShape's deliberately narrow first m8 exposure without importing CUTLASS into this host-only
// header.  In particular, (TM=16,WM=8) must not become "two m8 atoms": the device selector falls back to m16 there.
constexpr int instruction_m(Candidate c) {
  return c.tm == 8 && c.wm == 8 ? 8 : 16;
}

// The logical m8 tile still uses the AIU's physical 16-row A cube and .padz supplies rows 8..15.  Billing only TM
// rows here would admit tactics whose generated kernel needs more shared memory than the host model promised.
constexpr int physical_a_rows(Candidate c) {
  return c.tm < 16 ? 16 : c.tm;
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
  int const inst_m = instruction_m(c);
  if (c.tm % inst_m || c.tn % 16 || c.tactic_tile_k % 16 || c.wm % inst_m || c.wn % 16)
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
       + int64_t(c.tn) *
             ((c.tactic_tile_k + c.spec.minimum_group_size - 1) /
              c.spec.minimum_group_size) *
             2 * c.spec.metadata_planes;
}

// The PPU vectorized epilogue stages one fp32 accumulator for every element
// in its exact shared output tile.  Its M extent is one MMA instruction per
// M-warp, not necessarily the logical CTA TileM.  This mirrors
// CollectiveBuilder's SmemM/SmemLayoutO derivation without importing CUTLASS
// into this host-readable tactic authority.
constexpr int64_t epilogue_smem(Candidate c) {
  int const warp_on_m = c.tm / c.wm;
  return int64_t(warp_on_m) * instruction_m(c) * c.tn * 4;
}

constexpr Exclusion common_topology_exclusion(Candidate c, int stages = 2) {
  if (auto const e = common_non_smem_exclusion(c); e != Exclusion::None) return e;

  // A request of exactly 256 KiB is not launchable: the adapter must publish
  // it through hggcFuncAttributeMaxDynamicSharedMemorySize, and the PPU0010
  // runtime rejects that exact-cap request.  The Q4 TM256/TN256/WM16/WN128
  // device witness returned kErrorInternal from that sole initialize path;
  // its otherwise-isomorphic WM32 control used 172048 B and ran correctly.
  if (!fits_block_smem(epilogue_smem(c)))
    return Exclusion::EpilogueSharedStorage;

  // Fold cancels from B bytes.  Metadata uses the format-owned contract:
  // historical families retain the conservative gs16 scale+zero ladder,
  // while fixed-gs32 Q8_0 charges its one ScaleOnly fp16 plane.  Applying the
  // historical two-plane assumption to Q8 would silently delete legal rows.
  // This is deliberately a broad conservative model: it charges every
  // format at the minimum runtime group size and can therefore equal the
  // limit even when the compiled format-owned layout is smaller.  Preserve
  // equality here and let the exact compiled SharedStorageSize guard make
  // the strict boundary decision.
  if (common_per_stage_smem(c, physical_a_rows(c)) * stages >
      kBlockSmemBytes)
    return Exclusion::MinimumStageSmem;
  return Exclusion::None;
}

struct TacticSpace {
  static constexpr Exclusion kernel_exclusion(Candidate c) { return common_kernel_exclusion(c); }
  static constexpr Exclusion topology_exclusion(Candidate c, int stages = 2) {
    return common_topology_exclusion(c, stages);
  }
};
using DenseSpace = TacticSpace;
using GroupedSpace = TacticSpace;
static_assert(std::is_same_v<DenseSpace, GroupedSpace>,
              "dense and grouped must remain aliases of one tactic-space generator");

}  // namespace ppu_tactics
