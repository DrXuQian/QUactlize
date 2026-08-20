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

// The finite domain for the 025 sweep. Artifact folds are derived from (bits, ArtifactTileK), never selected. The
// consumer's TacticTileK is independent: it changes kernel geometry without changing stored bytes. Stage and split-K
// are tactic axes too; Stage=2 is used below as the existence test because it is the shallowest supported pipeline.
inline constexpr std::array<FormatSpec, 7> kFormats{{
    {Format::I1, "i1", 1, 0},
    {Format::I2, "i2", 2, 0},
    {Format::I4, "i4", 4, 0},
    // Q8_0 ScaleFirst stores signed int8 codes as resident q+128 bytes and a
    // separate fp16 scale per 32 weights.  Its one canonical artifact is
    // ArtifactTileK=32/FoldN=1; TacticTileK remains a reader axis.
    {Format::I8, "i8", 8, 0, 32, 1},
    {Format::Q3_K, ppu_formats::for_qtype(11).name,
                   ppu_formats::for_qtype(11).low_bits, ppu_formats::for_qtype(11).high_bits},
    {Format::Q5_K, ppu_formats::for_qtype(13).name,
                   ppu_formats::for_qtype(13).low_bits, ppu_formats::for_qtype(13).high_bits},
    {Format::Q6_K, ppu_formats::for_qtype(14).name,
                   ppu_formats::for_qtype(14).low_bits, ppu_formats::for_qtype(14).high_bits},
}};
inline constexpr std::array<int, 4> kTileK{{32, 64, 128, 256}};
// PPU0010's m8n16k16 atom is exposed only for the exact TileM=8/WarpM=8 family.  Keeping eight on both axes, rather
// than manufacturing that pair in an emitter, lets the one legality predicate reject every cross-family combination.
inline constexpr std::array<int, 6> kTileM{{8, 16, 32, 64, 128, 256}};
inline constexpr std::array<int, 5> kTileN{{16, 32, 64, 128, 256}};
inline constexpr std::array<int, 4> kWarpM{{8, 16, 32, 64}};
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
    case Exclusion::ProducerWarpN: return "the offline producer exposes only consumer-validated WarpN values through 64";
    case Exclusion::ProducerMap: return "the Q6 offline producer inverse at ArtifactTileK=256 is incomplete";
    case Exclusion::ProducerConsumerLayout:
      return "the tactic's folded B reader does not decode the canonical resident artifact byte map";
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

constexpr Exclusion common_topology_exclusion(Candidate c, int stages = 2) {
  if (auto const e = common_non_smem_exclusion(c); e != Exclusion::None) return e;

  // Fold cancels from B bytes.  Metadata uses the format-owned contract:
  // historical families retain the conservative gs16 scale+zero ladder,
  // while fixed-gs32 Q8_0 charges its one ScaleOnly fp16 plane.  Applying the
  // historical two-plane assumption to Q8 would silently delete legal rows.
  if (common_per_stage_smem(c, physical_a_rows(c)) * stages > kBlockSmemBytes)
    return Exclusion::MinimumStageSmem;
  return Exclusion::None;
}

// Denominator negative for Q8 metadata ownership.  This row fits only under
// the real fixed-gs32/one-plane contract; planting the historical
// gs16/scale+zero charge removes it.  The emitter's committed denominator
// check therefore has an independently named witness rather than merely
// observing that the final count is nonzero.
constexpr int64_t legacy_two_plane_per_stage_smem(Candidate c, int a_rows) {
  return int64_t(a_rows) * c.tactic_tile_k * 2
       + int64_t(c.tn) * c.tactic_tile_k *
             (c.spec.low_bits + c.spec.high_bits) / 8
       + int64_t(c.tn) *
             ((c.tactic_tile_k + kMinimumRuntimeGroupSize - 1) /
              kMinimumRuntimeGroupSize) * 2 * 2;
}
inline constexpr FormatSpec kQ8MetadataControl{
    Format::I8, "q8-metadata-control", 8, 0, 32, 1};
inline constexpr Candidate kQ8MetadataOnePlaneOnly{
    kQ8MetadataControl, 16, 128, 128, 16, 16, 32};
static_assert(common_topology_exclusion(kQ8MetadataOnePlaneOnly, 12) ==
                  Exclusion::None,
              "Q8 fixed-gs32 one-plane row must remain in the legal denominator");
static_assert(legacy_two_plane_per_stage_smem(
                  kQ8MetadataOnePlaneOnly,
                  physical_a_rows(kQ8MetadataOnePlaneOnly)) * 12 >
                  kBlockSmemBytes,
              "negative plant: the old gs16/two-plane charge must remove the Q8 witness");

constexpr Exclusion common_producer_exclusion(Candidate c) {
  // These are independent producer limits. The artifact-aware producer ABI names ArtifactTileK directly, and its
  // A256 inverse remains incomplete even though the A128/T256 consumer is now a complete resident-owner bijection.
  if (c.wn > 64) return Exclusion::ProducerWarpN;
  if (has_q6_plane_pair(c) && c.artifact_tile_k == 256) return Exclusion::ProducerMap;
  // Q4/A32 is a folded one-plane artifact.  FoldN=2 is necessary but not a
  // complete byte-layout descriptor: the reader's physical N tiling also
  // enters plane_map().  The canonical producer is TN64/WN32; exhaustive
  // cross-recovery over the finite TN/WN axes establishes exactly two
  // compatible reader classes under the current WN<=64 producer boundary:
  //
  //     TN64/WN32 and TN128/WN64  <=>  TN == 2*WN, WN >= 32.
  //
  // TN32/WN16 is worse than merely a different permutation: Ng=TN/F=16
  // while RPI=32, so Ng/RPI is zero and its derived consumer map has no
  // entries.  It was nevertheless admitted by FoldN|TN + delivery capacity,
  // producing the device-observed 65536/65536 wrong outputs.  Do not infer a
  // rule for Q2 or two-plane artifacts here: their maps have different
  // equivalence classes and retain their existing independently proved gates.
  if (c.spec.low_bits == 4 && c.spec.high_bits == 0 &&
      c.artifact_tile_k == 32 &&
      (c.wn < 32 || c.tn != 2 * c.wn))
    return Exclusion::ProducerConsumerLayout;
  return Exclusion::None;
}

// Constructive controls for the exact device failure.  The negative remains
// kernel-legal so only the producer/consumer byte-layout seam can reject it;
// the two positives are the complete equivalence class exposed by WN<=64.
inline constexpr FormatSpec kQ4A32LayoutControl{Format::I4, "q4-a32-layout-control", 4, 0, 32, 2};
inline constexpr Candidate kQ4A32EmptyConsumer{
    kQ4A32LayoutControl, 32, 32, 64, 16, 16, 32, 0};
inline constexpr Candidate kQ4A32CanonicalConsumer{
    kQ4A32LayoutControl, 64, 64, 64, 32, 32, 32, 0};
inline constexpr Candidate kQ4A32ScaledConsumer{
    kQ4A32LayoutControl, 64, 128, 256, 32, 64, 32, 0};
static_assert(common_kernel_exclusion(kQ4A32EmptyConsumer) == Exclusion::None &&
              common_producer_exclusion(kQ4A32EmptyConsumer) ==
                  Exclusion::ProducerConsumerLayout,
              "the exact TN32/WN16 Q4/A32 device failure must be a named static reject");
static_assert(common_producer_exclusion(kQ4A32CanonicalConsumer) == Exclusion::None &&
              common_producer_exclusion(kQ4A32ScaledConsumer) == Exclusion::None,
              "both canonical Q4/A32 reader equivalence classes must remain admitted");

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

// Atom-family controls: exactly 8x8 selects m8, both cross-family spellings remain illegal, and the m8 shared-memory
// charge retains the physical 16-row A cube rather than silently halving it with the logical TileM.
inline constexpr Candidate kM8AtomControl{kArtifactFoldControlI2, 8, 32, 64, 8, 32, 64};
static_assert(instruction_m(kM8AtomControl) == 8);
static_assert(common_kernel_exclusion(kM8AtomControl) == Exclusion::None);
static_assert(physical_a_rows(kM8AtomControl) == 16);
static_assert(common_per_stage_smem(kM8AtomControl, physical_a_rows(kM8AtomControl)) -
                  common_per_stage_smem(kM8AtomControl, kM8AtomControl.tm) == 8 * 64 * 2,
              "logical TM8 must not halve the physical A-cube shared-memory charge");
static_assert(common_kernel_exclusion(
                  Candidate{kArtifactFoldControlI2, 16, 32, 64, 8, 32, 64}) == Exclusion::AtomAlignment,
              "WM8 is legal only in the exact TM8 m8-atom family");
static_assert(common_kernel_exclusion(
                  Candidate{kArtifactFoldControlI2, 8, 32, 64, 16, 32, 64}) == Exclusion::AtomAlignment,
              "TM8 must not fall back silently to the m16 atom");

// Paired controls for the field-ownership regression. l115's shipping witness is exactly
//   Q6_K high A=128 T=256 tile=64x128x256 warp=64x64 F=1/1
// and requires logical=32768/32768, duplicates=unset=0, owner_diff=writer_diff=0 and COMPLETE. That discharges the
// former ConsumerMap gate. The matching A128/T128 row remains admitted, while a direct producer check pins the
// separate A256 ABI constraint so fixing the consumer side cannot erase it by accident.
inline constexpr FormatSpec kQ6MapControl{Format::Q6_K, "q6-map-control", 4, 2};
inline constexpr Candidate kQ6MapCrossTConsumer{kQ6MapControl, 64, 128, 256, 64, 64, 128};
inline constexpr Candidate kQ6MapGoodConsumer{kQ6MapControl, 64, 128, 128, 64, 64, 128};
inline constexpr Candidate kQ6MapBadProducer{kQ6MapControl, 64, 128, 256, 64, 64, 256};
static_assert(!(has_q6_plane_pair(kQ6MapCrossTConsumer) && kQ6MapCrossTConsumer.artifact_tile_k == 256),
              "negative control: the retired artifact-bound consumer predicate must miss A128/T256");
static_assert(common_kernel_exclusion(kQ6MapCrossTConsumer) == Exclusion::None &&
              common_producer_exclusion(kQ6MapCrossTConsumer) == Exclusion::None,
              "l115's Q6 A128/T256 COMPLETE witness releases this shipping consumer");
static_assert(common_kernel_exclusion(kQ6MapGoodConsumer) == Exclusion::None &&
              common_producer_exclusion(kQ6MapGoodConsumer) == Exclusion::None,
              "Q6 A128/T128 is the complete-map positive control");
static_assert(common_producer_exclusion(kQ6MapBadProducer) == Exclusion::ProducerMap,
              "Q6 ArtifactTileK=256 remains outside the offline producer ABI");
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
