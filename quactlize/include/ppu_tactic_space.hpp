// Host-readable description of the finite arrangement/tactic search domain shared by fpA dense and grouped GEMM.
//
// Keep this header free of CUTLASS/HGGC includes: dev/fold_derivation/emit_tactic_space.cpp compiles it with the host
// compiler and prints EVERY candidate, including exclusions.  The two launchers also name their own rule wrapper and
// static-assert the kernel part, so the emitter is not a transcription of constraints hidden in device-only code.
#pragma once

#include <array>
#include <cstdint>

namespace ppu_tactics {

enum class Format { I1, I2, I4, Q3_K, Q5_K, Q6_K };

struct FormatSpec {
  Format format;
  char const* name;
  int low_bits;
  int high_bits;
};

// The finite domain for the 025 sweep.  F is intentionally absent: it is derived from (bits, TK), never selected.
// Stage and split-K are tactic axes but do not change stored bytes; Stage=2 is used below as the existence test because
// it is the shallowest supported pipeline.  A topology that cannot fit at s2 cannot fit at any supported depth.
inline constexpr std::array<FormatSpec, 6> kFormats{{
    {Format::I1, "i1", 1, 0},
    {Format::I2, "i2", 2, 0},
    {Format::I4, "i4", 4, 0},
    {Format::Q3_K, "Q3_K", 2, 1},
    {Format::Q5_K, "Q5_K", 4, 1},
    {Format::Q6_K, "Q6_K", 4, 2},
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
  return run >= 32 ? 1 : 32 / run;
}

enum class Exclusion {
  None,
  AtomAlignment,
  WarpDoesNotDivideTile,
  TooManyWarps,
  AccumulatorRegisters,
  LowFoldDoesNotDivideTileN,
  HighFoldDoesNotDivideTileN,
  LowDelivery,
  HighDelivery,
  MinimumStageSmem,
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
    case Exclusion::LowFoldDoesNotDivideTileN: return "the derived low-plane fold does not divide TileN";
    case Exclusion::HighFoldDoesNotDivideTileN: return "the derived high-plane fold does not divide TileN";
    case Exclusion::LowDelivery: return "the low plane over-delivers the warp fragment";
    case Exclusion::HighDelivery: return "the high plane over-delivers the warp fragment";
    case Exclusion::MinimumStageSmem: return "the conservative gs16 scale+zero footprint exceeds 256KB even at stage 2";
    case Exclusion::ProducerWarpN: return "the offline producer exposes only consumer-validated WarpN values through 64";
    case Exclusion::ProducerMap: return "the Q6 two-plane inverse at TileK=256 is incomplete";
  }
  return "unknown exclusion";
}

struct Candidate {
  FormatSpec spec;
  int tm, tn, tk, wm, wn;
};

// These are kernel constraints, shared by the two launcher static_asserts and the host emitter.  They are kept apart
// from artifact reachability: a template may be a legal consumer while the shipping *_for_tile producer cannot yet
// make bytes for it.
constexpr Exclusion common_kernel_exclusion(Candidate c) {
  if (c.tm % 16 || c.tn % 16 || c.tk % 16 || c.wm % 16 || c.wn % 16)
    return Exclusion::AtomAlignment;
  if (c.wm > c.tm || c.wn > c.tn || c.tm % c.wm || c.tn % c.wn)
    return Exclusion::WarpDoesNotDivideTile;
  if ((c.tm / c.wm) * (c.tn / c.wn) > 32)
    return Exclusion::TooManyWarps;

  int const flo = fold_for(c.spec.low_bits, c.tk);
  int const fhi = c.spec.high_bits ? fold_for(c.spec.high_bits, c.tk) : 1;
  if (c.tn % flo) return Exclusion::LowFoldDoesNotDivideTileN;
  if (c.spec.high_bits && c.tn % fhi) return Exclusion::HighFoldDoesNotDivideTileN;
  // CheckDelivery's measured predicate: one 16-byte swzl delivery must fit the B fragment slots.
  if (int64_t(c.wn) * c.tk * c.spec.low_bits < 4096) return Exclusion::LowDelivery;
  if (c.spec.high_bits && int64_t(c.wn) * c.tk * c.spec.high_bits < 4096)
    return Exclusion::HighDelivery;
  return Exclusion::None;
}

constexpr Exclusion common_topology_exclusion(Candidate c, int stages = 2) {
  if (auto const e = common_kernel_exclusion(c); e != Exclusion::None) return e;
  if ((c.wm * c.wn) / 32 > 192)
    return Exclusion::AccumulatorRegisters;

  // Match moe_ok's conservative stage-2 existence test exactly. Fold cancels from B bytes; scale+zero is sized for
  // the smallest runtime group (16), because a too-loose filter produces a fake winner when initialize fails.
  int64_t const per_stage = int64_t(c.tm) * c.tk * 2
                          + int64_t(c.tn) * c.tk * (c.spec.low_bits + c.spec.high_bits) / 8
                          + int64_t(c.tn) * (c.tk / 16) * 2 * 2;
  if (per_stage * stages > 262144) return Exclusion::MinimumStageSmem;
  return Exclusion::None;
}

constexpr Exclusion common_sweep_exclusion(Candidate c) {
  if (auto const e = common_topology_exclusion(c, 2); e != Exclusion::None) return e;
  // The sweep packer places each exact candidate geometry; it is not restricted to ppu_dense_layout's canonical
  // artifact shape. WN=128 remains outside the producer's validated domain, and Q6/TK256 is the one known bad map.
  if (c.wn > 64) return Exclusion::ProducerWarpN;
  if (c.spec.format == Format::Q6_K && c.tk == 256) return Exclusion::ProducerMap;
  return Exclusion::None;
}

// Separate wrappers are intentional.  The emitter asks each launcher for its own answer and a comparator checks them;
// sharing only the current implementation makes equality explicit today without making future drift invisible.
struct DenseSpace {
  static constexpr Exclusion kernel_exclusion(Candidate c) { return common_kernel_exclusion(c); }
  static constexpr Exclusion topology_exclusion(Candidate c, int stages = 2) {
    return common_topology_exclusion(c, stages);
  }
  static constexpr Exclusion sweep_exclusion(Candidate c) { return common_sweep_exclusion(c); }
};
struct GroupedSpace {
  static constexpr Exclusion kernel_exclusion(Candidate c) { return common_kernel_exclusion(c); }
  static constexpr Exclusion topology_exclusion(Candidate c, int stages = 2) {
    return common_topology_exclusion(c, stages);
  }
  static constexpr Exclusion sweep_exclusion(Candidate c) { return common_sweep_exclusion(c); }
};

}  // namespace ppu_tactics
