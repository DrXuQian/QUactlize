// Host-only authority for the low-bit GEMV tactic search space.
//
// This file owns AXES and exclusions; it deliberately does not instantiate a
// kernel.  Static legality mirrors the compile-time contracts in
// gemv_details.hpp, gemv_wformat.hpp and gemv_kernel.hpp.  Problem-shape
// legality is a separate query: a candidate can be compilable yet inapplicable
// to one (N,K,group-size,route) tuple.
#pragma once

#include <array>
#include <cstdint>

#include "ppu_format_config.hpp"

namespace ppu_gemv::tactic_space {

enum class Format : std::uint8_t { Int4, Int2, Int1, Q3, Q6 };
enum class Layout : std::uint8_t { Native, TileK };
enum class Route : std::uint8_t { Dense, Grouped };

struct FormatTraits {
  Format format;
  char const* name;
  int qtype;
  int low_bits;
  int high_bits;
};

inline constexpr std::array<FormatTraits, 5> kFormats{{
    {Format::Int4, "int4", 12, 4, 0},
    {Format::Int2, "int2", 10, 2, 0},
    {Format::Int1, "int1", -1, 1, 0},  // synthetic: no GGUF shipping qtype
    {Format::Q3,   "q3",   11, 2, 1},
    {Format::Q6,   "q6",   14, 4, 2},
}};

// A layout and its artifact TileSizeK are one axis cell. Native has no
// artifact TileK. The v1 perf ABI deliberately names only the existing
// TileK256 benchmark artifact: TS128 is a correctness probe, while TS32/64
// belong to other producer ABIs. Calling those four one sweep axis would mix
// artifacts that no common producer/runtime contract currently supplies.
struct LayoutTile {
  Layout layout;
  int tile_size_k;
};
inline constexpr std::array<LayoutTile, 2> kLayoutTiles{{
    {Layout::Native, 0},
    {Layout::TileK, 256},
}};

inline constexpr std::array<int, 3> kStepKs{{8, 16, 32}};
inline constexpr std::array<int, 3> kThreads{{64, 128, 256}};
inline constexpr std::array<int, 15> kDenseCtaMs{{
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
inline constexpr std::array<int, 4> kGroupedCtaMs{{1, 2, 3, 4}};
inline constexpr std::array<int, 4> kCtaNs{{2, 4, 8, 16}};
inline constexpr std::array<int, 4> kChunks{{2, 4, 8, 16}};
inline constexpr std::array<int, 5> kCompiledGroupSizes{{0, 16, 32, 64, 128}};

struct Candidate {
  Format format;
  Layout layout;
  int tile_size_k;
  int step_k;
  int threads;
  Route route;
  int cta_m;
  int cta_n;
  int chunk;
};

// First-failure order is ABI: census output and pruning diagnostics use these
// stable names.  Axis-domain errors are retained even though the canonical
// enumerator cannot produce them; callers constructing one candidate directly
// still fail closed with a useful reason.
enum class Exclusion : std::uint8_t {
  None,
  UnknownFormat,
  LayoutTileMismatch,
  StepKOutsideAxis,
  ThreadsOutsideAxis,
  CtaMOutsideRouteAxis,
  CtaNOutsideAxis,
  ChunkOutsideAxis,
  StepTooSmallForSparsestPlane,
  LowPlaneNotWholeWord,
  HighPlaneNotWholeWord,
  TileKNotMultipleOfStepK,
  ThreadsDoNotCoverWholeTileK,
  CtaNNotWholeChunks,
};

constexpr char const* name_of(Format v) {
  for (auto const& f : kFormats) if (f.format == v) return f.name;
  return "unknown";
}
constexpr char const* name_of(Layout v) { return v == Layout::Native ? "native" : "tileK"; }
constexpr char const* name_of(Route v) { return v == Route::Dense ? "dense" : "grouped"; }
constexpr char const* name_of(Exclusion v) {
  switch (v) {
    case Exclusion::None: return "NONE";
    case Exclusion::UnknownFormat: return "UNKNOWN_FORMAT";
    case Exclusion::LayoutTileMismatch: return "LAYOUT_TILE_MISMATCH";
    case Exclusion::StepKOutsideAxis: return "STEP_K_OUTSIDE_AXIS";
    case Exclusion::ThreadsOutsideAxis: return "THREADS_OUTSIDE_AXIS";
    case Exclusion::CtaMOutsideRouteAxis: return "CTA_M_OUTSIDE_ROUTE_AXIS";
    case Exclusion::CtaNOutsideAxis: return "CTA_N_OUTSIDE_AXIS";
    case Exclusion::ChunkOutsideAxis: return "CHUNK_OUTSIDE_AXIS";
    case Exclusion::StepTooSmallForSparsestPlane: return "STEP_TOO_SMALL_FOR_SPARSEST_PLANE";
    case Exclusion::LowPlaneNotWholeWord: return "LOW_PLANE_NOT_WHOLE_WORD";
    case Exclusion::HighPlaneNotWholeWord: return "HIGH_PLANE_NOT_WHOLE_WORD";
    case Exclusion::TileKNotMultipleOfStepK: return "TILE_K_NOT_MULTIPLE_OF_STEP_K";
    case Exclusion::ThreadsDoNotCoverWholeTileK: return "THREADS_DO_NOT_COVER_WHOLE_TILE_K";
    case Exclusion::CtaNNotWholeChunks: return "CTA_N_NOT_WHOLE_CHUNKS";
  }
  return "UNKNOWN_EXCLUSION";
}

template <class Range>
constexpr bool contains(Range const& values, int value) {
  for (int v : values) if (v == value) return true;
  return false;
}

constexpr FormatTraits const* traits_of(Format format) {
  for (auto const& f : kFormats) if (f.format == format) return &f;
  return nullptr;
}

constexpr bool layout_tile_in_axis(Layout layout, int tile_size_k) {
  for (auto const& x : kLayoutTiles)
    if (x.layout == layout && x.tile_size_k == tile_size_k) return true;
  return false;
}

constexpr Exclusion static_exclusion(Candidate const& c) {
  auto const* f = traits_of(c.format);
  if (!f) return Exclusion::UnknownFormat;
  if (!layout_tile_in_axis(c.layout, c.tile_size_k)) return Exclusion::LayoutTileMismatch;
  if (!contains(kStepKs, c.step_k)) return Exclusion::StepKOutsideAxis;
  if (!contains(kThreads, c.threads)) return Exclusion::ThreadsOutsideAxis;
  if (c.route == Route::Dense ? !contains(kDenseCtaMs, c.cta_m)
                              : !contains(kGroupedCtaMs, c.cta_m))
    return Exclusion::CtaMOutsideRouteAxis;
  if (!contains(kCtaNs, c.cta_n)) return Exclusion::CtaNOutsideAxis;
  if (!contains(kChunks, c.chunk)) return Exclusion::ChunkOutsideAxis;

  int const min_bits = f->high_bits ? (f->low_bits < f->high_bits ? f->low_bits : f->high_bits)
                                    : f->low_bits;
  // KernelDetails: every plane needs >= one 32-bit word per lane.
  if (c.step_k * min_bits < 32) return Exclusion::StepTooSmallForSparsestPlane;
  if (c.step_k % (32 / f->low_bits)) return Exclusion::LowPlaneNotWholeWord;
  if (f->high_bits && c.step_k % (32 / f->high_bits)) return Exclusion::HighPlaneNotWholeWord;

  // WFormatTraits<TileK>: one lane step and one CTA must cover whole k-tiles.
  if (c.layout == Layout::TileK) {
    if (c.tile_size_k % c.step_k) return Exclusion::TileKNotMultipleOfStepK;
    if (c.threads % (c.tile_size_k / c.step_k))
      return Exclusion::ThreadsDoNotCoverWholeTileK;
  }

  // gemv_kernel: columns are converted in whole, even-sized chunks.  The
  // canonical chunk axis is already even and >=2, so only divisibility can
  // reject an enumerated cell.
  if (c.cta_n % c.chunk) return Exclusion::CtaNNotWholeChunks;
  return Exclusion::None;
}

enum class ShapeExclusion : std::uint8_t {
  None,
  StaticCandidateRejected,
  RouteMismatch,
  NoRows,
  NNotMultipleOfCtaN,
  KNotMultipleOfStepK,
  GroupSizeNotCompiled,
  KNotMultipleOfGroupSize,
  GroupSizeIncompatibleWithStep,
  KNotMultipleOfTileK,
};

constexpr char const* name_of(ShapeExclusion v) {
  switch (v) {
    case ShapeExclusion::None: return "NONE";
    case ShapeExclusion::StaticCandidateRejected: return "STATIC_CANDIDATE_REJECTED";
    case ShapeExclusion::RouteMismatch: return "ROUTE_MISMATCH";
    case ShapeExclusion::NoRows: return "NO_ROWS";
    case ShapeExclusion::NNotMultipleOfCtaN: return "N_NOT_MULTIPLE_OF_CTA_N";
    case ShapeExclusion::KNotMultipleOfStepK: return "K_NOT_MULTIPLE_OF_STEP_K";
    case ShapeExclusion::GroupSizeNotCompiled: return "GROUP_SIZE_NOT_COMPILED";
    case ShapeExclusion::KNotMultipleOfGroupSize: return "K_NOT_MULTIPLE_OF_GROUP_SIZE";
    case ShapeExclusion::GroupSizeIncompatibleWithStep: return "GROUP_SIZE_INCOMPATIBLE_WITH_STEP";
    case ShapeExclusion::KNotMultipleOfTileK: return "K_NOT_MULTIPLE_OF_TILE_K";
  }
  return "UNKNOWN_SHAPE_EXCLUSION";
}

struct Problem {
  Route route;
  int rows;
  int n;
  int k;
  int group_size;
};

constexpr bool group_step_ok(int group_size, int step_k, int cta_k) {
  if (group_size == 0) return true;
  return (group_size < step_k ? step_k % group_size == 0 : group_size % step_k == 0) &&
         cta_k % group_size == 0;
}

constexpr ShapeExclusion shape_exclusion(Candidate const& c, Problem const& p) {
  if (static_exclusion(c) != Exclusion::None) return ShapeExclusion::StaticCandidateRejected;
  if (p.route != c.route) return ShapeExclusion::RouteMismatch;
  if (p.rows <= 0) return ShapeExclusion::NoRows;
  if (p.n <= 0 || p.n % c.cta_n) return ShapeExclusion::NNotMultipleOfCtaN;
  if (p.k <= 0 || p.k % c.step_k) return ShapeExclusion::KNotMultipleOfStepK;
  if (!contains(kCompiledGroupSizes, p.group_size)) return ShapeExclusion::GroupSizeNotCompiled;
  if (p.group_size > 0 && p.k % p.group_size) return ShapeExclusion::KNotMultipleOfGroupSize;
  if (!group_step_ok(p.group_size, c.step_k, c.step_k * c.threads))
    return ShapeExclusion::GroupSizeIncompatibleWithStep;
  if (c.layout == Layout::TileK && p.k % c.tile_size_k)
    return ShapeExclusion::KNotMultipleOfTileK;
  return ShapeExclusion::None;
}

constexpr std::uint64_t cartesian_size() {
  return std::uint64_t(kFormats.size()) * kLayoutTiles.size() * kStepKs.size() * kThreads.size() *
         (kDenseCtaMs.size() + kGroupedCtaMs.size()) * kCtaNs.size() * kChunks.size();
}
static_assert(cartesian_size() == 27360, "GEMV tactic Cartesian domain changed; update its census gate");

// Production defaults are the four scale-first qtypes shipped by
// ppu_backend.cu.  Int1 deliberately has no anchor: it is a sweep format but
// no ppu_format_config row or public qtype currently ships it.
struct ShippingAnchor {
  Format format;
  int qtype;
  int group_size;
  int step_k;
  int threads;
  int cta_n;
  int chunk;
};

inline constexpr std::array<ShippingAnchor, 4> kShippingAnchors{{
    {Format::Int2, 10, ppu_formats::for_qtype(10).group_size, 16, 128, 8, 2},
    {Format::Q3,   11, ppu_formats::for_qtype(11).group_size, 32,  64, 8, 2},
    {Format::Int4, 12, ppu_formats::for_qtype(12).group_size, 16, 128, 8, 2},
    {Format::Q6,   14, ppu_formats::for_qtype(14).group_size, 16, 128, 8, 2},
}};

constexpr bool anchors_are_static_candidates() {
  for (auto const& a : kShippingAnchors) {
    Candidate const c{a.format, Layout::Native, 0, a.step_k, a.threads,
                      Route::Dense, 1, a.cta_n, a.chunk};
    if (static_exclusion(c) != Exclusion::None) return false;
  }
  return true;
}
static_assert(anchors_are_static_candidates(), "a shipping GEMV default fell outside the tactic authority");

}  // namespace ppu_gemv::tactic_space
