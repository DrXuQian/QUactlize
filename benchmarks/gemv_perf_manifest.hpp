#pragma once

// Host-only shape and identity authority for the exhaustive GEMV performance
// sweep.  Geometry and quantization semantics are deliberately two different
// objects: applying one shared `gs` to every format would silently turn Q2,
// Q3 and Q6 into non-shipping experiments while labelling them as shipping.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "gemv_lowbit/gemv_tactic_space.hpp"

namespace gemv_perf_manifest {

using ppu_gemv::tactic_space::Candidate;
using ppu_gemv::tactic_space::Format;
using ppu_gemv::tactic_space::Layout;
using ppu_gemv::tactic_space::Route;

enum class QuantOp : std::uint8_t { FinegrainedScaleOnly, FinegrainedScaleZero };
enum class SemanticClass : std::uint8_t { Shipping, ControlledUnshipped, Reference };
enum class ShapeSource : std::uint8_t {
  WorkloadsFixtures,
  HistoricalGrouped,
  ExternalDense,
  DenseReference,
};

struct Geometry {
  char const* id;
  char const* name;
  ShapeSource source;
  Route route;
  int experts;  // grouped expert address-space size; zero for dense
  int rows;     // grouped: routed token count; dense: M
  int n;
  int k;
  int topk;
  int active;
};

struct FormatSemantics {
  Format format;
  int group_size;
  QuantOp quant_op;
  SemanticClass semantic_class;
  char const* note;
};

struct ShapeCase {
  Geometry geometry;
  FormatSemantics semantics;
};

// S068--S079 are checked against workloads.py/fixtures.py, not trusted as a
// second handwritten authority.  The other four rows are explicitly named
// anchors requested by the sweep: one historical grouped comparison and three
// external dense decode projections plus the local 4096-square reference.
inline constexpr std::array<Geometry, 17> kGeometries{{
    {"S068", "T1 35B expert_gate+up", ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 1,  512, 2048, 8,  8},
    {"S069", "T1 122B expert_gate+up",ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 1,  512, 3072, 8,  8},
    {"S070", "T1 35B expert_down",    ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 1, 2048,  512, 8,  8},
    {"S071", "T1 122B expert_down",   ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 1, 3072,  512, 8,  8},
    {"S072", "T2 35B expert_gate+up", ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 2,  512, 2048, 8, 15},
    {"S073", "T2 122B expert_gate+up",ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 2,  512, 3072, 8, 15},
    {"S074", "T2 35B expert_down",    ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 2, 2048,  512, 8, 15},
    {"S075", "T2 122B expert_down",   ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 2, 3072,  512, 8, 15},
    {"S076", "T4 35B expert_gate+up", ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 4,  512, 2048, 8, 30},
    {"S077", "T4 122B expert_gate+up",ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 4,  512, 3072, 8, 30},
    {"S078", "T4 35B expert_down",    ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 4, 2048,  512, 8, 30},
    {"S079", "T4 122B expert_down",   ShapeSource::WorkloadsFixtures, Route::Grouped, 256, 4, 3072,  512, 8, 30},
    {"H-G8-2048", "historical grouped E8 active8", ShapeSource::HistoricalGrouped,
        Route::Grouped, 8, 1, 2048, 2048, 8, 8},
    {"D-EXT-O", "external dense K8192 N5120", ShapeSource::ExternalDense,
        Route::Dense, 0, 1, 5120, 8192, 0, 1},
    {"D-EXT-K1024", "external dense K1024 N5120", ShapeSource::ExternalDense,
        Route::Dense, 0, 1, 5120, 1024, 0, 1},
    {"D-EXT-Q", "external dense K5120 N8192", ShapeSource::ExternalDense,
        Route::Dense, 0, 1, 8192, 5120, 0, 1},
    {"D-4096", "dense M1 N4096 K4096", ShapeSource::DenseReference,
        Route::Dense, 0, 1, 4096, 4096, 0, 1},
}};

// These four rows are shipping semantics.  Int1 has no GGUF shipping qtype;
// it remains in the same controlled ScaleZero/gs16 experiment but is branded
// unshipped in both the shape identity and the output.  It must never inherit
// int4's gs32 label by convenience.
inline constexpr std::array<FormatSemantics, 5> kPrimarySemantics{{
    {Format::Int4, 32, QuantOp::FinegrainedScaleZero, SemanticClass::Shipping,
        "shipping int4 ScaleZero gs32"},
    {Format::Int2, 16, QuantOp::FinegrainedScaleZero, SemanticClass::Shipping,
        "shipping int2 ScaleZero gs16"},
    {Format::Int1, 16, QuantOp::FinegrainedScaleZero, SemanticClass::ControlledUnshipped,
        "controlled synthetic int1 ScaleZero gs16; not a shipping qtype"},
    {Format::Q3,   16, QuantOp::FinegrainedScaleZero, SemanticClass::Shipping,
        "shipping Q3 ScaleZero gs16"},
    {Format::Q6,   16, QuantOp::FinegrainedScaleZero, SemanticClass::Shipping,
        "shipping Q6 ScaleZero gs16"},
}};

inline constexpr FormatSemantics kInt4Gs128ScaleOnlyReference{
    Format::Int4, 128, QuantOp::FinegrainedScaleOnly, SemanticClass::Reference,
    "int4 gs128 ScaleOnly reference; not shipping semantics"};

constexpr char const* name_of(QuantOp v) {
  return v == QuantOp::FinegrainedScaleOnly ? "finegrained_scale_only"
                                             : "finegrained_scale_zero";
}
constexpr char const* name_of(SemanticClass v) {
  switch (v) {
    case SemanticClass::Shipping: return "shipping";
    case SemanticClass::ControlledUnshipped: return "controlled-unshipped";
    case SemanticClass::Reference: return "reference";
  }
  return "unknown";
}
constexpr char const* name_of(ShapeSource v) {
  switch (v) {
    case ShapeSource::WorkloadsFixtures: return "workloads-fixtures";
    case ShapeSource::HistoricalGrouped: return "historical-grouped";
    case ShapeSource::ExternalDense: return "external-dense";
    case ShapeSource::DenseReference: return "dense-reference";
  }
  return "unknown";
}

inline std::vector<ShapeCase> shape_cases() {
  std::vector<ShapeCase> out;
  out.reserve(kGeometries.size() * kPrimarySemantics.size() + 1);
  for (auto const& geometry : kGeometries)
    for (auto const& semantics : kPrimarySemantics)
      out.push_back({geometry, semantics});
  // The gs128 ScaleOnly row is an explicit int4-only comparison with the
  // classic dense Marlin reference, not a second set of pseudo-shipping rows.
  out.push_back({kGeometries.back(), kInt4Gs128ScaleOnlyReference});
  return out;
}

inline int real_rows(Geometry const& g) {
  return g.route == Route::Grouped ? g.rows * g.topk : g.rows;
}

namespace detail {
inline void append_json_string(std::string& out, char const* text) {
  static char const hex[] = "0123456789abcdef";
  out.push_back('"');
  for (unsigned char c : std::string(text ? text : "")) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out.push_back(hex[c >> 4]);
          out.push_back(hex[c & 0x0f]);
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}
}  // namespace detail

// JSON keys are emitted in lexical order.  This makes the text itself the
// canonical identity consumed by gemv_perf_samples.hpp and the Python driver;
// IDs are only readable aliases and cannot hide a missing semantic axis.
inline std::string shape_json(ShapeCase const& c) {
  auto const& g = c.geometry;
  auto const& s = c.semantics;
  std::string out = "{\"active\":" + std::to_string(g.active) +
      ",\"experts\":" + std::to_string(g.experts) + ",\"format\":";
  detail::append_json_string(out, ppu_gemv::tactic_space::name_of(s.format));
  out += ",\"group_size\":" + std::to_string(s.group_size) +
      ",\"k\":" + std::to_string(g.k) +
      ",\"m\":" + std::to_string(g.rows) +
      ",\"n\":" + std::to_string(g.n) +
      ",\"quant_op\":";
  detail::append_json_string(out, name_of(s.quant_op));
  out += ",\"real_rows\":" + std::to_string(real_rows(g)) +
      ",\"route\":";
  detail::append_json_string(out, ppu_gemv::tactic_space::name_of(g.route));
  out += ",\"semantic\":";
  detail::append_json_string(out, name_of(s.semantic_class));
  out += ",\"source\":";
  detail::append_json_string(out, name_of(g.source));
  out += ",\"topk\":" + std::to_string(g.topk) + "}";
  return out;
}

inline std::string shape_id(ShapeCase const& c) {
  std::string out = c.geometry.id;
  out += "/";
  out += ppu_gemv::tactic_space::name_of(c.semantics.format);
  out += "/gs" + std::to_string(c.semantics.group_size) + "/";
  out += c.semantics.quant_op == QuantOp::FinegrainedScaleOnly ? "scale-only" : "scale-zero";
  if (c.semantics.semantic_class != SemanticClass::Shipping) {
    out += "/";
    out += name_of(c.semantics.semantic_class);
  }
  return out;
}

inline std::string config_json(Candidate const& c) {
  std::string out = "{\"chunk\":" + std::to_string(c.chunk) +
      ",\"cta_m\":" + std::to_string(c.cta_m) +
      ",\"cta_n\":" + std::to_string(c.cta_n) +
      ",\"format\":";
  detail::append_json_string(out, ppu_gemv::tactic_space::name_of(c.format));
  out += ",\"layout\":";
  detail::append_json_string(out, ppu_gemv::tactic_space::name_of(c.layout));
  out += ",\"route\":";
  detail::append_json_string(out, ppu_gemv::tactic_space::name_of(c.route));
  out += ",\"step_k\":" + std::to_string(c.step_k) +
      ",\"threads\":" + std::to_string(c.threads) +
      ",\"tile_size_k\":" + std::to_string(c.tile_size_k) + "}";
  return out;
}

inline std::string config_id(Candidate const& c) {
  std::string out = ppu_gemv::tactic_space::name_of(c.format);
  out += "/";
  out += ppu_gemv::tactic_space::name_of(c.layout);
  out += c.layout == Layout::TileK ? std::to_string(c.tile_size_k) : "";
  out += "/s" + std::to_string(c.step_k) + "/t" + std::to_string(c.threads);
  out += "/";
  out += ppu_gemv::tactic_space::name_of(c.route);
  out += "/m" + std::to_string(c.cta_m) + "/n" + std::to_string(c.cta_n);
  out += "/c" + std::to_string(c.chunk);
  return out;
}

// A job fragment for gemv-sweep-manifest-v1.  The caller owns global counts,
// argv/env and shape pruning; this helper owns the complete identities so the
// benchmark writer and manifest cannot silently disagree on an axis.
inline std::string job_json(
    ShapeCase const& shape, std::vector<Candidate> const& candidates,
    char const* argv_value) {
  std::string out = "{\"argv\":[";
  detail::append_json_string(out, argv_value);
  out += "],\"env\":{},\"expected\":[";
  bool first = true;
  for (auto const& c : candidates) {
    if (!first) out.push_back(',');
    first = false;
    out += "{\"config\":" + config_json(c) + ",\"config_id\":";
    auto id = config_id(c);
    detail::append_json_string(out, id.c_str());
    out += ",\"format\":";
    detail::append_json_string(out, ppu_gemv::tactic_space::name_of(c.format));
    out += "}";
  }
  out += "],\"formats\":[";
  detail::append_json_string(out, ppu_gemv::tactic_space::name_of(shape.semantics.format));
  out += "],\"job_id\":";
  auto id = shape_id(shape);
  detail::append_json_string(out, id.c_str());
  out += ",\"shape\":" + shape_json(shape) + ",\"shape_id\":";
  detail::append_json_string(out, id.c_str());
  out += "}";
  return out;
}

}  // namespace gemv_perf_manifest
