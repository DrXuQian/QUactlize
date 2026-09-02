// One generated binary owns one (qtype, ArtifactTileK, BChunk) shard and any
// number of runtime dense shapes.  Static rejects remain in manifest.json;
// this executable emits every runtime NP/persistent/Split-K coordinate for
// every compiled row and fails closed on missing or extra coordinates.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "ppu_placed_arrangement.hpp"
#include "scalefirst_internal_sweep_bench.hpp"
#include "xplane_offline.hpp"
#include "scalefirst_registry.inc"

#ifndef SCALEFIRST_SWEEP_QTYPE
#error "SCALEFIRST_SWEEP_QTYPE must match the generated registry"
#endif
#ifndef SCALEFIRST_SWEEP_ARTIFACT_TK
#error "SCALEFIRST_SWEEP_ARTIFACT_TK must match the generated registry"
#endif
#ifndef SCALEFIRST_SWEEP_BCHUNK
#error "SCALEFIRST_SWEEP_BCHUNK must match the generated registry"
#endif
#ifndef SCALEFIRST_SWEEP_WEIGHT_LAYOUT
#define SCALEFIRST_SWEEP_WEIGHT_LAYOUT 0
#endif
#ifndef SCALEFIRST_GENERATED_WEIGHT_LAYOUT
// Older exact-selection fixtures predate the layout field in the generated
// registry; their build-axis definition remains the authority.
#define SCALEFIRST_GENERATED_WEIGHT_LAYOUT SCALEFIRST_SWEEP_WEIGHT_LAYOUT
#endif
static_assert(SCALEFIRST_SWEEP_WEIGHT_LAYOUT == 0 ||
                  SCALEFIRST_SWEEP_WEIGHT_LAYOUT == 1 ||
                  SCALEFIRST_SWEEP_WEIGHT_LAYOUT == 2);
static_assert(SCALEFIRST_SWEEP_WEIGHT_LAYOUT != 1 ||
                  (SCALEFIRST_SWEEP_QTYPE == 12 &&
                   SCALEFIRST_SWEEP_ARTIFACT_TK == 0 &&
                   SCALEFIRST_SWEEP_BCHUNK == 0),
              "K-pack4 ScaleFirst target is Q4/A0/bchunk0 only");
static_assert(SCALEFIRST_SWEEP_WEIGHT_LAYOUT != 2 ||
                  ((SCALEFIRST_SWEEP_QTYPE == 10 ||
                    SCALEFIRST_SWEEP_QTYPE == 11 ||
                    SCALEFIRST_SWEEP_QTYPE == 13 ||
                    SCALEFIRST_SWEEP_QTYPE == 14) &&
                   SCALEFIRST_SWEEP_ARTIFACT_TK == 0 &&
                   SCALEFIRST_SWEEP_BCHUNK == 0),
              "generic K-pack ScaleFirst target is Q2/Q3/Q5/Q6 A0/bchunk0 only");
static_assert(SCALEFIRST_SWEEP_QTYPE == SCALEFIRST_GENERATED_QTYPE);
static_assert(SCALEFIRST_SWEEP_ARTIFACT_TK == SCALEFIRST_GENERATED_ARTIFACT_TK);
static_assert(SCALEFIRST_SWEEP_BCHUNK == SCALEFIRST_GENERATED_BCHUNK);
static_assert(SCALEFIRST_SWEEP_WEIGHT_LAYOUT ==
                  SCALEFIRST_GENERATED_WEIGHT_LAYOUT);
static_assert(SCALEFIRST_SWEEP_QTYPE != 8 ||
                  SCALEFIRST_SWEEP_ARTIFACT_TK == 32,
              "Q8 has one canonical A32 artifact");

extern "C" int quactlize_ppu_prepare_dense_for_tile(
    std::uint8_t const*, std::uint8_t const*, std::uint8_t*, std::uint8_t*,
    int, int, int, int);
extern "C" int quactlize_ppu_recover_dense_for_tile(
    std::uint8_t const*, std::uint8_t const*, std::uint8_t*, std::uint8_t*,
    int, int, int, int);

namespace scalefirst_internal_sweep_generated {
#define SCALEFIRST_DECLARE(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN)         \
  bool FN(scalefirst_internal_sweep::DeviceInputs const&,             \
          scalefirst_internal_sweep::Options const&,                  \
          scalefirst_internal_sweep::RowResult&);
SCALEFIRST_REGISTRY_ROWS(SCALEFIRST_DECLARE)
#undef SCALEFIRST_DECLARE
}

namespace {

using namespace scalefirst_internal_sweep;

std::vector<RegistryRow> registry() {
  return {
#define SCALEFIRST_REGISTER(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN)        \
    {#FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN,                              \
     &scalefirst_internal_sweep_generated::FN},
    SCALEFIRST_REGISTRY_ROWS(SCALEFIRST_REGISTER)
#undef SCALEFIRST_REGISTER
  };
}

struct Shape { int m = 1, n = 4096, k = 4096; };
enum class FixtureMode {
  Exact,
  CodeOnly,
  ScaleOnly,
  ZeroOnly,
  MetadataOnly,
  TransportOnly,
  ScaleGroupTag,
  ScaleNTag,
  ZeroGroupTag,
  ZeroNTag,
  CodeK0Tag,
  CodeK1Tag,
  CodeK2Tag,
  CodeK3Tag,
  CodeN0Tag,
  CodeN1Tag,
  CodeN2Tag
};
enum class TagRound {
  None,
  Even,
  Odd,
  Stage,
  EvenNext,
  OddNext
};

char const* fixture_name(FixtureMode mode) {
  switch (mode) {
    case FixtureMode::Exact: return "exact";
    case FixtureMode::CodeOnly: return "code-only";
    case FixtureMode::ScaleOnly: return "scale-only";
    case FixtureMode::ZeroOnly: return "zero-only";
    case FixtureMode::MetadataOnly: return "metadata-only";
    case FixtureMode::TransportOnly: return "transport-only";
    case FixtureMode::ScaleGroupTag: return "scale-group-tag";
    case FixtureMode::ScaleNTag: return "scale-n-tag";
    case FixtureMode::ZeroGroupTag: return "zero-group-tag";
    case FixtureMode::ZeroNTag: return "zero-n-tag";
    case FixtureMode::CodeK0Tag: return "code-k0-tag";
    case FixtureMode::CodeK1Tag: return "code-k1-tag";
    case FixtureMode::CodeK2Tag: return "code-k2-tag";
    case FixtureMode::CodeK3Tag: return "code-k3-tag";
    case FixtureMode::CodeN0Tag: return "code-n0-tag";
    case FixtureMode::CodeN1Tag: return "code-n1-tag";
    case FixtureMode::CodeN2Tag: return "code-n2-tag";
  }
  return "unknown";
}

char const* tag_round_name(TagRound round) {
  switch (round) {
    case TagRound::None: return "none";
    case TagRound::Even: return "even";
    case TagRound::Odd: return "odd";
    case TagRound::Stage: return "stage";
    case TagRound::EvenNext: return "even-next";
    case TagRound::OddNext: return "odd-next";
  }
  return "unknown";
}

constexpr bool is_tag_fixture(FixtureMode mode) {
  return mode == FixtureMode::ScaleGroupTag || mode == FixtureMode::ScaleNTag ||
         mode == FixtureMode::ZeroGroupTag || mode == FixtureMode::ZeroNTag ||
         mode == FixtureMode::CodeK0Tag || mode == FixtureMode::CodeK1Tag ||
         mode == FixtureMode::CodeK2Tag || mode == FixtureMode::CodeK3Tag ||
         mode == FixtureMode::CodeN0Tag || mode == FixtureMode::CodeN1Tag ||
         mode == FixtureMode::CodeN2Tag;
}

constexpr bool is_code_tag(FixtureMode mode) {
  return mode == FixtureMode::CodeK0Tag || mode == FixtureMode::CodeK1Tag ||
         mode == FixtureMode::CodeK2Tag || mode == FixtureMode::CodeK3Tag ||
         mode == FixtureMode::CodeN0Tag || mode == FixtureMode::CodeN1Tag ||
         mode == FixtureMode::CodeN2Tag;
}

struct Cli {
  int iterations = 5, repeats = 2;
  std::uint64_t schedule_seed = UINT64_C(0x6a09e667f3bcc909);
  unsigned algorithm_mask = Options::kAllAlgorithms;
  FixtureMode fixture_mode = FixtureMode::Exact;
  TagRound tag_round = TagRound::None;
  bool fixture_binding = false;
  std::string symbol_file;
  std::vector<Shape> shapes;
};

bool parse_shape(char const* text, Shape& shape) {
  char tail = 0;
  return std::sscanf(text, "%dx%dx%d%c", &shape.m, &shape.n, &shape.k,
                     &tail) == 3 && shape.m > 0 && shape.n > 0 && shape.k > 0;
}

bool parse_cli(int argc, char** argv, Cli& cli) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strncmp(argv[i], "--shape=", 8)) {
      Shape shape;
      if (!parse_shape(argv[i] + 8, shape)) return false;
      cli.shapes.push_back(shape);
    } else if (!std::strncmp(argv[i], "--iterations=", 13)) {
      cli.iterations = std::atoi(argv[i] + 13);
    } else if (!std::strncmp(argv[i], "--correctness-repeats=", 22)) {
      cli.repeats = std::atoi(argv[i] + 22);
    } else if (!std::strncmp(argv[i], "--schedule-seed=", 16)) {
      char* end = nullptr;
      cli.schedule_seed = std::strtoull(argv[i] + 16, &end, 0);
      if (!end || *end) return false;
    } else if (!std::strncmp(argv[i], "--algorithm=", 12)) {
      char const* value = argv[i] + 12;
      if (!std::strcmp(value, "all"))
        cli.algorithm_mask = Options::kAllAlgorithms;
      else if (!std::strcmp(value, "nonpersistent"))
        cli.algorithm_mask = Options::kNonPersistent;
      else if (!std::strcmp(value, "persistent"))
        cli.algorithm_mask = Options::kPersistent;
      else if (!std::strcmp(value, "split"))
        cli.algorithm_mask = Options::kSplitK;
      else if (!std::strcmp(value, "full-output"))
        cli.algorithm_mask = Options::kNonPersistent | Options::kPersistent;
      else return false;
    } else if (!std::strncmp(argv[i], "--fixture=", 10)) {
      char const* value = argv[i] + 10;
      if (!std::strcmp(value, "exact"))
        cli.fixture_mode = FixtureMode::Exact;
      else if (!std::strcmp(value, "code-only"))
        cli.fixture_mode = FixtureMode::CodeOnly;
      else if (!std::strcmp(value, "scale-only"))
        cli.fixture_mode = FixtureMode::ScaleOnly;
      else if (!std::strcmp(value, "zero-only"))
        cli.fixture_mode = FixtureMode::ZeroOnly;
      else if (!std::strcmp(value, "metadata-only"))
        cli.fixture_mode = FixtureMode::MetadataOnly;
      else if (!std::strcmp(value, "transport-only"))
        cli.fixture_mode = FixtureMode::TransportOnly;
      else if (!std::strcmp(value, "scale-group-tag"))
        cli.fixture_mode = FixtureMode::ScaleGroupTag;
      else if (!std::strcmp(value, "scale-n-tag"))
        cli.fixture_mode = FixtureMode::ScaleNTag;
      else if (!std::strcmp(value, "zero-group-tag"))
        cli.fixture_mode = FixtureMode::ZeroGroupTag;
      else if (!std::strcmp(value, "zero-n-tag"))
        cli.fixture_mode = FixtureMode::ZeroNTag;
      else if (!std::strcmp(value, "code-k0-tag"))
        cli.fixture_mode = FixtureMode::CodeK0Tag;
      else if (!std::strcmp(value, "code-k1-tag"))
        cli.fixture_mode = FixtureMode::CodeK1Tag;
      else if (!std::strcmp(value, "code-k2-tag"))
        cli.fixture_mode = FixtureMode::CodeK2Tag;
      else if (!std::strcmp(value, "code-k3-tag"))
        cli.fixture_mode = FixtureMode::CodeK3Tag;
      else if (!std::strcmp(value, "code-n0-tag"))
        cli.fixture_mode = FixtureMode::CodeN0Tag;
      else if (!std::strcmp(value, "code-n1-tag"))
        cli.fixture_mode = FixtureMode::CodeN1Tag;
      else if (!std::strcmp(value, "code-n2-tag"))
        cli.fixture_mode = FixtureMode::CodeN2Tag;
      else return false;
    } else if (!std::strncmp(argv[i], "--tag-round=", 12)) {
      char const* value = argv[i] + 12;
      if (!std::strcmp(value, "even"))
        cli.tag_round = TagRound::Even;
      else if (!std::strcmp(value, "odd"))
        cli.tag_round = TagRound::Odd;
      else if (!std::strcmp(value, "stage"))
        cli.tag_round = TagRound::Stage;
      else if (!std::strcmp(value, "even-next"))
        cli.tag_round = TagRound::EvenNext;
      else if (!std::strcmp(value, "odd-next"))
        cli.tag_round = TagRound::OddNext;
      else return false;
    } else if (!std::strcmp(argv[i], "--fixture-binding")) {
      cli.fixture_binding = true;
    } else if (!std::strncmp(argv[i], "--symbol-file=", 14)) {
      cli.symbol_file = argv[i] + 14;
      if (cli.symbol_file.empty()) return false;
    } else return false;
  }
  if (cli.shapes.empty()) cli.shapes.push_back({1, 4096, 4096});
  bool const tag_contract = is_tag_fixture(cli.fixture_mode) ?
      cli.tag_round != TagRound::None && cli.fixture_binding :
      cli.tag_round == TagRound::None;
  return cli.iterations > 0 && cli.repeats > 0 && cli.algorithm_mask != 0 &&
         tag_contract;
}

bool selected_registry(Cli const& cli, std::vector<RegistryRow>& selected,
                       std::string& error) {
  auto const all = registry();
  if (cli.symbol_file.empty()) {
    selected = all;
    return true;
  }
  std::ifstream stream(cli.symbol_file);
  if (!stream) {
    error = "cannot open symbol file: " + cli.symbol_file;
    return false;
  }
  std::unordered_set<std::string> requested;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.find_first_of(" \t") != std::string::npos) {
      error = "symbol file contains an empty/whitespace-bearing record";
      return false;
    }
    if (!requested.insert(line).second) {
      error = "symbol file contains a duplicate: " + line;
      return false;
    }
  }
  if (!stream.eof() || requested.empty()) {
    error = requested.empty() ? "symbol file is empty" :
                                "failed while reading symbol file";
    return false;
  }
  for (auto const& row : all) {
    auto found = requested.find(row.symbol);
    if (found != requested.end()) {
      selected.push_back(row);
      requested.erase(found);
    }
  }
  if (!requested.empty()) {
    error = "symbol file names an unknown generated symbol: " +
            *requested.begin();
    return false;
  }
  return true;
}

constexpr int low_bits() {
  return SCALEFIRST_SWEEP_QTYPE == 8 ? 8 :
         SCALEFIRST_SWEEP_QTYPE == 10 || SCALEFIRST_SWEEP_QTYPE == 11 ? 2 : 4;
}
constexpr int high_bits() {
  return SCALEFIRST_SWEEP_QTYPE == 11 || SCALEFIRST_SWEEP_QTYPE == 13 ? 1 :
         SCALEFIRST_SWEEP_QTYPE == 14 ? 2 : 0;
}
constexpr int group_size() {
  return SCALEFIRST_SWEEP_QTYPE == 8 || SCALEFIRST_SWEEP_QTYPE == 12 ||
         SCALEFIRST_SWEEP_QTYPE == 13 ? 32 : 16;
}
constexpr bool has_zero() {
  return SCALEFIRST_SWEEP_QTYPE >= 10 && SCALEFIRST_SWEEP_QTYPE <= 14;
}
constexpr int converter_zero_multiplier() {
  return SCALEFIRST_SWEEP_QTYPE == 11 ? -4 :
         SCALEFIRST_SWEEP_QTYPE == 14 ? -24 : 0;
}

int code_value(int n, int k) {
  int const logical = ((13 * n + 7 * k + 3) % 15) - 7;
  switch (SCALEFIRST_SWEEP_QTYPE) {
    case 8: return logical + 128;
    case 10: return (logical + 8) & 3;
    case 11: return logical < -4 ? 0 : logical > 3 ? 7 : logical + 4;
    case 12: return logical + 8;
    // Exercise the Q5 high plane on every fixture.  The ScaleZero converter
    // still decodes code-8; choosing 17..31 changes only the test data, not
    // that bias convention.  The previous 1..15 range left the entire high
    // plane zero and let a disconnected high-plane reader pass.
    case 13: return logical + 24;
    case 14: return logical + 32;
  }
  return 0;
}

int decoded_value(int code) {
  switch (SCALEFIRST_SWEEP_QTYPE) {
    case 8: return code - 128;
    case 10: return code;
    case 11: return code - 4;
    case 12: return code - 8;
    case 13: return code - 8;
    case 14: return code - 32;
  }
  return 0;
}

int code_for_decoded_one() {
  switch (SCALEFIRST_SWEEP_QTYPE) {
    case 8: return 129;
    case 10: return 1;
    case 11: return 5;
    case 12: return 9;
    case 13: return 9;
    case 14: return 33;
  }
  return 0;
}

int code_for_decoded_zero() {
  switch (SCALEFIRST_SWEEP_QTYPE) {
    case 8: return 128;
    case 10: return 0;
    case 11: return 4;
    case 12: return 8;
    case 13: return 8;
    case 14: return 32;
  }
  return 0;
}

constexpr bool fixture_uses_constant_code(FixtureMode mode) {
  return mode != FixtureMode::Exact && mode != FixtureMode::CodeOnly &&
         !is_code_tag(mode);
}

constexpr bool fixture_uses_varied_scale(FixtureMode mode) {
  return mode == FixtureMode::Exact || mode == FixtureMode::ScaleOnly ||
         mode == FixtureMode::MetadataOnly ||
         mode == FixtureMode::ScaleGroupTag || mode == FixtureMode::ScaleNTag;
}

constexpr bool fixture_uses_varied_zero(FixtureMode mode) {
  return mode == FixtureMode::Exact || mode == FixtureMode::ZeroOnly ||
         mode == FixtureMode::MetadataOnly ||
         mode == FixtureMode::ZeroGroupTag || mode == FixtureMode::ZeroNTag;
}

int fixture_code(FixtureMode mode, int n, int k) {
  if (mode == FixtureMode::ZeroGroupTag || mode == FixtureMode::ZeroNTag)
    return code_for_decoded_zero();
  if (mode == FixtureMode::CodeK0Tag) return (k >> 0) & 15;
  if (mode == FixtureMode::CodeK1Tag) return (k >> 4) & 15;
  if (mode == FixtureMode::CodeK2Tag) return (k >> 8) & 15;
  if (mode == FixtureMode::CodeK3Tag) return (k >> 12) & 15;
  if (mode == FixtureMode::CodeN0Tag) return (n >> 0) & 15;
  if (mode == FixtureMode::CodeN1Tag) return (n >> 4) & 15;
  if (mode == FixtureMode::CodeN2Tag) return (n >> 8) & 15;
  return fixture_uses_constant_code(mode) ? code_for_decoded_one() :
                                            code_value(n, k);
}

void put_native(std::vector<std::uint8_t>& plane, int bits, int n, int k,
                int K, int value) {
  std::uint64_t const bit = (std::uint64_t(n) * K + k) * bits;
  plane[bit >> 3] |= std::uint8_t(value << (bit & 7));
}

template <bool Recover>
int transform_generic_kpack(std::uint8_t const* low_in,
                            std::uint8_t const* high_in,
                            std::uint8_t* low_out,
                            std::uint8_t* high_out,
                            int n, int k) {
  if constexpr (SCALEFIRST_SWEEP_QTYPE == 10)
    return kquant_kpack::transform<2, 0, 16, Recover>(
        low_in, high_in, low_out, high_out, n, k);
  if constexpr (SCALEFIRST_SWEEP_QTYPE == 11)
    return kquant_kpack::transform<2, 1, 16, Recover>(
        low_in, high_in, low_out, high_out, n, k);
  if constexpr (SCALEFIRST_SWEEP_QTYPE == 13)
    return kquant_kpack::transform<4, 1, 32, Recover>(
        low_in, high_in, low_out, high_out, n, k);
  if constexpr (SCALEFIRST_SWEEP_QTYPE == 14)
    return kquant_kpack::transform<4, 2, 16, Recover>(
        low_in, high_in, low_out, high_out, n, k);
  return 25;
}

struct Fixture {
  std::vector<half_t> a, scales, zeros, golden;
  std::vector<std::uint8_t> low_native, high_native, low, high;
  std::vector<int> probe_k;
  bool exact = false, roundtrip = false, high_plane_covered = false;
  bool isolation_covered = false;
};

std::uint64_t fnv1a_bytes(void const* data, std::size_t bytes) {
  auto const* p = static_cast<std::uint8_t const*>(data);
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

template <class T>
std::uint64_t fixture_hash(std::vector<T> const& values) {
  return fnv1a_bytes(values.data(), values.size() * sizeof(T));
}

Fixture make_fixture(Shape shape, FixtureMode mode, TagRound tag_round) {
  Fixture f;
  int constexpr LB = low_bits(), HB = high_bits(), GS = group_size();
  if (is_tag_fixture(mode) &&
      (SCALEFIRST_SWEEP_QTYPE != 12 ||
       SCALEFIRST_SWEEP_ARTIFACT_TK != 32 || !has_zero()))
    return f;
  std::size_t const codes = std::size_t(shape.n) * shape.k;
  f.a.assign(std::size_t(shape.m) * shape.k, half_t(0.f));
  f.scales.resize(std::size_t(shape.k / GS) * shape.n);
  if constexpr (has_zero()) f.zeros.resize(f.scales.size());
  f.golden.resize(std::size_t(shape.m) * shape.n);
  f.low_native.assign(codes * LB / 8, 0);
  f.high_native.assign(HB ? codes * HB / 8 : 0, 0);
  f.low.resize(f.low_native.size());
  f.high.resize(f.high_native.size());

  std::vector<std::vector<int>> active_by_m(std::size_t(shape.m));
  // Ordinary diagnostic fixtures use one exact nonzero in each K eighth.
  // Coordinate-tag fixtures instead use one A=1 impulse per M row.  Even and
  // odd rounds cover all 128 local K coordinates; even-next/odd-next repeat
  // those coordinates in the second TK128 tile; stage covers all 40 TK128
  // tiles of the exact 64x1024x5120 target.  The output is therefore the tag
  // attached to the B coordinate actually paired with that A coordinate.
  std::array<int, 8> active{};
  if (is_tag_fixture(mode)) {
    bool const next_tile = tag_round == TagRound::EvenNext ||
                           tag_round == TagRound::OddNext;
    if (shape.m != 64 || shape.k < (next_tile ? 256 : 128) ||
        tag_round == TagRound::None)
      return f;
    int const k_tiles = shape.k / 128;
    f.probe_k.resize(std::size_t(shape.m));
    for (int m = 0; m < shape.m; ++m) {
      int k = tag_round == TagRound::Even ? 2 * m :
              tag_round == TagRound::Odd ? 2 * m + 1 :
              tag_round == TagRound::EvenNext ? 128 + 2 * m :
              tag_round == TagRound::OddNext ? 129 + 2 * m :
              (m % k_tiles) * 128 + 11;
      f.probe_k[std::size_t(m)] = k;
      active_by_m[std::size_t(m)].push_back(k);
      f.a[std::size_t(m) * shape.k + k] = half_t(1.f);
    }
  } else {
    for (int s = 0; s < 8; ++s) {
      int const begin = s * shape.k / 8;
      int const span = shape.k / 8;
      active[s] = begin + ((37 * s + 11) % span);
      for (int m = 0; m < shape.m; ++m) {
        active_by_m[std::size_t(m)].push_back(active[s]);
        f.a[std::size_t(m) * shape.k + active[s]] =
            half_t(mode == FixtureMode::TransportOnly ?
                       ((m & 1) ? -0.5f : 0.5f) :
                       (((m + s) & 1) ? -0.5f : 0.5f));
      }
    }
  }
  for (int g = 0; g < shape.k / GS; ++g)
    for (int n = 0; n < shape.n; ++n) {
      float scale = !fixture_uses_varied_scale(mode) ? 1.f :
                    float(1 << ((17 * g + 29 * n + 1) % 3));
      if (mode == FixtureMode::ScaleGroupTag) scale = float(g + 1);
      if (mode == FixtureMode::ScaleNTag) scale = float(n + 1);
      f.scales[std::size_t(g) * shape.n + n] = half_t(scale);
      if constexpr (has_zero()) {
        float affine_zero = !fixture_uses_varied_zero(mode) ? 0.f :
                            float(((11 * g + 7 * n) % 3 - 1) * 3);
        if (mode == FixtureMode::ZeroGroupTag) affine_zero = float(g + 1);
        if (mode == FixtureMode::ZeroNTag) affine_zero = float(n + 1);
        f.zeros[std::size_t(g) * shape.n + n] = half_t(
            affine_zero + converter_zero_multiplier() * scale);
      }
    }

  std::vector<std::uint8_t> kmajor(codes);
  for (int n = 0; n < shape.n; ++n)
    for (int k = 0; k < shape.k; ++k) {
      int const code = fixture_code(mode, n, k);
      kmajor[std::size_t(k) * shape.n + n] = std::uint8_t(code);
      put_native(f.low_native, LB, n, k, shape.k,
                 code & ((1 << LB) - 1));
      if constexpr (HB != 0)
        put_native(f.high_native, HB, n, k, shape.k, code >> LB);
    }
  bool const code_varied = std::adjacent_find(
      kmajor.begin(), kmajor.end(), std::not_equal_to<std::uint8_t>{}) !=
      kmajor.end();
  bool const code_is_one = std::all_of(
      kmajor.begin(), kmajor.end(), [](std::uint8_t code) {
        return int(code) == code_for_decoded_one();
      });
  bool const scale_varied = std::adjacent_find(
      f.scales.begin(), f.scales.end(), [](half_t a, half_t b) {
        return float(a) != float(b);
      }) != f.scales.end();
  bool const scale_is_one = std::all_of(
      f.scales.begin(), f.scales.end(), [](half_t value) {
        return float(value) == 1.f;
      });
  bool zero_varied = !has_zero();
  bool zero_is_zero = !has_zero();
  if constexpr (has_zero()) {
    std::vector<float> affine_zero(f.zeros.size());
    for (std::size_t i = 0; i < f.zeros.size(); ++i)
      affine_zero[i] = float(f.zeros[i]) -
          converter_zero_multiplier() * float(f.scales[i]);
    zero_varied = std::adjacent_find(
        affine_zero.begin(), affine_zero.end(), std::not_equal_to<float>{}) !=
        affine_zero.end();
    zero_is_zero = std::all_of(
        affine_zero.begin(), affine_zero.end(),
        [](float value) { return value == 0.f; });
  }
  if (is_tag_fixture(mode)) {
    bool const code_is_zero = std::all_of(
        kmajor.begin(), kmajor.end(), [](std::uint8_t code) {
          return int(code) == code_for_decoded_zero();
        });
    bool const scale_tag = mode == FixtureMode::ScaleGroupTag ||
                           mode == FixtureMode::ScaleNTag;
    bool const zero_tag = mode == FixtureMode::ZeroGroupTag ||
                          mode == FixtureMode::ZeroNTag;
    bool const tag_code_ok = scale_tag ? code_is_one :
                             zero_tag ? code_is_zero :
                             code_varied;
    bool const tag_scale_ok = scale_tag ? scale_varied : scale_is_one;
    bool const tag_zero_ok = zero_tag ? zero_varied : zero_is_zero;
    f.isolation_covered = f.probe_k.size() == std::size_t(shape.m) &&
                          tag_code_ok && tag_scale_ok && tag_zero_ok;
  } else {
    bool const code_covered = fixture_uses_constant_code(mode) ?
        code_is_one : code_varied;
    bool const scale_covered = fixture_uses_varied_scale(mode) ?
        scale_varied : scale_is_one;
    bool const zero_covered = fixture_uses_varied_zero(mode) ?
        zero_varied : zero_is_zero;
    f.isolation_covered = code_covered && scale_covered && zero_covered;
  }
#if SCALEFIRST_SWEEP_WEIGHT_LAYOUT == 1
  {
    auto const arrangement = ppu_arrangements::q4_kpack4_transpose_v1();
    std::vector<std::uint8_t> direct(f.low.size(), std::uint8_t(0xcd));
    int const direct_rc = q4_kpack4::prepare(
        f.low_native.data(), direct.data(), shape.n, shape.k);
    int const abi_rc = quactlize_ppu_prepare_dense_for_arrangement_v2(
        f.low_native.data(), nullptr, f.low.data(), nullptr,
        shape.n, shape.k, SCALEFIRST_SWEEP_QTYPE, &arrangement);
    if (direct_rc != 0 || abi_rc != 0 || direct != f.low) return f;
    std::vector<std::uint8_t> direct_back(f.low_native.size(),
                                          std::uint8_t(0xab));
    std::vector<std::uint8_t> low_back(f.low_native.size());
    int const direct_recover_rc = q4_kpack4::recover(
        f.low.data(), direct_back.data(), shape.n, shape.k);
    int const abi_recover_rc = quactlize_ppu_recover_dense_for_arrangement_v2(
        f.low.data(), nullptr, low_back.data(), nullptr,
        shape.n, shape.k, SCALEFIRST_SWEEP_QTYPE, &arrangement);
    f.roundtrip = direct_recover_rc == 0 && abi_recover_rc == 0 &&
                  direct_back == low_back && low_back == f.low_native;
  }
#elif SCALEFIRST_SWEEP_WEIGHT_LAYOUT == 2
  {
    auto const arrangement =
        ppu_arrangements::kquant_kpack_transpose_v1(
            SCALEFIRST_SWEEP_QTYPE);
    std::vector<std::uint8_t> direct_low(f.low.size(), std::uint8_t(0xcd));
    std::vector<std::uint8_t> direct_high(
        f.high.size(), std::uint8_t(0xcd));
    int const direct_rc = transform_generic_kpack<false>(
        f.low_native.data(), HB ? f.high_native.data() : nullptr,
        direct_low.data(), HB ? direct_high.data() : nullptr,
        shape.n, shape.k);
    int const abi_rc = quactlize_ppu_prepare_dense_for_arrangement_v2(
        f.low_native.data(), HB ? f.high_native.data() : nullptr,
        f.low.data(), HB ? f.high.data() : nullptr,
        shape.n, shape.k, SCALEFIRST_SWEEP_QTYPE, &arrangement);
    if (direct_rc != 0 || abi_rc != 0 || direct_low != f.low ||
        direct_high != f.high) return f;
    std::vector<std::uint8_t> direct_low_back(f.low_native.size(),
                                              std::uint8_t(0xab));
    std::vector<std::uint8_t> direct_high_back(f.high_native.size(),
                                               std::uint8_t(0xab));
    std::vector<std::uint8_t> low_back(f.low_native.size());
    std::vector<std::uint8_t> high_back(f.high_native.size());
    int const direct_recover_rc = transform_generic_kpack<true>(
        f.low.data(), HB ? f.high.data() : nullptr,
        direct_low_back.data(), HB ? direct_high_back.data() : nullptr,
        shape.n, shape.k);
    int const abi_recover_rc = quactlize_ppu_recover_dense_for_arrangement_v2(
        f.low.data(), HB ? f.high.data() : nullptr,
        low_back.data(), HB ? high_back.data() : nullptr,
        shape.n, shape.k, SCALEFIRST_SWEEP_QTYPE, &arrangement);
    f.roundtrip = direct_recover_rc == 0 && abi_recover_rc == 0 &&
                  direct_low_back == low_back && low_back == f.low_native &&
                  direct_high_back == high_back &&
                  high_back == f.high_native;
  }
#else
#if SCALEFIRST_SWEEP_QTYPE == 8
  {
    xplane::place_derived<8,64,64,32,32,32,1,32>(
        reinterpret_cast<std::int8_t*>(f.low.data()), kmajor,
        shape.n, shape.k);
    std::vector<std::uint8_t> back;
    xplane::recover_derived<8,64,64,32,32,32,1,32>(
        reinterpret_cast<std::int8_t const*>(f.low.data()), back,
        shape.n, shape.k);
    f.roundtrip = back == kmajor;
  }
#else
  {
    if (quactlize_ppu_prepare_dense_for_tile(
            f.low_native.data(), HB ? f.high_native.data() : nullptr,
            f.low.data(), HB ? f.high.data() : nullptr,
            shape.n, shape.k, SCALEFIRST_SWEEP_QTYPE,
            SCALEFIRST_SWEEP_ARTIFACT_TK) != 0) return f;
    std::vector<std::uint8_t> low_back(f.low_native.size());
    std::vector<std::uint8_t> high_back(f.high_native.size());
    f.roundtrip = quactlize_ppu_recover_dense_for_tile(
        f.low.data(), HB ? f.high.data() : nullptr,
        low_back.data(), HB ? high_back.data() : nullptr,
        shape.n, shape.k, SCALEFIRST_SWEEP_QTYPE,
        SCALEFIRST_SWEEP_ARTIFACT_TK) == 0 &&
        low_back == f.low_native && high_back == f.high_native;
  }
#endif
#endif
  if constexpr (HB == 0) {
    f.high_plane_covered = true;
  } else if (fixture_uses_constant_code(mode)) {
    // These arms intentionally hold the decoded code constant; high-plane
    // coverage belongs to exact/code-only, not to component isolation.
    f.high_plane_covered = true;
  } else {
    f.high_plane_covered = std::any_of(
        f.high_native.begin(), f.high_native.end(),
        [](std::uint8_t value) { return value != 0; });
  }

  bool exact = true;
  for (int m = 0; m < shape.m; ++m)
    for (int n = 0; n < shape.n; ++n) {
      float sum = 0;
      for (int k : active_by_m[std::size_t(m)]) {
        int const g = k / GS;
        float const scale = float(f.scales[std::size_t(g) * shape.n + n]);
        float const zero = has_zero() ?
            float(f.zeros[std::size_t(g) * shape.n + n]) -
                converter_zero_multiplier() * scale : 0.f;
        sum += float(f.a[std::size_t(m) * shape.k + k]) *
            (scale * decoded_value(fixture_code(mode, n, k)) + zero);
      }
      half_t rounded(sum);
      exact &= float(rounded) == sum;
      f.golden[std::size_t(m) * shape.n + n] = rounded;
    }
  f.exact = exact;
  return f;
}

template <class T>
void copy_to(cutlass::DeviceAllocation<T>& allocation,
             std::vector<T> const& source) {
  if (!source.empty()) allocation.copy_from_host(source.data());
}

void print_samples(std::vector<double> const& samples) {
  std::printf("[");
  for (std::size_t i = 0; i < samples.size(); ++i)
    std::printf("%s%.9f", i ? "," : "", samples[i]);
  std::printf("]");
}

bool dump_tag_output(cutlass::DeviceAllocation<half_t> const& output,
                     Fixture const& fixture, Shape shape,
                     FixtureMode mode, TagRound round) {
  std::vector<half_t> host(std::size_t(shape.m) * shape.n);
  if (hggcMemcpy(host.data(), output.get(), host.size() * sizeof(half_t),
                 hggcMemcpyDeviceToHost) != hggcSuccess)
    return false;
  int const row_count = std::min(shape.m, 64);
  int const col_count = std::min(shape.n, 64);
  std::printf("SF_TAG_ROWS mode=%s tag_round=%s n=0 count=%d got=",
              fixture_name(mode), tag_round_name(round), row_count);
  for (int m = 0; m < row_count; ++m)
    std::printf("%s%04x", m ? "," : "",
                unsigned(host[std::size_t(m) * shape.n].raw()));
  std::printf(" want=");
  for (int m = 0; m < row_count; ++m)
    std::printf("%s%04x", m ? "," : "",
                unsigned(fixture.golden[std::size_t(m) * shape.n].raw()));
  std::printf("\n");
  std::printf("SF_TAG_COLS mode=%s tag_round=%s m=0 count=%d got=",
              fixture_name(mode), tag_round_name(round), col_count);
  for (int n = 0; n < col_count; ++n)
    std::printf("%s%04x", n ? "," : "", unsigned(host[std::size_t(n)].raw()));
  std::printf(" want=");
  for (int n = 0; n < col_count; ++n)
    std::printf("%s%04x", n ? "," : "",
                unsigned(fixture.golden[std::size_t(n)].raw()));
  std::printf("\n");
  std::fflush(stdout);
  return true;
}

int run_shape(Shape shape, Cli const& cli, int device, int cu,
              std::vector<RegistryRow> const& rows) {
  if (shape.n % 256 || shape.k % 256 || shape.k % 8) {
    std::fprintf(stderr, "shape %dx%dx%d violates resident/split alignment\n",
                 shape.m, shape.n, shape.k);
    return 2;
  }
  Fixture fixture = make_fixture(shape, cli.fixture_mode, cli.tag_round);
  if (cli.fixture_binding) {
    std::printf(
        "SF_FIXTURE mode=%s first_golden=0x%04x tag_round=%s "
        "probe_count=%zu probe_fnv=%016llx "
        "a_fnv=%016llx low_native_fnv=%016llx low_placed_fnv=%016llx "
        "scale_fnv=%016llx zero_fnv=%016llx golden_fnv=%016llx "
        "roundtrip=%d exact=%d isolation=%d\n",
        fixture_name(cli.fixture_mode),
        fixture.golden.empty() ? 0u : unsigned(fixture.golden[0].raw()),
        tag_round_name(cli.tag_round), fixture.probe_k.size(),
        static_cast<unsigned long long>(fixture_hash(fixture.probe_k)),
        static_cast<unsigned long long>(fixture_hash(fixture.a)),
        static_cast<unsigned long long>(fixture_hash(fixture.low_native)),
        static_cast<unsigned long long>(fixture_hash(fixture.low)),
        static_cast<unsigned long long>(fixture_hash(fixture.scales)),
        static_cast<unsigned long long>(fixture_hash(fixture.zeros)),
        static_cast<unsigned long long>(fixture_hash(fixture.golden)),
        int(fixture.roundtrip), int(fixture.exact),
        int(fixture.isolation_covered));
    std::fflush(stdout);
  }
  if (!fixture.roundtrip || !fixture.exact || !fixture.high_plane_covered ||
      !fixture.isolation_covered) {
    std::fprintf(stderr,
                 "fixture failed roundtrip=%d exact=%d high_plane_covered=%d "
                 "isolation_covered=%d\n",
                 int(fixture.roundtrip), int(fixture.exact),
                 int(fixture.high_plane_covered),
                 int(fixture.isolation_covered));
    return 2;
  }
  cutlass::DeviceAllocation<half_t> dA(fixture.a.size());
  cutlass::DeviceAllocation<std::uint8_t> dLow(fixture.low.size());
  cutlass::DeviceAllocation<std::uint8_t> dHigh(fixture.high.size());
  cutlass::DeviceAllocation<half_t> dScale(fixture.scales.size());
  cutlass::DeviceAllocation<half_t> dZero(fixture.zeros.size());
  cutlass::DeviceAllocation<half_t> dOutput(std::size_t(shape.m) * shape.n);
  std::size_t const workspace_bytes =
      std::size_t(shape.m) * shape.n * 8 * sizeof(float) + 4096;
  cutlass::DeviceAllocation<char> dWorkspace(workspace_bytes);
  copy_to(dA, fixture.a); copy_to(dLow, fixture.low); copy_to(dHigh, fixture.high);
  copy_to(dScale, fixture.scales); copy_to(dZero, fixture.zeros);
  DeviceInputs inputs{
      dA.get(), dLow.get(), fixture.high.empty() ? nullptr : dHigh.get(),
      dScale.get(), fixture.zeros.empty() ? nullptr : dZero.get(),
      dOutput.get(), dWorkspace.get(), workspace_bytes, fixture.golden.data(),
      shape.m, shape.n, shape.k, device, cu};
  Options options{cli.iterations, cli.repeats, true, cli.algorithm_mask};
  std::vector<std::size_t> order(rows.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 rng(cli.schedule_seed ^ std::uint64_t(shape.m) ^
                      (std::uint64_t(shape.n) << 17) ^
                      (std::uint64_t(shape.k) << 33));
  std::shuffle(order.begin(), order.end(), rng);
  std::size_t runtime_cells = 0, measured_cells = 0, records = 0;
  for (std::size_t ordinal = 0; ordinal < order.size(); ++ordinal) {
    auto const& registry_row = rows[order[ordinal]];
    RowResult result;
    std::printf("SF_ATTEMPT shape=%dx%dx%d ordinal=%zu/%zu symbol=%s\n",
                shape.m, shape.n, shape.k, ordinal + 1, order.size(),
                registry_row.symbol);
    std::fflush(stdout);
    bool const row_ok = registry_row.run(inputs, options, result);
    if (is_tag_fixture(cli.fixture_mode) && !result.cells.empty() &&
        (row_ok || result.cells.back().raw_bad != 0) &&
        !dump_tag_output(dOutput, fixture, shape, cli.fixture_mode,
                         cli.tag_round)) {
      std::fprintf(stderr,
                   "SF_FATAL symbol=%s shape=%dx%dx%d algorithm=TAG-DUMP "
                   "state=OUTPUT_COPY step=TAG_OUTPUT_COPY\n",
                   registry_row.symbol, shape.m, shape.n, shape.k);
      return 1;
    }
    if (!row_ok) {
      if (result.cells.empty()) {
        std::fprintf(stderr,
                     "SF_FATAL symbol=%s shape=%dx%dx%d algorithm=NONE "
                     "state=INPUT_OR_SETUP step=BEFORE_CELL\n",
                     registry_row.symbol, shape.m, shape.n, shape.k);
      } else {
        auto const& failed = result.cells.back();
        std::fprintf(
            stderr,
            "SF_FATAL symbol=%s shape=%dx%dx%d algorithm=%s state=%s "
            "step=%s repeat=%d raw_bad=%llu first_bad=%zu "
            "want=0x%04x got=0x%04x\n",
            registry_row.symbol, shape.m, shape.n, shape.k,
            failed.algorithm, state_name(failed.state), failed.failure_step,
            failed.failure_repeat,
            static_cast<unsigned long long>(failed.raw_bad),
            failed.first_bad_index, unsigned(failed.first_bad_want),
            unsigned(failed.first_bad_got));
      }
      return 1;
    }
    runtime_cells += result.cells.size();
    for (auto const& cell : result.cells) {
      bool const measured = cell.state == State::Measured;
      measured_cells += measured;
      int const sample_count = measured ? int(cell.samples_us.size()) : 1;
      if (measured && sample_count != cli.iterations) {
        std::fprintf(stderr, "sample denominator drift for %s/%s\n",
                     registry_row.symbol, cell.algorithm);
        return 1;
      }
      for (int sample = 0; sample < sample_count; ++sample) {
        double const us = measured ? cell.samples_us[std::size_t(sample)] : 0.;
        double const tflops = us > 0 ?
            (2. * shape.m * shape.n * shape.k) / (us * 1.e6) : 0.;
        double const mfu = tflops / 500. * 100.;
        double const distinct_bytes =
            double(shape.m) * shape.k * 2. +
            double(shape.n) * shape.k * (low_bits() + high_bits()) / 8. +
            double(shape.n) * (shape.k / group_size()) *
                (has_zero() ? 4. : 2.) +
            double(shape.m) * shape.n * 2.;
        double const mbu = us > 0 ? distinct_bytes / (us * 1.e3) / 2766. * 100. : 0.;
        std::printf(
            "SF_CELL {\"shape\":\"%dx%dx%d\",\"qtype\":%d,"
            "\"artifact_tile_k\":%d,\"bchunk\":%d,\"symbol\":\"%s\","
            "\"a_provider\":%d,\"resolved_delivery_n\":%d,"
            "\"config\":\"%dx%dx%d_w%dx%d_s%d_bc%d_ap%d_dn%d\","
            "\"algorithm\":\"%s\",\"metric_scope\":\"%s\","
            "\"policy\":\"%s\",\"split\":%d,\"grid\":%d,"
            "\"occupancy\":%d,\"capacity_b_mask\":\"0x%llx\","
            "\"balanced_b_mask\":\"0x%llx\",\"status\":\"%s\","
            "\"reason\":\"%s\",\"sample\":%d,\"sample_us\":%.9f,"
            "\"MFU_pct\":%.9f,\"distinct_MBU_model_pct\":%.9f,"
            "\"raw_bad\":%llu,\"fingerprint\":\"0x%llx\","
            "\"reducer_correctness_untimed\":%d,\"partial_bytes\":%zu,"
            "\"shipping_smem\":%zu,\"persistent_smem\":%zu,"
            "\"split_smem\":%zu,\"execution_ordinal\":%zu}\n",
            shape.m, shape.n, shape.k, SCALEFIRST_SWEEP_QTYPE,
            SCALEFIRST_SWEEP_ARTIFACT_TK, SCALEFIRST_SWEEP_BCHUNK,
            registry_row.symbol, registry_row.a_provider,
            registry_row.resolved_delivery_n, registry_row.tm, registry_row.tn,
            registry_row.tk, registry_row.wm, registry_row.wn,
            registry_row.stages, registry_row.bchunk,
            registry_row.a_provider, registry_row.resolved_delivery_n,
            cell.algorithm,
            cell.metric_scope, cell.policy, cell.split, cell.grid,
            cell.occupancy,
            static_cast<unsigned long long>(cell.capacity_b_mask),
            static_cast<unsigned long long>(cell.balanced_b_mask),
            measured ? "MEASURED" : "INADMISSIBLE", state_name(cell.state),
            sample, us, mfu, mbu,
            static_cast<unsigned long long>(cell.raw_bad),
            static_cast<unsigned long long>(cell.fingerprint),
            int(cell.reducer_correctness_untimed), cell.partial_bytes,
            cell.shipping_smem, cell.persistent_smem, cell.split_smem, ordinal);
        ++records;
      }
    }
  }
  std::printf(
      "SF_COMPLETE status=COMPLETE shape=%dx%dx%d typed_rows=%zu "
      "runtime_cells=%zu measured_cells=%zu records=%zu iterations=%d "
      "fixture=ORDER-INDEPENDENT+FP16-EXACT fixture_mode=%s roundtrip=PASS "
      "high_plane_coverage=PASS isolation_coverage=PASS\n",
      shape.m, shape.n, shape.k, rows.size(), runtime_cells, measured_cells,
      records, cli.iterations, fixture_name(cli.fixture_mode));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Cli cli;
  if (!parse_cli(argc, argv, cli)) {
    std::fprintf(stderr,
        "usage: %s [--shape=MxNxK] [--iterations=N] "
        "[--correctness-repeats=N] [--schedule-seed=N] "
        "[--algorithm=all|nonpersistent|persistent|split|full-output] "
        "[--fixture=exact|code-only|scale-only|zero-only|metadata-only|transport-only|"
        "scale-{group,n}-tag|zero-{group,n}-tag|"
        "code-k{0,1,2,3}-tag|code-n{0,1,2}-tag] "
        "[--tag-round=even|odd|stage|even-next|odd-next] [--fixture-binding] "
        "[--symbol-file=PATH]\n", argv[0]);
    return 2;
  }
  std::vector<RegistryRow> rows;
  std::string selection_error;
  if (!selected_registry(cli, rows, selection_error)) {
    std::fprintf(stderr, "SF_SELECTION_FAIL reason=%s\n",
                 selection_error.c_str());
    return 2;
  }
  int device = 0;
  if (hggcGetDevice(&device) != hggcSuccess) return 2;
  int const cu = cutlass::KernelHardwareInfo::
      query_device_multiprocessor_count(device);
  if (cu <= 0) return 2;
  std::printf(
      "SF_SHARD qtype=%d artifact_tile_k=%d bchunk=%d typed_rows=%d "
      "weight_layout=%d weight_mapping_id=0x%016llx "
      "selected_rows=%zu algorithm_mask=0x%x device=%d cu=%d "
      "iterations=%d correctness_repeats=%d "
      "schedule_seed=0x%llx\n",
      SCALEFIRST_SWEEP_QTYPE, SCALEFIRST_SWEEP_ARTIFACT_TK,
      SCALEFIRST_SWEEP_BCHUNK, SCALEFIRST_GENERATED_TYPED_ROWS,
      SCALEFIRST_SWEEP_WEIGHT_LAYOUT,
      static_cast<unsigned long long>(
          SCALEFIRST_SWEEP_WEIGHT_LAYOUT == 1 ? q4_kpack4::kMappingId :
          SCALEFIRST_SWEEP_WEIGHT_LAYOUT == 2 ? kquant_kpack::kMappingId : 0),
      rows.size(), cli.algorithm_mask, device, cu, cli.iterations, cli.repeats,
      static_cast<unsigned long long>(cli.schedule_seed));
  for (auto const& shape : cli.shapes) {
    int const rc = run_shape(shape, cli, device, cu, rows);
    if (rc) return rc;
  }
  return 0;
}
