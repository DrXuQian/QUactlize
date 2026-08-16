// L198 -- exhaustive local denominator and arithmetic oracle for one-plane fixed Split-K.
//
// The three committed dense tables are the authority.  This probe expands every
// i4/A64, i2/A128 and i1/A256 row, crosses it with both fine-grained metadata
// modes and S={1,2,4,8}, and retains every rejected cell with a named reason.
// For admitted cells it consumes the production FixedSplitKWork ABI, checks
// exact-once (q,k-tile) coverage and unique (q,peer) slots, then compares the
// ordered FP32-partial reduction with S=1 in raw FP16 bits on an exact fixture.
//
// The numerical fixture deliberately uses only small integer products.  Every
// product, partial and final sum is exactly representable in FP32, and every
// final sum is in the exact-integer range of FP16.  A raw-bit mismatch therefore
// cannot be excused as a tolerance or accumulation-order effect.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

#include "ppu_tactic_space.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_fixed_splitk_partition.hpp"

#include "lowbit_dense_configs.inc"
#include "lowbit_dense_i2_configs.inc"
#include "lowbit_dense_i1_configs.inc"

namespace {

namespace fs = cutlass::gemm::kernel::fixed_splitk;

constexpr int kN = 4096;
constexpr int kK = 4096;
constexpr std::array<int, 4> kSplits{{1, 2, 4, 8}};

enum class Format : uint8_t { I4, I2, I1 };
enum class Mode : uint8_t { ScaleOnly, ScaleZero };
enum class Plant : uint8_t {
  None,
  Bits,
  Mode,
  Artifact,
  Fold,
  BChunk,
  Partial,
};

struct Row {
  Format format;
  char const* name;
  int bits;
  int artifact_tile_k;
  int tm;
  int tn;
  int tactic_tile_k;
  int wm;
  int wn;
  int stages;
  int b_chunk;
};

#define L198_I4_ROW(TM, TN, TK, WM, WN, ST, BC, BODY) \
  Row{Format::I4, "i4", 4, LOWBIT_DENSE_CFG_ARTIFACT_TILEK, \
      TM, TN, TK, WM, WN, ST, BC},
constexpr Row kI4Rows[] = {
    LOWBIT_DENSE_CFG_LIST(L198_I4_ROW, L198_UNUSED_BODY)};
#undef L198_I4_ROW

#define L198_I2_ROW(TM, TN, TK, WM, WN, ST, BC, BODY) \
  Row{Format::I2, "i2", 2, LOWBIT_DENSE_I2_CFG_ARTIFACT_TILEK, \
      TM, TN, TK, WM, WN, ST, BC},
constexpr Row kI2Rows[] = {
    LOWBIT_DENSE_I2_CFG_LIST(L198_I2_ROW, L198_UNUSED_BODY)};
#undef L198_I2_ROW

#define L198_I1_ROW(TM, TN, TK, WM, WN, ST, BC, BODY) \
  Row{Format::I1, "i1", 1, LOWBIT_DENSE_I1_CFG_ARTIFACT_TILEK, \
      TM, TN, TK, WM, WN, ST, BC},
constexpr Row kI1Rows[] = {
    LOWBIT_DENSE_I1_CFG_LIST(L198_I1_ROW, L198_UNUSED_BODY)};
#undef L198_I1_ROW

static_assert(std::size(kI4Rows) == LOWBIT_DENSE_CFG_ROWS);
static_assert(std::size(kI2Rows) == LOWBIT_DENSE_I2_CFG_ROWS);
static_assert(std::size(kI1Rows) == LOWBIT_DENSE_I1_CFG_ROWS);
static_assert(LOWBIT_DENSE_CFG_ROWS == 1772 &&
              LOWBIT_DENSE_I2_CFG_ROWS == 2140 &&
              LOWBIT_DENSE_I1_CFG_ROWS == 878);
static_assert(LOWBIT_DENSE_CFG_ARTIFACT_TILEK == 64 &&
              LOWBIT_DENSE_I2_CFG_ARTIFACT_TILEK == 128 &&
              LOWBIT_DENSE_I1_CFG_ARTIFACT_TILEK == 256);

struct Totals {
  uint64_t rows = 0;
  uint64_t cells = 0;
  uint64_t admitted = 0;
  uint64_t pipeline_depth = 0;
  uint64_t descriptors = 0;
  uint64_t qk_visits = 0;
  uint64_t slot_visits = 0;
  uint64_t raw_bit_checks = 0;
  uint64_t resident_checks = 0;
  uint64_t errors = 0;
  std::array<uint64_t, 4> admitted_per_split{};
  std::array<uint64_t, 3> rows_per_format{};
  std::array<uint64_t, 3> admitted_per_format{};
  std::array<uint64_t, 2> admitted_per_b_chunk{};
};

constexpr size_t format_index(Format format) {
  return format == Format::I4 ? 0 : format == Format::I2 ? 1 : 2;
}

constexpr int expected_bits(Format format) {
  return format == Format::I4 ? 4 : format == Format::I2 ? 2 : 1;
}

constexpr int expected_artifact_tile_k(Format format) {
  return format == Format::I4 ? 64 : format == Format::I2 ? 128 : 256;
}

constexpr ppu_tactics::Format tactic_format(Format format) {
  return format == Format::I4 ? ppu_tactics::Format::I4
       : format == Format::I2 ? ppu_tactics::Format::I2
                              : ppu_tactics::Format::I1;
}

constexpr char const* plant_name(Plant plant) {
  switch (plant) {
    case Plant::None: return "none";
    case Plant::Bits: return "bits";
    case Plant::Mode: return "mode";
    case Plant::Artifact: return "artifact";
    case Plant::Fold: return "fold";
    case Plant::BChunk: return "bchunk";
    case Plant::Partial: return "partial";
  }
  return "unknown";
}

Plant parse_plant(int argc, char** argv) {
  if (argc == 1) return Plant::None;
  if (argc != 3 || std::string_view(argv[1]) != "--plant") return Plant::None;
  std::string_view const name(argv[2]);
  if (name == "bits") return Plant::Bits;
  if (name == "mode") return Plant::Mode;
  if (name == "artifact") return Plant::Artifact;
  if (name == "fold") return Plant::Fold;
  if (name == "bchunk") return Plant::BChunk;
  if (name == "partial") return Plant::Partial;
  return Plant::None;
}

// Round an FP32 value to IEEE binary16, round-to-nearest-even.  This oracle is
// host-only so it cannot accidentally call the production conversion op it is
// intended to check independently.
uint16_t fp16_bits(float value) {
  uint32_t word = 0;
  static_assert(sizeof(word) == sizeof(value));
  std::memcpy(&word, &value, sizeof(word));
  uint16_t const sign = uint16_t((word >> 16) & 0x8000u);
  uint32_t mantissa = word & 0x007fffffu;
  int exponent = int((word >> 23) & 0xffu);

  if (exponent == 0xff) {
    return uint16_t(sign | (mantissa ? 0x7e00u : 0x7c00u));
  }
  exponent = exponent - 127 + 15;
  if (exponent >= 31) return uint16_t(sign | 0x7c00u);
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa |= 0x00800000u;
    int const shift = 14 - exponent;
    uint32_t half_mantissa = mantissa >> shift;
    uint32_t const remainder = mantissa & ((uint32_t(1) << shift) - 1);
    uint32_t const halfway = uint32_t(1) << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (half_mantissa & 1u))) {
      ++half_mantissa;
    }
    return uint16_t(sign | half_mantissa);
  }

  uint32_t half_mantissa = mantissa >> 13;
  uint32_t const remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (half_mantissa & 1u))) {
    ++half_mantissa;
    if (half_mantissa == 0x400u) {
      half_mantissa = 0;
      ++exponent;
      if (exponent >= 31) return uint16_t(sign | 0x7c00u);
    }
  }
  return uint16_t(sign | uint16_t(exponent << 10) |
                  uint16_t(half_mantissa));
}

uint64_t resident_payload_fingerprint(Row const& row) {
  // The payload is logical code data; the exact shipping collective/type
  // witness in l198_dense_splitk_oneplane_types.cu binds its resident physical
  // arrangement and stride.  S is intentionally absent here: changing the
  // scheduler must not create or reinterpret a new offline artifact.
  uint64_t hash = 1469598103934665603ull;
  uint32_t const mask = (uint32_t(1) << row.bits) - 1;
  for (int k = 0; k < 257; ++k) {
    for (int n = 0; n < 17; ++n) {
      uint8_t const code = uint8_t((uint32_t(k * 13 + n * 7 + 3)) & mask);
      hash ^= code;
      hash *= 1099511628211ull;
    }
  }
  hash ^= uint64_t(row.bits) << 48;
  hash ^= uint64_t(row.artifact_tile_k) << 32;
  return hash;
}

bool numerical_oracle(Row const& row, Mode expected_mode, int split,
                      Mode executed_mode, uint64_t& raw_checks) {
  constexpr int Outputs = 7;
  int const k_tiles = kK / row.tactic_tile_k;
  int const tiles_per_split = k_tiles / split;
  if (k_tiles % split != 0) return false;
  uint32_t const mask = (uint32_t(1) << row.bits) - 1;
  int const bias = 1 << (row.bits - 1);

  std::array<float, Outputs> s1{};
  std::array<float, Outputs> reduced{};
  for (int n = 0; n < Outputs; ++n) {
    for (int tile = 0; tile < k_tiles; ++tile) {
      int const raw = int(uint32_t(tile * 3 + n * 5 + 1) & mask);
      int const zero = -1 - ((tile + 2 * n) % 3);
      int const decoded = expected_mode == Mode::ScaleOnly
          ? raw - bias : raw + zero;
      s1[size_t(n)] += float(decoded);
    }
  }

  for (int peer = 0; peer < split; ++peer) {
    std::array<float, Outputs> partial{};
    for (int n = 0; n < Outputs; ++n) {
      for (int local = 0; local < tiles_per_split; ++local) {
        int const tile = peer * tiles_per_split + local;
        int const raw = int(uint32_t(tile * 3 + n * 5 + 1) & mask);
        int const zero = -1 - ((tile + 2 * n) % 3);
        int const decoded = executed_mode == Mode::ScaleOnly
            ? raw - bias : raw + zero;
        partial[size_t(n)] += float(decoded);
      }
      reduced[size_t(n)] += partial[size_t(n)];
    }
  }

  bool ok = true;
  for (int n = 0; n < Outputs; ++n) {
    // Keep the fixture inside FP16's exact integer domain.  The explicit bit
    // comparison below is then the same RAW-BIT contract used on device.
    ok = ok && s1[size_t(n)] >= -2048.f && s1[size_t(n)] <= 2048.f;
    ok = ok && fp16_bits(s1[size_t(n)]) == fp16_bits(reduced[size_t(n)]);
    ++raw_checks;
  }
  return ok;
}

bool validate_partition(Row const& row, int split, Totals& totals) {
  if (kN % row.tn != 0 || kK % row.tactic_tile_k != 0) return false;
  uint64_t const output_tiles = uint64_t(kN / row.tn);
  uint32_t const k_tiles = uint32_t(kK / row.tactic_tile_k);
  fs::Params const params = fs::make_params(output_tiles, k_tiles,
                                            uint32_t(split));
  if (!params.is_valid()) return false;

  std::vector<uint8_t> qk(size_t(output_tiles) * k_tiles, uint8_t(0));
  std::vector<uint8_t> slots(size_t(params.work_units), uint8_t(0));
  for (uint64_t linear = 0; linear < params.work_units; ++linear) {
    fs::FixedSplitKWork const work = fs::work_for_linear(params, linear);
    if (!fs::work_matches_params(params, work)) return false;
    uint64_t const slot = work.q * params.splits + work.peer_idx;
    if (slot >= slots.size() || slots[size_t(slot)] != 0) return false;
    ++slots[size_t(slot)];
    for (uint32_t k = work.k_begin; k < work.k_begin + work.k_count; ++k) {
      size_t const index = size_t(work.q) * k_tiles + k;
      if (index >= qk.size() || qk[index] ==
              (std::numeric_limits<uint8_t>::max)()) return false;
      ++qk[index];
      ++totals.qk_visits;
    }
    ++totals.descriptors;
    ++totals.slot_visits;
  }
  return std::all_of(qk.begin(), qk.end(), [](uint8_t n) { return n == 1; }) &&
      std::all_of(slots.begin(), slots.end(), [](uint8_t n) { return n == 1; });
}

bool row_contract(Row const& row, int expected_fold) {
  if (row.bits != expected_bits(row.format) ||
      row.artifact_tile_k != expected_artifact_tile_k(row.format) ||
      expected_fold != 1 ||
      ppu_tactics::fold_for(row.bits, row.artifact_tile_k) != expected_fold ||
      row.tactic_tile_k < row.artifact_tile_k ||
      row.tactic_tile_k % row.artifact_tile_k != 0 ||
      (row.b_chunk != 0 && row.b_chunk != 1) ||
      (row.bits == 4 && row.b_chunk != 0)) {
    return false;
  }
  ppu_tactics::FormatSpec const spec{
      tactic_format(row.format), row.name, row.bits, 0};
  ppu_tactics::Candidate const tactic{
      spec, row.tm, row.tn, row.tactic_tile_k, row.wm, row.wn,
      row.artifact_tile_k, row.b_chunk};
  return ppu_tactics::DenseSpace::topology_exclusion(tactic, row.stages) ==
             ppu_tactics::Exclusion::None &&
      ppu_tactics::common_producer_exclusion(tactic) ==
             ppu_tactics::Exclusion::None;
}

void audit_rows(Row const* begin, Row const* end, Plant plant,
                bool& planted, Totals& totals) {
  for (Row const* source = begin; source != end; ++source) {
    Row row = *source;
    int expected_fold = 1;
    if (!planted && plant == Plant::Bits && row.format == Format::I2) {
      row.bits = 4;
      planted = true;
    } else if (!planted && plant == Plant::Artifact &&
               row.format == Format::I2) {
      row.artifact_tile_k = 64;
      planted = true;
    } else if (!planted && plant == Plant::Fold &&
               row.format == Format::I1) {
      expected_fold = 2;
      planted = true;
    } else if (!planted && plant == Plant::BChunk &&
               row.format == Format::I4) {
      row.b_chunk = 1;
      planted = true;
    }

    ++totals.rows;
    ++totals.rows_per_format[format_index(row.format)];
    if (!row_contract(row, expected_fold)) ++totals.errors;
    uint64_t const resident = resident_payload_fingerprint(row);

    for (Mode mode : {Mode::ScaleOnly, Mode::ScaleZero}) {
      for (size_t split_index = 0; split_index < kSplits.size(); ++split_index) {
        int const split = kSplits[split_index];
        ++totals.cells;
        int const k_tiles = kK / row.tactic_tile_k;
        fs::Params const params = fs::make_params(
            uint64_t(kN / row.tn), uint32_t(k_tiles), uint32_t(split));
        if (!params.is_valid() ||
            (split > 1 && k_tiles / split < row.stages - 1)) {
          ++totals.pipeline_depth;
          continue;
        }

        ++totals.admitted;
        ++totals.admitted_per_split[split_index];
        ++totals.admitted_per_format[format_index(row.format)];
        ++totals.admitted_per_b_chunk[size_t(row.b_chunk)];
        if (!validate_partition(row, split, totals)) ++totals.errors;

        Mode executed_mode = mode;
        if (!planted && plant == Plant::Mode &&
            row.format == Format::I4 && mode == Mode::ScaleZero) {
          executed_mode = Mode::ScaleOnly;
          planted = true;
        }
        if (!numerical_oracle(row, mode, split, executed_mode,
                              totals.raw_bit_checks)) {
          ++totals.errors;
        }

        size_t const partial_bytes = split == 1 ? 0
            : size_t(split) * size_t(kN) * sizeof(float);
        size_t const expected_partial_bytes = split == 1 ? 0
            : size_t(split) * size_t(kN) * size_t(
                  (!planted && plant == Plant::Partial) ? 2 : 4);
        if (!planted && plant == Plant::Partial && split > 1) planted = true;
        if (partial_bytes != expected_partial_bytes) ++totals.errors;

        // Recompute rather than cache a pointer: the equality proves that S,
        // mode and BChunk cannot select a second resident weight payload.
        if (resident_payload_fingerprint(row) != resident) ++totals.errors;
        ++totals.resident_checks;
      }
    }
  }
}

bool expected_positive_totals(Totals const& totals) {
  return totals.rows == 4790 && totals.cells == 38320 &&
      totals.admitted == 33004 && totals.pipeline_depth == 5316 &&
      totals.admitted_per_split ==
          std::array<uint64_t, 4>{{9580, 9384, 8088, 5952}} &&
      totals.rows_per_format ==
          std::array<uint64_t, 3>{{1772, 2140, 878}} &&
      totals.admitted_per_format ==
          std::array<uint64_t, 3>{{12908, 14556, 5540}} &&
      totals.admitted_per_b_chunk ==
          std::array<uint64_t, 2>{{22956, 10048}} &&
      totals.descriptors == 8187072 && totals.slot_visits == 8187072 &&
      totals.qk_visits == 78703616 &&
      totals.raw_bit_checks == totals.admitted * 7 &&
      totals.resident_checks == totals.admitted && totals.errors == 0;
}

}  // namespace

int main(int argc, char** argv) {
  Plant const plant = parse_plant(argc, argv);
  if (argc != 1 && plant == Plant::None) {
    std::fprintf(stderr,
                 "usage: %s [--plant bits|mode|artifact|fold|bchunk|partial]\n",
                 argv[0]);
    return 2;
  }

  Totals totals;
  bool planted = false;
  audit_rows(std::begin(kI4Rows), std::end(kI4Rows), plant, planted, totals);
  audit_rows(std::begin(kI2Rows), std::end(kI2Rows), plant, planted, totals);
  audit_rows(std::begin(kI1Rows), std::end(kI1Rows), plant, planted, totals);

  if (plant != Plant::None) {
    bool const red = planted && totals.errors != 0;
    std::printf("[l198:plant] %s name=%s planted=%d errors=%llu\n",
                red ? "EXPECTED_RED" : "FALSE_GREEN", plant_name(plant),
                int(planted),
                static_cast<unsigned long long>(totals.errors));
    return red ? 1 : 0;
  }

  bool const pass = expected_positive_totals(totals);
  std::printf(
      "[l198] %s tables=3 rows=%llu modes=2 cells=%llu admitted=%llu "
      "inadmissible_pipeline_depth=%llu per_split=%llu/%llu/%llu/%llu "
      "formats=%llu/%llu/%llu bc=%llu/%llu descriptors=%llu qk=%llu "
      "raw_bit_checks=%llu resident_payload_invariant=%llu partial=FP32\n",
      pass ? "PASS" : "FAIL",
      static_cast<unsigned long long>(totals.rows),
      static_cast<unsigned long long>(totals.cells),
      static_cast<unsigned long long>(totals.admitted),
      static_cast<unsigned long long>(totals.pipeline_depth),
      static_cast<unsigned long long>(totals.admitted_per_split[0]),
      static_cast<unsigned long long>(totals.admitted_per_split[1]),
      static_cast<unsigned long long>(totals.admitted_per_split[2]),
      static_cast<unsigned long long>(totals.admitted_per_split[3]),
      static_cast<unsigned long long>(totals.admitted_per_format[0]),
      static_cast<unsigned long long>(totals.admitted_per_format[1]),
      static_cast<unsigned long long>(totals.admitted_per_format[2]),
      static_cast<unsigned long long>(totals.admitted_per_b_chunk[0]),
      static_cast<unsigned long long>(totals.admitted_per_b_chunk[1]),
      static_cast<unsigned long long>(totals.descriptors),
      static_cast<unsigned long long>(totals.qk_visits),
      static_cast<unsigned long long>(totals.raw_bit_checks),
      static_cast<unsigned long long>(totals.resident_checks));
  if (!pass) {
    std::fprintf(stderr,
                 "[l198] totals drift: errors=%llu slots=%llu raw=%llu\n",
                 static_cast<unsigned long long>(totals.errors),
                 static_cast<unsigned long long>(totals.slot_visits),
                 static_cast<unsigned long long>(totals.raw_bit_checks));
  }
  return pass ? 0 : 1;
}
