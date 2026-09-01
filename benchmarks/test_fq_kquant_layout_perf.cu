// Production-C-ABI Xplane vs K-pack performance closure for K-quants.  The
// production runner measures Q2/Q3/Q5/Q6 dense+grouped and Q4 grouped; Q4
// dense already has its own real-shape K-pack4 closure.
//
// One format-selected binary contains both physical readers.  Every timed
// pair therefore shares compiler, code inventory, input values, config,
// stream and event policy; only the arrangement descriptor and resident code
// bytes differ.  Offline placement, fixture construction, correctness and
// warmup are outside every event span.

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "gguf_packed_unit.hpp"
#include "moe_router_fixture.hpp"
#include "ppu_dense_shipping_policy.hpp"
#include "ppu_format_config.hpp"
#include "ppu_placed_arrangement.hpp"
#include "quactlize_ppu_config.h"
#include "quactlize_ppu_device.h"

#ifndef FQ_KQUANT_PERF_QTYPE
#error "FQ_KQUANT_PERF_QTYPE must select Q2/Q3/Q4/Q5/Q6"
#endif
static_assert(FQ_KQUANT_PERF_QTYPE == 10 ||
              FQ_KQUANT_PERF_QTYPE == 11 ||
              FQ_KQUANT_PERF_QTYPE == 12 ||
              FQ_KQUANT_PERF_QTYPE == 13 ||
              FQ_KQUANT_PERF_QTYPE == 14);

namespace {

using half_t = cutlass::half_t;
constexpr int kQtype = FQ_KQUANT_PERF_QTYPE;

template <int Q> struct Format;
template <> struct Format<10> {
  static constexpr auto Type = gguf_scale::KType::Q2_K;
  static constexpr int LowBits = 2, HighBits = 0, Group = 16;
};
template <> struct Format<11> {
  static constexpr auto Type = gguf_scale::KType::Q3_K;
  static constexpr int LowBits = 2, HighBits = 1, Group = 16;
};
template <> struct Format<12> {
  static constexpr auto Type = gguf_scale::KType::Q4_K;
  static constexpr int LowBits = 4, HighBits = 0, Group = 32;
};
template <> struct Format<13> {
  static constexpr auto Type = gguf_scale::KType::Q5_K;
  static constexpr int LowBits = 4, HighBits = 1, Group = 32;
};
template <> struct Format<14> {
  static constexpr auto Type = gguf_scale::KType::Q6_K;
  static constexpr int LowBits = 4, HighBits = 2, Group = 16;
};
using F = Format<kQtype>;

struct DenseCase { int m = 0, n = 0, k = 0; };
struct GroupedCase { int tokens = 0, n = 0, k = 0, experts = 0, topk = 0; };

struct Cli {
  int iterations = 11;
  int warmups = 3;
  int round = 1;
  bool kpack_first = false;
  bool all_configs = false;
  bool policy_v2 = false;
  std::vector<DenseCase> dense;
  std::vector<GroupedCase> grouped;
};

bool positive(char const* text, int& value) {
  char* end = nullptr;
  long parsed = std::strtol(text, &end, 10);
  if (!text[0] || !end || *end || parsed <= 0 || parsed > INT32_MAX)
    return false;
  value = int(parsed);
  return true;
}

template <class Case>
bool parse_tuple(char const* text, int fields, Case& value) {
  std::vector<int> out;
  std::string copy(text ? text : "");
  std::size_t begin = 0;
  while (begin <= copy.size()) {
    std::size_t const end = copy.find(',', begin);
    std::string const token = copy.substr(
        begin, end == std::string::npos ? end : end - begin);
    int parsed = 0;
    if (!positive(token.c_str(), parsed)) return false;
    out.push_back(parsed);
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  if (int(out.size()) != fields) return false;
  if constexpr (std::is_same_v<Case, DenseCase>)
    value = {out[0], out[1], out[2]};
  else
    value = {out[0], out[1], out[2], out[3], out[4]};
  return true;
}

bool parse_cli(int argc, char** argv, Cli& cli) {
  for (int i = 1; i < argc; ++i) {
    auto value = [&](char const* prefix) -> char const* {
      std::size_t const n = std::strlen(prefix);
      return std::strncmp(argv[i], prefix, n) == 0 ? argv[i] + n : nullptr;
    };
    if (char const* v = value("--iterations=")) {
      if (!positive(v, cli.iterations)) return false;
    } else if (char const* v = value("--warmups=")) {
      if (!positive(v, cli.warmups)) return false;
    } else if (char const* v = value("--round=")) {
      if (!positive(v, cli.round)) return false;
    } else if (char const* v = value("--order=")) {
      if (!std::strcmp(v, "xplane-first")) cli.kpack_first = false;
      else if (!std::strcmp(v, "kpack-first")) cli.kpack_first = true;
      else return false;
    } else if (char const* v = value("--all-configs=")) {
      if (!std::strcmp(v, "0")) cli.all_configs = false;
      else if (!std::strcmp(v, "1")) cli.all_configs = true;
      else return false;
    } else if (char const* v = value("--profile=")) {
      if (!std::strcmp(v, "kpack-policy-v2")) cli.policy_v2 = true;
      else return false;
    } else if (char const* v = value("--dense=")) {
      DenseCase row;
      if (!parse_tuple(v, 3, row)) return false;
      cli.dense.push_back(row);
    } else if (char const* v = value("--grouped=")) {
      GroupedCase row;
      if (!parse_tuple(v, 5, row)) return false;
      cli.grouped.push_back(row);
    } else return false;
  }
  if (cli.dense.empty() && cli.grouped.empty()) return false;
  for (auto const& row : cli.dense)
    if (row.n % 256 || row.k % 256) return false;
  for (auto const& row : cli.grouped)
    if (row.n % 256 || row.k % 256 || row.topk > row.experts) return false;
  if (!cli.grouped.empty())
    for (auto const& row : cli.grouped)
      if (row.experts != cli.grouped.front().experts) return false;
  if (cli.policy_v2 &&
      (kQtype != 12 || !cli.kpack_first || !cli.all_configs ||
       !cli.grouped.empty() ||
       cli.dense.empty()))
    return false;
  return true;
}

int code_value(int n, int k) {
  int const logical = ((13 * n + 7 * k + 3) & 7) - 3;
  if constexpr (kQtype == 10) return logical & 3;
  if constexpr (kQtype == 11) return std::max(0, std::min(7, logical + 4));
  if constexpr (kQtype == 12) return logical + 8;
  if constexpr (kQtype == 13) return logical & 31;
  return std::max(0, std::min(63, logical + 32));
}

int decoded_value(int code) {
  if constexpr (kQtype == 11) return code - 4;
  if constexpr (kQtype == 14) return code - 32;
  return code;
}

void put_native(std::vector<uint8_t>& plane, int bits, int n, int k,
                int K, int code) {
  std::uint64_t const bit = (std::uint64_t(n) * K + k) * bits;
  plane[bit >> 3] |= std::uint8_t(code << (bit & 7));
}

template <gguf_scale::KType T>
std::vector<uint8_t> make_units(int n, int k) {
  using U = gguf_scale::packed_unit::Unit<T>;
  int const superblocks = k / 256;
  int const num_units = superblocks / U::kSbPerUnit;
  std::vector<uint8_t> units(
      std::size_t(num_units) * n * U::kUnitTotal, uint8_t(0));
  int constexpr scale_code = T == gguf_scale::KType::Q3_K ? 33 : 1;
  for (int u = 0; u < num_units; ++u)
    for (int col = 0; col < n; ++col) {
      uint8_t* unit = units.data() +
          (std::int64_t(u) * n + col) * U::kUnitTotal;
      for (int sb = 0; sb < U::kSbPerUnit; ++sb) {
        uint8_t* p = unit + sb * U::kSbBytes;
        auto const one = half_t(1.f).raw();
        p[0] = uint8_t(one); p[1] = uint8_t(one >> 8);
        if constexpr (U::kHasMin) { p[2] = 0; p[3] = 0; }
        for (int g = 0; g < U::kGroups; ++g) {
          gguf_scale::packed_unit::put_code<T>(p, g, 0, scale_code);
          if constexpr (U::kHasMin)
            gguf_scale::packed_unit::put_code<T>(p, g, 1, 0);
        }
      }
    }
  return units;
}

quactlize_ppu_placed_arrangement_v2 arrangement(bool kpack) {
  if (kpack) {
    if constexpr (kQtype == 12)
      return ppu_arrangements::q4_kpack4_transpose_v1();
    return ppu_arrangements::kquant_kpack_transpose_v1(kQtype);
  }
  auto const& format = ppu_formats::for_qtype(kQtype);
  return {QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2,
          QUACTLIZE_PPU_LAYOUT_XPLANE_V1,
          format.low_bits, format.high_bits,
          format.fully_quantized_tile_k, 0, 0, 0, 0};
}

struct HostWeights {
  std::vector<uint8_t> low, high, units;
  quactlize_ppu_placed_arrangement_v2 descriptor{};
  std::uint64_t low_hash = 0, high_hash = 0, unit_hash = 0;
  bool exact = false;
};

std::uint64_t hash_bytes(void const* data, std::size_t bytes);

HostWeights make_weights(int n, int k, int experts, bool kpack) {
  HostWeights out;
  std::size_t const low_bytes = std::size_t(n) * k * F::LowBits / 8;
  std::size_t const high_bytes = std::size_t(n) * k * F::HighBits / 8;
  std::vector<uint8_t> native_low(low_bytes, 0), native_high(high_bytes, 0);
  for (int col = 0; col < n; ++col)
    for (int kk = 0; kk < k; ++kk) {
      int const code = code_value(col, kk);
      put_native(native_low, F::LowBits, col, kk, k,
                 code & ((1 << F::LowBits) - 1));
      if constexpr (F::HighBits != 0)
        put_native(native_high, F::HighBits, col, kk, k,
                   code >> F::LowBits);
    }
  out.descriptor = arrangement(kpack);
  std::vector<uint8_t> placed_low(low_bytes, 0xcd), placed_high(high_bytes, 0xcd);
  int const rc = quactlize_ppu_prepare_dense_for_arrangement_v2(
      native_low.data(), F::HighBits ? native_high.data() : nullptr,
      placed_low.data(), F::HighBits ? placed_high.data() : nullptr,
      n, k, kQtype, &out.descriptor);
  std::vector<uint8_t> back_low(low_bytes, 0xab), back_high(high_bytes, 0xab);
  int const recover = rc ? rc : quactlize_ppu_recover_dense_for_arrangement_v2(
      placed_low.data(), F::HighBits ? placed_high.data() : nullptr,
      back_low.data(), F::HighBits ? back_high.data() : nullptr,
      n, k, kQtype, &out.descriptor);
  out.exact = rc == 0 && recover == 0 && back_low == native_low &&
              back_high == native_high;
  if (!out.exact) return out;
  out.low_hash = hash_bytes(placed_low.data(), placed_low.size());
  out.high_hash = hash_bytes(placed_high.data(), placed_high.size());
  out.low.resize(low_bytes * experts);
  out.high.resize(high_bytes * experts);
  for (int e = 0; e < experts; ++e) {
    std::copy(placed_low.begin(), placed_low.end(),
              out.low.begin() + std::size_t(e) * low_bytes);
    if constexpr (F::HighBits != 0)
      std::copy(placed_high.begin(), placed_high.end(),
                out.high.begin() + std::size_t(e) * high_bytes);
  }
  std::vector<uint8_t> one_units = make_units<F::Type>(n, k);
  out.unit_hash = hash_bytes(one_units.data(), one_units.size());
  out.units.resize(one_units.size() * experts);
  for (int e = 0; e < experts; ++e)
    std::copy(one_units.begin(), one_units.end(),
              out.units.begin() + std::size_t(e) * one_units.size());
  return out;
}

struct DeviceWeights {
  cutlass::DeviceAllocation<uint8_t> low, high, units;
  quactlize_ppu_placed_arrangement_v2 descriptor{};
  explicit DeviceWeights(HostWeights const& host)
      : low(host.low.size()), high(std::max<std::size_t>(host.high.size(), 1)),
        units(host.units.size()), descriptor(host.descriptor) {
    low.copy_from_host(host.low.data());
    if (!host.high.empty()) high.copy_from_host(host.high.data());
    units.copy_from_host(host.units.data());
  }
};

struct ProblemData {
  std::vector<half_t> a, golden;
};

std::uint64_t hash_bytes(void const* data, std::size_t bytes) {
  auto const* p = static_cast<std::uint8_t const*>(data);
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

ProblemData make_problem(int rows, int n, int k) {
  ProblemData out;
  out.a.assign(std::size_t(rows) * k, half_t(0.f));
  out.golden.resize(std::size_t(rows) * n);
  int const superblocks = k / 256;
  int const count = std::min(superblocks, 32);
  std::vector<std::pair<int,int>> active;
  for (int sample = 0; sample < count; ++sample) {
    int const sb = (sample * superblocks) / count;
    int const kk = sb * 256 + ((37 * sb + 11) & 255);
    active.emplace_back(sb, kk);
  }
  std::vector<half_t> base(n);
  for (int col = 0; col < n; ++col) {
    int sum = 0;
    for (auto [sb, kk] : active)
      sum += (sb & 1 ? -1 : 1) * decoded_value(code_value(col, kk));
    if (std::abs(sum) >= 2048) return {};
    base[col] = half_t(float(sum));
  }
  for (int row = 0; row < rows; ++row) {
    for (auto [sb, kk] : active)
      out.a[std::size_t(row) * k + kk] = half_t((row + sb) & 1 ? -1.f : 1.f);
    for (int col = 0; col < n; ++col) {
      float const value = float(base[col]);
      // The device accumulates from +0.  Negating an exactly-zero host golden
      // would instead manufacture fp16 -0 on odd rows, which is numerically
      // equal but fails this benchmark's intentional raw-bit comparison.  At
      // Q3 K=3072 the fixture has 64 zero columns, so the old expression made
      // exactly 4 odd rows * 64 columns = 256 false mismatches.  Canonicalize
      // only the oracle's exact zero; finite nonzero signs remain row-sensitive.
      float const signed_value = value == 0.f ? 0.f :
          (row & 1 ? -value : value);
      out.golden[std::size_t(row) * n + col] =
          half_t(signed_value);
    }
  }
  return out;
}

__global__ void compare_raw_kernel(uint16_t const* got,
                                   uint16_t const* want,
                                   std::size_t count,
                                   unsigned int* bad) {
  std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t const stride = std::size_t(blockDim.x) * gridDim.x;
  unsigned int local = 0;
  for (; index < count; index += stride) local += got[index] != want[index];
  if (local) atomicAdd(bad, local);
}

bool raw_bad(uint16_t const* got, uint16_t const* want, std::size_t count,
             cutlass::DeviceAllocation<unsigned int>& storage,
             unsigned int& bad) {
  if (hggcMemset(storage.get(), 0, sizeof(unsigned int)) != hggcSuccess)
    return false;
  int const blocks = int(std::min<std::size_t>((count + 255) / 256, 65535));
  compare_raw_kernel<<<blocks, 256>>>(got, want, count, storage.get());
  if (hggcDeviceSynchronize() != hggcSuccess ||
      hggcMemcpy(&bad, storage.get(), sizeof(bad),
                 hggcMemcpyDeviceToHost) != hggcSuccess)
    return false;
  return true;
}

struct RawMismatch {
  std::uint64_t bad = 0;
  std::size_t first = std::size_t(-1);
  std::uint16_t want = 0, got = 0;
  std::uint64_t got_hash = 0, want_hash = 0;
  std::uint64_t bad_row_mask = 0, bad_col_mod64_mask = 0;
  int bad_rows = 0, first_row_bad = 0;
  int bad_col_min = INT_MAX, bad_col_max = -1;
};

bool inspect_raw(uint16_t const* got_device,
                 std::vector<half_t> const& want,
                 int n, RawMismatch& out) {
  if (!got_device || want.empty() || n <= 0 || want.size() % std::size_t(n))
    return false;
  std::vector<std::uint16_t> got(want.size());
  if (hggcMemcpy(got.data(), got_device, got.size() * sizeof(got[0]),
                 hggcMemcpyDeviceToHost) != hggcSuccess)
    return false;
  std::vector<unsigned int> row_bad(want.size() / std::size_t(n), 0);
  out = {};
  out.first = std::size_t(-1);
  out.bad_col_min = INT_MAX;
  out.bad_col_max = -1;
  for (std::size_t i = 0; i < got.size(); ++i) {
    std::uint16_t const expected = want[i].raw();
    if (got[i] == expected) continue;
    int const row = int(i / std::size_t(n));
    int const col = int(i % std::size_t(n));
    if (out.bad++ == 0) {
      out.first = i;
      out.want = expected;
      out.got = got[i];
    }
    ++row_bad[std::size_t(row)];
    if (row < 64) out.bad_row_mask |= UINT64_C(1) << row;
    out.bad_col_mod64_mask |= UINT64_C(1) << (col & 63);
    out.bad_col_min = std::min(out.bad_col_min, col);
    out.bad_col_max = std::max(out.bad_col_max, col);
  }
  for (unsigned int count : row_bad) out.bad_rows += count != 0;
  if (out.first != std::size_t(-1))
    out.first_row_bad = int(row_bad[out.first / std::size_t(n)]);
  out.got_hash = hash_bytes(got.data(), got.size() * sizeof(got[0]));
  out.want_hash = hash_bytes(want.data(), want.size() * sizeof(want[0]));
  return true;
}

void print_mismatch(char const* operator_name, char const* layout,
                    int n, RawMismatch const& row, int first_expert) {
  std::size_t const first_row = row.first == std::size_t(-1)
      ? std::size_t(-1) : row.first / std::size_t(n);
  std::size_t const first_col = row.first == std::size_t(-1)
      ? std::size_t(-1) : row.first % std::size_t(n);
  std::printf(
      "FQ_KQUANT_LAYOUT_MISMATCH q=%d operator=%s layout=%s raw_bad=%llu "
      "first_bad=%zu first_row=%zu first_col=%zu first_expert=%d "
      "first_want=0x%04x first_got=0x%04x first_row_bad=%d bad_rows=%d "
      "bad_col_range=[%d,%d] bad_row_mask=0x%016llx "
      "bad_col_mod64_mask=0x%016llx got_hash=0x%016llx want_hash=0x%016llx\n",
      kQtype, operator_name, layout,
      static_cast<unsigned long long>(row.bad), row.first, first_row,
      first_col, first_expert, unsigned(row.want), unsigned(row.got),
      row.first_row_bad, row.bad_rows,
      row.bad_col_min == INT_MAX ? -1 : row.bad_col_min, row.bad_col_max,
      static_cast<unsigned long long>(row.bad_row_mask),
      static_cast<unsigned long long>(row.bad_col_mod64_mask),
      static_cast<unsigned long long>(row.got_hash),
      static_cast<unsigned long long>(row.want_hash));
}

struct Timing {
  double median = 0, minimum = 0, maximum = 0;
  std::vector<double> samples;
};

template <class Launch>
bool measure(Launch&& launch, int warmups, int iterations, Timing& timing) {
  for (int i = 0; i < warmups; ++i) if (launch() != 0) return false;
  if (hggcDeviceSynchronize() != hggcSuccess) return false;
  timing.samples.clear();
  timing.samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    hggcEvent_t begin{}, end{};
    if (hggcEventCreate(&begin) != hggcSuccess ||
        hggcEventCreate(&end) != hggcSuccess ||
        hggcEventRecord(begin, nullptr) != hggcSuccess ||
        launch() != 0 ||
        hggcEventRecord(end, nullptr) != hggcSuccess ||
        hggcEventSynchronize(end) != hggcSuccess) {
      if (begin) hggcEventDestroy(begin);
      if (end) hggcEventDestroy(end);
      return false;
    }
    float ms = 0;
    bool const ok = hggcEventElapsedTime(&ms, begin, end) == hggcSuccess &&
                    ms > 0 && std::isfinite(ms);
    hggcEventDestroy(begin); hggcEventDestroy(end);
    if (!ok) return false;
    timing.samples.push_back(double(ms) * 1000.0);
  }
  std::sort(timing.samples.begin(), timing.samples.end());
  timing.minimum = timing.samples.front();
  timing.maximum = timing.samples.back();
  std::size_t const n = timing.samples.size();
  timing.median = n & 1 ? timing.samples[n / 2]
                        : .5 * (timing.samples[n / 2 - 1] + timing.samples[n / 2]);
  return true;
}

void print_samples(std::vector<double> const& samples) {
  std::printf("[");
  for (std::size_t i = 0; i < samples.size(); ++i)
    std::printf("%s%.9f", i ? "," : "", samples[i]);
  std::printf("]");
}

std::string dense_default_name(int m) {
  auto const id = ppu_dense_shipping::default_config_for_m(m);
  for (auto const& row : ppu_dense_shipping::kConfigs)
    if (row.id == id) return row.name;
  return "MISSING";
}

struct ConfigName {
  std::string label;
  char const* wire = nullptr;
};

std::vector<ConfigName> dense_configs(
    int m, int n, int k, quactlize_ppu_placed_arrangement_v2 const& desc,
    bool all, bool allow_split_k = false) {
  if (!all) return {{dense_default_name(m), nullptr}};
  int count = quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2_v4(
      nullptr, 0, m, n, k, F::Group, kQtype, &desc);
  if (count <= 0) return {};
  std::vector<quactlize_ppu_config_v4> rows(count);
  if (quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2_v4(
          rows.data(), count, m, n, k, F::Group, kQtype, &desc) != count)
    return {};
  std::vector<ConfigName> result;
  for (auto const& row : rows)
    if (!row.enable_cuda_kernel &&
        (allow_split_k || row.split_k_slices == 1) && row.name)
      result.push_back({row.name, row.name});
  return result;
}

std::vector<ConfigName> grouped_configs(
    int total, int n, int k, int experts, int max_rows,
    quactlize_ppu_placed_arrangement_v2 const& desc, bool all) {
  if (!all) return {{"16x128:16x16:s2", nullptr}};
  int count = quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2(
      nullptr, 0, total, n, k, F::Group, experts, max_rows, kQtype, &desc);
  if (count <= 0) return {};
  std::vector<quactlize_ppu_config_v3> rows(count);
  if (quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2(
          rows.data(), count, total, n, k, F::Group, experts, max_rows,
          kQtype, &desc) != count)
    return {};
  std::vector<ConfigName> result;
  for (auto const& row : rows)
    if (!row.enable_cuda_kernel && row.name)
      result.push_back({row.name, row.name});
  return result;
}

char const* layout_name(bool kpack) { return kpack ? "kpack" : "xplane"; }

bool run_dense_cell(DenseCase shape, bool kpack, DeviceWeights& weights,
                    Cli const& cli) {
  auto fail = [&](char const* phase, int code, char const* config) {
    std::printf(
        "FQ_KQUANT_LAYOUT_FAILURE q=%d round=%d operator=dense layout=%s "
        "shape=%dx%dx%d config=%s phase=%s code=%d\n",
        kQtype, cli.round, layout_name(kpack), shape.m, shape.n, shape.k,
        config, phase, code);
    return false;
  };
  ProblemData host = make_problem(shape.m, shape.n, shape.k);
  if (host.a.empty() || host.golden.empty()) return fail("FIXTURE", -1, "NONE");
  cutlass::DeviceAllocation<half_t> a(host.a.size()), out(host.golden.size()),
      golden(host.golden.size());
  a.copy_from_host(host.a.data()); golden.copy_from_host(host.golden.data());
  int64_t const ws_bytes =
      quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2(
          shape.m, shape.n, shape.k, kQtype, &weights.descriptor);
  if (ws_bytes <= 0) return fail("WORKSPACE_QUERY", int(ws_bytes), "NONE");
  cutlass::DeviceAllocation<uint8_t> workspace{std::size_t(ws_bytes)};
  cutlass::DeviceAllocation<unsigned int> counter(1);
  auto configs = dense_configs(shape.m, shape.n, shape.k,
                               weights.descriptor, cli.all_configs,
                               cli.policy_v2);
  if (configs.empty()) return fail("CONFIG_QUERY", 0, "NONE");
  for (auto const& config : configs) {
    int last_launch_rc = 0;
    auto launch = [&] {
      last_launch_rc = quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2(
          reinterpret_cast<uint16_t const*>(a.get()), weights.low.get(),
          F::HighBits ? weights.high.get() : nullptr, weights.units.get(),
          reinterpret_cast<uint16_t*>(out.get()), shape.m, shape.n, shape.k,
          kQtype, workspace.get(), ws_bytes, nullptr, config.wire,
          &weights.descriptor);
      return last_launch_rc;
    };
    hggcError_t const memset_status =
        hggcMemset(out.get(), 0x7b, host.golden.size() * sizeof(half_t));
    if (memset_status != hggcSuccess)
      return fail("OUTPUT_POISON", int(memset_status), config.label.c_str());
    if (launch() != 0)
      return fail("LAUNCH", last_launch_rc, config.label.c_str());
    hggcError_t const sync_status = hggcDeviceSynchronize();
    if (sync_status != hggcSuccess)
      return fail("SYNCHRONIZE", int(sync_status), config.label.c_str());
    unsigned int bad = 0;
    if (!raw_bad(reinterpret_cast<uint16_t const*>(out.get()),
                 reinterpret_cast<uint16_t const*>(golden.get()),
                 host.golden.size(), counter, bad))
      return fail("RAW_COMPARE", -1, config.label.c_str());
    if (bad) {
      RawMismatch mismatch;
      if (!inspect_raw(reinterpret_cast<uint16_t const*>(out.get()),
                       host.golden, shape.n, mismatch) || mismatch.bad != bad)
        return fail("RAW_DIAGNOSTIC", int(bad), config.label.c_str());
      print_mismatch("dense", layout_name(kpack), shape.n, mismatch, -1);
      return fail("RAW_MISMATCH", int(bad), config.label.c_str());
    }
    Timing timing;
    if (!measure(launch, cli.warmups, cli.iterations, timing))
      return fail("TIMING", last_launch_rc, config.label.c_str());
    bool const packed_a = !kpack && kQtype == 10 && shape.m == 1 &&
        config.label == "8x128:8x32:s3";
    std::printf(
        "FQ_KQUANT_LAYOUT_DENSE q=%d round=%d order=%s layout=%s "
        "mapping_id=0x%016llx shape=%dx%dx%d config=%s provider=%s "
        "iterations=%d raw_bad=%u median_us=%.9f min_us=%.9f max_us=%.9f samples=",
        kQtype, cli.round, cli.kpack_first ? "kpack-first" : "xplane-first",
        layout_name(kpack),
        static_cast<unsigned long long>(weights.descriptor.mapping_id),
        shape.m, shape.n, shape.k, config.label.c_str(),
        packed_a ? "packed-row" : "standard-aiu", cli.iterations, bad,
        timing.median, timing.minimum, timing.maximum);
    print_samples(timing.samples); std::printf("\n");
  }
  return true;
}

bool run_grouped_cell(GroupedCase shape, bool kpack, DeviceWeights& weights,
                      Cli const& cli) {
  moe_router_fixture::Rows route;
  auto fail = [&](char const* phase, int code, char const* config) {
    std::printf(
        "FQ_KQUANT_LAYOUT_FAILURE q=%d round=%d operator=grouped layout=%s "
        "tokens=%d shape=%dx%dx%d experts=%d topk=%d active=%d zero=%d "
        "max_rows=%d config=%s phase=%s code=%d\n",
        kQtype, cli.round, layout_name(kpack), shape.tokens, route.total,
        shape.n, shape.k, shape.experts, shape.topk, route.active, route.zero,
        route.max, config, phase, code);
    return false;
  };
  char why[160]{};
  if (!moe_router_fixture::route(
          shape.tokens, shape.topk, shape.experts, route, why, sizeof why))
    return fail("ROUTER", -1, "NONE");
  std::vector<int> offsets(std::size_t(shape.experts + 1), 0);
  for (int e = 0; e < shape.experts; ++e)
    offsets[std::size_t(e + 1)] = offsets[std::size_t(e)] +
                                  route.per_expert[std::size_t(e)];
  if (offsets.back() != route.total) return fail("OFFSETS", -1, "NONE");
  ProblemData host = make_problem(route.total, shape.n, shape.k);
  if (host.a.empty() || host.golden.empty()) return fail("FIXTURE", -1, "NONE");
  cutlass::DeviceAllocation<half_t> a(host.a.size()), out(host.golden.size()),
      golden(host.golden.size());
  cutlass::DeviceAllocation<int> d_offsets(offsets.size());
  a.copy_from_host(host.a.data()); golden.copy_from_host(host.golden.data());
  d_offsets.copy_from_host(offsets.data());
  int64_t const ws_bytes =
      quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2(
          route.total, route.max, shape.n, shape.k, shape.experts, kQtype,
          &weights.descriptor);
  if (ws_bytes <= 0) return fail("WORKSPACE_QUERY", int(ws_bytes), "NONE");
  cutlass::DeviceAllocation<uint8_t> workspace{std::size_t(ws_bytes)};
  cutlass::DeviceAllocation<unsigned int> counter(1);
  auto configs = grouped_configs(route.total, shape.n, shape.k, shape.experts,
                                 route.max, weights.descriptor,
                                 cli.all_configs);
  if (configs.empty()) return fail("CONFIG_QUERY", 0, "NONE");
  for (auto const& config : configs) {
    int last_launch_rc = 0;
    auto launch = [&] {
      last_launch_rc = quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2(
          reinterpret_cast<uint16_t const*>(a.get()), weights.low.get(),
          F::HighBits ? weights.high.get() : nullptr, weights.units.get(),
          d_offsets.get(), reinterpret_cast<uint16_t*>(out.get()),
          route.total, shape.n, shape.k, shape.experts, route.max, kQtype,
          workspace.get(), ws_bytes, nullptr, config.wire,
          &weights.descriptor);
      return last_launch_rc;
    };
    hggcError_t const memset_status =
        hggcMemset(out.get(), 0x7b, host.golden.size() * sizeof(half_t));
    if (memset_status != hggcSuccess)
      return fail("OUTPUT_POISON", int(memset_status), config.label.c_str());
    if (launch() != 0)
      return fail("LAUNCH", last_launch_rc, config.label.c_str());
    hggcError_t const sync_status = hggcDeviceSynchronize();
    if (sync_status != hggcSuccess)
      return fail("SYNCHRONIZE", int(sync_status), config.label.c_str());
    unsigned int bad = 0;
    if (!raw_bad(reinterpret_cast<uint16_t const*>(out.get()),
                 reinterpret_cast<uint16_t const*>(golden.get()),
                 host.golden.size(), counter, bad))
      return fail("RAW_COMPARE", -1, config.label.c_str());
    if (bad) {
      RawMismatch mismatch;
      if (!inspect_raw(reinterpret_cast<uint16_t const*>(out.get()),
                       host.golden, shape.n, mismatch) || mismatch.bad != bad)
        return fail("RAW_DIAGNOSTIC", int(bad), config.label.c_str());
      int first_expert = -1;
      if (mismatch.first != std::size_t(-1)) {
        int const first_row = int(mismatch.first / std::size_t(shape.n));
        for (int e = 0; e < shape.experts; ++e)
          if (offsets[std::size_t(e)] <= first_row &&
              first_row < offsets[std::size_t(e + 1)]) {
            first_expert = e;
            break;
          }
      }
      print_mismatch("grouped", layout_name(kpack), shape.n, mismatch,
                     first_expert);
      return fail("RAW_MISMATCH", int(bad), config.label.c_str());
    }
    Timing timing;
    if (!measure(launch, cli.warmups, cli.iterations, timing))
      return fail("TIMING", last_launch_rc, config.label.c_str());
    std::printf(
        "FQ_KQUANT_LAYOUT_GROUPED q=%d round=%d order=%s layout=%s "
        "mapping_id=0x%016llx tokens=%d shape=%dx%dx%d experts=%d topk=%d "
        "active=%d zero=%d max_rows=%d config=%s provider=standard-aiu "
        "iterations=%d raw_bad=%u median_us=%.9f min_us=%.9f max_us=%.9f samples=",
        kQtype, cli.round, cli.kpack_first ? "kpack-first" : "xplane-first",
        layout_name(kpack),
        static_cast<unsigned long long>(weights.descriptor.mapping_id),
        shape.tokens, route.total, shape.n, shape.k, shape.experts,
        shape.topk, route.active, route.zero, route.max,
        config.label.c_str(), cli.iterations, bad, timing.median,
        timing.minimum, timing.maximum);
    print_samples(timing.samples); std::printf("\n");
  }
  return true;
}

template <class Cases, class Run>
bool run_families(Cases const& cases, int experts, Cli const& cli, Run&& run) {
  using Case = typename Cases::value_type;
  std::map<std::pair<int,int>, std::vector<Case>> families;
  for (auto const& row : cases) families[{row.n, row.k}].push_back(row);
  for (auto const& family : families) {
    int const n = family.first.first, k = family.first.second;
    if (cli.policy_v2) {
      HostWeights hk = make_weights(n, k, experts, true);
      if (!hk.exact) return false;
      DeviceWeights dk(hk);
      std::printf(
          "FQ_KQUANT_POLICY_WEIGHT schema=kpack-policy-v2 q=%d n=%d k=%d "
          "experts=%d mapping_id=0x%016llx low_bytes=%zu high_bytes=%zu "
          "unit_bytes=%zu low_hash=0x%016llx high_hash=0x%016llx "
          "unit_hash=0x%016llx roundtrip=PASS\n",
          kQtype, n, k, experts,
          static_cast<unsigned long long>(hk.descriptor.mapping_id),
          hk.low.size(), hk.high.size(), hk.units.size(),
          static_cast<unsigned long long>(hk.low_hash),
          static_cast<unsigned long long>(hk.high_hash),
          static_cast<unsigned long long>(hk.unit_hash));
      for (auto const& row : family.second)
        if (!run(row, true, dk, cli)) return false;
      continue;
    }
    HostWeights hx = make_weights(n, k, experts, false);
    HostWeights hk = make_weights(n, k, experts, true);
    if (!hx.exact || !hk.exact) return false;
    DeviceWeights dx(hx), dk(hk);
    std::printf(
        "FQ_KQUANT_LAYOUT_WEIGHT q=%d n=%d k=%d experts=%d "
        "xplane_mapping=0x%016llx kpack_mapping=0x%016llx "
        "low_bytes=%zu high_bytes=%zu unit_bytes=%zu "
        "xplane_low_hash=0x%016llx xplane_high_hash=0x%016llx "
        "kpack_low_hash=0x%016llx kpack_high_hash=0x%016llx "
        "unit_hash=0x%016llx roundtrip=PASS\n",
        kQtype, n, k, experts,
        static_cast<unsigned long long>(hx.descriptor.mapping_id),
        static_cast<unsigned long long>(hk.descriptor.mapping_id),
        hx.low.size(), hx.high.size(), hx.units.size(),
        static_cast<unsigned long long>(hx.low_hash),
        static_cast<unsigned long long>(hx.high_hash),
        static_cast<unsigned long long>(hk.low_hash),
        static_cast<unsigned long long>(hk.high_hash),
        static_cast<unsigned long long>(hx.unit_hash));
    for (auto const& row : family.second) {
      if (cli.kpack_first) {
        if (!run(row, true, dk, cli) || !run(row, false, dx, cli)) return false;
      } else {
        if (!run(row, false, dx, cli) || !run(row, true, dk, cli)) return false;
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Cli cli;
  if (!parse_cli(argc, argv, cli)) {
    std::fprintf(stderr,
        "usage: %s [--dense=M,N,K ...] [--grouped=tokens,N,K,experts,topk ...] "
        "[--iterations=N] [--warmups=N] [--round=N] "
        "[--order=xplane-first|kpack-first] [--all-configs=0|1] "
        "[--profile=kpack-policy-v2]\n",
        argv[0]);
    return 2;
  }
  auto const& format = ppu_formats::for_qtype(kQtype);
  if (format.qtype != kQtype || format.low_bits != F::LowBits ||
      format.high_bits != F::HighBits || format.group_size != F::Group)
    return 2;
  bool ok = true;
  // Grouped is the less broadly exercised deployment surface and owns the
  // larger admission contract.  Run it first so a rejected real ragged row
  // fails before the 77-cell dense timing board, not forty minutes later.
  if (!cli.grouped.empty())
    ok = run_families(cli.grouped, cli.grouped.front().experts, cli,
                      run_grouped_cell);
  if (ok && !cli.dense.empty())
    ok = run_families(cli.dense, 1, cli, run_dense_cell);
  if (cli.policy_v2)
    std::printf(
        "FQ_KQUANT_POLICY_RUN schema=kpack-policy-v2 q=%d round=%d "
        "layout=kpack order=kpack-first iterations=%d warmups=%d all_configs=%d "
        "dense_cases=%zu grouped_cases=%zu status=%s\n",
        kQtype, cli.round, cli.iterations, cli.warmups,
        int(cli.all_configs), cli.dense.size(), cli.grouped.size(),
        ok ? "PASS" : "FAIL");
  else
    std::printf(
        "FQ_KQUANT_LAYOUT_RUN q=%d round=%d order=%s iterations=%d warmups=%d "
        "all_configs=%d dense_cases=%zu grouped_cases=%zu status=%s\n",
        kQtype, cli.round, cli.kpack_first ? "kpack-first" : "xplane-first",
        cli.iterations, cli.warmups, int(cli.all_configs), cli.dense.size(),
        cli.grouped.size(), ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
