// One generated build covers one (qtype, ArtifactTileK, BChunk) tuple and any
// number of runtime M,N,K shapes.  The orchestration runner merges all tuples,
// static rejects and the four explicit Q8 unsupported cells into one exact
// denominator.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "fully_quantized_splitk_producer_bench.hpp"
#include "gguf_bc_vecdot.hpp"
#include "gguf_packed_unit.hpp"

#include "fq_tc_registry.inc"

#ifndef FQ_SWEEP_QTYPE
#error "FQ_SWEEP_QTYPE must match the generated registry"
#endif
#ifndef FQ_SWEEP_ARTIFACT_TK
#error "FQ_SWEEP_ARTIFACT_TK must match the generated registry"
#endif
#ifndef FQ_SWEEP_BCHUNK
#error "FQ_SWEEP_BCHUNK must match the generated registry"
#endif
static_assert(FQ_SWEEP_QTYPE == FQ_TC_GENERATED_QTYPE);
static_assert(FQ_SWEEP_ARTIFACT_TK == FQ_TC_GENERATED_ARTIFACT_TK);
static_assert(FQ_SWEEP_BCHUNK == FQ_TC_GENERATED_BCHUNK);

extern "C" int quactlize_ppu_prepare_dense_for_tile(
    uint8_t const*, uint8_t const*, uint8_t*, uint8_t*,
    int, int, int, int);
extern "C" int quactlize_ppu_recover_dense_for_tile(
    uint8_t const*, uint8_t const*, uint8_t*, uint8_t*,
    int, int, int, int);

namespace fq_internal_sweep_generated {
#define FQ_TC_DECLARE(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP)                  \
  bool FN(fq_internal_sweep::DeviceInputs const&,                      \
          fq_internal_sweep::Options const&,                           \
          fq_internal_sweep::RowResult&);
FQ_TC_REGISTRY_ROWS(FQ_TC_DECLARE)
#undef FQ_TC_DECLARE
}

namespace {

using namespace fq_internal_sweep;

std::vector<RegistryRow> registry() {
  return {
#define FQ_TC_REGISTER(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP)                 \
    {#FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,                                 \
     &fq_internal_sweep_generated::FN},
    FQ_TC_REGISTRY_ROWS(FQ_TC_REGISTER)
#undef FQ_TC_REGISTER
  };
}

template <int Q> struct KTypeFor;
template <> struct KTypeFor<10> { static constexpr auto value = gguf_scale::KType::Q2_K; };
template <> struct KTypeFor<11> { static constexpr auto value = gguf_scale::KType::Q3_K; };
template <> struct KTypeFor<12> { static constexpr auto value = gguf_scale::KType::Q4_K; };
template <> struct KTypeFor<13> { static constexpr auto value = gguf_scale::KType::Q5_K; };
template <> struct KTypeFor<14> { static constexpr auto value = gguf_scale::KType::Q6_K; };

struct Shape { int m=1,n=4096,k=4096; };

bool parse_shape(char const* text, Shape& out) {
  char tail = 0;
  return std::sscanf(text, "%dx%dx%d%c", &out.m, &out.n, &out.k, &tail) == 3 &&
      out.m > 0 && out.n > 0 && out.k > 0;
}

struct Cli {
  int iterations = 7;
  int repeats = 2;
  int only_split = 0;
  int tm8_max_m = ppu_dense_shipping::kDecodeDefaultExclusiveM - 1;
  bool force_custom_splitk_s1 = false;
  enum class BcMode { All, Skip, Only } bc_mode = BcMode::All;
  std::string symbols_file;
  std::vector<Shape> shapes;
};

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
    } else if (!std::strncmp(argv[i], "--only-split=", 13)) {
      cli.only_split = std::atoi(argv[i] + 13);
    } else if (!std::strncmp(argv[i], "--tm8-max-m=", 12)) {
      cli.tm8_max_m = std::atoi(argv[i] + 12);
    } else if (!std::strncmp(argv[i], "--symbols-file=", 15)) {
      cli.symbols_file = argv[i] + 15;
    } else if (!std::strcmp(argv[i], "--force-custom-splitk-s1")) {
      cli.force_custom_splitk_s1 = true;
    } else if (!std::strncmp(argv[i], "--bc-mode=", 10)) {
      char const* mode = argv[i] + 10;
      if (!std::strcmp(mode, "all")) cli.bc_mode = Cli::BcMode::All;
      else if (!std::strcmp(mode, "skip")) cli.bc_mode = Cli::BcMode::Skip;
      else if (!std::strcmp(mode, "only")) cli.bc_mode = Cli::BcMode::Only;
      else return false;
    } else return false;
  }
  if (cli.shapes.empty()) cli.shapes.push_back({1,4096,4096});
  return cli.iterations > 0 && cli.repeats > 0 && cli.tm8_max_m > 0 &&
      (cli.only_split == 0 || cli.only_split == 1 || cli.only_split == 2 ||
       cli.only_split == 4 || cli.only_split == 8) &&
      !(cli.bc_mode == Cli::BcMode::Only && cli.only_split != 0);
}

bool select_registry(Cli const& cli, std::vector<RegistryRow> const& all,
                     std::vector<RegistryRow>& selected) {
  if (cli.symbols_file.empty()) {
    selected = all;
    return true;
  }
  std::ifstream stream(cli.symbols_file);
  if (!stream) return false;
  std::set<std::string> wanted;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty() || line.find_first_of(" \t\r") != std::string::npos ||
        !wanted.insert(line).second) return false;
  }
  if (wanted.empty()) return false;
  for (auto const& row : all)
    if (wanted.erase(row.symbol)) selected.push_back(row);
  return wanted.empty() && !selected.empty();
}

int code_value(int qtype, int n, int k) {
  int logical = ((13 * n + 7 * k + 3) & 7) - 3;
  switch (qtype) {
    case 10: return logical & 3;
    case 11: return std::max(0, std::min(7, logical + 4));
    case 12: return logical & 15;
    case 13: return logical & 31;
    case 14: return std::max(0, std::min(63, logical + 32));
  }
  return 0;
}

int decoded_value(int qtype, int code) {
  return qtype == 11 ? code - 4 : qtype == 14 ? code - 32 : code;
}

void put_native(std::vector<uint8_t>& plane, int bits, int n, int k,
                int N, int K, int code) {
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
        auto one = half_t(1.f).raw();
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

struct Fixture {
  std::vector<half_t> a, golden;
  std::array<std::vector<float>, kSplits.size()> partial_golden;
  std::vector<uint8_t> low_native, high_native, low, high, units;
  bool exact = false, roundtrip = false;
};

Fixture make_fixture(Shape shape, bool build_partial_golden = false) {
  constexpr int qtype = FQ_SWEEP_QTYPE;
  constexpr int low_bits = qtype == 10 || qtype == 11 ? 2 : 4;
  constexpr int high_bits = qtype == 11 || qtype == 13 ? 1 : qtype == 14 ? 2 : 0;
  Fixture f;
  f.a.assign(std::size_t(shape.m) * shape.k, half_t(0.f));
  f.golden.resize(std::size_t(shape.m) * shape.n);
  if (build_partial_golden)
    for (std::size_t slot = 0; slot < kSplits.size(); ++slot)
      f.partial_golden[slot].assign(
          std::size_t(kSplits[slot]) * shape.m * shape.n, 0.f);
  f.low_native.assign(std::size_t(shape.n) * shape.k * low_bits / 8, 0);
  f.high_native.assign(high_bits ? std::size_t(shape.n) * shape.k * high_bits / 8 : 0, 0);
  f.low.resize(f.low_native.size());
  f.high.resize(f.high_native.size());
  std::vector<int> active;
  int const superblocks = shape.k / 256;
  int const active_superblocks = std::min(superblocks, 32);
  for (int sample = 0; sample < active_superblocks; ++sample) {
    // A bounded exact fixture must remain exact for the largest model K, but
    // it must not accidentally exercise only the first Split-K slice.  Spread
    // at most 32 nonzeros across the whole K range; Q6 then stays below 992,
    // safely inside fp16's unit-spaced integer interval.
    int const sb = (sample * superblocks) / active_superblocks;
    int const k = sb * 256 + ((37 * sb + 11) & 255);
    active.push_back(k);
    for (int m = 0; m < shape.m; ++m)
      f.a[std::size_t(m) * shape.k + k] = half_t((m + sb) & 1 ? -1.f : 1.f);
  }
  for (int n = 0; n < shape.n; ++n)
    for (int k = 0; k < shape.k; ++k) {
      int const code = code_value(qtype, n, k);
      put_native(f.low_native, low_bits, n, k, shape.n, shape.k,
                 code & ((1 << low_bits) - 1));
      if constexpr (high_bits != 0)
        put_native(f.high_native, high_bits, n, k, shape.n, shape.k,
                   code >> low_bits);
    }
  if (quactlize_ppu_prepare_dense_for_tile(
          f.low_native.data(), high_bits ? f.high_native.data() : nullptr,
          f.low.data(), high_bits ? f.high.data() : nullptr,
          shape.n, shape.k, qtype, FQ_SWEEP_ARTIFACT_TK) != 0) return f;
  std::vector<uint8_t> low_back(f.low_native.size()), high_back(f.high_native.size());
  f.roundtrip = quactlize_ppu_recover_dense_for_tile(
      f.low.data(), high_bits ? f.high.data() : nullptr,
      low_back.data(), high_bits ? high_back.data() : nullptr,
      shape.n, shape.k, qtype, FQ_SWEEP_ARTIFACT_TK) == 0 &&
      low_back == f.low_native && high_back == f.high_native;
  if constexpr (qtype == 10) f.units = make_units<gguf_scale::KType::Q2_K>(shape.n, shape.k);
  if constexpr (qtype == 11) f.units = make_units<gguf_scale::KType::Q3_K>(shape.n, shape.k);
  if constexpr (qtype == 12) f.units = make_units<gguf_scale::KType::Q4_K>(shape.n, shape.k);
  if constexpr (qtype == 13) f.units = make_units<gguf_scale::KType::Q5_K>(shape.n, shape.k);
  if constexpr (qtype == 14) f.units = make_units<gguf_scale::KType::Q6_K>(shape.n, shape.k);
  int max_abs = 0;
  for (int m = 0; m < shape.m; ++m)
    for (int n = 0; n < shape.n; ++n) {
      int sum = 0;
      for (int k : active) {
        int const contribution =
            int(float(f.a[std::size_t(m) * shape.k + k])) *
            decoded_value(qtype, code_value(qtype, n, k));
        sum += contribution;
        if (build_partial_golden)
          for (std::size_t slot = 0; slot < kSplits.size(); ++slot) {
            int const splits = kSplits[slot];
            int const plane = int(std::int64_t(k) * splits / shape.k);
            std::size_t const offset =
                (std::size_t(plane) * shape.m + m) * shape.n + n;
            f.partial_golden[slot][offset] += float(contribution);
          }
      }
      max_abs = std::max(max_abs, std::abs(sum));
      f.golden[std::size_t(m) * shape.n + n] = half_t(float(sum));
    }
  f.exact = max_abs < 2048;
  return f;
}

std::uint64_t hash_float_bits(float const* values, std::size_t count) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (std::size_t index = 0; index < count; ++index) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, values + index, sizeof(bits));
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= std::uint8_t(bits >> (8 * byte));
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
}

template <int QType, int ArtifactTileK, int RowsPerWarp>
bool run_bc(Shape shape, uint8_t const* low, uint8_t const* high,
            uint8_t const* units, half_t const* a, float* output,
            std::vector<half_t> const& golden, int iterations,
            CellResult& result, std::uint64_t& bad) {
  constexpr auto T = KTypeFor<QType>::value;
  int const bpr = shape.k / 256;
  auto launch = [&] {
    gguf_scale::bc_vecdot::launch_fixed<T, ArtifactTileK, RowsPerWarp, false>(
        low, high, units,
        reinterpret_cast<gguf_scale::vecdot::VecdotActivation const*>(a),
        nullptr, output, shape.n, bpr, 1, shape.m, nullptr);
    return cutlass::Status::kSuccess;
  };
  if (launch() != cutlass::Status::kSuccess ||
      hggcDeviceSynchronize() != hggcSuccess) return false;
  std::vector<float> host(std::size_t(shape.m) * shape.n);
  if (hggcMemcpy(host.data(), output, host.size() * sizeof(float),
                 hggcMemcpyDeviceToHost) != hggcSuccess) return false;
  bad = 0;
  for (std::size_t index = 0; index < host.size(); ++index)
    bad += host[index] != float(golden[index]);
  if (bad || !fq_internal_sweep::measure(launch, iterations, result)) return false;
  return true;
}

void print_samples(std::vector<double> const& samples) {
  std::printf("[");
  for (std::size_t i = 0; i < samples.size(); ++i)
    std::printf("%s%.9f", i ? "," : "", samples[i]);
  std::printf("]");
}

template <int QType, int ArtifactTileK>
bool run_bc_family(Shape shape, uint8_t const* low, uint8_t const* high,
                   uint8_t const* units, half_t const* a, float* output,
                   std::vector<half_t> const& golden, int iterations) {
  constexpr auto T = KTypeFor<QType>::value;
  if constexpr (!gguf_scale::bc_vecdot::arrangement_supported_v<
                    T, ArtifactTileK>) {
    // Static-reject cells are supplied by the matrix authority.  Emitting a
    // runtime record here would both instantiate an illegal reader and create
    // an unconsumed extra key in the analyzer.
    return true;
  } else {
    bool family_ok = true;
#define FQ_RUN_BC_RPW(RPW) do {                                        \
    CellResult bc_result; std::uint64_t bad = 0;                        \
    bool const supported = shape.m < 8;                               \
    bool const ok = supported &&                                       \
        run_bc<QType,ArtifactTileK,RPW>(                               \
            shape, low, high, units, a, output, golden,                 \
            iterations, bc_result, bad);                               \
    constexpr int threads = QType == 12 && RPW == 4 ? 128 : 256;       \
    std::printf(                                                       \
        "FQ_BC_CELL q=%d A=%d shape=%dx%dx%d rpw=%d threads=%d "      \
        "scope=FULL_OUTPUT launches=%d batch_policy=%s "               \
        "state=%s us=%.9f raw_bad=%llu samples=",                     \
        QType, ArtifactTileK, shape.m, shape.n, shape.k, RPW, threads, \
        supported ? 1 : 0, "native-grid-y-m-lt8",                     \
        !supported ? "UNSUPPORTED_M_GE_8" : ok ? "MEASURED" : "FAILED", \
        bc_result.median_us, static_cast<unsigned long long>(bad));     \
    print_samples(bc_result.samples_us);                               \
    std::printf("\n");                                                \
    family_ok = family_ok && (!supported || ok);                       \
  } while (false)
    FQ_RUN_BC_RPW(1);
    FQ_RUN_BC_RPW(2);
    FQ_RUN_BC_RPW(4);
    FQ_RUN_BC_RPW(8);
#undef FQ_RUN_BC_RPW
    return family_ok;
  }
}

int run_shape(Shape shape, Cli const& cli,
              std::vector<RegistryRow> const& rows,
              std::size_t typed_rows) {
  Fixture fixture = make_fixture(shape, cli.force_custom_splitk_s1);
  if (!fixture.exact || !fixture.roundtrip) {
    std::fprintf(stderr,
        "FQ_FIXTURE_FAIL q=%d A=%d shape=%dx%dx%d exact=%d roundtrip=%d\n",
        FQ_SWEEP_QTYPE, FQ_SWEEP_ARTIFACT_TK, shape.m, shape.n, shape.k,
        int(fixture.exact), int(fixture.roundtrip));
    return 1;
  }
  cutlass::DeviceAllocation<half_t> dA(fixture.a.size());
  cutlass::DeviceAllocation<uint8_t> dLow(fixture.low.size());
  cutlass::DeviceAllocation<uint8_t> dHigh(std::max<std::size_t>(fixture.high.size(), 1));
  cutlass::DeviceAllocation<uint8_t> dUnits(fixture.units.size());
  cutlass::DeviceAllocation<half_t> dOut(std::size_t(shape.m) * shape.n);
  cutlass::DeviceAllocation<float> dBcOut(std::size_t(shape.m) * shape.n);
  std::size_t const partial_bytes = std::size_t(shape.m) * shape.n * 8 * sizeof(float);
  cutlass::DeviceAllocation<char> dWorkspace(std::max<std::size_t>(partial_bytes, 1));
  dA.copy_from_host(fixture.a.data()); dLow.copy_from_host(fixture.low.data());
  if (!fixture.high.empty()) dHigh.copy_from_host(fixture.high.data());
  dUnits.copy_from_host(fixture.units.data());
  std::array<float const*, kSplits.size()> partial_golden_ptrs{};
  if (cli.force_custom_splitk_s1)
    for (std::size_t slot = 0; slot < kSplits.size(); ++slot)
      partial_golden_ptrs[slot] = fixture.partial_golden[slot].data();
  DeviceInputs inputs{
      dA.get(), dLow.get(), fixture.high.empty() ? nullptr : dHigh.get(),
      dUnits.get(), dOut.get(), dWorkspace.get(), partial_bytes,
      partial_golden_ptrs,
      fixture.golden.data(), shape.m, shape.n, shape.k};
  Options options{cli.iterations, cli.repeats, cli.only_split, true,
                  cli.tm8_max_m, cli.force_custom_splitk_s1};
  bool all_runtime_ok = true;
  std::printf("FQ_SHARD q=%d A=%d bchunk=%d shape=%dx%dx%d "
              "typed_rows=%zu selected_rows=%zu only_split=%d bc_mode=%s "
              "bc_batch=native-grid-y-m-lt8 split_timing=ordered-close "
              "iterations=%d correctness_repeats=%d\n",
              FQ_SWEEP_QTYPE, FQ_SWEEP_ARTIFACT_TK, FQ_SWEEP_BCHUNK,
              shape.m, shape.n, shape.k, typed_rows, rows.size(),
              cli.only_split,
              cli.bc_mode == Cli::BcMode::All ? "all" :
              cli.bc_mode == Cli::BcMode::Skip ? "skip" : "only",
              cli.iterations, cli.repeats);
  if (cli.force_custom_splitk_s1) {
    std::printf(
        "FQ_CUSTOM_SPLIT_COUNT_PROBE route=GemmUniversalMixedInputSplitKParallel "
        "runtime_splits=1,2,4 shipping_s1_bypassed=1 output=FP32_PARTIAL_THEN_REDUCER\n");
    std::printf(
        "FQ_CUSTOM_SPLIT_COUNT_ORACLE exact=1 "
        "S1=0x%016llx S2=0x%016llx S4=0x%016llx S8=0x%016llx\n",
        static_cast<unsigned long long>(hash_float_bits(
            fixture.partial_golden[0].data(),
            fixture.partial_golden[0].size())),
        static_cast<unsigned long long>(hash_float_bits(
            fixture.partial_golden[1].data(),
            fixture.partial_golden[1].size())),
        static_cast<unsigned long long>(hash_float_bits(
            fixture.partial_golden[2].data(),
            fixture.partial_golden[2].size())),
        static_cast<unsigned long long>(hash_float_bits(
            fixture.partial_golden[3].data(),
            fixture.partial_golden[3].size())));
  }
  if (cli.bc_mode != Cli::BcMode::Only) for (auto const& entry : rows) {
    RowResult result;
    bool const ok = entry.run(inputs, options, result);
    all_runtime_ok = all_runtime_ok && ok;
    for (auto const& cell : result.cells) {
      if (cell.split == 0) continue;
      char const* scope = cell.full_output ? "FULL_OUTPUT" :
          "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS";
      std::printf(
          "FQ_TC_CELL q=%d A=%d bchunk=%d shape=%dx%dx%d symbol=%s "
          "tm=%d tn=%d tk=%d wm=%d wn=%d stages=%d provider=%s S=%d scope=%s "
          "provider_capacity_rows=%d "
          "state=%s us=%.9f raw_bad=%llu reducer_untimed=%d "
          "failure_step=%s failure_repeat=%d first_bad=%zu "
          "first_want=0x%04x first_got=0x%04x "
          "shipping_smem=%zu split_smem=%zu partial_bytes=%zu samples=",
          entry.qtype, entry.artifact_tile_k, entry.bchunk,
          shape.m, shape.n, shape.k, entry.symbol,
          entry.tm, entry.tn, entry.tk, entry.wm, entry.wn, entry.stages,
          entry.a_provider ? "packed-row" : "standard-aiu",
          cell.split, scope, cell.a_provider_capacity_rows,
          state_name(cell.state), cell.median_us,
          static_cast<unsigned long long>(cell.raw_bad),
          int(cell.reducer_correctness_untimed), cell.failure_step,
          cell.failure_repeat, cell.first_bad_index,
          unsigned(cell.first_bad_want), unsigned(cell.first_bad_got),
          cell.shipping_smem,
          cell.split_smem, cell.partial_bytes);
      print_samples(cell.samples_us);
      std::printf("\n");
      if (cli.force_custom_splitk_s1 &&
          (cell.split == 1 || cell.split == 2 || cell.split == 4)) {
        std::printf(
            "FQ_CUSTOM_SPLIT_COUNT_CELL symbol=%s provider=%s S=%d "
            "kernel=GemmUniversalMixedInputSplitKParallel state=%s "
            "raw_bad=%llu failure_step=%s failure_repeat=%d "
            "first_bad=%zu first_want=0x%04x first_got=0x%04x "
            "partial_probe=%s partial_value_raw_bad=%llu "
            "partial_bad_plane_mask=0x%x partial_first_bad_plane=%d "
            "partial_first_bad_index=%zu partial_first_bad_want=0x%08x "
            "partial_first_bad_got=0x%08x reducer_replay_raw_bad=%llu "
            "reducer_replay_first_bad=%zu "
            "partial_bytes=%zu\n",
            entry.symbol,
            entry.a_provider ? "packed-row" : "standard-aiu",
            cell.split, state_name(cell.state),
            static_cast<unsigned long long>(cell.raw_bad),
            cell.failure_step, cell.failure_repeat, cell.first_bad_index,
            unsigned(cell.first_bad_want), unsigned(cell.first_bad_got),
            !cell.partial_probe_attempted ? "NOT_TRIGGERED" :
                cell.partial_probe_complete ? "COMPLETE" : "API_FAIL",
            static_cast<unsigned long long>(cell.partial_value_raw_bad),
            unsigned(cell.partial_bad_plane_mask),
            cell.partial_first_bad_plane, cell.partial_first_bad_index,
            unsigned(cell.partial_first_bad_want),
            unsigned(cell.partial_first_bad_got),
            static_cast<unsigned long long>(cell.reducer_replay_raw_bad),
            cell.reducer_replay_first_bad,
            cell.partial_bytes);
      }
    }
  }
  if constexpr (FQ_SWEEP_BCHUNK == 0) {
    if (cli.bc_mode != Cli::BcMode::Skip) {
    all_runtime_ok = all_runtime_ok &&
        run_bc_family<FQ_SWEEP_QTYPE,FQ_SWEEP_ARTIFACT_TK>(
            shape, dLow.get(), fixture.high.empty() ? nullptr : dHigh.get(),
            dUnits.get(), dA.get(), dBcOut.get(), fixture.golden,
            cli.iterations);
    }
  }
  std::printf("FQ_SHAPE_DONE q=%d A=%d bchunk=%d shape=%dx%dx%d "
              "typed_rows=%zu selected_rows=%zu only_split=%d bc_mode=%s "
              "bc_batch=native-grid-y-m-lt8 split_timing=ordered-close "
              "iterations=%d status=%s\n",
              FQ_SWEEP_QTYPE, FQ_SWEEP_ARTIFACT_TK, FQ_SWEEP_BCHUNK,
              shape.m, shape.n, shape.k, typed_rows, rows.size(),
              cli.only_split,
              cli.bc_mode == Cli::BcMode::All ? "all" :
              cli.bc_mode == Cli::BcMode::Skip ? "skip" : "only",
              cli.iterations,
              all_runtime_ok ? "PASS" : "FAIL");
  return all_runtime_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Cli cli;
  if (!parse_cli(argc, argv, cli)) {
    std::fprintf(stderr,
        "usage: %s [--shape=MxNxK ...] [--iterations=N] "
        "[--correctness-repeats=N] [--only-split=0|1|2|4|8] "
        "[--tm8-max-m=N] [--symbols-file=PATH] "
        "[--bc-mode=all|skip|only] [--force-custom-splitk-s1]\n", argv[0]);
    return 2;
  }
  auto const all_rows = registry();
  std::vector<RegistryRow> selected_rows;
  if (!select_registry(cli, all_rows, selected_rows)) {
    std::fprintf(stderr, "FQ_SELECTION_FAIL typed_rows=%zu symbols_file=%s\n",
                 all_rows.size(), cli.symbols_file.c_str());
    return 2;
  }
  if (cli.bc_mode == Cli::BcMode::Only) selected_rows.clear();
  int rc = 0;
  for (auto shape : cli.shapes)
    rc |= run_shape(shape, cli, selected_rows, all_rows.size());
  return rc;
}
