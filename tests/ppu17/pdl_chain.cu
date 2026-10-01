// Physical-device dependent-chain gate. NOT a one-launch simulator input.
#include "fixture.cuh"
#include "launch.cuh"
#include <memory>

namespace {
struct ChainOptions : Options {
  int replays = 3;
  bool negative = false;
};

__host__ __device__ float activation_value(int row, int k, int epoch) {
  return float((row * 3 + k * 5 + epoch * 7) % 29 - 14) / 16.f;
}

// Deliberately publish the launch hint BEFORE producing dependent data. A
// bounded delay exposes missing waits without requiring concurrent progress
// or spinning on another CTA/kernel. Each launch has different activations
// and expert row ownership; stale inputs from a prior replay cannot pass.
template <bool Pdl>
__global__ void produce_inputs(q17_call_v1 c, int const *directories,
    uint8_t const *unit_source, size_t unit_bytes, uint8_t *output_storage,
    size_t output_bytes, int epoch, unsigned long long delay) {
  wait_for_predecessors<Pdl>();
  release_successors<Pdl>();
  auto start = clock64();
  while (clock64() - start < delay * (1 + blockIdx.x % 3)) {}
  size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = size_t(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < size_t(c.problem.m) * c.problem.k; i += step) {
    int row = i / c.problem.k, k = i % c.problem.k;
    store_element(const_cast<void *>(c.a), c.problem.input_type,
                  int64_t(row) * c.a_stride + k, activation_value(row, k, epoch));
  }
  for (size_t i = tid; i <= size_t(c.problem.experts); i += step)
    const_cast<int32_t *>(c.offsets)[i] = directories[(epoch % 2) * (c.problem.experts + 1) + i];
  for (size_t i = tid; i < unit_bytes; i += step)
    const_cast<uint8_t *>(static_cast<uint8_t const *>(c.units))[i] = unit_source[i];
  for (size_t i = tid; i < output_bytes; i += step) output_storage[i] = 0x5a;
  if (c.problem.route == Q17_SF) {
    size_t count = size_t(c.problem.experts) * c.problem.n * (c.problem.k / c.problem.arrangement.group_size);
    for (size_t i = tid; i < count; i += step) {
      static_cast<uint16_t *>(const_cast<void *>(c.scale))[i] = 0x7fff;
      static_cast<uint16_t *>(const_cast<void *>(c.zero))[i] = 0x7fff;
    }
  }
}

template <bool Pdl> __global__ void consume_output(
    uint8_t const *source, uint8_t *snapshot, size_t bytes) {
  wait_for_predecessors<Pdl>();
  release_successors<Pdl>();
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < bytes; i += size_t(gridDim.x) * blockDim.x) snapshot[i] = source[i];
}

__global__ void delayed_marker(int *value) {
  release_successors<true>();
  auto start = clock64();
  while (clock64() - start < 1000000) {}
  if (threadIdx.x == 0) *value = 42;
}
template <bool Wait> __global__ void read_marker(int const *value, int *snapshot) {
  wait_for_predecessors<Wait>();
  if (threadIdx.x == 0) *snapshot = *value;
}

int negative_control() {
  int initial = -1, result = 0;
  DeviceMemory value(sizeof(initial), &initial), output(sizeof(result), &result);
  cudaStream_t stream;
  gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "negative stream");
  // Lack of overlap is legal. Report an unobserved negative as inconclusive,
  // never manufacture a correctness proof by requiring concurrent progress.
  bool red = false;
  for (int repeat = -1; repeat < 8 && !red; ++repeat) {
    gpu(cudaMemcpyAsync(value.p, &initial, sizeof(initial), cudaMemcpyHostToDevice, stream), "negative reset");
    gpu(cudaError_t(enqueue_kernel<false>(delayed_marker, 1, 32, 0, stream, static_cast<int *>(value.p))), "negative producer");
    auto reader = repeat < 0 ? read_marker<true> : read_marker<false>;
    gpu(cudaError_t(enqueue_kernel<true>(reader, 1, 32, 0, stream,
                              static_cast<int const *>(value.p), static_cast<int *>(output.p))), "negative reader");
    gpu(cudaStreamSynchronize(stream), "negative completion");
    gpu(cudaMemcpy(&result, output.p, sizeof(result), cudaMemcpyDeviceToHost), "negative result");
    int produced = 0;
    gpu(cudaMemcpy(&produced, value.p, sizeof(produced), cudaMemcpyDeviceToHost), "producer control");
    if (produced != 42 || (repeat < 0 && result != 42))
      throw std::runtime_error("PDL waited control or marker producer failed");
    red = repeat >= 0 && result != 42;
  }
  gpu(cudaStreamDestroy(stream), "negative stream destroy");
  std::printf("PPU17_PDL_NEGATIVE missing_wait=%s observed=%d waited_control=PASS producer=PASS timing=NOT_MEASURED\n",
              red ? "EXPECTED_RED" : "UNRESOLVED_NO_OVERLAP", result);
  return red ? 0 : 77;
}

template <int Q, class T> void test(ChainOptions const &o) {
  std::vector<int> counts = o.grouped ?
      (o.rows.empty() ? std::vector<int>{0, 1, 7, 8, 0} : o.rows) : std::vector<int>{o.m};
  std::vector<int32_t> directories;
  int m = 0, max_rows = 0, experts = int(counts.size());
  for (int n : counts) {
    if (n < 0 || int64_t(m) + n > INT32_MAX - 256) throw std::runtime_error("invalid row counts");
    m += n; max_rows = std::max(max_rows, n);
  }
  for (int epoch = 0; epoch < 2; ++epoch) {
    int prefix = 0;
    directories.push_back(0);
    for (int e = 0; e < experts; ++e) {
      prefix += counts[(e + epoch) % experts];
      directories.push_back(prefix);
    }
  }
  q17_call_v1 c{};
  c.version = 1; c.struct_bytes = sizeof(c);
  auto &p = c.problem;
  p.version = 1; p.struct_bytes = sizeof(p);
  p.qtype = Q; p.operation = o.grouped ? Q17_GROUPED : Q17_DENSE;
  p.route = o.sf ? Q17_SF : Q17_FQ;
  p.compute_type = o.compute; p.input_type = o.input; p.output_type = o.output;
  p.m = m; p.n = o.n; p.k = o.k; p.experts = experts; p.max_rows = max_rows;
  check(q17_canonical_arrangement_v1(Q, &p.arrangement), "arrangement");
  check(q17_config_get_v1(&p, o.config, &c.config), "config");
  q17_sizes_v1 sizes{};
  check(q17_sizes_query_v1(&p, &sizes), "sizes");
  Fixture<Q, T> fixture(o.n, o.k, experts);
  c.a_stride = o.k + 8; c.output_stride = o.n + 8;
  int input_bytes = o.input == Q17_F32 ? 4 : 2, output_bytes = o.output == Q17_F32 ? 4 : 2;
  size_t bytes = size_t(m) * c.output_stride * output_bytes + 64;
  std::vector<uint8_t> activation(size_t(m) * c.a_stride * input_bytes, 0x5a), output(bytes, 0x5a);
  std::vector<std::vector<float>> expected(2, std::vector<float>(size_t(m) * o.n));
  for (int epoch = 0; epoch < 2; ++epoch) {
    auto offsets = directories.data() + epoch * (experts + 1);
    for (int e = 0; e < experts; ++e) for (int row = offsets[e]; row < offsets[e + 1]; ++row)
      for (int col = 0; col < o.n; ++col) {
        double sum = 0;
        for (int k = 0; k < o.k; ++k) {
          uint32_t storage = 0;
          store_element(&storage, o.input, 0, activation_value(row, k, epoch));
          T a(load_element(&storage, o.input, 0));
          sum += double(float(a)) * float(Fixture<Q, T>::oracle_weight(e, col, k));
        }
        expected[epoch][size_t(row) * o.n + col] = float(sum);
      }
  }
  if (o.cpu_only) {
    std::printf("PPU17_PDL_CHAIN CPU_FIXTURE_PASS target_launches=0\n");
    return;
  }
  if (!q17_pdl_compiled_v1()) throw std::runtime_error("bundle was not built with --enable-pdl");
  gpu(cudaGetDevice(&c.device), "device");
  std::vector<uint8_t> units(fixture.units.size(), 0xff);
  std::vector<int32_t> offsets(experts + 1, 0);
  std::vector<T> metadata(fixture.scale.size(), T(-123.f));
  DeviceMemory a(activation.size(), activation.data()), l(fixture.low.size(), fixture.low.data()),
      h(fixture.high.size(), fixture.high.data()), u(units.size(), units.data()),
      u_source(fixture.units.size(), fixture.units.data()), d(directories.size() * 4, directories.data()),
      ro(offsets.size() * 4, offsets.data()), out(bytes, output.data()), snapshot(bytes, output.data()),
      s(metadata.size() * sizeof(T), metadata.data()), z(metadata.size() * sizeof(T), metadata.data());
  cudaStream_t stream;
  gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "chain stream");
  c.a = a.p; c.low = l.p; c.high = h.p; c.units = u.p; c.scale = s.p; c.zero = z.p;
  c.offsets = static_cast<int32_t const *>(ro.p);
  c.output = static_cast<uint8_t *>(out.p) + 32; c.stream = stream;
  q17_launch_options_v1 options{1, sizeof(options), 1, 0};
  std::vector<std::vector<uint8_t>> baseline(2);
  int comparisons = 0;
  for (bool graph : {false, true}) for (bool pdl : {false, true}) {
    q17_plan_v1 *raw = nullptr;
    check(pdl ? q17_prepare_with_options_v1(&c, &options, &raw) : q17_prepare_v1(&c, &raw), "prepare chain");
    std::unique_ptr<q17_plan_v1, decltype(&q17_destroy_v1)> plan(raw, q17_destroy_v1);
    auto enqueue = [&](int epoch) {
      auto producer = pdl ? produce_inputs<true> : produce_inputs<false>;
      int rc = pdl ? enqueue_kernel<true>(producer, 7, 128, 0, stream, c,
                        static_cast<int const *>(d.p), static_cast<uint8_t const *>(u_source.p), units.size(),
                        static_cast<uint8_t *>(out.p), bytes, epoch, 50000ull)
                   : enqueue_kernel<false>(producer, 7, 128, 0, stream, c,
                        static_cast<int const *>(d.p), static_cast<uint8_t const *>(u_source.p), units.size(),
                        static_cast<uint8_t *>(out.p), bytes, epoch, 50000ull);
      gpu(cudaError_t(rc), "producer launch");
      if (o.sf) check(pdl ? q17_expand_scales_with_options_v1(&c, &options) : q17_expand_scales_v1(&c), "expansion");
      check(q17_execute_v1(plan.get()), "dependent compute");
      rc = pdl ? enqueue_kernel<true>(consume_output<true>, 3, 128, 0, stream,
                    static_cast<uint8_t const *>(out.p), static_cast<uint8_t *>(snapshot.p), bytes)
               : enqueue_kernel<false>(consume_output<false>, 3, 128, 0, stream,
                    static_cast<uint8_t const *>(out.p), static_cast<uint8_t *>(snapshot.p), bytes);
      gpu(cudaError_t(rc), "consumer launch");
    };
    cudaGraph_t graphs[2]{};
    cudaGraphExec_t executables[2]{};
    if (graph) for (int epoch = 0; epoch < 2; ++epoch) {
      gpu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture begin");
      enqueue(epoch);
      gpu(cudaStreamEndCapture(stream, &graphs[epoch]), "capture end");
      gpu(cudaGraphInstantiate(&executables[epoch], graphs[epoch], 0), "instantiate");
    }
    for (int replay = 0; replay < o.replays; ++replay) for (int epoch = 0; epoch < 2; ++epoch) {
      if (graph) gpu(cudaGraphLaunch(executables[epoch], stream), "replay");
      else enqueue(epoch);
      gpu(cudaStreamSynchronize(stream), "chain completion");
      gpu(cudaMemcpy(output.data(), snapshot.p, bytes, cudaMemcpyDeviceToHost), "snapshot download");
      for (size_t i = 0; i < bytes; ++i) {
        bool payload = i >= 32 && i < bytes - 32;
        size_t index = payload ? (i - 32) / output_bytes : 0;
        if ((!payload || index % c.output_stride >= size_t(o.n)) && output[i] != 0x5a)
          throw std::runtime_error("chain output guard changed");
      }
      for (int row = 0; row < m; ++row) for (int col = 0; col < o.n; ++col) {
        float got = load_element(output.data() + 32, o.output, int64_t(row) * c.output_stride + col);
        float want = expected[epoch][size_t(row) * o.n + col];
        float tolerance = (o.output == Q17_BF16 ? .008f : o.output == Q17_F16 ? .001f : .0001f) * std::max(1.f, std::abs(want));
        if (!std::isfinite(got) || std::abs(got - want) > tolerance)
          throw std::runtime_error("chain CPU oracle mismatch row=" + std::to_string(row) + " col=" + std::to_string(col));
      }
      if (baseline[epoch].empty()) baseline[epoch] = output;
      else if (baseline[epoch] != output) throw std::runtime_error("PDL/graph/replay differs from ordinary raw output");
      ++comparisons;
    }
    if (graph) for (int epoch = 0; epoch < 2; ++epoch) {
      gpu(cudaGraphExecDestroy(executables[epoch]), "executable destroy");
      gpu(cudaGraphDestroy(graphs[epoch]), "graph destroy");
    }
  }
  gpu(cudaStreamDestroy(stream), "stream destroy");
  std::printf("PPU17_PDL_CHAIN PASS q=%d compute=%d input=%d output=%d config=%d route=%d grouped=%d comparisons=%d raw_equal=1 cpu_oracle=PASS stream=PASS graph=PASS timing=NOT_MEASURED backend=%s\n",
      Q, o.compute, o.input, o.output, o.config, p.route, o.grouped, comparisons, q17_build_target_v1());
}
} // namespace

int main(int argc, char **argv) {
  try {
    ChainOptions o;
    o.m = 1;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--cpu-only") o.cpu_only = true;
      else if (key == "--grouped") o.grouped = true;
      else if (key == "--sf") o.sf = true;
      else if (key == "--negative-missing-wait") o.negative = true;
      else {
        if (++i == argc) throw std::runtime_error("missing value for " + key);
        if (key == "--rows") {
          o.grouped = true;
          std::istringstream fields(argv[i]); std::string field;
          while (std::getline(fields, field, ',')) o.rows.push_back(std::stoi(field));
          if (o.rows.empty()) throw std::runtime_error("empty rows");
          continue;
        }
        int value = std::stoi(argv[i]);
        if (key == "--qtype") o.q = value;
        else if (key == "--compute") o.compute = value;
        else if (key == "--input") o.input = value;
        else if (key == "--output") o.output = value;
        else if (key == "--config") o.config = value;
        else if (key == "--m") o.m = value;
        else if (key == "--n") o.n = value;
        else if (key == "--k") o.k = value;
        else if (key == "--replays") o.replays = value;
        else throw std::runtime_error("unknown option " + key);
      }
    }
    if (o.replays < 2 || o.replays > 32) throw std::runtime_error("replays must be in [2,32]");
    if (o.compute != Q17_F16 && o.compute != Q17_BF16) throw std::runtime_error("invalid compute type");
    if (o.negative) {
      if (o.cpu_only) throw std::runtime_error("negative requires a physical device");
      return negative_control();
    }
    switch (o.q) {
#define CASE(Q) case Q: if (o.compute == Q17_F16) test<Q, cutlass::half_t>(o); else test<Q, cutlass::bfloat16_t>(o); break;
      CASE(8) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14)
#undef CASE
      default: throw std::runtime_error("unsupported qtype");
    }
    return 0;
  } catch (std::exception const &e) {
    std::fprintf(stderr, "PPU17_PDL_CHAIN FAIL: %s\n", e.what());
    return 1;
  }
}
