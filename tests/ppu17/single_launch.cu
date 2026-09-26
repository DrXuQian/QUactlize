// One target launch per process. All fixtures, expansion references and dot
// products are made on CPU. No GPU warmup, memset, reference or retry.
#include "fixture.cuh"

template <int Q, class T> void test(Options const &o) {
  std::vector<int32_t> offsets{0};
  auto row_counts = o.grouped ? (o.rows.empty() ? std::vector<int>{0, 1, 9, 65} : o.rows) : std::vector<int>{o.m};
  int max_rows = 0;
  for (int count : row_counts) {
    if (count < 0 || int64_t(offsets.back()) + count > INT32_MAX - 256)
      throw std::runtime_error("invalid grouped row count");
    max_rows = std::max(max_rows, count);
    offsets.push_back(offsets.back() + count);
  }
  int m = offsets.back(), experts = int(offsets.size()) - 1;
  q17_call_v1 c{};
  c.version = 1; c.struct_bytes = sizeof(c);
  auto &p = c.problem;
  p.version = 1; p.struct_bytes = sizeof(p);
  p.qtype = Q; p.operation = o.grouped ? Q17_GROUPED : Q17_DENSE;
  p.route = o.sf ? Q17_SF : Q17_FQ;
  p.compute_type = o.compute; p.input_type = o.input; p.output_type = o.output;
  p.m = m; p.n = o.n; p.k = o.k; p.experts = experts; p.max_rows = max_rows;
  check(q17_canonical_arrangement_v1(Q, &p.arrangement), "arrangement");
  q17_sizes_v1 sizes{};
  check(q17_sizes_query_v1(&p, &sizes), "sizes");
  check(q17_config_get_v1(&p, o.config, &c.config), "config");
  Fixture<Q, T> fixture(o.n, o.k, experts);
  if (sizes.low_bytes != fixture.low.size() || sizes.high_bytes != fixture.high.size() || sizes.units_bytes != fixture.units.size())
    throw std::runtime_error("fixture sizes differ from public ABI");
  c.a_stride = o.k + 8; c.output_stride = o.n + 8;
  int input_size = o.input == Q17_F32 ? 4 : 2, output_size = o.output == Q17_F32 ? 4 : 2;
  std::vector<uint8_t> activation(size_t(m) * c.a_stride * input_size, 0x5a);
  std::vector<uint8_t> output(size_t(m) * c.output_stride * output_size + 64, 0x5a);
  std::vector<float> expected(size_t(m) * o.n);
  for (int row = 0; row < m; ++row) for (int k = 0; k < o.k; ++k) {
    float v = float((row * 3 + k * 5) % 29 - 14) / 16.f;
    // Exercise values above FP16 range without introducing an FP16 input.
    if (o.compute == Q17_BF16 && o.input != Q17_F16 && o.output != Q17_F16 && row == 0 && k % 64 == 0) v *= 262144.f;
    store_element(activation.data(), o.input, int64_t(row) * c.a_stride + k, v);
  }
  for (int e = 0; e < experts; ++e) for (int row = offsets[e]; row < offsets[e + 1]; ++row)
    for (int col = 0; col < o.n; ++col) {
      double sum = 0;
      for (int k = 0; k < o.k; ++k) {
        T a(load_element(activation.data(), o.input, int64_t(row) * c.a_stride + k));
        sum += double(float(a)) * float(Fixture<Q, T>::oracle_weight(e, col, k));
      }
      expected[size_t(row) * o.n + col] = float(sum);
    }
  std::printf("PPU17_CASE q=%d m=%d n=%d k=%d experts=%d config=%d compute=%d input=%d output=%d route=%d phase=%s\n",
      Q, m, o.n, o.k, experts, o.config, o.compute, o.input, o.output, p.route, o.expansion ? "scales" : "gemm");
  std::fflush(stdout);
  if (o.cpu_only) {
    std::printf("PPU17_CASE CPU_FIXTURE_PASS device_admission=NOT_RUN target_launches=0\n");
    return;
  }
  gpu(cudaGetDevice(&c.device), "device");
  DeviceMemory a(activation.size(), activation.data()), l(fixture.low.size(), fixture.low.data()),
      h(fixture.high.size(), fixture.high.data()), u(fixture.units.size(), fixture.units.data()),
      ro(offsets.size() * sizeof(int32_t), offsets.data()), out(output.size(), output.data());
  std::vector<T> scale_out(fixture.scale.size(), T(-123.f)), zero_out(fixture.zero.size(), T(-123.f));
  DeviceMemory s(fixture.scale.size() * sizeof(T), o.expansion ? scale_out.data() : fixture.scale.data()),
      z(fixture.zero.size() * sizeof(T), o.expansion ? zero_out.data() : fixture.zero.data());
  c.a = a.p; c.low = l.p; c.high = h.p; c.units = u.p;
  c.scale = s.p; c.zero = z.p; c.offsets = static_cast<int32_t const *>(ro.p);
  c.output = static_cast<uint8_t *>(out.p) + 32;
  gpu(cudaDeviceSynchronize(), "upload completion");
  if (o.expansion) {
    check(q17_expand_scales_v1(&c), "scale expansion");
  } else {
    q17_plan_v1 *plan = nullptr;
    check(q17_prepare_v1(&c, &plan), "prepare");
    int status = q17_execute_v1(plan);
    q17_destroy_v1(plan);
    check(status, "execute");
  }
  gpu(cudaDeviceSynchronize(), "target completion");
  double max_abs = 0;
  if (o.expansion) {
    gpu(cudaMemcpy(scale_out.data(), s.p, fixture.scale.size() * sizeof(T), cudaMemcpyDeviceToHost), "scale download");
    gpu(cudaMemcpy(zero_out.data(), z.p, fixture.zero.size() * sizeof(T), cudaMemcpyDeviceToHost), "zero download");
    if (std::memcmp(scale_out.data(), fixture.scale.data(), fixture.scale.size() * sizeof(T)) ||
        std::memcmp(zero_out.data(), fixture.zero.data(), fixture.zero.size() * sizeof(T)))
      throw std::runtime_error("expanded metadata differs from CPU typed oracle");
  } else {
    gpu(cudaMemcpy(output.data(), out.p, output.size(), cudaMemcpyDeviceToHost), "output download");
    for (size_t i = 0; i < output.size(); ++i) {
      bool payload = i >= 32 && i < output.size() - 32;
      size_t local = payload ? (i - 32) / output_size : 0;
      if ((!payload || local % c.output_stride >= size_t(o.n)) && output[i] != 0x5a)
        throw std::runtime_error("output guard overwritten");
    }
    for (int row = 0; row < m; ++row) for (int col = 0; col < o.n; ++col) {
      float got = load_element(output.data() + 32, o.output, int64_t(row) * c.output_stride + col);
      float want = expected[size_t(row) * o.n + col];
      double tolerance = (o.output == Q17_BF16 ? .008 : o.output == Q17_F16 ? .001 : .0001) * std::max(1.f, std::abs(want));
      max_abs = std::max(max_abs, double(std::abs(got - want)));
      if (!std::isfinite(got) || !std::isfinite(want) || std::abs(got - want) > tolerance)
        throw std::runtime_error("CPU dot oracle mismatch at row=" + std::to_string(row) + " col=" + std::to_string(col));
    }
  }
  std::printf("PPU17_CASE PASS target_launches=1 max_abs=%.9g timing=NOT_MEASURED backend=%s\n", max_abs, q17_build_target_v1());
}

int main(int argc, char **argv) {
  try {
    Options o;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--grouped") o.grouped = true;
      else if (key == "--sf") o.sf = true;
      else if (key == "--expand") o.expansion = true;
      else if (key == "--cpu-only") o.cpu_only = true;
      else {
        if (++i == argc) throw std::runtime_error("missing value for " + key);
        if (key == "--rows") {
          o.grouped = true;
          std::istringstream fields(argv[i]);
          std::string count;
          while (std::getline(fields, count, ',')) o.rows.push_back(std::stoi(count));
          if (o.rows.empty()) throw std::runtime_error("empty --rows list");
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
        else throw std::runtime_error("unknown option " + key);
      }
    }
    if (o.compute != Q17_F16 && o.compute != Q17_BF16) throw std::runtime_error("compute must be 0=F16 or 1=BF16");
    switch (o.q) {
#define CASE(Q) case Q: if (o.compute == Q17_F16) test<Q, cutlass::half_t>(o); else test<Q, cutlass::bfloat16_t>(o); break;
      CASE(8) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14)
#undef CASE
      default: throw std::runtime_error("unsupported qtype");
    }
    return 0;
  } catch (std::exception const &e) {
    std::fprintf(stderr, "PPU17_CASE FAIL: %s\n", e.what());
    return 1;
  }
}
