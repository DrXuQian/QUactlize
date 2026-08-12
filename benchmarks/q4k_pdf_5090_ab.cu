// INBOX 132B: same-machine RTX 5090 A/B between a PDF-equivalent Q4_K
// reconstruction and gemv_lowbit's production Native int4+S/Z representation.
// See q4k_pdf_reconstruction.cuh for the source boundary.  This executable
// refuses to time either representation until both packers and every output
// element pass an independent raw-Q4_K golden.

#include "q4k_pdf_ab_fixture.hpp"

#define GEMV_GS_LIST(EMIT) EMIT(32)
#define GEMV_QUANT_LIST(EMIT, G) EMIT(QuantOp::FinegrainedScaleZero, G)
#define GEMV_CTAM_MAX 1
#define GEMV_GROUPED_CTAM_MAX 1
#define GEMV_ENABLE_BIAS 0
#include "gemv_lowbit/gemv_launcher.hpp"

#include <nvml.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using q4k_pdf_ab::HostProblem;
using q4k_pdf_ab::Shape;
using q4k_pdf_reconstruction::block_q4_K;
using OursDetails = ppu_gemv::KernelDetails<
    ppu_gemv::FP16DetailsA, ppu_gemv::WFormat::Int4,
    ppu_gemv::WLayout::Native, 16, 128>;

[[noreturn]] void fail(std::string const& message) { throw std::runtime_error(message); }

void cuda_ok(cudaError_t status, char const* where) {
  if (status != cudaSuccess) fail(std::string(where) + ": " + cudaGetErrorString(status));
}

void nvml_ok(nvmlReturn_t status, char const* where) {
  if (status != NVML_SUCCESS) fail(std::string(where) + ": " + nvmlErrorString(status));
}

struct CudaBuffer {
  void* ptr = nullptr;
  std::size_t bytes = 0;
  CudaBuffer() = default;
  explicit CudaBuffer(std::size_t n) : bytes(n) {
    if (n) cuda_ok(cudaMalloc(&ptr, n), "cudaMalloc");
  }
  CudaBuffer(CudaBuffer const&) = delete;
  CudaBuffer& operator=(CudaBuffer const&) = delete;
  CudaBuffer(CudaBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = nullptr; other.bytes = 0;
  }
  CudaBuffer& operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr) cudaFree(ptr);
      ptr = other.ptr; bytes = other.bytes;
      other.ptr = nullptr; other.bytes = 0;
    }
    return *this;
  }
  ~CudaBuffer() { if (ptr) cudaFree(ptr); }
  template <class T> T* as() const { return static_cast<T*>(ptr); }
};

template <class T>
CudaBuffer upload_repeated(std::vector<T> const& host, int copies) {
  std::size_t const one = host.size() * sizeof(T);
  CudaBuffer out(one * std::size_t(copies));
  if (!one) return out;
  cuda_ok(cudaMemcpy(out.ptr, host.data(), one, cudaMemcpyHostToDevice), "H2D operand");
  for (int i = 1; i < copies; ++i)
    cuda_ok(cudaMemcpy(static_cast<std::uint8_t*>(out.ptr) + std::size_t(i) * one,
                       out.ptr, one, cudaMemcpyDeviceToDevice), "D2D operand replica");
  return out;
}

__global__ void flush_l2(std::uint8_t const* p, std::size_t bytes,
                         unsigned long long* sink) {
  std::size_t i = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 128;
  unsigned long long value = 0;
  for (; i < bytes; i += std::size_t(gridDim.x) * blockDim.x * 128) value += p[i];
  if (value) atomicAdd(sink, value);
}

struct NvmlClock {
  nvmlDevice_t device{};
  std::string pci;
  NvmlClock() {
    nvml_ok(nvmlInit_v2(), "nvmlInit_v2");
    int ordinal = 0;
    cuda_ok(cudaGetDevice(&ordinal), "cudaGetDevice");
    char bus[32]{};
    cuda_ok(cudaDeviceGetPCIBusId(bus, sizeof(bus), ordinal), "cudaDeviceGetPCIBusId");
    pci = bus;
    nvml_ok(nvmlDeviceGetHandleByPciBusId_v2(bus, &device), "nvmlDeviceGetHandleByPciBusId_v2");
  }
  ~NvmlClock() { nvmlShutdown(); }
  unsigned sample() const {
    unsigned mhz = 0;
    nvml_ok(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &mhz), "nvmlDeviceGetClockInfo");
    return mhz;
  }
};

struct DeviceProblem {
  HostProblem const* host = nullptr;
  int copies = 1;
  CudaBuffer act, raw, low, scales, zeros, offsets, flush, sink;
  std::size_t raw_one = 0, low_one = 0, scales_one = 0, zeros_one = 0;
  std::size_t flush_bytes = 0;

  DeviceProblem(HostProblem const& h, int count, std::size_t flush_n)
      : host(&h), copies(count),
        act(upload_repeated(h.act, 1)), raw(upload_repeated(h.raw, count)),
        low(upload_repeated(h.low, count)), scales(upload_repeated(h.scales, count)),
        zeros(upload_repeated(h.zeros, count)), flush(flush_n), sink(sizeof(unsigned long long)),
        raw_one(h.raw.size() * sizeof(block_q4_K)), low_one(h.low.size()),
        scales_one(h.scales.size() * sizeof(half)), zeros_one(h.zeros.size() * sizeof(half)),
        flush_bytes(flush_n) {
    cuda_ok(cudaMemset(flush.ptr, 0x5a, flush.bytes), "init flush buffer");
    cuda_ok(cudaMemset(sink.ptr, 0, sink.bytes), "init flush sink");
    if (h.shape.l > 1) {
      std::vector<int> rows(std::size_t(h.shape.l + 1));
      std::iota(rows.begin(), rows.end(), 0);
      offsets = upload_repeated(rows, 1);
    }
    cuda_ok(cudaDeviceSynchronize(), "operand initialization");
  }

  block_q4_K const* raw_at(int copy, int expert = 0) const {
    std::size_t const e_bytes = raw_one / std::size_t(host->shape.l);
    return reinterpret_cast<block_q4_K const*>(
        static_cast<std::uint8_t const*>(raw.ptr) + std::size_t(copy) * raw_one
        + std::size_t(expert) * e_bytes);
  }
  std::uint8_t const* low_at(int copy, int expert = 0) const {
    std::size_t const e_bytes = low_one / std::size_t(host->shape.l);
    return static_cast<std::uint8_t const*>(low.ptr) + std::size_t(copy) * low_one
         + std::size_t(expert) * e_bytes;
  }
  half const* scale_at(int copy, int expert = 0) const {
    std::size_t const e_bytes = scales_one / std::size_t(host->shape.l);
    return reinterpret_cast<half const*>(
        static_cast<std::uint8_t const*>(scales.ptr) + std::size_t(copy) * scales_one
        + std::size_t(expert) * e_bytes);
  }
  half const* zero_at(int copy, int expert = 0) const {
    std::size_t const e_bytes = zeros_one / std::size_t(host->shape.l);
    return reinterpret_cast<half const*>(
        static_cast<std::uint8_t const*>(zeros.ptr) + std::size_t(copy) * zeros_one
        + std::size_t(expert) * e_bytes);
  }
};

enum class ArmKind { PdfScalar, PdfPair, OursDense, OursGrouped };

struct Arm {
  std::string name;
  ArmKind kind{};
  int kernels_per_workload = 1;
  std::string representation;
  std::size_t representation_bytes = 0;
  CudaBuffer output;
  std::uint64_t correctness_hash = 0;
};

template <bool PairMetadata>
void launch_pdf(DeviceProblem const& d, int copy, half* out) {
  Shape const& s = d.host->shape;
  for (int e = 0; e < s.l; ++e) {
    half const* a = d.act.as<half>() + std::size_t(e) * s.k;
    half* o = out + std::size_t(e) * s.n;
    if (s.pdf_cta_n == 2 && s.pdf_warps_n == 8 && s.pdf_warps_k == 1)
      q4k_pdf_reconstruction::launch_q4k_gemv<2, 8, 1, PairMetadata>(
          a, d.raw_at(copy, e), o, 1, s.n, s.k);
    else if (s.pdf_cta_n == 4 && s.pdf_warps_n == 8 && s.pdf_warps_k == 1)
      q4k_pdf_reconstruction::launch_q4k_gemv<4, 8, 1, PairMetadata>(
          a, d.raw_at(copy, e), o, 1, s.n, s.k);
    else
      fail("uninstantiated PDF configuration");
  }
}

ppu_gemv::Params ours_params(DeviceProblem const& d, int copy, int expert, half* out) {
  Shape const& s = d.host->shape;
  ppu_gemv::Params p;
  p.act = d.act.as<half>() + std::size_t(expert) * s.k;
  p.weight = d.low_at(copy, expert);
  p.scales = d.scale_at(copy, expert);
  p.zeros = d.zero_at(copy, expert);
  p.out = out + std::size_t(expert) * s.n;
  p.m = 1; p.n = s.n; p.k = s.k; p.groupsize = 32;
  p.format = ppu_gemv::WFormat::Int4;
  p.quant = ppu_gemv::QuantOp::FinegrainedScaleZero;
  p.layout = ppu_gemv::WLayout::Native;
  return p;
}

void launch_ours_dense(DeviceProblem const& d, int copy, half* out) {
  Shape const& s = d.host->shape;
  for (int e = 0; e < s.l; ++e) {
    auto p = ours_params(d, copy, e, out);
    if (!ppu_gemv::launch_gemv<OursDetails, 8, 2>(p, nullptr)) fail("gemv_lowbit dense launch refused");
  }
}

void launch_ours_grouped(DeviceProblem const& d, int copy, half* out) {
  Shape const& s = d.host->shape;
  ppu_gemv::Params p;
  p.act = d.act.ptr; p.weight = d.low_at(copy); p.scales = d.scale_at(copy);
  p.zeros = d.zero_at(copy); p.out = out;
  p.m = s.l; p.n = s.n; p.k = s.k; p.groupsize = 32;
  p.format = ppu_gemv::WFormat::Int4;
  p.quant = ppu_gemv::QuantOp::FinegrainedScaleZero;
  p.layout = ppu_gemv::WLayout::Native;
  p.num_experts = s.l; p.row_offsets = d.offsets.as<int>(); p.max_rows = 1;
  p.w_bytes_per_expert = std::int64_t(d.low_one / std::size_t(s.l));
  p.scale_elems_per_expert = std::int64_t(d.host->scales.size() / std::size_t(s.l));
  if (!ppu_gemv::launch_gemv<OursDetails, 8, 2>(p, nullptr)) fail("gemv_lowbit grouped launch refused");
}

void launch_arm(Arm const& arm, DeviceProblem const& d, int copy) {
  switch (arm.kind) {
    case ArmKind::PdfScalar: launch_pdf<false>(d, copy, arm.output.as<half>()); break;
    case ArmKind::PdfPair: launch_pdf<true>(d, copy, arm.output.as<half>()); break;
    case ArmKind::OursDense: launch_ours_dense(d, copy, arm.output.as<half>()); break;
    case ArmKind::OursGrouped: launch_ours_grouped(d, copy, arm.output.as<half>()); break;
  }
}

std::vector<Arm> make_arms(HostProblem const& h) {
  Shape const& s = h.shape;
  std::size_t const outputs = std::size_t(s.l) * s.n * sizeof(half);
  std::size_t const raw_bytes = h.raw.size() * sizeof(block_q4_K);
  std::size_t const ours_bytes = h.low.size() + (h.scales.size() + h.zeros.size()) * sizeof(half);
  std::vector<Arm> arms;
  auto add = [&](char const* name, ArmKind kind, int kernels, char const* rep, std::size_t bytes) {
    Arm a;
    a.name = name; a.kind = kind; a.kernels_per_workload = kernels;
    a.representation = rep; a.representation_bytes = bytes; a.output = CudaBuffer(outputs);
    arms.push_back(std::move(a));
  };
  add(s.l == 1 ? "pdf_scalar_dense1" : "pdf_scalar_dense8", ArmKind::PdfScalar, s.l,
      "native_q4k_144B_per_256", raw_bytes);
  add(s.l == 1 ? "pdf_pair_dense1" : "pdf_pair_dense8", ArmKind::PdfPair, s.l,
      "native_q4k_144B_per_256", raw_bytes);
  add(s.l == 1 ? "ours_native_dense1" : "ours_native_dense8", ArmKind::OursDense, s.l,
      "affine_int4_plus_fp16_scale_zero_gs32", ours_bytes);
  if (s.l > 1)
    add("ours_native_grouped1", ArmKind::OursGrouped, 1,
        "affine_int4_plus_fp16_scale_zero_gs32", ours_bytes);
  return arms;
}

void correctness_gate(DeviceProblem const& d, std::vector<Arm>& arms) {
  HostProblem const& h = *d.host;
  std::string why;
  if (!q4k_pdf_ab::verify_representation(h, why)) fail("host representation gate: " + why);
  int const before = ppu_gemv::gemv_fail_count();
  for (Arm& arm : arms) {
    cuda_ok(cudaMemset(arm.output.ptr, 0xa5, arm.output.bytes), "poison output");
    launch_arm(arm, d, 0);
    cuda_ok(cudaGetLastError(), "correctness launch");
    cuda_ok(cudaDeviceSynchronize(), "correctness sync");
    std::vector<half> got(std::size_t(h.shape.l) * h.shape.n);
    cuda_ok(cudaMemcpy(got.data(), arm.output.ptr, arm.output.bytes, cudaMemcpyDeviceToHost),
            "correctness D2H");
    std::size_t bad = 0;
    float max_conditioned = 0.f;
    for (std::size_t i = 0; i < got.size(); ++i) {
      float const value = __half2float(got[i]);
      float const want = __half2float(h.golden[i]);
      float const conditioned = std::fabs(value - want) / std::max(1.f, std::fabs(want));
      max_conditioned = std::max(max_conditioned, conditioned);
      bad += !std::isfinite(value) || conditioned > (1.f / 128.f);
    }
    if (bad) {
      std::ostringstream os;
      os << arm.name << " correctness failed: " << bad << "/" << got.size()
         << " max_conditioned=" << max_conditioned << " (fixed limit=2^-7)";
      fail(os.str());
    }
    arm.correctness_hash = q4k_pdf_ab::fnv1a_half(got);
    std::printf("CORRECT shape=%s arm=%s outputs=%zu max_conditioned=%.8g hash=%016llx\n",
                h.shape.id, arm.name.c_str(), got.size(), double(max_conditioned),
                static_cast<unsigned long long>(arm.correctness_hash));
  }
  if (ppu_gemv::gemv_fail_count() != before) fail("gemv_lowbit refused a correctness launch");
}

struct Options {
  std::string output = "/tmp/q4k_pdf_5090_ab_raw.csv";
  std::string shape;
  std::string git_sha = "UNKNOWN";
  std::string binary_sha = "UNKNOWN";
  int samples = 31;
  int warm_batch = 64;
  int warmup = 100;
  int precondition_ms = 50;
  int cold_budget_mib = 512;
};

Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    auto value = [&](char const* flag) -> char const* {
      if (i + 1 >= argc) fail(std::string("missing value for ") + flag);
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--output")) o.output = value(argv[i]);
    else if (!std::strcmp(argv[i], "--shape")) o.shape = value(argv[i]);
    else if (!std::strcmp(argv[i], "--samples")) o.samples = std::atoi(value(argv[i]));
    else if (!std::strcmp(argv[i], "--warm-batch")) o.warm_batch = std::atoi(value(argv[i]));
    else if (!std::strcmp(argv[i], "--warmup")) o.warmup = std::atoi(value(argv[i]));
    else if (!std::strcmp(argv[i], "--precondition-ms")) o.precondition_ms = std::atoi(value(argv[i]));
    else if (!std::strcmp(argv[i], "--cold-budget-mib")) o.cold_budget_mib = std::atoi(value(argv[i]));
    else if (!std::strcmp(argv[i], "--git-sha")) o.git_sha = value(argv[i]);
    else if (!std::strcmp(argv[i], "--binary-sha")) o.binary_sha = value(argv[i]);
    else fail(std::string("unknown argument: ") + argv[i]);
  }
  if (o.samples <= 0 || o.warm_batch <= 0 || o.warmup < 0 || o.precondition_ms < 0 ||
      o.cold_budget_mib <= 0)
    fail("samples/batches/budget must be positive");
  return o;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct RawSample {
  std::string state;
  int pass = 0;
  int order = 0;
  int arm = 0;
  int batch = 0;
  unsigned clock_mhz = 0;
  int pending = 0;
  cudaEvent_t start{}, stop{};
  float elapsed_ms = 0.f;
};

void write_header(std::FILE* f) {
  std::fprintf(f,
      "schema,git_sha,binary_sha,device_pci,driver,shape_id,m,n,k,l,arm,cache_state,pass,arm_order,"
      "logical_workloads,batch,kernels_per_workload,representation,representation_bytes,distinct_bytes,"
      "cold_copy_count,l2_bytes,flush_bytes,event_ms_bits,event_total_us,event_us_per_workload,"
      "sm_clock_mhz,event_pending_after_clock_query,correctness_hash,pdf_config,pdf_config_authority,"
      "timing_scope,clock_scope\n");
  std::fflush(f);
}

void emit_samples(std::FILE* f, Options const& options, Shape const& shape,
                  std::vector<Arm> const& arms, std::vector<RawSample>& samples,
                  NvmlClock const& clock, int driver, std::size_t l2_bytes,
                  std::size_t flush_bytes, int cold_copies) {
  for (RawSample& sample : samples) {
    cuda_ok(cudaEventElapsedTime(&sample.elapsed_ms, sample.start, sample.stop), "cudaEventElapsedTime");
    Arm const& arm = arms[std::size_t(sample.arm)];
    double const total_us = double(sample.elapsed_ms) * 1000.0;
    double const per = total_us / sample.batch;
    std::size_t const ad = std::size_t(shape.l) * (shape.k + shape.n) * sizeof(half);
    std::size_t const distinct = arm.representation_bytes + ad;
    char cfg[32];
    std::snprintf(cfg, sizeof(cfg), "%dx%dx%d", shape.pdf_cta_n, shape.pdf_warps_n, shape.pdf_warps_k);
    std::fprintf(f,
      "q4k-pdf-ab-raw-v1,%s,%s,%s,%d,%s,1,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%s,%zu,%zu,%d,%zu,%zu,"
      "0x%08x,%.9g,%.9g,%u,%d,%016llx,%s,%s,cuda_event_gpu_span,nvml_adjacent_snapshot\n",
      options.git_sha.c_str(), options.binary_sha.c_str(), clock.pci.c_str(), driver, shape.id,
      shape.n, shape.k, shape.l, arm.name.c_str(), sample.state.c_str(), sample.pass, sample.order,
      sample.batch, sample.batch, arm.kernels_per_workload, arm.representation.c_str(),
      arm.representation_bytes, distinct, cold_copies, l2_bytes, flush_bytes,
      float_bits(sample.elapsed_ms), total_us, per, sample.clock_mhz, sample.pending,
      static_cast<unsigned long long>(arm.correctness_hash), cfg,
      shape.document_winner ? "pdf_p22_winner" : "pdf_documented_default_unmeasured_shape");
    std::fflush(f);
    cudaEventDestroy(sample.start);
    cudaEventDestroy(sample.stop);
  }
}

void measure_state(std::FILE* f, Options const& options, DeviceProblem const& d,
                   std::vector<Arm> const& arms, NvmlClock const& clock,
                   char const* state, int batch, int driver, std::size_t l2_bytes) {
  std::vector<RawSample> raw;
  raw.reserve(std::size_t(options.samples) * arms.size());
  int const before = ppu_gemv::gemv_fail_count();
  for (int pass = 0; pass < options.samples; ++pass) {
    std::vector<int> order(arms.size());
    std::iota(order.begin(), order.end(), 0);
    if (pass & 1) std::reverse(order.begin(), order.end());
    for (int rank = 0; rank < int(order.size()); ++rank) {
      int const ai = order[std::size_t(rank)];
      RawSample sample;
      sample.state = state; sample.pass = pass; sample.order = rank;
      sample.arm = ai; sample.batch = batch;
      cuda_ok(cudaEventCreate(&sample.start), "cudaEventCreate start");
      cuda_ok(cudaEventCreate(&sample.stop), "cudaEventCreate stop");
      if (sample.state == "weight_metadata_cold")
        flush_l2<<<4096, 256>>>(d.flush.as<std::uint8_t>(), d.flush_bytes,
                               d.sink.as<unsigned long long>());
      cuda_ok(cudaEventRecord(sample.start), "record start");
      for (int b = 0; b < batch; ++b)
        launch_arm(arms[std::size_t(ai)], d,
                   sample.state == "weight_metadata_cold" ? b : 0);
      cuda_ok(cudaEventRecord(sample.stop), "record stop");
      sample.clock_mhz = clock.sample();
      cudaError_t const query = cudaEventQuery(sample.stop);
      sample.pending = query == cudaErrorNotReady ? 1 : 0;
      if (query != cudaSuccess && query != cudaErrorNotReady) cuda_ok(query, "query stop event");
      raw.push_back(sample);
    }
  }
  cuda_ok(cudaGetLastError(), "queued timing launches");
  cuda_ok(cudaDeviceSynchronize(), "timing final synchronize");
  if (ppu_gemv::gemv_fail_count() != before) fail("gemv_lowbit refused a timed launch");
  emit_samples(f, options, d.host->shape, arms, raw, clock, driver, l2_bytes,
               d.flush_bytes, d.copies);
}

void warmup(DeviceProblem const& d, std::vector<Arm> const& arms, int count) {
  for (int i = 0; i < count; ++i)
    for (Arm const& arm : arms) launch_arm(arm, d, 0);
  cuda_ok(cudaGetLastError(), "warmup launches");
  cuda_ok(cudaDeviceSynchronize(), "warmup synchronize");
}

void clock_precondition(DeviceProblem const& d, std::vector<Arm> const& arms, int milliseconds) {
  auto const until = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
  std::size_t launches = 0;
  while (std::chrono::steady_clock::now() < until) {
    launch_arm(arms[launches % arms.size()], d, 0);
    ++launches;
  }
  cuda_ok(cudaGetLastError(), "clock precondition launches");
  cuda_ok(cudaDeviceSynchronize(), "clock precondition synchronize");
  std::printf("CLOCK-PRECONDITION shape=%s host_enqueue_ms=%d logical_workloads=%zu\n",
              d.host->shape.id, milliseconds, launches);
}

void print_summary(Shape const& shape, std::vector<Arm> const& arms) {
  std::printf("SHAPE-DONE id=%s M=1 N=%d K=%d L=%d config=%dx%dx%d authority=%s arms=%zu\n",
              shape.id, shape.n, shape.k, shape.l,
              shape.pdf_cta_n, shape.pdf_warps_n, shape.pdf_warps_k,
              shape.document_winner ? "pdf-p22-winner" : "document-default-not-measured", arms.size());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options const options = parse_options(argc, argv);
    int ordinal = 0;
    cuda_ok(cudaSetDevice(ordinal), "cudaSetDevice");
    cudaDeviceProp prop{};
    cuda_ok(cudaGetDeviceProperties(&prop, ordinal), "cudaGetDeviceProperties");
    if (prop.major != 12 || prop.minor != 0) fail("this evidence target requires sm_120 / RTX 5090");
    int driver = 0;
    cuda_ok(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    std::size_t const flush_bytes = std::max<std::size_t>(std::size_t(prop.l2CacheSize) * 2,
                                                          std::size_t(128) << 20);
    NvmlClock clock;
    std::FILE* f = std::fopen(options.output.c_str(), "w");
    if (!f) fail("cannot open raw CSV: " + options.output + ": " + std::strerror(errno));
    write_header(f);
    bool ran = false;
    for (Shape const& shape : q4k_pdf_ab::kShapes) {
      if (!options.shape.empty() && options.shape != shape.id) continue;
      ran = true;
      std::printf("BUILD-FIXTURE id=%s M=1 N=%d K=%d L=%d\n", shape.id, shape.n, shape.k, shape.l);
      HostProblem host = q4k_pdf_ab::make_problem(shape);
      std::size_t const raw_bytes = host.raw.size() * sizeof(block_q4_K);
      std::size_t const ours_bytes = host.low.size()
                                   + (host.scales.size() + host.zeros.size()) * sizeof(half);
      std::size_t const maximum = std::max(raw_bytes, ours_bytes);
      std::size_t const budget = std::size_t(options.cold_budget_mib) << 20;
      int const copies = std::max(1, std::min(64, int(budget / maximum)));
      DeviceProblem device(host, copies, flush_bytes);
      std::vector<Arm> arms = make_arms(host);
      correctness_gate(device, arms);
      warmup(device, arms, options.warmup);
      clock_precondition(device, arms, options.precondition_ms);
      measure_state(f, options, device, arms, clock, "weight_metadata_cold", copies,
                    driver, prop.l2CacheSize);
      warmup(device, arms, options.warmup);
      clock_precondition(device, arms, options.precondition_ms);
      measure_state(f, options, device, arms, clock, "warm", options.warm_batch,
                    driver, prop.l2CacheSize);
      print_summary(shape, arms);
    }
    std::fclose(f);
    if (!ran) fail("shape filter selected nothing");
    std::printf("RAW-CSV %s\n", options.output.c_str());
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "q4k_pdf_5090_ab: %s\n", e.what());
    return 1;
  }
}
