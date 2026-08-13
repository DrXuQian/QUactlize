// INBOX 144: measurement-only RTX 5090 dense vLLM Marlin axis probe.
//
// The kernel, selector, launch ABI, and helper types have one source authority:
// /root/ref5090/marlin/fullrun/marlin_fullrun.cu.  This translation unit includes
// that pinned source after renaming its main; it does not copy those definitions.
// The Python runner verifies the authority SHA before compiling this file.

#define BUILD_DENSE 1
#define main quactlize_pinned_vllm_marlin_fullrun_main
#include "/root/ref5090/marlin/fullrun/marlin_fullrun.cu"
#undef main

#include <nvml.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <set>

namespace {

constexpr char kSchema[] = "vllm-marlin-dense-axis-raw-v1";
constexpr char kVllmCommit[] = "11ba93f3646d4c5476c3b3fd56835589701f0fb1";
constexpr int kAxisGroupSize = 32;
constexpr char kCorrectnessFixture[] =
    "exact_q9_a1_scale2m8_expectedKover256_fp16bits_v1";
constexpr char kTimingFixture[] =
    "pinned_fullrun_seeds_b57a41d9b_s16334a2f_a91104f23_v1";

__global__ void fill_half_constant_kernel(half* dst, std::size_t count,
                                          half value) {
  std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t stride = std::size_t(blockDim.x) * gridDim.x;
  for (; i < count; i += stride) dst[i] = value;
}

__global__ void flush_l2_kernel(std::uint8_t const* src, std::size_t bytes,
                                unsigned long long* sink) {
  std::size_t i =
      (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 128;
  unsigned long long value = 0;
  for (; i < bytes; i += std::size_t(gridDim.x) * blockDim.x * 128)
    value += src[i];
  if (value) atomicAdd(sink, value);
}

void fill_half_constant(Buffer& buffer, std::size_t count, float value,
                        cudaStream_t stream) {
  int blocks = int(std::min<std::size_t>(
      4096, std::max<std::size_t>(1, (count + 255) / 256)));
  fill_half_constant_kernel<<<blocks, 256, 0, stream>>>(
      buffer.as<half>(), count, __float2half(value));
  check(cudaGetLastError(), "fill_half_constant_kernel");
}

struct NvmlIdentity {
  nvmlDevice_t device{};
  std::string pci;
  std::string name;
  std::string driver;

  NvmlIdentity() {
    nvmlReturn_t rc = nvmlInit_v2();
    if (rc != NVML_SUCCESS)
      throw std::runtime_error(std::string("nvmlInit_v2: ") +
                               nvmlErrorString(rc));
    int ordinal = 0;
    check(cudaGetDevice(&ordinal), "cudaGetDevice");
    char bus[32]{};
    check(cudaDeviceGetPCIBusId(bus, sizeof(bus), ordinal),
          "cudaDeviceGetPCIBusId");
    pci = bus;
    rc = nvmlDeviceGetHandleByPciBusId_v2(bus, &device);
    if (rc != NVML_SUCCESS)
      throw std::runtime_error(std::string("nvmlDeviceGetHandleByPciBusId_v2: ") +
                               nvmlErrorString(rc));
    char device_name[NVML_DEVICE_NAME_BUFFER_SIZE]{};
    char driver_version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE]{};
    rc = nvmlDeviceGetName(device, device_name, sizeof(device_name));
    if (rc != NVML_SUCCESS)
      throw std::runtime_error(std::string("nvmlDeviceGetName: ") +
                               nvmlErrorString(rc));
    rc = nvmlSystemGetDriverVersion(driver_version, sizeof(driver_version));
    if (rc != NVML_SUCCESS)
      throw std::runtime_error(std::string("nvmlSystemGetDriverVersion: ") +
                               nvmlErrorString(rc));
    name = device_name;
    driver = driver_version;
  }

  ~NvmlIdentity() { nvmlShutdown(); }

  unsigned sm_clock_mhz() const {
    unsigned value = 0;
    nvmlReturn_t rc =
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &value);
    if (rc != NVML_SUCCESS)
      throw std::runtime_error(std::string("nvmlDeviceGetClockInfo: ") +
                               nvmlErrorString(rc));
    return value;
  }
};

struct Options {
  std::string output;
  std::string shapes;
  std::string repo_sha;
  std::string authority_sha;
  std::string binary_sha;
  int samples = 31;
  int warm_batch = 64;
  int warmups = 100;
  int precondition_ms = 50;
  int cold_budget_mib = 512;
};

char const* take_value(int argc, char** argv, int& i) {
  if (++i >= argc) throw std::runtime_error("missing option value");
  return argv[i];
}

Options parse_axis_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--output"))
      options.output = take_value(argc, argv, i);
    else if (!std::strcmp(argv[i], "--shapes"))
      options.shapes = take_value(argc, argv, i);
    else if (!std::strcmp(argv[i], "--repo-sha"))
      options.repo_sha = take_value(argc, argv, i);
    else if (!std::strcmp(argv[i], "--authority-sha"))
      options.authority_sha = take_value(argc, argv, i);
    else if (!std::strcmp(argv[i], "--binary-sha"))
      options.binary_sha = take_value(argc, argv, i);
    else if (!std::strcmp(argv[i], "--samples"))
      options.samples = std::stoi(take_value(argc, argv, i));
    else if (!std::strcmp(argv[i], "--warm-batch"))
      options.warm_batch = std::stoi(take_value(argc, argv, i));
    else if (!std::strcmp(argv[i], "--warmups"))
      options.warmups = std::stoi(take_value(argc, argv, i));
    else if (!std::strcmp(argv[i], "--precondition-ms"))
      options.precondition_ms = std::stoi(take_value(argc, argv, i));
    else if (!std::strcmp(argv[i], "--cold-budget-mib"))
      options.cold_budget_mib = std::stoi(take_value(argc, argv, i));
    else
      throw std::runtime_error(std::string("unknown option: ") + argv[i]);
  }
  if (options.output.empty() || options.shapes.empty() ||
      options.repo_sha.empty() || options.authority_sha.empty() ||
      options.binary_sha.empty())
    throw std::runtime_error(
        "--output/--shapes/--repo-sha/--authority-sha/--binary-sha are required");
  if (options.samples <= 0 || options.warm_batch <= 0 ||
      options.warmups < 0 || options.precondition_ms < 0 ||
      options.cold_budget_mib <= 0)
    throw std::runtime_error("invalid timing protocol count");
  return options;
}

std::vector<std::pair<int, int>> parse_shapes(std::string const& text) {
  std::vector<std::pair<int, int>> shapes;
  std::set<std::pair<int, int>> seen;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    std::size_t x = item.find('x');
    if (x == std::string::npos)
      throw std::runtime_error("shape must be KxN: " + item);
    int k = std::stoi(item.substr(0, x));
    int n = std::stoi(item.substr(x + 1));
    if (k <= 0 || n <= 0 || k % kAxisGroupSize || n % 64)
      throw std::runtime_error("invalid Marlin shape: " + item);
    if (!seen.emplace(k, n).second)
      throw std::runtime_error("duplicate shape: " + item);
    shapes.emplace_back(k, n);
  }
  if (shapes.empty()) throw std::runtime_error("shape list is empty");
  return shapes;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t fnv1a_half(std::vector<half> const& values) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (half value : values) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift : {0, 8}) {
      hash ^= std::uint8_t(bits >> shift);
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
}

std::string csv_token(std::string value) {
  for (char& c : value)
    if (c == ',') c = ';';
  return value;
}

struct RawSample {
  char const* state = nullptr;
  int pass = 0;
  int batch = 0;
  int cold_copies = 0;
  cudaEvent_t start{};
  cudaEvent_t stop{};
  unsigned sm_clock = 0;
  int pending = 0;
  float elapsed_ms = 0;
};

struct ShapeRun {
  int k = 0;
  int n = 0;
  int sms = 0;
  int max_shared = 0;
  int copies = 1;
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  std::size_t c_bytes = 0;
  std::size_t scale_bytes = 0;
  std::size_t distinct_bytes = 0;
  Config cfg;
  Buffer a;
  Buffer b;
  Buffer c;
  Buffer scales;
  Buffer ctmp;
  Buffer workspace;
  Buffer a_scales;
  Buffer dummy;
  Buffer flush;
  Buffer flush_sink;
  std::uint64_t correctness_hash = 0;
  std::uint16_t expected_bits = 0;

  ShapeRun(int k_, int n_, int sms_, int max_shared_, std::size_t flush_bytes,
           int cold_budget_mib, cudaStream_t stream)
      : k(k_), n(n_), sms(sms_), max_shared(max_shared_),
        a_bytes(std::size_t(k) * sizeof(half)),
        b_bytes(std::size_t(k) * n / 2),
        c_bytes(std::size_t(n) * sizeof(half)),
        scale_bytes(std::size_t(k / kAxisGroupSize) * n * sizeof(half)),
        distinct_bytes(a_bytes + b_bytes + c_bytes + scale_bytes),
        cfg(select_dense_config(1, n, k, max_shared, sms)),
        a(a_bytes),
        b(b_bytes * std::size_t(std::max(
                      1, std::min(64, int((std::size_t(cold_budget_mib) << 20) /
                                          (b_bytes + scale_bytes)))))),
        c(c_bytes),
        scales(scale_bytes * std::size_t(std::max(
                             1, std::min(64, int((std::size_t(cold_budget_mib) << 20) /
                                                 (b_bytes + scale_bytes)))))),
        ctmp(std::size_t(sms) * 16 * kMaxThreadN * sizeof(float)),
        workspace(std::size_t(sms) * sizeof(int)),
        a_scales(sizeof(float)),
        dummy(16),
        flush(flush_bytes),
        flush_sink(sizeof(unsigned long long)) {
    copies = int(b.bytes / b_bytes);
    if (copies <= 0 || scales.bytes / scale_bytes != std::size_t(copies))
      throw std::runtime_error("cold replica allocation invariant failed");
    check(cudaFuncSetAttribute(cfg.function,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               cfg.dynamic_smem),
          "cudaFuncSetAttribute(dense axis)");
    fill_half_constant(a, std::size_t(k), 1.0f, stream);
    check(cudaMemsetAsync(b.ptr, 0x99, b.bytes, stream), "fill q=9 weights");
    fill_half_constant(scales, scales.bytes / sizeof(half), 1.0f / 256.0f,
                       stream);
    check(cudaMemsetAsync(c.ptr, 0xa5, c.bytes, stream), "poison output");
    check(cudaMemsetAsync(ctmp.ptr, 0, ctmp.bytes, stream), "zero ctmp");
    check(cudaMemsetAsync(workspace.ptr, 0, workspace.bytes, stream),
          "zero locks");
    check(cudaMemsetAsync(a_scales.ptr, 0, a_scales.bytes, stream),
          "zero a_scales");
    check(cudaMemsetAsync(dummy.ptr, 0, dummy.bytes, stream), "zero dummy");
    check(cudaMemsetAsync(flush.ptr, 0x5a, flush.bytes, stream),
          "initialize L2 flush buffer");
    check(cudaMemsetAsync(flush_sink.ptr, 0, flush_sink.bytes, stream),
          "initialize L2 flush sink");
    check(cudaStreamSynchronize(stream), "fixture initialization");
  }

  DenseLaunch launch_for(int copy) const {
    if (copy < 0 || copy >= copies)
      throw std::runtime_error("cold copy index out of range");
    DenseLaunch launch;
    launch.cfg = cfg;
    launch.a = reinterpret_cast<int4 const*>(a.ptr);
    launch.b = reinterpret_cast<int4 const*>(
        static_cast<char const*>(b.ptr) + std::size_t(copy) * b_bytes);
    launch.c = reinterpret_cast<int4*>(c.ptr);
    launch.c_tmp = reinterpret_cast<int4*>(ctmp.ptr);
    launch.bias = dummy.as_const<int4>();
    launch.a_scales = a_scales.as_const<float>();
    launch.b_scales = reinterpret_cast<int4 const*>(
        static_cast<char const*>(scales.ptr) + std::size_t(copy) * scale_bytes);
    launch.global_scale = dummy.as_const<float>();
    launch.zp = dummy.as_const<int4>();
    launch.g_idx = dummy.as_const<int>();
    launch.num_groups = k / kAxisGroupSize;
    launch.m = 1;
    launch.n = n;
    launch.k = k;
    launch.lda = k;
    launch.locks = reinterpret_cast<int*>(workspace.ptr);
    launch.max_shared_mem = cfg.dynamic_smem;
    launch.sms = sms;
    return launch;
  }

  void launch(int copy, cudaStream_t stream) const {
    launch_for(copy).launch(stream);
  }
};

void correctness_gate(ShapeRun& shape, cudaStream_t stream) {
  check(cudaMemsetAsync(shape.c.ptr, 0xa5, shape.c.bytes, stream),
        "poison correctness output");
  check(cudaMemsetAsync(shape.ctmp.ptr, 0, shape.ctmp.bytes, stream),
        "reset correctness ctmp");
  check(cudaMemsetAsync(shape.workspace.ptr, 0, shape.workspace.bytes, stream),
        "reset correctness locks");
  shape.launch(0, stream);
  check(cudaGetLastError(), "correctness launch");
  check(cudaStreamSynchronize(stream), "correctness synchronize");
  std::vector<half> got(std::size_t(shape.n));
  check(cudaMemcpy(got.data(), shape.c.ptr, shape.c.bytes,
                   cudaMemcpyDeviceToHost),
        "correctness D2H");
  half expected = __float2half(float(shape.k) / 256.0f);
  std::memcpy(&shape.expected_bits, &expected, sizeof(expected));
  std::size_t bad = 0;
  for (half value : got) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bad += bits != shape.expected_bits;
  }
  if (bad) {
    std::ostringstream out;
    out << "constant q=9 correctness failed: " << bad << '/' << got.size()
        << " expected=" << __half2float(expected)
        << " expected_bits=0x" << std::hex << shape.expected_bits;
    throw std::runtime_error(out.str());
  }
  shape.correctness_hash = fnv1a_half(got);
  std::cout << "CORRECT K=" << shape.k << " N=" << shape.n
            << " outputs=" << got.size() << " bad=0 expected="
            << __half2float(expected) << " expected_bits=0x" << std::hex
            << shape.expected_bits << " hash=0x" << shape.correctness_hash
            << std::dec << '\n';
}

void prepare_timing_fixture(ShapeRun& shape, cudaStream_t stream) {
  // These are the exact three fill operations and seeds used by the pinned
  // dense fullrun.  The first B/scale copy is therefore fixture-identical to
  // that archive; later copies are non-overlapping random continuations for
  // the cold batch rather than duplicated constant/compressible operands.
  fill_u32(shape.b, 0x57a41d9bU, stream);
  fill_half(shape.scales, shape.scales.bytes / sizeof(half), 0x16334a2fU,
            true, stream);
  fill_half(shape.a, std::size_t(shape.k), 0x91104f23U, false, stream);
  check(cudaMemsetAsync(shape.c.ptr, 0, shape.c.bytes, stream),
        "reset timing C");
  check(cudaMemsetAsync(shape.ctmp.ptr, 0, shape.ctmp.bytes, stream),
        "reset timing Ctmp");
  check(cudaMemsetAsync(shape.workspace.ptr, 0, shape.workspace.bytes, stream),
        "reset timing locks");
  check(cudaStreamSynchronize(stream), "timing fixture synchronize");
  std::cout << "TIMING-FIXTURE K=" << shape.k << " N=" << shape.n
            << " identity=" << kTimingFixture << '\n';
}

void warmup(ShapeRun const& shape, int count, cudaStream_t stream) {
  for (int i = 0; i < count; ++i) shape.launch(0, stream);
  check(cudaGetLastError(), "warmup launch");
  check(cudaStreamSynchronize(stream), "warmup synchronize");
}

void precondition(ShapeRun const& shape, int milliseconds,
                  cudaStream_t stream) {
  auto until = std::chrono::steady_clock::now() +
               std::chrono::milliseconds(milliseconds);
  std::size_t launches = 0;
  while (std::chrono::steady_clock::now() < until) {
    shape.launch(0, stream);
    ++launches;
  }
  check(cudaGetLastError(), "clock precondition launch");
  check(cudaStreamSynchronize(stream), "clock precondition synchronize");
  std::cout << "CLOCK-PRECONDITION K=" << shape.k << " N=" << shape.n
            << " host_enqueue_ms=" << milliseconds
            << " logical_workloads=" << launches << '\n';
}

std::vector<RawSample> measure_state(ShapeRun const& shape, char const* state,
                                     int batch, int samples,
                                     NvmlIdentity const& identity,
                                     cudaStream_t stream) {
  std::vector<RawSample> raw;
  raw.reserve(samples);
  for (int pass = 0; pass < samples; ++pass) {
    RawSample sample;
    sample.state = state;
    sample.pass = pass;
    sample.batch = batch;
    sample.cold_copies = !std::strcmp(state, "weight_metadata_cold")
                             ? shape.copies
                             : 0;
    check(cudaEventCreate(&sample.start), "create start event");
    check(cudaEventCreate(&sample.stop), "create stop event");
    if (sample.cold_copies)
      flush_l2_kernel<<<4096, 256>>>(
          shape.flush.as_const<std::uint8_t>(), shape.flush.bytes,
          reinterpret_cast<unsigned long long*>(shape.flush_sink.ptr));
    check(cudaEventRecord(sample.start, stream), "record start event");
    for (int i = 0; i < batch; ++i)
      shape.launch(sample.cold_copies ? i : 0, stream);
    check(cudaEventRecord(sample.stop, stream), "record stop event");
    sample.sm_clock = identity.sm_clock_mhz();
    cudaError_t pending = cudaEventQuery(sample.stop);
    sample.pending = pending == cudaErrorNotReady ? 1 : 0;
    if (pending != cudaSuccess && pending != cudaErrorNotReady)
      check(pending, "query stop event");
    raw.push_back(sample);
  }
  check(cudaGetLastError(), "queued timing launches");
  check(cudaStreamSynchronize(stream), "timing final synchronize");
  for (RawSample& sample : raw) {
    check(cudaEventElapsedTime(&sample.elapsed_ms, sample.start, sample.stop),
          "elapsed event time");
    cudaEventDestroy(sample.start);
    cudaEventDestroy(sample.stop);
  }
  return raw;
}

void write_header(std::ofstream& out) {
  out << "schema,repo_git_sha,authority_source_sha,binary_sha,vllm_commit,"
         "device_name,device_pci,driver_version,shape_id,m,n,k,group_size,"
         "cache_state,pass,arm_order,protocol_order,batch,cold_copy_count,"
         "l2_bytes,flush_bytes,event_ms_bits,event_total_us,"
         "event_us_per_workload,sm_clock_mhz,event_pending_after_clock_query,"
         "correctness_hash,correctness_expected_bits,correctness_fixture,"
         "timing_fixture,distinct_bytes,"
         "weight_bytes,scale_bytes,activation_bytes,output_bytes,kernel_config,"
         "samples_requested,warmup_rounds,precondition_host_enqueue_ms,"
         "cold_budget_mib,timing_scope,clock_scope\n";
}

void write_samples(std::ofstream& out, Options const& options,
                   ShapeRun const& shape, std::vector<RawSample> const& raw,
                   NvmlIdentity const& identity, std::size_t l2_bytes) {
  std::string config = csv_token(shape.cfg.label("vllm::marlin"));
  for (RawSample const& sample : raw) {
    double total_us = double(sample.elapsed_ms) * 1000.0;
    out << kSchema << ',' << options.repo_sha << ',' << options.authority_sha
        << ',' << options.binary_sha << ',' << kVllmCommit << ','
        << csv_token(identity.name) << ',' << identity.pci << ','
        << identity.driver << ",K" << shape.k << "_N" << shape.n
        << ",1," << shape.n << ',' << shape.k << ',' << kAxisGroupSize << ','
        << sample.state << ',' << sample.pass
        << ",0,single_arm_no_counterbalance," << sample.batch << ','
        << sample.cold_copies << ',' << l2_bytes << ',' << shape.flush.bytes
        << ",0x" << std::hex << std::setw(8) << std::setfill('0')
        << float_bits(sample.elapsed_ms) << std::dec << std::setfill(' ')
        << ',' << std::setprecision(12) << total_us << ','
        << total_us / sample.batch << ',' << sample.sm_clock << ','
        << sample.pending << ',' << std::hex << std::setw(16)
        << std::setfill('0') << shape.correctness_hash << ",0x"
        << std::setw(4) << shape.expected_bits << std::dec << std::setfill(' ')
        << ',' << kCorrectnessFixture << ',' << kTimingFixture
        << ',' << shape.distinct_bytes << ',' << shape.b_bytes << ','
        << shape.scale_bytes << ',' << shape.a_bytes << ',' << shape.c_bytes
        << ',' << config << ',' << options.samples << ',' << options.warmups
        << ',' << options.precondition_ms << ',' << options.cold_budget_mib
        << ",cuda_event_gpu_span,nvml_adjacent_snapshot\n";
  }
  out.flush();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options const options = parse_axis_options(argc, argv);
    auto const shapes = parse_shapes(options.shapes);
    check(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    if (prop.major != 12 || prop.minor != 0)
      throw std::runtime_error("INBOX 144 evidence requires sm_120 / RTX 5090");
    int sms = 0;
    int max_shared = 0;
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0),
          "cudaDevAttrMultiProcessorCount");
    check(cudaDeviceGetAttribute(&max_shared,
                                 cudaDevAttrMaxSharedMemoryPerBlockOptin, 0),
          "cudaDevAttrMaxSharedMemoryPerBlockOptin");
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    NvmlIdentity identity;
    std::size_t flush_bytes = std::max<std::size_t>(
        std::size_t(prop.l2CacheSize) * 2, std::size_t(128) << 20);
    std::ofstream output(options.output, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open raw output");
    write_header(output);
    for (auto const& [k, n] : shapes) {
      ShapeRun shape(k, n, sms, max_shared, flush_bytes,
                     options.cold_budget_mib, stream);
      correctness_gate(shape, stream);
      prepare_timing_fixture(shape, stream);
      warmup(shape, options.warmups, stream);
      precondition(shape, options.precondition_ms, stream);
      auto cold = measure_state(shape, "weight_metadata_cold", shape.copies,
                                options.samples, identity, stream);
      write_samples(output, options, shape, cold, identity, prop.l2CacheSize);
      warmup(shape, options.warmups, stream);
      precondition(shape, options.precondition_ms, stream);
      auto warm = measure_state(shape, "warm", options.warm_batch,
                                options.samples, identity, stream);
      write_samples(output, options, shape, warm, identity, prop.l2CacheSize);
      std::cout << "SHAPE-DONE K=" << k << " N=" << n
                << " cold_batch=" << shape.copies
                << " warm_batch=" << options.warm_batch
                << " distinct_bytes=" << shape.distinct_bytes << '\n';
    }
    cudaStreamDestroy(stream);
    std::cout << "RAW-CSV " << options.output << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "vllm_marlin_dense_axis_5090: " << error.what() << '\n';
    return 1;
  }
}
