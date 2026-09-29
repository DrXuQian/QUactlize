// Physical Hopper only. This is deliberately separate from the single-launch
// simulator/numerical application. Correctness precedes all timing.
#include "fixture.cuh"
#include <memory>

struct Plan {
  q17_plan_v1 *p = nullptr;
  explicit Plan(q17_call_v1 const &call) { check(q17_prepare_v1(&call, &p), "prepare"); }
  ~Plan() { q17_destroy_v1(p); }
};

template <int Q, class T> void benchmark(Options const &o, bool rotating) {
  auto counts = o.grouped ? std::vector<int>{0, 1, 8, o.m} : std::vector<int>{o.m};
  std::vector<int32_t> offsets{0};
  for (int rows : counts) offsets.push_back(offsets.back() + rows);
  int m = offsets.back(), e = int(counts.size());
  q17_call_v1 c{};
  c.version = 1; c.struct_bytes = sizeof(c);
  auto &p = c.problem;
  p.version = 1; p.struct_bytes = sizeof(p);
  p.qtype = Q; p.operation = o.grouped ? Q17_GROUPED : Q17_DENSE;
  p.route = o.sf ? Q17_SF : Q17_FQ;
  p.compute_type = o.compute; p.input_type = o.compute; p.output_type = Q17_F32;
  p.m = m; p.n = o.n; p.k = o.k; p.experts = e;
  p.max_rows = *std::max_element(counts.begin(), counts.end());
  check(q17_canonical_arrangement_v1(Q, &p.arrangement), "arrangement");
  q17_sizes_v1 sizes{};
  check(q17_sizes_query_v1(&p, &sizes), "sizes");
  check(q17_config_get_v1(&p, o.config, &c.config), "config");
  Fixture<Q, T> fixture(o.n, o.k, e);
  c.a_stride = o.k + 8; c.output_stride = o.n + 8;
  std::vector<T> a(size_t(m) * c.a_stride, T(0.f));
  for (int row = 0; row < m; ++row) for (int k = 0; k < o.k; ++k)
    a[size_t(row) * c.a_stride + k] = T(float((row * 3 + k * 5) % 29 - 14) / 16.f);
  std::vector<float> initial(size_t(m) * c.output_stride + 16, -12345.f), result(initial), expected(size_t(m) * o.n);
  // Complete typed-weight CPU dot oracle, outside every timed interval.
  for (int expert = 0; expert < e; ++expert) for (int row = offsets[expert]; row < offsets[expert+1]; ++row)
    for (int col = 0; col < o.n; ++col) {
      double sum = 0;
      for (int k = 0; k < o.k; ++k)
        sum += double(float(a[size_t(row) * c.a_stride + k])) * float(Fixture<Q,T>::oracle_weight(expert,col,k));
      expected[size_t(row) * o.n + col] = float(sum);
    }
  gpu(cudaGetDevice(&c.device), "device");
  cudaDeviceProp prop{};
  gpu(cudaGetDeviceProperties(&prop, c.device), "properties");
  if (prop.major != 9 || prop.minor != 0 || prop.l2CacheSize <= 0)
    throw std::runtime_error("physical SM90 with reported L2 size required");
  size_t active = std::count_if(counts.begin(), counts.end(), [](int rows) { return rows > 0; });
  size_t weights = (fixture.low.size() + fixture.high.size() +
      (o.sf ? (fixture.scale.size()+fixture.zero.size())*sizeof(T) : fixture.units.size())) / e * active;
  // Every used weight address recurs only after >2x the measured L2 capacity.
  int copies = rotating ? int(2*size_t(prop.l2CacheSize)/weights)+1 : 1;
  if (copies > 1024) throw std::runtime_error("rotating inventory exceeds bounded capacity");
  int launches = copies * ((32 + copies - 1) / copies);
  cudaStream_t stream{};
  gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
  c.stream = stream;
  DeviceMemory da(a.size()*sizeof(T), a.data()), dout(initial.size()*sizeof(float), initial.data()),
      ro(offsets.size()*sizeof(int32_t), offsets.data());
  c.a = da.p; c.output = static_cast<float *>(dout.p)+8; c.offsets = static_cast<int32_t *>(ro.p);
  std::vector<std::unique_ptr<DeviceMemory>> storage;
  std::vector<std::unique_ptr<Plan>> plans;
  auto upload = [&](auto const &host) {
    auto block = std::make_unique<DeviceMemory>(host.size()*sizeof(host[0]), host.data());
    void *ptr = block->p; storage.push_back(std::move(block)); return ptr;
  };
  for (int i = 0; i < copies; ++i) {
    c.low=upload(fixture.low); c.high=upload(fixture.high); c.units=upload(fixture.units);
    c.scale=upload(fixture.scale); c.zero=upload(fixture.zero);
    plans.push_back(std::make_unique<Plan>(c));
  }
  gpu(cudaDeviceSynchronize(), "upload completion");
  // The runner verifies our live GPU PID before admitting any kernel work.
  std::puts("PPU17_BENCH_READY"); std::fflush(stdout);
  if (std::getchar() != 'R') throw std::runtime_error("idle guard did not admit execution");
  auto verify = [&] {
    gpu(cudaStreamSynchronize(stream), "correctness completion");
    gpu(cudaMemcpy(result.data(),dout.p,result.size()*sizeof(float),cudaMemcpyDeviceToHost),"download");
    for (int row=0; row<m; ++row) for (int col=0; col<o.n; ++col) {
      float got=result[8+size_t(row)*c.output_stride+col], want=expected[size_t(row)*o.n+col];
      if (!std::isfinite(got) || std::abs(got-want)>1.e-4*std::max(1.f,std::abs(want)))
        throw std::runtime_error("independent CPU dot mismatch");
    }
    for (size_t i=0; i<result.size(); ++i) {
      bool body=i>=8 && i<result.size()-8;
      if ((!body || (i-8)%c.output_stride>=size_t(o.n)) && result[i]!=-12345.f)
        throw std::runtime_error("output guard overwritten");
    }
  };
  for (auto const &plan: plans) { check(q17_execute_v1(plan->p),"correctness execute"); verify(); }
  cudaGraph_t graph{}; cudaGraphExec_t executable{};
  gpu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"capture begin");
  for (int i=0; i<launches; ++i) check(q17_execute_v1(plans[i%copies]->p),"capture execute");
  gpu(cudaStreamEndCapture(stream,&graph),"capture end");
  gpu(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0),"instantiate");
  for (int i=0; i<3; ++i) gpu(cudaGraphLaunch(executable,stream),"warmup");
  verify();
  cudaEvent_t begin{}, end{};
  gpu(cudaEventCreate(&begin),"event create"); gpu(cudaEventCreate(&end),"event create");
  std::vector<double> samples;
  for (int i=0; i<11; ++i) {
    gpu(cudaEventRecord(begin,stream),"begin");
    gpu(cudaGraphLaunch(executable,stream),"timed graph");
    gpu(cudaEventRecord(end,stream),"end"); gpu(cudaEventSynchronize(end),"event wait");
    float ms=0;
    gpu(cudaEventElapsedTime(&ms,begin,end),"event interval");
    if (!(ms>0) || !std::isfinite(ms)) throw std::runtime_error("invalid device timing");
    samples.push_back(ms*1000./launches);
  }
  verify();
  auto sorted=samples; std::sort(sorted.begin(),sorted.end());
  std::printf("PPU17_BENCH {\"q\":%d,\"compute\":%d,\"config\":%d,\"route\":%d,\"m\":%d,\"n\":%d,\"k\":%d,\"experts\":%d,\"sm_count\":%d,\"l2_bytes\":%d,\"cache\":\"%s\",\"weight_bytes\":%zu,\"weight_copies\":%d,\"launches_per_graph\":%d,\"warmup_graphs\":3,\"correctness\":\"CPU_DOT_AND_GUARDS_PASS\",\"median_us\":%.9f,\"samples_us\":[",
    Q,o.compute,o.config,p.route,m,o.n,o.k,e,prop.multiProcessorCount,prop.l2CacheSize,rotating?"rotating":"warm",weights,copies,launches,sorted[sorted.size()/2]);
  for (size_t i=0;i<samples.size();++i) std::printf("%s%.9f",i?",":"",samples[i]);
  std::printf("],\"scope\":\"PHYSICAL_HOPPER_GRAPH_AVERAGE_NOT_PPU17\"}\n");
  gpu(cudaEventDestroy(begin),"event destroy"); gpu(cudaEventDestroy(end),"event destroy");
  gpu(cudaGraphExecDestroy(executable),"graph destroy"); gpu(cudaGraphDestroy(graph),"graph destroy");
  gpu(cudaStreamDestroy(stream),"stream destroy");
}

int main(int argc,char **argv) {
  try {
    if (!std::getenv("Q17_BENCH_GUARDED"))
      throw std::runtime_error("use run_benchmark.py to monitor exclusive GPU use");
    Options o; o.n=512; o.k=2048; o.m=128;
    bool rotating=false;
    for (int i=1;i<argc;++i) {
      std::string key=argv[i];
      if(key=="--grouped") o.grouped=true;
      else if(key=="--sf") o.sf=true;
      else if(key=="--rotating") rotating=true;
      else {
        if(++i==argc) throw std::runtime_error("missing value");
        int v=std::stoi(argv[i]);
        if(key=="--qtype") o.q=v;
        else if(key=="--compute") o.compute=v;
        else if(key=="--config") o.config=v;
        else if(key=="--m") o.m=v;
        else if(key=="--n") o.n=v;
        else if(key=="--k") o.k=v;
        else throw std::runtime_error("unknown option");
      }
    }
    if(o.compute!=0 && o.compute!=1) throw std::runtime_error("compute must be F16/BF16");
    switch(o.q) {
#define CASE(Q) case Q: if(o.compute==0) benchmark<Q,cutlass::half_t>(o,rotating); else benchmark<Q,cutlass::bfloat16_t>(o,rotating); break;
      CASE(8) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14)
#undef CASE
      default: throw std::runtime_error("unsupported qtype");
    }
    return 0;
  } catch(std::exception const &e) { std::fprintf(stderr,"PPU17_BENCH FAIL: %s\n",e.what()); return 1; }
}
