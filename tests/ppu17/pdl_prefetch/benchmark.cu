// Physical Hopper experiment: complete GEMM -> [SF] -> GEMM chains.
// No synthetic delay, no production policy change, no GPU reference.
#include "../fixture.cuh"
#include "tma.cuh"
#include <array>
#include <memory>

namespace {
using Handle = std::unique_ptr<q17_plan_v1, decltype(&q17_destroy_v1)>;
struct Calls {
  q17_call_v1 producer, consumer;
  std::array<Handle, 3> first{{{nullptr,q17_destroy_v1},{nullptr,q17_destroy_v1},{nullptr,q17_destroy_v1}}};
  std::array<Handle, 3> last{{{nullptr,q17_destroy_v1},{nullptr,q17_destroy_v1},{nullptr,q17_destroy_v1}}};
};
char const *arm_name(int arm) { return arm == 0 ? "ordinary" : arm == 1 ? "protocol" : "prefetch"; }
q17_launch_options_v1 const pdl_options{1, sizeof(q17_launch_options_v1), 1, 0};

template <class T> void benchmark(Options const &o, bool rotating, int carveout, bool serial_sf) {
  constexpr int Q = 12, producer_k = 256;
  auto counts = o.grouped ? std::vector<int>{0,1,8,o.m} : std::vector<int>{o.m};
  std::vector<int32_t> offsets{0};
  for (int n: counts) offsets.push_back(offsets.back()+n);
  int m=offsets.back(), experts=counts.size();
  auto make_call = [&](int n, int k, int route, int output, int config) {
    q17_call_v1 c{}; c.version=1; c.struct_bytes=sizeof(c);
    auto &p=c.problem; p.version=1; p.struct_bytes=sizeof(p);
    p.qtype=Q; p.operation=o.grouped?Q17_GROUPED:Q17_DENSE; p.route=route;
    p.compute_type=p.input_type=o.compute; p.output_type=output;
    p.m=m; p.n=n; p.k=k; p.experts=experts;
    p.max_rows=*std::max_element(counts.begin(),counts.end());
    check(q17_canonical_arrangement_v1(Q,&p.arrangement),"arrangement");
    check(q17_config_get_v1(&p,config,&c.config),"config");
    c.a_stride=k+8; c.output_stride=n+8;
    gpu(cudaGetDevice(&c.device),"device");
    return c;
  };
  auto first=make_call(o.k,producer_k,Q17_FQ,o.compute,2);
  auto last=make_call(o.n,o.k,o.sf?Q17_SF:Q17_FQ,Q17_F32,5);
  cudaDeviceProp prop{}; gpu(cudaGetDeviceProperties(&prop,last.device),"properties");
  if(prop.major!=9 || prop.minor!=0 || prop.l2CacheSize<=0) throw std::runtime_error("physical SM90/L2 required");
  Fixture<Q,T> before(o.k,producer_k,experts), after(o.n,o.k,experts);
  std::vector<T> input(size_t(m)*first.a_stride,T(-123.f));
  for(int r=0;r<m;++r) for(int k=0;k<producer_k;++k)
    input[size_t(r)*first.a_stride+k]=T(float((r*3+k*5)%29-14)/16.f);
  std::vector<T> intermediate(size_t(m)*last.a_stride+64,T(-123.f)), expected_a(intermediate);
  std::vector<float> output(size_t(m)*last.output_stride+16,-12345.f), expected(size_t(m)*o.n);
  // Decode once on CPU, then perform the full independent two-dot oracle.
  for(int e=0;e<experts;++e) {
    std::vector<float> w0(size_t(o.k)*producer_k), w1(size_t(o.n)*o.k);
    for(int n=0;n<o.k;++n) for(int k=0;k<producer_k;++k) w0[size_t(n)*producer_k+k]=float(Fixture<Q,T>::oracle_weight(e,n,k));
    for(int n=0;n<o.n;++n) for(int k=0;k<o.k;++k) w1[size_t(n)*o.k+k]=float(Fixture<Q,T>::oracle_weight(e,n,k));
    for(int r=offsets[e];r<offsets[e+1];++r) {
      for(int n=0;n<o.k;++n) {
        double sum=0;
        for(int k=0;k<producer_k;++k) sum+=double(float(input[size_t(r)*first.a_stride+k]))*w0[size_t(n)*producer_k+k];
        expected_a[32+size_t(r)*last.a_stride+n]=T(float(sum));
      }
      for(int n=0;n<o.n;++n) {
        double sum=0;
        for(int k=0;k<o.k;++k) sum+=double(float(expected_a[32+size_t(r)*last.a_stride+k]))*w1[size_t(n)*o.k+k];
        expected[size_t(r)*o.n+n]=float(sum);
      }
    }
  }
  cudaStream_t stream{}; gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
  first.stream=last.stream=stream;
  DeviceMemory a(input.size()*sizeof(T),input.data()), mid(intermediate.size()*sizeof(T),intermediate.data()),
    c(output.size()*4,output.data()), directory(offsets.size()*4,offsets.data());
  first.a=a.p; first.output=static_cast<T *>(mid.p)+32;
  last.a=first.output; last.output=static_cast<float *>(c.p)+8;
  first.offsets=last.offsets=static_cast<int32_t const *>(directory.p);
  size_t active=std::count_if(counts.begin(),counts.end(),[](int n){return n>0;});
  size_t weights=(before.low.size()+before.units.size()+after.low.size()+after.units.size()+
    (o.sf?(after.scale.size()+after.zero.size())*sizeof(T):0))/experts*active;
  int copies=rotating?int(2*size_t(prop.l2CacheSize)/weights)+1:1;
  if(copies>256) throw std::runtime_error("weight ring exceeds bounded capacity");
  int chains=copies*((16+copies-1)/copies);
  std::vector<std::unique_ptr<DeviceMemory>> storage;
  auto upload=[&](auto const &host) {
    auto block=std::make_unique<DeviceMemory>(host.size()*sizeof(host[0]),host.data());
    void *p=block->p; storage.push_back(std::move(block)); return p;
  };
  std::vector<std::unique_ptr<Calls>> ring;
  for(int i=0;i<copies;++i) {
    auto row=std::make_unique<Calls>(); row->producer=first; row->consumer=last;
    auto &p=row->producer; auto &t=row->consumer;
    p.low=upload(before.low); p.units=upload(before.units);
    t.low=upload(after.low); t.units=upload(after.units);
    t.scale=upload(after.scale); t.zero=upload(after.zero);
    for(int arm=0;arm<3;++arm) {
      q17_plan_v1 *raw=nullptr;
      check(q17_prepare_with_options_v1(&p,arm?&pdl_options:nullptr,&raw),"prepare producer");
      row->first[arm].reset(raw); raw=nullptr;
      // All target arms use the same translation unit and resource policy;
      // ordinary/protocol bodies are checked against the immutable library.
      std::function<int()> execute;
      if(arm==2) execute=o.sf?prepare_tma<Q,T,true,true,true>(t,carveout):prepare_tma<Q,T,false,true,true>(t,carveout);
      else if(arm==1) execute=o.sf?prepare_tma<Q,T,true,true,false>(t,carveout):prepare_tma<Q,T,false,true,false>(t,carveout);
      else execute=o.sf?prepare_tma<Q,T,true,false,false>(t,carveout):prepare_tma<Q,T,false,false,false>(t,carveout);
      raw=new q17_plan_v1{t.device,std::move(execute)};
      row->last[arm].reset(raw);
    }
    ring.push_back(std::move(row));
  }
  gpu(cudaDeviceSynchronize(),"uploads");
  std::puts("PPU17_BENCH_READY"); std::fflush(stdout);
  if(std::getchar()!='R') throw std::runtime_error("idle guard did not admit execution");
  auto enqueue=[&](Calls const &row,int arm) {
    check(q17_execute_v1(row.first[arm].get()),"producer");
    if(o.sf) check(q17_expand_scales_with_options_v1(&row.consumer,arm&&!serial_sf?&pdl_options:nullptr),"SF expansion");
    check(q17_execute_v1(row.last[arm].get()),"consumer");
  };
  std::vector<float> raw_baseline;
  auto verify=[&]() {
    gpu(cudaStreamSynchronize(stream),"verify completion");
    gpu(cudaMemcpy(output.data(),c.p,output.size()*4,cudaMemcpyDeviceToHost),"output");
    gpu(cudaMemcpy(intermediate.data(),mid.p,intermediate.size()*sizeof(T),cudaMemcpyDeviceToHost),"intermediate");
    for(size_t i=0;i<intermediate.size();++i)
      if(intermediate[i].raw()!=expected_a[i].raw()) throw std::runtime_error("producer typed CPU dot/guard mismatch");
    for(int r=0;r<m;++r) for(int n=0;n<o.n;++n) {
      float got=output[8+size_t(r)*last.output_stride+n],want=expected[size_t(r)*o.n+n];
      if(!std::isfinite(got)||std::abs(got-want)>1.e-4f*std::max(1.f,std::abs(want))) throw std::runtime_error("consumer CPU dot mismatch");
    }
    for(size_t i=0;i<output.size();++i) {
      bool live=i>=8&&i<output.size()-8&&(i-8)%last.output_stride<size_t(o.n);
      if(!live&&output[i]!=-12345.f) throw std::runtime_error("consumer guard mismatch");
    }
    if(raw_baseline.empty()) raw_baseline=output;
    else if(std::memcmp(raw_baseline.data(),output.data(),output.size()*4)) throw std::runtime_error("raw arm/replay mismatch");
  };
  for(int arm=0;arm<3;++arm) for(auto const &row:ring) {enqueue(*row,arm);verify();}
  cudaGraph_t graphs[3]{}; cudaGraphExec_t execs[3]{};
  for(int arm=0;arm<3;++arm) {
    gpu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"capture");
    for(int i=0;i<chains;++i) enqueue(*ring[i%copies],arm);
    gpu(cudaStreamEndCapture(stream,&graphs[arm]),"capture end");
    gpu(cudaGraphInstantiate(&execs[arm],graphs[arm],0),"instantiate");
    for(int i=0;i<3;++i) gpu(cudaGraphLaunch(execs[arm],stream),"warmup");
    verify();
  }
  cudaEvent_t begin{},end{}; gpu(cudaEventCreate(&begin),"event"); gpu(cudaEventCreate(&end),"event");
  std::array<std::vector<double>,3> samples;
  for(int repeat=0;repeat<11;++repeat) for(int order=0;order<3;++order) {
    int arm=repeat%2?2-order:order;
    gpu(cudaEventRecord(begin,stream),"begin"); gpu(cudaGraphLaunch(execs[arm],stream),"sample");
    gpu(cudaEventRecord(end,stream),"end"); gpu(cudaEventSynchronize(end),"sample wait");
    float ms=0; gpu(cudaEventElapsedTime(&ms,begin,end),"interval");
    if(!(ms>0)||!std::isfinite(ms)) throw std::runtime_error("invalid elapsed time");
    samples[arm].push_back(ms*1000./chains);
  }
  for(int arm=0;arm<3;++arm) {gpu(cudaGraphLaunch(execs[arm],stream),"final replay");verify();}
  auto median=[](std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
  auto print_samples=[](std::vector<double> const &v){std::printf("[");for(size_t i=0;i<v.size();++i)std::printf("%s%.9f",i?",":"",v[i]);std::printf("]");};
  std::printf("PPU17_BENCH {\"q\":12,\"compute\":%d,\"route\":%d,\"m\":%d,\"n\":%d,\"k\":%d,\"experts\":%d,\"sm_count\":%d,\"l2_bytes\":%d,\"cache\":\"%s\",\"weight_bytes\":%zu,\"weight_copies\":%d,\"launches_per_graph\":%d,\"chains_per_graph\":%d,\"warmup_graphs_per_arm\":3,\"producer_config\":2,\"consumer_config\":5,\"correctness\":\"CPU_DOT_AND_GUARDS_PASS\",\"raw_equal\":true,\"scope\":\"PHYSICAL_HOPPER_GRAPH_AVERAGE_NOT_PPU17\",\"metric_scope\":\"GEMM_OPTIONAL_SF_GEMM_CHAIN\",\"median_us\":%.9f,\"samples_us\":",
    o.compute,int(o.sf),m,o.n,o.k,experts,prop.multiProcessorCount,prop.l2CacheSize,rotating?"rotating":"warm",weights,copies,chains,chains,median(samples[0]));
  print_samples(samples[0]); std::printf(",\"shared_carveout\":%d,\"serial_sf\":%s,\"inline_controls\":true,\"arms\":{",carveout,serial_sf?"true":"false");
  for(int arm=0;arm<3;++arm) {
    std::printf("%s\"%s\":{\"median_us\":%.9f,\"samples_us\":",arm?",":"",arm_name(arm),median(samples[arm]));
    print_samples(samples[arm]);std::printf("}");
  }
  std::printf("}}\n");
  gpu(cudaEventDestroy(begin),"destroy");gpu(cudaEventDestroy(end),"destroy");
  for(int arm=0;arm<3;++arm) {gpu(cudaGraphExecDestroy(execs[arm]),"destroy");gpu(cudaGraphDestroy(graphs[arm]),"destroy");}
  gpu(cudaStreamDestroy(stream),"destroy");
}
} // namespace

int main(int argc,char **argv) {
  try {
    if(!std::getenv("Q17_BENCH_GUARDED")) throw std::runtime_error("use the guarded physical-device runner");
    Options o; o.m=1;o.n=512;o.k=2048;bool rotating=false,serial_sf=false;int carveout=-1;
    for(int i=1;i<argc;++i) {
      std::string key=argv[i];
      if(key=="--grouped")o.grouped=true;
      else if(key=="--sf")o.sf=true;
      else if(key=="--rotating")rotating=true;
      else if(key=="--serial-sf")serial_sf=true;
      else {if(++i==argc)throw std::runtime_error("missing value");int v=std::stoi(argv[i]);
        if(key=="--compute")o.compute=v;else if(key=="--m")o.m=v;
        else if(key=="--n")o.n=v;else if(key=="--k")o.k=v;
        else if(key=="--carveout")carveout=v;
        else throw std::runtime_error("unknown option");}
    }
    if(o.m<1||o.m>128||o.n<16||o.n>4096||o.k<256||o.k>8192||o.k%256||o.n%16)throw std::runtime_error("outside bounded experiment");
    if(o.compute==Q17_F16)benchmark<cutlass::half_t>(o,rotating,carveout,serial_sf);
    else if(o.compute==Q17_BF16)benchmark<cutlass::bfloat16_t>(o,rotating,carveout,serial_sf);
    else throw std::runtime_error("compute must be F16/BF16");
    return 0;
  } catch(std::exception const &e) {std::fprintf(stderr,"PPU17_PREFETCH_BENCH FAIL: %s\n",e.what());return 1;}
}
