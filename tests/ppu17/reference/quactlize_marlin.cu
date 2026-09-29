// Prepared-call adapter for the complete Hopper provider. Device code lives
// in the optional backend, not in this benchmark or in the reference DSO.
#include "common.cuh"
#include "quactlize/backends/ppu17/gptq_marlin.cuh"

namespace q17_reference {
using Provider = quactlize::ppu17::MarlinGptq;
struct MarlinPlan : Plan {
  Provider::Configuration config;
  Provider::Arguments args;
  Provider::Workspace size;
  cudaStream_t stream;
  int sms = 0, shared = 0;
  explicit MarlinPlan(q17_ref_call const& c)
      : config(Provider::configuration(c.m,c.config)),
        args{c.m,c.n,c.k,c.a,c.b,c.scales,c.output},
        stream(static_cast<cudaStream_t>(c.stream)) {
    if (c.n%config.tile_n || c.k%config.tile_k)
      throw std::invalid_argument("Marlin GPTQ tile divisibility");
    int device = 0;
    checked(cudaGetDevice(&device),"Marlin device");
    checked(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"Marlin SM count");
    checked(cudaDeviceGetAttribute(&shared,cudaDevAttrMaxSharedMemoryPerBlockOptin,device),"Marlin shared capacity");
    if (Provider::shared_bytes(config) > shared)
      throw std::invalid_argument("Marlin GPTQ shared capacity");
    checked(cudaFuncSetAttribute(config.kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared),"Marlin opt-in shared");
    size = Provider::workspace(c.m,sms);
    allocate(size.total_bytes);
    checked(cudaMemsetAsync(workspace,0,bytes,stream),"initial Marlin scratch");
  }
  void execute() override {
    checked(Provider::run(config,args,sms,shared,workspace,size,stream),"Marlin GPTQ full call");
  }
};
}

Q17_REF_EXPORT int q17_ref_config_count() { return 3; }
Q17_REF_EXPORT char const* q17_ref_config_name(int i) {
  constexpr char const* names[] = {"classic-marlin-choice0","classic-marlin-choice1","classic-marlin-choice2"};
  return i >= 0 && i < 3 ? names[i] : "INVALID";
}
Q17_REF_EXPORT int q17_ref_prepare(q17_ref_call const* c, void** out) {
  return q17_reference::guard([&] {
    if (!c || !out) throw std::invalid_argument("null Marlin call/plan");
    *out = nullptr;
    q17_reference::validate(*c);
    *out = new q17_reference::MarlinPlan(*c);
  });
}
Q17_REF_EXPORT int q17_ref_pack(int dtype, int n, int k, uint8_t const* input, uint8_t* out) {
  return q17_reference::guard([&] {
    if (dtype != 0) throw std::invalid_argument("Marlin GPTQ FP16 contract");
    q17_reference::Provider::pack(n,k,input,out);
  });
}
