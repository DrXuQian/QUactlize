#include <cuda_runtime.h>
#include "target.cuh"
#include "host.hpp"
#include "prepared.hpp"
#include <memory>

namespace quactlize::ppu17 {
#define Q17_DECLARE(Q) int launch_q##Q(q17_call_v1 const &, bool); \
  std::function<int()> prepare_q##Q(q17_call_v1 const &);
Q17_DECLARE(8)
Q17_DECLARE(10)
Q17_DECLARE(11)
Q17_DECLARE(12)
Q17_DECLARE(13)
Q17_DECLARE(14)
#undef Q17_DECLARE

int check_device(q17_call_v1 const *c) {
  int device = -1;
  cudaError_t runtime_status = cudaGetDevice(&device);
  if (runtime_status != cudaSuccess) return error(Q17_WRONG_DEVICE, cudaGetErrorString(runtime_status));
  if (device != c->device) return error(Q17_WRONG_DEVICE, "caller device is not current; no implicit device switch");
  cudaDeviceProp properties{};
  runtime_status = cudaGetDeviceProperties(&properties, device);
  if (runtime_status != cudaSuccess) return error(Q17_WRONG_DEVICE, cudaGetErrorString(runtime_status));
  if (properties.major != 9 || properties.minor != 0)
    return error(Q17_WRONG_DEVICE, "backend requires a verified SM90-compatible device");
  if (properties.maxThreadsPerBlock < 256)
    return error(Q17_UNSUPPORTED, "CTA exceeds device thread limit");
  return Q17_SUCCESS;
}

int run_expansion(q17_call_v1 const *c) {
  int status = validate_call(c, true);
  if (status) return status;
  status = check_device(c);
  if (status) return status;
  int rc;
  switch (c->problem.qtype) {
#define Q17_CASE(Q) case Q: rc = launch_q##Q(*c, true); break;
    Q17_CASE(8) Q17_CASE(10) Q17_CASE(11) Q17_CASE(12) Q17_CASE(13) Q17_CASE(14)
#undef Q17_CASE
    default: return error(Q17_UNSUPPORTED, "qtype not compiled");
  }
  return rc ? error(Q17_LAUNCH_ERROR, cudaGetErrorString(cudaError_t(rc))) : error(Q17_SUCCESS, "");
}
} // namespace quactlize::ppu17

extern "C" int q17_prepare_v1(q17_call_v1 const *c, q17_plan_v1 **out) {
  using namespace quactlize::ppu17;
  if (!out) return error(Q17_INVALID, "NULL plan output");
  *out = nullptr;
  int status = validate_call(c, false);
  if (status) return status;
  status = check_device(c);
  if (status) return status;
  try {
    auto plan = std::make_unique<q17_plan_v1>();
    plan->device = c->device;
    switch (c->problem.qtype) {
#define Q17_CASE(Q) case Q: plan->enqueue = prepare_q##Q(*c); break;
      Q17_CASE(8) Q17_CASE(10) Q17_CASE(11) Q17_CASE(12) Q17_CASE(13) Q17_CASE(14)
#undef Q17_CASE
      default: return error(Q17_UNSUPPORTED, "qtype not compiled");
    }
    *out = plan.release();
  } catch (std::exception const &e) {
    return error(Q17_INVALID, e.what());
  }
  return error(Q17_SUCCESS, "");
}
extern "C" int q17_execute_v1(q17_plan_v1 const *p) {
  using namespace quactlize::ppu17;
  if (!p || !p->enqueue) return error(Q17_INVALID, "NULL or empty prepared plan");
  int device = -1;
  auto rc = cudaGetDevice(&device);
  if (rc != cudaSuccess) return error(Q17_WRONG_DEVICE, cudaGetErrorString(rc));
  if (device != p->device) return error(Q17_WRONG_DEVICE, "prepared plan belongs to another device");
  rc = cudaError_t(p->enqueue());
  return rc != cudaSuccess ? error(Q17_LAUNCH_ERROR, cudaGetErrorString(rc)) : error(Q17_SUCCESS, "");
}
extern "C" void q17_destroy_v1(q17_plan_v1 *p) { delete p; }
extern "C" int q17_launch_v1(q17_call_v1 const *c) {
  q17_plan_v1 *plan = nullptr;
  int status = q17_prepare_v1(c, &plan);
  if (status) return status;
  status = q17_execute_v1(plan);
  q17_destroy_v1(plan);
  return status;
}
extern "C" int q17_expand_scales_v1(q17_call_v1 const *c) { return quactlize::ppu17::run_expansion(c); }
extern "C" char const *q17_build_target_v1() {
#ifdef Q17_NATIVE
  return "ppu17-native";
#else
  return "cuda-sm90a-source-validation-not-ppu-device-admission";
#endif
}
