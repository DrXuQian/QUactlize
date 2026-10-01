#pragma once

#include "kernels.cuh"
#include "resource_policy.hpp"
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace pdl_experiment {
enum class SfTrigger { Early, Late, Serial };

inline char const *name(SfTrigger trigger) {
  switch (trigger) {
    case SfTrigger::Early: return "early";
    case SfTrigger::Late: return "late";
    case SfTrigger::Serial: return "serial";
  }
  throw std::invalid_argument("invalid SF trigger");
}
inline SfTrigger parse_trigger(std::string const &value) {
  if (value == "early") return SfTrigger::Early;
  if (value == "late") return SfTrigger::Late;
  if (value == "serial") return SfTrigger::Serial;
  throw std::invalid_argument("SF trigger must be early, late or serial");
}
inline int parse_carveout(std::string const &value) {
  if (value == "-1" || value == "default") return default_carveout;
  if (value == "-2" || value == "auto") return occupancy_preserving_l1;
  if (value == "25") return 25;
  throw std::invalid_argument("carveout must be default, 25 or auto");
}

inline void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct ResourceRecord {
  std::string role;
  int arm, device, threads;
  size_t dynamic_bytes;
  cudaFuncAttributes attributes;
  ResourceChoice choice;
};

inline ResourceRecord configure_resource(void const *kernel, char const *role,
    int arm, int request, int threads, size_t dynamic_bytes, int expected_device) {
  int device = -1;
  check_cuda(cudaGetDevice(&device));
  if (device != expected_device) throw std::runtime_error("resource configuration device differs from plan");
  cudaFuncAttributes attrs{};
  check_cuda(cudaFuncGetAttributes(&attrs, kernel));
  if (threads <= 0 || threads > attrs.maxThreadsPerBlock || dynamic_bytes > size_t(attrs.maxDynamicSharedSizeBytes))
    throw std::invalid_argument("resource query launch geometry is unsupported");
  const auto set = [&](int pct) {
    check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
  };
  const auto query = [&] {
    int blocks = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, dynamic_bytes));
    return blocks;
  };
  auto choice = choose_resources(request, attrs.preferredShmemCarveout, set, query);
  check_cuda(cudaFuncGetAttributes(&attrs, kernel));
  if (attrs.preferredShmemCarveout != choice.selected)
    throw std::runtime_error("resource attribute readback differs");
  return {role, arm, device, threads, dynamic_bytes, attrs, choice};
}

inline void print_resources(std::vector<ResourceRecord> const &records) {
  std::printf("[");
  for (size_t i = 0; i < records.size(); ++i) {
    auto const &r = records[i];
    std::printf("%s{\"role\":\"%s\",\"arm\":%d,\"device\":%d,\"threads\":%d,\"dynamic_bytes\":%zu,\"static_bytes\":%zu,\"registers\":%d,\"request\":%d,\"selected\":%d,\"default_blocks\":%d,\"selected_blocks\":%d}",
        i ? "," : "", r.role.c_str(), r.arm, r.device, r.threads, r.dynamic_bytes,
        r.attributes.sharedSizeBytes, r.attributes.numRegs, r.choice.requested,
        r.choice.selected, r.choice.default_blocks, r.choice.selected_blocks);
  }
  std::printf("]");
}

// A separate symbol preserves the existing public expansion specialization.
// Trigger is not a publication fence: the successor still waits before reads.
template <int Q, class T>
__global__ void expand_scales_late(q17_call_v1 c) {
  quactlize::ppu17::wait_for_predecessors<true>();
  quactlize::ppu17::expand_scale_values<Q,T>(c);
  quactlize::ppu17::release_successors<true>();
}

template <int Q, class T>
std::function<int(q17_call_v1 const &)> prepare_expansion(q17_call_v1 const &c,
    int arm, SfTrigger trigger, int carveout, std::vector<ResourceRecord> &records) {
  using namespace quactlize::ppu17;
  const bool pdl = arm != 0 && trigger != SfTrigger::Serial;
  auto kernel = !pdl ? expand_scales<Q,T,false>
      : trigger == SfTrigger::Late ? expand_scales_late<Q,T> : expand_scales<Q,T,true>;
  records.push_back(configure_resource(reinterpret_cast<void const *>(kernel), "sf", arm,
                                      carveout, 256, 0, c.device));
  return [kernel, pdl](q17_call_v1 const &call) {
    size_t total = size_t(call.problem.experts) * call.problem.n * (call.problem.k / Format<Q>::group_size);
    unsigned blocks = unsigned(std::min<size_t>((total + 255) / 256, 65535));
    auto stream = static_cast<cudaStream_t>(call.stream);
    return pdl ? enqueue_kernel<true>(kernel, blocks, 256, 0, stream, call)
               : enqueue_kernel<false>(kernel, blocks, 256, 0, stream, call);
  };
}
} // namespace pdl_experiment
