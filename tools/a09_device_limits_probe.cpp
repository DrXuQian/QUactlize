// Focused runtime limit witness for the A09 launch/init diagnostic.
//
// This is intentionally a host-only utility: it launches no kernel and only
// reports the properties of the one device selected by CUDA_VISIBLE_DEVICES.

#include <hggc_runtime.h>

#include <cstdio>

int main() {
  int device = -1;
  hggcError_t status = hggcGetDevice(&device);
  if (status != hggcSuccess) {
    std::fprintf(stderr, "hggcGetDevice failed status=%d\n", int(status));
    return 2;
  }
  hggcDeviceProp properties{};
  status = hggcGetDeviceProperties(&properties, device);
  if (status != hggcSuccess) {
    std::fprintf(stderr, "hggcGetDeviceProperties failed status=%d\n",
                 int(status));
    return 3;
  }
  std::printf(
      "A09_DEVICE_LIMITS ordinal=%d name=%s compute_units=%d "
      "max_threads_per_block=%d max_threads_dim=%d,%d,%d\n",
      device, properties.name, properties.multiProcessorCount,
      properties.maxThreadsPerBlock, properties.maxThreadsDim[0],
      properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
  return 0;
}
