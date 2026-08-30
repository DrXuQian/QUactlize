// One-variable PPU image-admission arm.  Included by four distinct .cu wrappers so hgcc gives every arm its own
// object and fat image; using one source with four -D values would collide in PPUToolchain.cmake's path-keyed object
// name and turn this into a first-object-wins comparison.
#include <hggc_runtime.h>

#include <cstdint>
#include <cstdio>

#if PPU_PREPASS_IMAGE_ARM != 0
#include "gguf_scale_prepass.hpp"
#endif

#if PPU_PREPASS_IMAGE_ARM < 0 || PPU_PREPASS_IMAGE_ARM > 3
#error "PPU_PREPASS_IMAGE_ARM must be marker=0, header=1, raw=2 or packed=3"
#endif

namespace {

constexpr uint16_t kPoison = 0x7bffu;

__global__ void prepass_image_marker(uint16_t* out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = 0x3c00u;
}

struct LaunchStatus {
  int before;
  int immediate;
  int synchronize;
};

int clear_launch_state() { return int(hggcGetLastError()); }

LaunchStatus finish_launch(int before) {
  int const immediate = int(hggcGetLastError());
  int const synchronize = int(hggcDeviceSynchronize());
  return {before, immediate, synchronize};
}

char const* arm_name() {
#if PPU_PREPASS_IMAGE_ARM == 0
  return "marker";
#elif PPU_PREPASS_IMAGE_ARM == 1
  return "header";
#elif PPU_PREPASS_IMAGE_ARM == 2
  return "raw";
#else
  return "packed";
#endif
}

}  // namespace

int main() {
  uint16_t* device_marker = nullptr;
  uint16_t marker = kPoison;
  if (hggcMalloc(reinterpret_cast<void**>(&device_marker), sizeof(marker)) != hggcSuccess) return 2;
  if (hggcMemcpy(device_marker, &marker, sizeof(marker), hggcMemcpyHostToDevice) != hggcSuccess) return 3;

  int const marker_before = clear_launch_state();
  prepass_image_marker<<<1, 32>>>(device_marker);
  LaunchStatus const marker_launch = finish_launch(marker_before);
  if (hggcMemcpy(&marker, device_marker, sizeof(marker), hggcMemcpyDeviceToHost) != hggcSuccess) return 4;

  LaunchStatus subject_launch{0, 0, 0};
#if PPU_PREPASS_IMAGE_ARM == 2
  // The zero-column argument makes every thread return before dereferencing a pointer.  The exact production
  // specialization is still emitted and registered, which is the only variable this image-admission rung tests.
  gguf_scale::prepass::PrepassKernelArgs args{
      nullptr, nullptr, nullptr, nullptr, nullptr,
      0, 0, 0, 0, 0, 0, 0, 1};
  int const subject_before = clear_launch_state();
  gguf_scale::prepass::prepass_kernel<gguf_scale::KType::Q2_K, 0><<<1, 256>>>(args);
  subject_launch = finish_launch(subject_before);
#elif PPU_PREPASS_IMAGE_ARM == 3
  // Likewise, zero experts returns before the packed source or destination is touched while retaining the exact
  // production kernel symbol, ABI and body in the image.
  gguf_scale::prepass::UnitPrepassKernelArgs args{
      nullptr, nullptr, nullptr, 0, 0, 0, 0, 8, 2};
  int const subject_before = clear_launch_state();
  gguf_scale::prepass::prepass_unit_kernel<gguf_scale::KType::Q2_K, 0><<<1, 256>>>(args);
  subject_launch = finish_launch(subject_before);
#endif

  std::printf(
      "FQ_PREPASS_PPU_IMAGE arm=%s marker_bad=%d marker_value=0x%04x "
      "marker=[before:%d,immediate:%d,sync:%d] subject=[before:%d,immediate:%d,sync:%d]\n",
      arm_name(), marker != uint16_t(0x3c00u), unsigned(marker),
      marker_launch.before, marker_launch.immediate, marker_launch.synchronize,
      subject_launch.before, subject_launch.immediate, subject_launch.synchronize);

  hggcFree(device_marker);
  bool const marker_ok = marker == uint16_t(0x3c00u) && marker_launch.before == 0 &&
                         marker_launch.immediate == 0 && marker_launch.synchronize == 0;
  bool const subject_ok = subject_launch.before == 0 && subject_launch.immediate == 0 &&
                          subject_launch.synchronize == 0;
  bool const pass = marker_ok && subject_ok;
  std::printf("FQ_PREPASS_PPU_IMAGE_VERDICT arm=%s verdict=%s\n", arm_name(), pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
