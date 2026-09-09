#pragma once
#include <cstdint>
#include <limits>

namespace quactlize::runtime {

// All offsets are bytes. Partial planes are [split][all real rows][N],
// whereas output descriptors are [split][expert]. Empty experts own no data.
struct GroupedWorkspace {
  uint64_t shapes = 0, outputs = 0, strides = 0, rows = 0, partials = 0;
  uint64_t partial_bytes = 0, total = 0;
};

inline bool grouped_workspace(int experts, int splits, int m, int n,
    uint64_t scheduler, uint64_t shape_size, uint64_t stride_size,
    bool device_rows, GroupedWorkspace& out) {
  out = {};
  if (experts <= 0 || m <= 0 || n <= 0 ||
      (splits != 1 && splits != 2 && splits != 4 && splits != 8)) return false;
  constexpr uint64_t limit = uint64_t(INT64_MAX) - 15;
  uint64_t cursor = 0;
  auto append = [&](uint64_t count, uint64_t size, uint64_t& offset) {
    if (size && count > limit / size) return false;
    uint64_t bytes = (count * size + 15) & ~UINT64_C(15);
    if (cursor > limit - bytes) return false;
    offset = cursor; cursor += bytes; return true;
  };
  uint64_t ignored = 0;
  uint64_t const descriptors = uint64_t(experts) * splits;
  if (!append(1, scheduler, ignored) ||
      !append(experts, shape_size, out.shapes) ||
      !append(descriptors, sizeof(void*), out.outputs) ||
      !append(descriptors, stride_size, out.strides) ||
      !append(device_rows ? experts : 0, sizeof(int32_t), out.rows)) return false;
  if (splits > 1) {
    if (uint64_t(m) * uint64_t(n) > limit / uint64_t(splits) / sizeof(float)) return false;
    out.partial_bytes = uint64_t(m) * n * splits * sizeof(float);
  }
  if (!append(1, out.partial_bytes, out.partials)) return false;
  out.total = cursor;
  return true;
}

}  // namespace quactlize::runtime
