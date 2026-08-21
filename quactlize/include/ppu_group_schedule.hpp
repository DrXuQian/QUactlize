#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"
#include "actlize_extensions/cutlass/gemm/quactlize_dispatch_policy.hpp"

// One definition of the fine-grained group-size ladder used by dense, grouped and the shipping backend.
// The schedule tag controls scale-tile reload cadence; ScaleGroups controls how many groups a K tile covers.
// Keep both compile-time so an unsupported group size fails while building a kernel instead of asserting on device.
namespace ppu_group_schedule {

template <int GroupSize>
struct Selector {
  static_assert(GroupSize == 16 || GroupSize == 32 || GroupSize == 64 || GroupSize == 128,
                "PPU mixed-input fine-grained kernels support group sizes 16, 32, 64 and 128");
};

template <>
struct Selector<128> {
  using Type = cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs128;
};

template <>
struct Selector<64> {
  using Type = cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs64;
};

template <>
struct Selector<32> {
  using Type = cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32;
};

template <>
struct Selector<16> {
  // The arithmetic path is the same per-MMA-atom path as gs32, but the tag
  // must still carry the declared group size.  Reusing Gs32 made the runtime
  // extent and the static schedule look contradictory to admission checks.
  using Type = cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs16;
};

template <int GroupSize>
using FinegrainedSchedule = typename Selector<GroupSize>::Type;

template <int TileK, int GroupSize>
struct ScaleGroups {
  static_assert(TileK > 0, "TileK must be positive");
  static_assert(GroupSize > 0, "group size must be positive");
  static constexpr int value = (TileK + GroupSize - 1) / GroupSize;
  static_assert(value > 0,
                "a scale tile must contain at least one group; use ceiling division for TileK/group_size");
};

template <int TileK, int GroupSize>
inline constexpr int scale_groups_v = ScaleGroups<TileK, GroupSize>::value;

}  // namespace ppu_group_schedule
