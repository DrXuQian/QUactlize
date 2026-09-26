// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/barrier.h"
#include "cutlass/block_striped.h"
#include "cutlass/workspace.h"
// Import ONLY the device-free algebra. Do not put actlize on the SM90 include
// search path: this header's cutlass/cutlass.h resolves to PPU CUTLASS3.6.
#include "../../../third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp"

namespace quactlize::ppu17 {

struct MarlinScheduler {};

// Adapt the existing K-fast/reverse-peer scheduler to two Hopper consumer
// warpgroups. The TMA/WGMMA collective and its async lifetimes stay unchanged.
// One CTA/SM and Cluster1 are deliberate initial capabilities, not a claim
// that a Marlin scheduler alone ports the standalone Marlin mainloop.
template<class TileShape, class ClusterShape, int GroupK = 128>
class CooperativeMarlinScheduler {
 public:
  using Core = cutlass::gemm::kernel::detail::MarlinStripeSchedulerCore;
  using Base = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90;
  using RasterOrder = typename Base::RasterOrder;
  using RasterOrderOptions = typename Base::RasterOrderOptions;
  using BarrierType = typename cutlass::NamedBarrierManager<1>::T;
  static constexpr bool IsDynamicPersistent = false;
  static constexpr int KAlignment = GroupK / cute::size<2>(TileShape{});
  static constexpr uint64_t TileElements = cute::size<0>(TileShape{}) * cute::size<1>(TileShape{});
  static constexpr uint32_t ConsumerGroups = 2;
  static_assert(cute::size(ClusterShape{}) == 1, "Marlin adapter requires Cluster1");
  static_assert(GroupK >= cute::size<2>(TileShape{}) && GroupK % cute::size<2>(TileShape{}) == 0);

  struct Arguments {
    static constexpr int max_swizzle_size = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::AlongN;
  };
  struct Params : Core::Params {
    void* workspace = nullptr;
    int log_swizzle_size_ = 0;
    RasterOrder raster_order_ = RasterOrder::AlongN;
  };
  struct WorkTileInfo : Core::WorkTileInfo {
    CUTLASS_HOST_DEVICE int reduction_subtile_idx() const { return -1; }
  };

  // Integer endpoints are expressed in whole scale groups in the core, and
  // converted to mainloop K-tile ordinals at this one seam. Scale bits and
  // group size do not change; splits never begin inside a metadata group.
  CUTLASS_HOST_DEVICE static constexpr Params make_params(
      uint64_t tiles_m, uint64_t tiles_n, uint64_t tiles_l,
      uint64_t k_tiles, uint64_t sm_count, void* workspace = nullptr) {
    Params p;
    if (k_tiles % KAlignment != 0 || k_tiles > std::numeric_limits<uint32_t>::max()) return p;
    static_cast<Core::Params&>(p) = Core::make_params_for_tiles(
        tiles_m, tiles_n, tiles_l, k_tiles / KAlignment, sm_count);
    p.workspace = workspace;
    if (p.output_tiles_ > uint64_t(std::numeric_limits<int>::max()) / ConsumerGroups) p.valid_ = false;
    return p;
  }

  CUTLASS_HOST_DEVICE static constexpr WorkTileInfo work_for_block(Params const& p, uint64_t block) {
    return convert(Core::get_work_for_block(p, block));
  }
  CUTLASS_HOST_DEVICE static constexpr WorkTileInfo next_work(Params const& p, WorkTileInfo const& w) {
    return convert(Core::fetch_next_work(p, w));
  }
  CUTLASS_HOST_DEVICE static constexpr uint64_t lock_index(WorkTileInfo const& w, uint32_t group) {
    return w.output_tile_idx * ConsumerGroups + group;
  }

  template<class Shape>
  static Params to_underlying_arguments(Shape shape, TileShape, ClusterShape,
      cutlass::KernelHardwareInfo hw, Arguments const&, void* workspace,
      uint32_t = 1, uint32_t = 1) {
    auto s = cute::append<4>(shape, 1);
    if (cute::get<0>(s) <= 0 || cute::get<1>(s) <= 0 || cute::get<2>(s) <= 0 ||
        cute::get<3>(s) <= 0 || cute::get<2>(s) % GroupK) return {};
    if (hw.sm_count <= 0) hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
    if (hw.sm_count <= 0) return {};
    return make_params(Core::ceil_div_u64(cute::get<0>(s), cute::size<0>(TileShape{})),
        Core::ceil_div_u64(cute::get<1>(s), cute::size<1>(TileShape{})), cute::get<3>(s),
        cute::get<2>(s) / cute::size<2>(TileShape{}), hw.sm_count, workspace);
  }
  static bool can_implement(Arguments const& args) {
    return args.raster_order == RasterOrderOptions::AlongN;
  }

  template<class Shape, class Accumulator>
  static size_t get_workspace_size(Arguments const& args, Shape shape,
      cutlass::KernelHardwareInfo const& hw, uint32_t groups, uint32_t = 1, uint32_t = 1) {
    static_assert(std::is_same_v<Accumulator, float>, "Marlin fixup remains FP32");
    Params p = to_underlying_arguments(shape, {}, {}, hw, args, nullptr);
    if (!p.valid_ || groups != ConsumerGroups || !needs_workspace(p)) return 0;
    return reduction_bytes(p) + p.output_tiles_ * ConsumerGroups * sizeof(BarrierType);
  }
  template<class Shape, class Accumulator>
  static cutlass::Status initialize_workspace(Arguments const& args, void* workspace,
      cudaStream_t stream, Shape shape, cutlass::KernelHardwareInfo const& hw,
      uint32_t groups, uint32_t = 1, uint32_t = 1, cutlass::CudaHostAdapter* adapter = nullptr) {
    static_assert(std::is_same_v<Accumulator, float>);
    Params p = to_underlying_arguments(shape, {}, {}, hw, args, workspace);
    if (!p.valid_ || groups != ConsumerGroups) return cutlass::Status::kErrorInvalidProblem;
    if (!needs_workspace(p)) return cutlass::Status::kSuccess;
    if (!workspace) return cutlass::Status::kErrorWorkspaceNull;
    // Clear only locks; the highest-K peer owns initialization of FP32 data.
    return cutlass::zero_workspace(static_cast<uint8_t*>(workspace) + reduction_bytes(p),
        p.output_tiles_ * ConsumerGroups * sizeof(BarrierType), stream, adapter);
  }
  template<class Shape>
  static dim3 get_grid_shape(Params const& p, Shape, TileShape, ClusterShape,
      cutlass::KernelHardwareInfo const&, Arguments const&) {
    return p.valid_ ? dim3(unsigned(p.grid_blocks_), 1, 1) : dim3(0, 0, 0);
  }

  CUTLASS_DEVICE explicit CooperativeMarlinScheduler(Params const& p) : params_(p) {}
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) const {
    return work_for_block(params_, blockIdx.x);
  }
  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo const& w) const {
    return cute::make_tuple(next_work(params_, w), true);
  }
  CUTLASS_HOST_DEVICE static bool requires_separate_reduction(Params const&) { return false; }
  CUTLASS_HOST_DEVICE static bool valid_warpgroup_in_work_tile(WorkTileInfo const& w) { return w.is_valid(); }
  CUTLASS_HOST_DEVICE static uint32_t get_work_k_tile_start(WorkTileInfo const& w) { return w.K_idx; }
  template<class Shape>
  CUTLASS_HOST_DEVICE static uint32_t get_work_k_tile_count(WorkTileInfo const& w, Shape, TileShape) {
    return w.k_tile_count;
  }
  CUTLASS_HOST_DEVICE static bool compute_epilogue(WorkTileInfo const& w, Params const&) {
    return w.is_valid() && w.slice_idx + 1 == w.slice_count;
  }

  template<class Fragment>
  CUTLASS_DEVICE static void fixup(Params const& p, WorkTileInfo const& w,
      Fragment& accumulators, uint32_t groups, uint32_t group) {
    if (!w.is_valid() || w.slice_count == 1) return;
    using Accumulator = typename Fragment::value_type;
    static_assert(std::is_same_v<Accumulator, float>);
    static_assert(cute::size(Fragment{}) * 128 * ConsumerGroups == TileElements,
                  "two complete consumer fragments must cover one output tile");
    CUTLASS_ASSERT(groups == ConsumerGroups && group < ConsumerGroups);
    using Array = cutlass::Array<Accumulator, cute::size(Fragment{})>;
    using Striped = cutlass::BlockStripedReduce<128, Array>;
    using Barrier = cutlass::NamedBarrierManager<128,
        int(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0), ConsumerGroups>;
    auto* data = reinterpret_cast<Array*>(static_cast<Accumulator*>(p.workspace) +
        w.output_tile_idx * TileElements + group * 128 * cute::size(Fragment{}));
    auto* locks = reinterpret_cast<BarrierType*>(static_cast<uint8_t*>(p.workspace) + reduction_bytes(p));
    auto& fragment = *reinterpret_cast<Array*>(accumulators.data());
    uint32_t thread = threadIdx.x % 128;
    int lock = int(lock_index(w, group));
    if (w.slice_idx == 0) {
      Striped::store(data, fragment, thread);
      Barrier::arrive_inc(group, locks, thread, lock, 1);
    } else if (!compute_epilogue(w, p)) {
      Barrier::wait_eq(group, locks, thread, lock, w.slice_idx);
      Striped::load_add(fragment, data, thread);
      Striped::store(data, fragment, thread);
      Barrier::arrive_inc(group, locks, thread, lock, 1);
    } else {
      Barrier::wait_eq_reset(group, locks, thread, lock, w.slice_idx);
      Striped::load_add(fragment, data, thread);
    }
  }

 private:
  Params params_;
  CUTLASS_HOST_DEVICE static constexpr WorkTileInfo convert(Core::WorkTileInfo w) {
    WorkTileInfo result;
    static_cast<Core::WorkTileInfo&>(result) = w;
    result.K_idx *= KAlignment;
    result.k_tile_count *= KAlignment;
    return result;
  }
  CUTLASS_HOST_DEVICE static bool needs_workspace(Params const& p) {
    return p.iters_per_block_ != p.k_tiles_per_output_;
  }
  CUTLASS_HOST_DEVICE static uint64_t reduction_bytes(Params const& p) {
    return p.output_tiles_ * TileElements * sizeof(float);
  }
};
} // namespace quactlize::ppu17

namespace cutlass::gemm::kernel::detail {
template<class Tile, class Cluster>
struct TileSchedulerSelector<quactlize::ppu17::MarlinScheduler, cutlass::arch::Sm90, Tile, Cluster> {
  using Scheduler = quactlize::ppu17::CooperativeMarlinScheduler<Tile, Cluster>;
};
} // namespace cutlass::gemm::kernel::detail
