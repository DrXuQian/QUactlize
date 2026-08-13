/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/workspace.h"
#include "cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp"

namespace cutlass::gemm::kernel::marlin {

// Scheduler seam for the standalone Marlin-CuTe PPU kernel.
//
// This type deliberately reuses only MarlinStripeSchedulerCore's proven
// integer decomposition.  Its workspace is one CTA-wide lock per global
// output-tile q.  In particular, it has no accumulator type and cannot name,
// allocate, or index an FP32 partial tile: the standalone cooperative hands
// partials through its own Marlin C-chain protocol.
//
// The first target is intentionally narrow.  It admits the classic dense
// decode geometry (one partial M tile, compact full N/K tiles, L == 1) for the
// fixed 16x128x128 CTA tile.  General residue, grouped and batched shapes must
// add an explicit contract rather than silently inheriting this scheduler.
template <class TileShape_, class ClusterShape_>
class MarlinSchedulerPPU {
public:
  using TileShape = TileShape_;
  using ClusterShape = ClusterShape_;
  using StripeCore = cutlass::gemm::kernel::detail::MarlinStripeSchedulerCore;
  using StripeParams = typename StripeCore::Params;
  using Barrier = cutlass::Barrier;
  using BarrierType = typename Barrier::T;

  enum class WorkFlag : uint8_t {
    Valid = 1u << 0,
    FirstPeer = 1u << 1,
    FinalPeer = 1u << 2,
    Split = 1u << 3,
  };

  // Hot device descriptor.  Fixed dense M/L coordinates are compile-time
  // facts, and global q is simultaneously N_idx, output-tile id and lock id;
  // none is stored twice.  First/final/split are lowered once rather than
  // re-derived at every cooperative call site.  There are no compatibility
  // fields from the retired 44-byte descriptor in this hot ABI.
  struct WorkTileInfo {
    int32_t N_idx = -1;
    uint32_t K_idx = 0;
    uint32_t k_tile_count = 0;
    uint32_t peer_idx = 0;
    uint8_t flags = 0;

    CUTLASS_HOST_DEVICE constexpr bool has_flag(WorkFlag flag) const {
      return (flags & uint8_t(flag)) != 0;
    }
    CUTLASS_HOST_DEVICE constexpr bool is_valid() const {
      return has_flag(WorkFlag::Valid);
    }
    CUTLASS_HOST_DEVICE static constexpr WorkTileInfo invalid_work_tile() {
      return {};
    }
  };

  static constexpr uint64_t TileM = uint64_t(cute::size<0>(TileShape{}));
  static constexpr uint64_t TileN = uint64_t(cute::size<1>(TileShape{}));
  static constexpr uint64_t TileK = uint64_t(cute::size<2>(TileShape{}));

  static_assert(cute::is_static<TileShape>::value,
                "standalone Marlin scheduler requires a static CTA tile");
  static_assert(cute::is_static<ClusterShape>::value,
                "standalone Marlin scheduler requires a static cluster");
  static_assert(TileM == 16 && TileN == 128 && TileK == 128,
                "the first standalone Marlin scheduler target is fixed at 16x128x128");
  static_assert(cute::size(ClusterShape{}) == 1,
                "the first standalone Marlin scheduler does not support clusters");
  static_assert(std::is_same_v<BarrierType, int>,
                "Marlin's global q-lock ABI is one 32-bit integer per output tile");

  struct Arguments {
    // B=1 is the classic launch contract.  A caller may request a larger B,
    // but the final kernel must first validate it against that exact kernel's
    // maximum_active_blocks(); the scheduler must never multiply occupancy in
    // implicitly or substitute a different value on device.
    uint32_t blocks_per_cu = 1;
  };

  struct Params {
    uint32_t k_tiles_per_output_ = 0;
    uint32_t output_tiles_ = 0;
    uint32_t total_k_tiles_ = 0;
    uint32_t grid_blocks_ = 0;
    uint32_t active_blocks_ = 0;
    uint32_t iters_per_block_ = 0;
    BarrierType* locks_ = nullptr;
    bool valid_ = false;
  };

  enum class PeerReleaseAction : uint8_t {
    None,
    Arrive,
    Reset,
  };

  static_assert(std::is_standard_layout_v<Arguments> &&
                    std::is_trivially_copyable_v<Arguments> &&
                    sizeof(Arguments) == 4 && alignof(Arguments) == 4 &&
                    offsetof(Arguments, blocks_per_cu) == 0,
                "standalone Marlin scheduler Arguments ABI changed");
  static_assert(std::is_standard_layout_v<Params> &&
                    std::is_trivially_copyable_v<Params>,
                "standalone Marlin scheduler arguments must cross the host/device ABI unchanged");
  static_assert(sizeof(Params) == 40 && alignof(Params) == alignof(BarrierType*) &&
                    offsetof(Params, k_tiles_per_output_) == 0 &&
                    offsetof(Params, output_tiles_) == 4 &&
                    offsetof(Params, total_k_tiles_) == 8 &&
                    offsetof(Params, grid_blocks_) == 12 &&
                    offsetof(Params, active_blocks_) == 16 &&
                    offsetof(Params, iters_per_block_) == 20 &&
                    offsetof(Params, locks_) == 24 &&
                    offsetof(Params, valid_) == 32,
                "standalone Marlin scheduler device ABI must stay compact");
  static_assert(std::is_standard_layout_v<WorkTileInfo> &&
                    std::is_trivially_copyable_v<WorkTileInfo> &&
                    sizeof(WorkTileInfo) == 20 && alignof(WorkTileInfo) == 4 &&
                    offsetof(WorkTileInfo, N_idx) == 0 &&
                    offsetof(WorkTileInfo, K_idx) == 4 &&
                    offsetof(WorkTileInfo, k_tile_count) == 8 &&
                    offsetof(WorkTileInfo, peer_idx) == 12 &&
                    offsetof(WorkTileInfo, flags) == 16,
                "standalone Marlin hot work descriptor ABI changed");

private:
  Params scheduler_params_{};

  CUTLASS_HOST_DEVICE static constexpr bool compact_core_params(
      StripeParams const& p) {
    uint64_t const u32 = uint64_t(std::numeric_limits<uint32_t>::max());
    uint64_t grid_span = 0;
    return p.valid_ && p.tiles_m_ == 1 && p.tiles_l_ == 1 &&
           p.tiles_n_ <= uint64_t(std::numeric_limits<int32_t>::max()) &&
           p.k_tiles_per_output_ <= u32 && p.output_tiles_ <= u32 &&
           p.total_k_tiles_ <= u32 && p.grid_blocks_ <= u32 &&
           p.active_blocks_ <= u32 && p.iters_per_block_ <= u32 &&
           p.iters_per_block_ <= p.k_tiles_per_output_ &&
           StripeCore::mul_u64(p.grid_blocks_, p.iters_per_block_, grid_span) &&
           grid_span <= u32;
  }

  // Direct transcription of Awesome-CuTe TileWorkDesc::init for the fixed
  // dense M/L contract.  K is the flattened fast dimension, q is global N,
  // and the lowest-block/lowest-K peer owns peer_idx zero.
  CUTLASS_HOST_DEVICE static constexpr WorkTileInfo make_reference_work(
      Params const& p, uint32_t block, uint32_t q,
      uint32_t block_begin, uint32_t block_end) {
    if (block_begin >= block_end || q >= p.output_tiles_) {
      return WorkTileInfo::invalid_work_tile();
    }
    uint32_t const kt = uint32_t(p.k_tiles_per_output_);
    uint32_t const iters = uint32_t(p.iters_per_block_);
    uint32_t const q_begin = q * kt;
    uint32_t const q_end = q_begin + kt;
    uint32_t const segment_begin = block_begin > q_begin ? block_begin : q_begin;
    uint32_t const segment_end = block_end < q_end ? block_end : q_end;
    uint32_t const first_peer = q_begin / iters;
    if (segment_begin >= segment_end || block < first_peer) {
      return WorkTileInfo::invalid_work_tile();
    }

    WorkTileInfo work;
    work.N_idx = int32_t(q);
    work.K_idx = segment_begin - q_begin;
    work.k_tile_count = segment_end - segment_begin;
    work.peer_idx = block - first_peer;
    bool const first_k = work.K_idx == 0;
    bool const final_k = work.K_idx + work.k_tile_count == kt;
    bool const split = !(first_k && final_k);
    work.flags = uint8_t(WorkFlag::Valid) |
                 (first_k ? uint8_t(WorkFlag::FirstPeer) : uint8_t(0)) |
                 (final_k ? uint8_t(WorkFlag::FinalPeer) : uint8_t(0)) |
                 (split ? uint8_t(WorkFlag::Split) : uint8_t(0));
    if (work.k_tile_count == 0 || first_k != (work.peer_idx == 0) ||
        final_k != (segment_end == q_end)) {
      return WorkTileInfo::invalid_work_tile();
    }
    return work;
  }

  CUTLASS_HOST_DEVICE static constexpr Params invalid_params() {
    return {};
  }

  CUTLASS_HOST_DEVICE static constexpr bool needs_peer_locks(Params const& p) {
    // With this launch policy, I == Kt is the only no-handoff schedule:
    // every CTA owns one complete output tile.  Other valid I values place at
    // least one CTA boundary inside an output tile.
    return p.valid_ && p.iters_per_block_ != p.k_tiles_per_output_;
  }

  CUTLASS_HOST_DEVICE static constexpr uint64_t lock_workspace_bytes_u64(
      Params const& p) {
    if (!needs_peer_locks(p)) {
      return 0;
    }
    uint64_t bytes = 0;
    return StripeCore::mul_u64(p.output_tiles_, sizeof(BarrierType), bytes)
        ? bytes
        : 0;
  }

public:
  CUTLASS_HOST_DEVICE constexpr MarlinSchedulerPPU() = default;
  CUTLASS_HOST_DEVICE constexpr explicit MarlinSchedulerPPU(Params const& p)
      : scheduler_params_(p) {}

  CUTLASS_HOST_DEVICE static constexpr bool arguments_supported(
      Arguments const& args) {
    return args.blocks_per_cu > 0;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static constexpr bool problem_supported(
      ProblemShape problem_shape) {
    static_assert(cute::rank(ProblemShape{}) == 3 ||
                      cute::rank(ProblemShape{}) == 4,
                  "standalone Marlin scheduler requires dense <M,N,K[,L]>");
    auto shape = cute::append<4>(problem_shape, cute::Int<1>{});
    int64_t const m = int64_t(cute::get<0>(shape));
    int64_t const n = int64_t(cute::get<1>(shape));
    int64_t const k = int64_t(cute::get<2>(shape));
    int64_t const l = int64_t(cute::get<3>(shape));
    return m > 0 && m <= int64_t(TileM) &&
           n >= int64_t(TileN) && n % int64_t(TileN) == 0 &&
           k >= int64_t(TileK) && k % int64_t(TileK) == 0 &&
           l == 1;
  }

  // This is the single G/BPC lowering seam.  It calls the already-proved
  // K-fast integer core verbatim: G=max(Q,CU*B), I=ceil(Q*Kt/G), and q is the
  // globally flattened N-fast output-tile ordinal.
  CUTLASS_HOST_DEVICE static constexpr Params make_params_for_tiles(
      uint64_t tiles_m, uint64_t tiles_n, uint64_t tiles_l,
      uint64_t k_tiles, uint64_t cu_count,
      BarrierType* locks = nullptr, uint32_t blocks_per_cu = 1) {
    // The first fixed target has exactly one (possibly residual) M tile and
    // no batch/group dimension.  Reject rather than lower an unsupported
    // problem into an apparently valid dense schedule.
    if (tiles_m != 1 || tiles_n == 0 || tiles_l != 1 || k_tiles == 0 ||
        cu_count == 0 || blocks_per_cu == 0) {
      return invalid_params();
    }
    StripeParams const core = StripeCore::make_params_for_tiles(
        tiles_m, tiles_n, tiles_l, k_tiles, cu_count,
        uint64_t(blocks_per_cu));
    if (!compact_core_params(core)) {
      return invalid_params();
    }
    return {
        uint32_t(core.k_tiles_per_output_), uint32_t(core.output_tiles_),
        uint32_t(core.total_k_tiles_), uint32_t(core.grid_blocks_),
        uint32_t(core.active_blocks_), uint32_t(core.iters_per_block_),
        locks, true};
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static constexpr Params make_params_for_problem_shape(
      ProblemShape problem_shape, uint64_t cu_count,
      BarrierType* locks = nullptr, uint32_t blocks_per_cu = 1) {
    if (!problem_supported(problem_shape)) {
      return invalid_params();
    }
    auto shape = cute::append<4>(problem_shape, cute::Int<1>{});
    uint64_t const m = uint64_t(cute::get<0>(shape));
    uint64_t const n = uint64_t(cute::get<1>(shape));
    uint64_t const k = uint64_t(cute::get<2>(shape));
    return make_params_for_tiles(
        StripeCore::ceil_div_u64(m, TileM), n / TileN, 1, k / TileK,
        cu_count, locks, blocks_per_cu);
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static constexpr Params to_underlying_arguments(
      ProblemShape problem_shape, KernelHardwareInfo const& hw_info,
      Arguments const& args, void* workspace) {
    return make_params_for_problem_shape(
        problem_shape,
        hw_info.cu_count > 0 ? uint64_t(hw_info.cu_count) : uint64_t(0),
        reinterpret_cast<BarrierType*>(workspace), args.blocks_per_cu);
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static constexpr bool can_implement(
      ProblemShape problem_shape, KernelHardwareInfo const& hw_info,
      Arguments const& args) {
    return arguments_supported(args) && hw_info.cu_count > 0 &&
           make_params_for_problem_shape(
               problem_shape, uint64_t(hw_info.cu_count), nullptr,
               args.blocks_per_cu)
               .valid_;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(
      Arguments const& args, ProblemShape problem_shape,
      KernelHardwareInfo const& hw_info) {
    Params const p = to_underlying_arguments(
        problem_shape, hw_info, args, nullptr);
    if (!p.valid_) {
      return 0;
    }
    uint64_t const bytes = lock_workspace_bytes_u64(p);
    return bytes <= uint64_t(std::numeric_limits<size_t>::max())
        ? size_t(bytes)
        : 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      Arguments const& args, void* workspace, hggcStream_t stream,
      ProblemShape const& problem_shape, KernelHardwareInfo const& hw_info,
      HostAdapter* host_adapter = nullptr) {
    if (!can_implement(problem_shape, hw_info, args)) {
      return Status::kErrorInvalidProblem;
    }
    Params const p = to_underlying_arguments(
        problem_shape, hw_info, args, workspace);
    uint64_t const bytes_u64 = lock_workspace_bytes_u64(p);
    if (bytes_u64 == 0) {
      return Status::kSuccess;
    }
    if (workspace == nullptr ||
        bytes_u64 > uint64_t(std::numeric_limits<size_t>::max())) {
      return workspace == nullptr
          ? Status::kErrorWorkspaceNull
          : Status::kErrorInvalidProblem;
    }
    return zero_workspace(workspace, size_t(bytes_u64), stream, host_adapter);
  }

  CUTLASS_HOST_DEVICE static constexpr bool requires_handoff(
      WorkTileInfo const& work) {
    return work.is_valid() && work.has_flag(WorkFlag::Split);
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_first_peer(
      WorkTileInfo const& work) {
    return work.is_valid() && work.has_flag(WorkFlag::FirstPeer);
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_final_peer(
      WorkTileInfo const& work) {
    return work.is_valid() && work.has_flag(WorkFlag::FinalPeer);
  }

  CUTLASS_HOST_DEVICE static constexpr BarrierType peer_wait_value(
      WorkTileInfo const& work) {
    return requires_handoff(work) ? BarrierType(work.peer_idx) : BarrierType(0);
  }

  CUTLASS_HOST_DEVICE static constexpr PeerReleaseAction peer_release_action(
      WorkTileInfo const& work) {
    return !requires_handoff(work)
        ? PeerReleaseAction::None
        : (is_final_peer(work) ? PeerReleaseAction::Reset
                               : PeerReleaseAction::Arrive);
  }

  CUTLASS_HOST_DEVICE static constexpr uint64_t output_tile_index(
      WorkTileInfo const& work) {
    return work.is_valid() ? uint64_t(work.N_idx) : uint64_t(-1);
  }

  CUTLASS_HOST_DEVICE static constexpr int barrier_lock_index(
      WorkTileInfo const& work) {
    // MarlinStripeSchedulerCore has already rejected Q > INT_MAX.  The lock
    // is derived from the one stored global q; there is no independent local
    // coordinate that can silently alias another output tile.
    return work.is_valid() && work.N_idx >= 0 ? int(work.N_idx) : -1;
  }

  CUTLASS_HOST_DEVICE static constexpr BarrierType* lock_workspace(
      Params const& p) {
    return p.locks_;
  }

  CUTLASS_DEVICE static void acquire_peer_turn(
      Params const& p, WorkTileInfo const& work, int thread_idx) {
    if (requires_handoff(work)) {
      CUTLASS_ASSERT(p.locks_ != nullptr);
      int const lock = barrier_lock_index(work);
      CUTLASS_ASSERT(lock >= 0);
      Barrier::wait_eq(p.locks_, thread_idx, lock, peer_wait_value(work));
    }
  }

  CUTLASS_DEVICE static void release_peer_turn(
      Params const& p, WorkTileInfo const& work, int thread_idx) {
    if (!requires_handoff(work)) {
      return;
    }
    CUTLASS_ASSERT(p.locks_ != nullptr);
    int const lock = barrier_lock_index(work);
    CUTLASS_ASSERT(lock >= 0);
    PeerReleaseAction const action = peer_release_action(work);
    if (action == PeerReleaseAction::Reset) {
      // Awesome-CuTe/classic Marlin reset the q lock with one thread after
      // the final peer's wait_eq acquisition.  Do not repeat that acquisition
      // through wait_eq_reset: it adds an atomic-CAS spin and another CTA
      // rendezvous to the hot path.  The kernel's result-staging barrier
      // follows this call, and a later launch on the same stream cannot begin
      // before this kernel completes.
      if (thread_idx == 0) {
        p.locks_[lock] = BarrierType(0);
      }
    }
    else if (action == PeerReleaseAction::Arrive) {
      Barrier::arrive_inc(p.locks_, thread_idx, lock, 1);
    }
  }

  CUTLASS_HOST_DEVICE static dim3 get_grid_shape(Params const& p) {
    return p.valid_ &&
                   p.grid_blocks_ <= uint64_t(std::numeric_limits<unsigned>::max())
        ? dim3(unsigned(p.grid_blocks_), 1, 1)
        : dim3(0, 0, 0);
  }

  CUTLASS_HOST_DEVICE static constexpr WorkTileInfo get_work_for_block(
      Params const& p, uint64_t block_idx) {
    if (!p.valid_ || block_idx >= p.grid_blocks_) {
      return WorkTileInfo::invalid_work_tile();
    }
    uint32_t const block = uint32_t(block_idx);
    uint32_t const iters = uint32_t(p.iters_per_block_);
    uint32_t const total = uint32_t(p.total_k_tiles_);
    uint32_t const begin = block * iters;
    if (begin >= total) {
      return WorkTileInfo::invalid_work_tile();
    }
    uint32_t const raw_end = begin + iters;
    uint32_t const end = raw_end < total ? raw_end : total;
    uint32_t const end_q = (end - 1) / uint32_t(p.k_tiles_per_output_);
    return make_reference_work(p, block, end_q, begin, end);
  }

  CUTLASS_HOST_DEVICE constexpr WorkTileInfo get_work_for_block_index(
      uint64_t block_idx) const {
    return get_work_for_block(scheduler_params_, block_idx);
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    return get_work_for_block_index(uint64_t(blockIdx.x));
  }

  CUTLASS_HOST_DEVICE static constexpr WorkTileInfo fetch_next_work_for_params(
      Params const& p, WorkTileInfo const& work) {
    if (!p.valid_ || !work.is_valid() || work.N_idx <= 0 ||
        work.K_idx != 0 || work.k_tile_count >= p.iters_per_block_) {
      return WorkTileInfo::invalid_work_tile();
    }
    // G=max(Q,CU*B) guarantees I<=Kt.  Consequently a stripe touches at
    // most two output tiles.  A reverse successor exists exactly when the
    // high-q segment starts at K=0 but contains fewer than I cells.
    uint32_t const iters = uint32_t(p.iters_per_block_);
    uint32_t const kt = uint32_t(p.k_tiles_per_output_);
    uint32_t const remaining = iters - work.k_tile_count;
    if (remaining == 0 || remaining >= kt) {
      return WorkTileInfo::invalid_work_tile();
    }
    uint32_t const q = uint32_t(work.N_idx) - 1;
    uint32_t const q_begin = q * kt;
    uint32_t const begin = q_begin + kt - remaining;
    uint32_t const block = begin / iters;
    return make_reference_work(p, block, q, begin, q_begin + kt);
  }

  CUTLASS_HOST_DEVICE constexpr WorkTileInfo get_next_work(
      WorkTileInfo const& work) const {
    return fetch_next_work_for_params(scheduler_params_, work);
  }

  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo const& work) const {
    return cute::make_tuple(get_next_work(work), true);
  }

  CUTLASS_HOST_DEVICE static constexpr uint32_t get_work_k_tile_start(
      WorkTileInfo const& work) {
    return work.K_idx;
  }

  CUTLASS_HOST_DEVICE static constexpr uint32_t get_work_k_tile_count(
      WorkTileInfo const& work) {
    return work.k_tile_count;
  }
};

namespace detail {

// Compile-time oracle for the traversal adapter above.  This reconstruction
// is intentionally independent of StripeCore::fetch_next_work: it implements
// the Awesome-CuTe reference literally as
//
//   end_tile = (block_iter_end - 1) / Kt;
//   begin_tile = block_iter_begin / Kt;
//   for (tile_idx = end_tile; tile_idx >= begin_tile; --tile_idx)
//
// and checks every active CTA/segment, including segment K bounds and global
// q lock identity.  These cases cover the classic 20-SM diagram, the fixed
// 72-CU decode target, and an explicit BPC sweep point.
using MarlinSchedulerPPUOracle = MarlinSchedulerPPU<
    cute::Shape<cute::_16, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_1, cute::_1>>;

CUTLASS_HOST_DEVICE constexpr bool marlin_scheduler_ppu_reverse_reference(
    uint64_t tiles_n, uint64_t k_tiles, uint64_t cu_count,
    uint32_t blocks_per_cu) {
  using Scheduler = MarlinSchedulerPPUOracle;
  using Work = typename Scheduler::WorkTileInfo;
  auto const p = Scheduler::make_params_for_tiles(
      1, tiles_n, 1, k_tiles, cu_count, nullptr, blocks_per_cu);
  if (!p.valid_ || p.iters_per_block_ > p.k_tiles_per_output_) {
    return false;
  }

  uint64_t visited = 0;
  for (uint64_t block = 0; block < p.grid_blocks_; ++block) {
    uint64_t const begin = block * p.iters_per_block_;
    uint64_t end = begin + p.iters_per_block_;
    if (end > p.total_k_tiles_ || end < begin) {
      end = p.total_k_tiles_;
    }
    Work work = Scheduler::get_work_for_block(p, block);
    if (begin >= end) {
      if (work.is_valid()) {
        return false;
      }
      continue;
    }

    uint64_t expected_q = (end - 1) / p.k_tiles_per_output_;
    uint64_t const begin_q = begin / p.k_tiles_per_output_;
    while (true) {
      if (!work.is_valid() || uint32_t(work.N_idx) != expected_q ||
          Scheduler::barrier_lock_index(work) != int(expected_q)) {
        return false;
      }
      uint64_t const q_begin = expected_q * p.k_tiles_per_output_;
      uint64_t const q_end = q_begin + p.k_tiles_per_output_;
      uint64_t const segment_begin = begin > q_begin ? begin : q_begin;
      uint64_t const segment_end = end < q_end ? end : q_end;
      if (work.K_idx != segment_begin - q_begin ||
          work.k_tile_count != segment_end - segment_begin) {
        return false;
      }
      visited += work.k_tile_count;
      work = Scheduler::fetch_next_work_for_params(p, work);
      if (expected_q == begin_q) {
        break;
      }
      --expected_q;
    }
    if (work.is_valid()) {
      return false;
    }
  }
  return visited == p.total_k_tiles_;
}

static_assert(marlin_scheduler_ppu_reverse_reference(16, 16, 20, 1),
              "Marlin scheduler must reproduce the reference 20-SM reverse-q traversal");
static_assert(marlin_scheduler_ppu_reverse_reference(32, 32, 72, 1),
              "Marlin scheduler must reproduce fixed-target B=1 reverse-q traversal");
static_assert(marlin_scheduler_ppu_reverse_reference(32, 32, 72, 4),
              "Marlin scheduler must preserve reverse-q traversal under explicit BPC");

CUTLASS_HOST_DEVICE constexpr bool marlin_scheduler_ppu_has_reverse_witness() {
  using Scheduler = MarlinSchedulerPPUOracle;
  auto const p = Scheduler::make_params_for_tiles(1, 32, 1, 32, 72);
  auto const stripe = Scheduler::StripeCore::make_params_for_tiles(
      1, 32, 1, 32, 72);
  for (uint64_t block = 0; block < p.grid_blocks_; ++block) {
    auto const forward = Scheduler::StripeCore::get_work_for_block(stripe, block);
    auto const forward_second =
        Scheduler::StripeCore::fetch_next_work(stripe, forward);
    auto const reverse = Scheduler::get_work_for_block(p, block);
    if (forward.is_valid() && forward_second.is_valid()) {
      return reverse.is_valid() &&
             uint64_t(reverse.N_idx) == forward_second.output_tile_idx &&
             uint64_t(reverse.N_idx) > forward.output_tile_idx;
    }
  }
  return false;
}

static_assert(marlin_scheduler_ppu_has_reverse_witness(),
              "reverse-q oracle needs a stripe where forward and reference orders differ");
static_assert(!MarlinSchedulerPPUOracle::make_params_for_tiles(
                  2, 32, 1, 32, 72)
                   .valid_ &&
              !MarlinSchedulerPPUOracle::make_params_for_tiles(
                  1, 32, 2, 32, 72)
                   .valid_ &&
              !MarlinSchedulerPPUOracle::make_params_for_tiles(
                  1, 32, 1, 32, 72, nullptr, 0)
                   .valid_,
              "first standalone scheduler must fail closed on M/L/BPC extensions");

} // namespace detail

} // namespace cutlass::gemm::kernel::marlin
