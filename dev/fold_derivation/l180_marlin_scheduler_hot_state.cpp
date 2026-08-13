// L180 -- production-equivalence oracle for the standalone Marlin scheduler's
// cold public Params and compact device traversal state.
//
// This does not implement another stripe scheduler.  Both sides call the
// production scheduler: the reference side enters through Params, while the
// device side constructs the shipping scheduler object and uses its cached
// traversal state.  Equality is required for every descriptor and reverse-q
// successor in the complete bounded domain below.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "quactlize_extensions/cutlass/gemm/kernel/marlin_scheduler_ppu.hpp"

using Scheduler = cutlass::gemm::kernel::marlin::MarlinSchedulerPPU<
    cute::Shape<cute::_16, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_1, cute::_1>>;
using Params = Scheduler::Params;
using State = Scheduler::DeviceTraversalState;
using Work = Scheduler::WorkTileInfo;

static_assert(sizeof(Params) == 40 && alignof(Params) == 8,
              "L180 preserves the public scheduler Params ABI");
static_assert(sizeof(State) == 16 && alignof(State) == 4,
              "L180 requires a four-word pointer-free device traversal state");
static_assert(std::is_standard_layout_v<State> &&
                  std::is_trivially_copyable_v<State>,
              "L180 device traversal state must remain a plain ABI value");
static_assert(sizeof(Scheduler) == 16 && alignof(Scheduler) == 4 &&
                  std::is_standard_layout_v<Scheduler> &&
                  std::is_trivially_copyable_v<Scheduler>,
              "L180 shipping scheduler object must contain only hot traversal state");

namespace {

bool same_work(Work const& a, Work const& b) {
  return a.N_idx == b.N_idx && a.K_idx == b.K_idx &&
         a.k_tile_count == b.k_tile_count && a.peer_idx == b.peer_idx &&
         a.flags == b.flags;
}

int fail(char const* reason, uint64_t schedules, uint64_t blocks,
         uint64_t segments) {
  std::fprintf(stderr,
               "[l180:red] schedules=%llu blocks=%llu segments=%llu "
               "reason=%s result=RED\n",
               static_cast<unsigned long long>(schedules),
               static_cast<unsigned long long>(blocks),
               static_cast<unsigned long long>(segments), reason);
  return 1;
}

}  // namespace

int main() {
  uint64_t schedules = 0;
  uint64_t blocks = 0;
  uint64_t segments = 0;

  // A fabricated invalid Params must not smuggle live traversal words into
  // the scheduler object.  This is the fail-closed half of the lowering.
  Params invalid = Scheduler::make_params_for_tiles(1, 32, 1, 32, 72);
  invalid.valid_ = false;
  State const invalid_state = Scheduler::make_device_traversal_state(invalid);
  Scheduler const invalid_scheduler(invalid);
  if (invalid_state.k_tiles_per_output_ != 0 ||
      invalid_state.output_tiles_ != 0 ||
      invalid_state.total_k_tiles_ != 0 ||
      invalid_state.iters_per_block_ != 0 ||
      invalid_scheduler.get_work_for_block_index(0).is_valid()) {
    return fail("invalid Params survived device-state lowering", 0, 0, 0);
  }

  // Complete bounded domain: every dense fixed-target tile count, K-tile
  // count, CU count and explicit BPC in these inclusive ranges.  This is
  // 64*32*32*4 = 262,144 distinct decompositions, not representative samples.
  for (uint32_t output_tiles = 1; output_tiles <= 64; ++output_tiles) {
    for (uint32_t k_tiles = 1; k_tiles <= 32; ++k_tiles) {
      for (uint32_t cu = 1; cu <= 32; ++cu) {
        for (uint32_t bpc = 1; bpc <= 4; ++bpc) {
          Params const p = Scheduler::make_params_for_tiles(
              1, output_tiles, 1, k_tiles, cu, nullptr, bpc);
          if (!p.valid_) {
            return fail("bounded production Params unexpectedly invalid",
                        schedules, blocks, segments);
          }
          ++schedules;

          State const state = Scheduler::make_device_traversal_state(p);
          if (state.k_tiles_per_output_ != p.k_tiles_per_output_ ||
              state.output_tiles_ != p.output_tiles_ ||
              state.total_k_tiles_ != p.total_k_tiles_ ||
              state.iters_per_block_ != p.iters_per_block_) {
            return fail("device traversal words differ from public Params",
                        schedules, blocks, segments);
          }

          Scheduler const device_scheduler(p);
          // Include the first two out-of-grid indices.  They must remain
          // invalid on both APIs, so compacting state cannot broaden work.
          for (uint64_t block = 0; block < uint64_t(p.grid_blocks_) + 2;
               ++block) {
            ++blocks;
            Work from_params = Scheduler::get_work_for_block(p, block);
            Work from_state =
                device_scheduler.get_work_for_block_index(block);
            if (!same_work(from_params, from_state)) {
              return fail("first descriptor differs after state lowering",
                          schedules, blocks, segments);
            }

            while (from_params.is_valid()) {
              ++segments;
              Work const next_params =
                  Scheduler::fetch_next_work_for_params(p, from_params);
              Work const next_state =
                  device_scheduler.get_next_work(from_state);
              if (!same_work(next_params, next_state)) {
                return fail("reverse-q successor differs after state lowering",
                            schedules, blocks, segments);
              }
              from_params = next_params;
              from_state = next_state;
            }
          }
        }
      }
    }
  }

  std::printf(
      "[l180] PASS: public-params=%zuB device-traversal=%zuB "
      "schedules=%llu blocks=%llu segments=%llu exact-descriptor-equality=1 "
      "invalid-lowering=zero\n",
      sizeof(Params), sizeof(State),
      static_cast<unsigned long long>(schedules),
      static_cast<unsigned long long>(blocks),
      static_cast<unsigned long long>(segments));
  return 0;
}
