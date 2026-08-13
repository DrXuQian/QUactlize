// L177 -- production-bound Marlin cooperative lock-lifecycle oracle.
//
// The kernel caches Split/First/Final once and executes the ordered D-chain
// only for split output tiles.  This host oracle consumes the production
// WorkTileInfo descriptors and compares every protocol field against an
// independent reconstruction from (q, K_idx, k_tile_count).  It then models
// the exact acquire -> handoff -> arrive/reset lifecycle.  No device timing or
// floating-point result is involved.

// Runtime plants deliberately reproduce the three silent failures this gate
// exists to reject: a local-q lock id, resetting before the final peer, and a
// missing handoff operation.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "quactlize_extensions/cutlass/gemm/kernel/marlin_scheduler_ppu.hpp"

using Scheduler = cutlass::gemm::kernel::marlin::MarlinSchedulerPPU<
    cute::Shape<cute::_16, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_1, cute::_1>>;
using Work = Scheduler::WorkTileInfo;
using Action = Scheduler::PeerReleaseAction;

namespace {

char const* plant_name(int argc, char** argv) {
  constexpr char prefix[] = "--plant=";
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], prefix, sizeof(prefix) - 1) == 0) {
      return argv[i] + sizeof(prefix) - 1;
    }
  }
  return "none";
}

bool is_plant(char const* plant, char const* name) {
  return std::strcmp(plant, name) == 0;
}

int fail(char const* plant, char const* reason) {
  std::fprintf(stderr,
               "[l177:red] plant=%s caught=1 reason=%s result=RED\n",
               plant, reason);
  return 1;
}

struct Protocol {
  uint32_t lock;
  uint32_t wait;
  uint32_t split;
  uint32_t first;
  uint32_t final;
  uint32_t action;
};

uint32_t popcount32(uint32_t x) {
  uint32_t count = 0;
  while (x != 0) {
    count += x & 1u;
    x >>= 1;
  }
  return count;
}

uint32_t protocol_bitdiff(Protocol const& a, Protocol const& b) {
  return popcount32(a.lock ^ b.lock) + popcount32(a.wait ^ b.wait) +
         popcount32(a.split ^ b.split) + popcount32(a.first ^ b.first) +
         popcount32(a.final ^ b.final) + popcount32(a.action ^ b.action);
}

Protocol expected_protocol(Work const& work, uint32_t kt) {
  bool const first = work.K_idx == 0;
  bool const final = work.K_idx + work.k_tile_count == kt;
  bool const split = !(first && final);
  return {
      uint32_t(work.N_idx), work.peer_idx, uint32_t(split), uint32_t(first),
      uint32_t(final),
      uint32_t(!split ? Action::None
                      : (final ? Action::Reset : Action::Arrive))};
}

Protocol production_protocol(Work const& work, char const* plant) {
  uint32_t lock = uint32_t(Scheduler::barrier_lock_index(work));
  if (is_plant(plant, "local-q")) {
    lock &= 15u;
  }
  bool const split = Scheduler::requires_handoff(work);
  bool const first = Scheduler::is_first_peer(work);
  bool const final = Scheduler::is_final_peer(work);
  Action action = Scheduler::peer_release_action(work);
  if (is_plant(plant, "early-reset") && split && !final) {
    action = Action::Reset;
  }
  return {
      lock, uint32_t(Scheduler::peer_wait_value(work)), uint32_t(split),
      uint32_t(first), uint32_t(final), uint32_t(action)};
}

struct Census {
  uint32_t segments = 0;
  uint32_t split_segments = 0;
  uint32_t acquire_calls = 0;
  uint32_t handoff_calls = 0;
  uint32_t release_calls = 0;
  uint32_t arrive_calls = 0;
  uint32_t reset_calls = 0;
  uint32_t write_calls = 0;
  uint32_t protocol_bitdiff = 0;
};

bool run_schedule(
    Scheduler::Params const& p, char const* plant, Census& census) {
  std::array<int, 128> locks{};
  if (!p.valid_ || p.output_tiles_ > locks.size()) {
    return false;
  }
  bool skipped = false;
  for (uint32_t block = 0; block < p.grid_blocks_; ++block) {
    Work work = Scheduler::get_work_for_block(p, block);
    while (work.is_valid()) {
      ++census.segments;
      Protocol observed = production_protocol(work, plant);
      Protocol const expected = expected_protocol(
          work, uint32_t(p.k_tiles_per_output_));
      census.protocol_bitdiff += protocol_bitdiff(observed, expected);

      // This is the kernel's one cached decision.  Nothing below recomputes
      // split/first/final from the descriptor.
      bool const split = observed.split != 0;
      bool const final = observed.final != 0;
      if (split) {
        ++census.split_segments;
        ++census.acquire_calls;
        if (observed.lock >= locks.size() ||
            locks[observed.lock] != int(observed.wait)) {
          return false;
        }

        bool const skip = is_plant(plant, "skip-handoff") && !skipped;
        if (!skip) {
          ++census.handoff_calls;
        } else {
          skipped = true;
        }

        ++census.release_calls;
        if (Action(observed.action) == Action::Reset) {
          locks[observed.lock] = 0;
          ++census.reset_calls;
        } else if (Action(observed.action) == Action::Arrive) {
          ++locks[observed.lock];
          ++census.arrive_calls;
        } else {
          return false;
        }
      }
      if (final) {
        ++census.write_calls;
      }
      work = Scheduler::fetch_next_work_for_params(p, work);
    }
  }
  for (uint32_t q = 0; q < p.output_tiles_; ++q) {
    if (locks[q] != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  char const* plant = plant_name(argc, argv);

  auto const split_params =
      Scheduler::make_params_for_tiles(1, 32, 1, 32, 72);
  Census split{};
  if (!run_schedule(split_params, plant, split)) {
    return fail(plant, "split lock lifecycle diverged");
  }
  if (split.protocol_bitdiff != 0 || split.segments != 98 ||
      split.split_segments != 98 || split.acquire_calls != 98 ||
      split.handoff_calls != 98 || split.release_calls != 98 ||
      split.arrive_calls != 66 || split.reset_calls != 32 ||
      split.write_calls != 32) {
    return fail(plant, "split protocol/census is not bit-exact");
  }

  // Q >= CU and Kt=1 gives I==Kt: every segment is a complete DP tile.
  // The lock/D-chain call count must be exactly zero, not merely harmless.
  auto const whole_params =
      Scheduler::make_params_for_tiles(1, 72, 1, 1, 72);
  Census whole{};
  if (!run_schedule(whole_params, plant, whole) ||
      whole.protocol_bitdiff != 0 || whole.segments != 72 ||
      whole.split_segments != 0 || whole.acquire_calls != 0 ||
      whole.handoff_calls != 0 || whole.release_calls != 0 ||
      whole.arrive_calls != 0 || whole.reset_calls != 0 ||
      whole.write_calls != 72) {
    return fail(plant, "Q>=CU unsplit path entered the handoff protocol");
  }

  if (!is_plant(plant, "none")) {
    return fail(plant, "named plant did not perturb its intended invariant");
  }
  std::printf(
      "[l177] PASS: split={segments:%u acquire:%u handoff:%u release:%u "
      "arrive:%u reset:%u writes:%u bitdiff:%u} "
      "unsplit={segments:%u acquire:%u handoff:%u release:%u writes:%u}\n",
      split.split_segments, split.acquire_calls, split.handoff_calls,
      split.release_calls, split.arrive_calls, split.reset_calls,
      split.write_calls, split.protocol_bitdiff, whole.segments,
      whole.acquire_calls, whole.handoff_calls, whole.release_calls,
      whole.write_calls);
  return 0;
}
