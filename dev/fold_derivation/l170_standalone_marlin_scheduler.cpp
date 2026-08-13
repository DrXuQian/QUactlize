// L170 -- production-bound host oracle for MarlinSchedulerPPU.
//
// L168 independently reconstructs the Awesome-CuTe/classic stripe.  This
// oracle answers the other half: does the production descriptor expose that
// exact traversal, ABI and peer-lock state machine?  It calls the production
// helpers directly; none of the work descriptors below are re-created by a
// parallel scheduler model.

#include <array>
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
using Work = Scheduler::WorkTileInfo;
using Action = Scheduler::PeerReleaseAction;

#if defined(L170_PLANT_LEGACY_44_DESCRIPTOR)
struct DescriptorAbiWitness {
  uint32_t legacy_words[11];
};
#else
using DescriptorAbiWitness = Work;
#endif

static_assert(std::is_same_v<DescriptorAbiWitness, Work> &&
                  sizeof(DescriptorAbiWitness) == 20 &&
                  alignof(DescriptorAbiWitness) == 4,
              "L170 rejects the legacy 44-byte work descriptor");

constexpr bool cached_split_flag_is_authoritative() {
  auto const p = Scheduler::make_params_for_tiles(1, 32, 1, 32, 72);
  Work work = Scheduler::get_work_for_block(p, 2);
  if (!work.is_valid() || !Scheduler::requires_handoff(work)) {
    return false;
  }
  work.flags &= ~uint8_t(Scheduler::WorkFlag::Split);
#if defined(L170_PLANT_RECOMPUTED_PREDICATE)
  bool const observed = work.is_valid() &&
                        !(Scheduler::is_first_peer(work) &&
                          Scheduler::is_final_peer(work));
#else
  bool const observed = Scheduler::requires_handoff(work);
#endif
  return !observed;
}

static_assert(cached_split_flag_is_authoritative(),
              "L170 requires the cached split predicate, not recomputation");

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
  std::fprintf(stderr, "[l170:red] plant=%s caught=1 reason=%s result=RED\n",
               plant, reason);
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  char const* plant = plant_name(argc, argv);
  Params const p = Scheduler::make_params_for_tiles(1, 32, 1, 32, 72);
  if (!p.valid_ || p.k_tiles_per_output_ != 32 || p.output_tiles_ != 32 ||
      p.total_k_tiles_ != 1024 || p.grid_blocks_ != 72 ||
      p.active_blocks_ != 69 || p.iters_per_block_ != 15) {
    return fail(plant, "production fixed-target Params changed");
  }

  std::array<uint8_t, 1024> visits{};
  std::array<uint32_t, 32> peers{};
  std::array<int, 32> lock_state{};
  uint32_t segments = 0;
  uint32_t cells = 0;
  uint32_t handoffs = 0;
  uint32_t active = 0;
  uint32_t idle = 0;
  uint32_t reverse_witness = 0;
  uint32_t final_resets = 0;

  for (uint32_t block = 0; block < p.grid_blocks_; ++block) {
    Work work = Scheduler::get_work_for_block(p, block);
    if (!work.is_valid()) {
      ++idle;
      if (is_plant(plant, "inactive-valid") && block == 69) {
        return fail(plant, "inactive CTA 69 was deliberately required valid");
      }
      continue;
    }
    ++active;
    uint32_t previous_q = uint32_t(work.N_idx);
    bool first_segment = true;
    while (work.is_valid()) {
      ++segments;
      uint32_t const q = uint32_t(work.N_idx);
      if (!first_segment) {
        if (q >= previous_q) {
          return fail(plant, "production traversal is not reverse-q");
        }
        ++reverse_witness;
        if (is_plant(plant, "forward-q")) {
          return fail(plant, "forward-q plant rejected by production ordering");
        }
      }
      first_segment = false;
      previous_q = q;

      uint32_t expected_slice = block - (q * p.k_tiles_per_output_) /
                                            p.iters_per_block_;
      if (is_plant(plant, "wrong-slice")) {
        ++expected_slice;
      }
      if (work.peer_idx != expected_slice ||
          Scheduler::peer_wait_value(work) != int(expected_slice)) {
        return fail(plant, "slice/wait ordinal mismatch");
      }
      int expected_lock = int(q);
      if (is_plant(plant, "aliased-lock")) {
        expected_lock &= 15;
      }
      if (Scheduler::barrier_lock_index(work) != expected_lock) {
        return fail(plant, "global q lock was aliased or changed");
      }
      if (lock_state[q] != int(work.peer_idx)) {
        return fail(plant, "peer arrived out of lock order");
      }

      ++peers[q];
      bool const first = Scheduler::is_first_peer(work);
      bool const final = Scheduler::is_final_peer(work);
      bool const split = Scheduler::requires_handoff(work);
      if (!split || first != (work.K_idx == 0) ||
          final != (work.K_idx + work.k_tile_count == 32)) {
        return fail(plant, "production handoff/first/final predicate changed");
      }
      Action action = Scheduler::peer_release_action(work);
      if (is_plant(plant, "no-reset") && action == Action::Reset) {
        action = Action::Arrive;
      }
      if (is_plant(plant, "early-reset") && action == Action::Arrive) {
        action = Action::Reset;
      }
      if (action == Action::Arrive) {
        ++lock_state[q];
        ++handoffs;
      } else if (action == Action::Reset) {
        lock_state[q] = 0;
        ++final_resets;
      } else {
        return fail(plant, "split segment exposed no release action");
      }

      for (uint32_t k = work.K_idx;
           k < work.K_idx + work.k_tile_count; ++k) {
        uint32_t const cell = q * 32 + k;
        if (is_plant(plant, "cell-hole") && cell == 511) {
          continue;
        }
        if (++visits[cell] != 1) {
          return fail(plant, "duplicate (q,k) cell");
        }
        ++cells;
      }
      work = Scheduler::fetch_next_work_for_params(p, work);
    }
  }

  for (uint32_t cell = 0; cell < visits.size(); ++cell) {
    if (visits[cell] != 1) {
      return fail(plant, "hole in (q,k) coverage");
    }
  }
  uint32_t hist3 = 0;
  uint32_t hist4 = 0;
  for (uint32_t q = 0; q < peers.size(); ++q) {
    hist3 += peers[q] == 3;
    hist4 += peers[q] == 4;
    if (lock_state[q] != 0) {
      return fail(plant, "q lock did not finish reset to zero");
    }
  }

  Work w = Scheduler::get_work_for_block(p, 2);
  Work w2 = Scheduler::fetch_next_work_for_params(p, w);
  if (!(w.N_idx == 1 && w.K_idx == 0 &&
        w.k_tile_count == 13 && w.peer_idx == 0 &&
        w2.N_idx == 0 && w2.K_idx == 30 &&
        w2.k_tile_count == 2 && w2.peer_idx == 2 &&
        Scheduler::is_final_peer(w2))) {
    return fail(plant, "block-2 reverse-q witness changed");
  }

  Params const no_handoff =
      Scheduler::make_params_for_tiles(1, 72, 1, 1, 72);
  if (!no_handoff.valid_ || no_handoff.iters_per_block_ != 1 ||
      no_handoff.active_blocks_ != 72) {
    return fail(plant, "Q>=CU no-handoff Params changed");
  }
  for (uint32_t block = 0; block < 72; ++block) {
    Work const whole = Scheduler::get_work_for_block(no_handoff, block);
    if (!whole.is_valid() || Scheduler::requires_handoff(whole) ||
        Scheduler::peer_release_action(whole) != Action::None) {
      return fail(plant, "Q>=CU path unexpectedly requires a peer lock");
    }
  }

  if (active != 69 || idle != 3 || segments != 98 || cells != 1024 ||
      handoffs != 66 || final_resets != 32 || reverse_witness != 29 ||
      hist3 != 30 || hist4 != 2) {
    return fail(plant, "fixed-target census changed");
  }
  if (!is_plant(plant, "none")) {
    return fail(plant, "named plant did not perturb its intended invariant");
  }

  std::printf(
      "[l170] PASS: G=%u I=%u active=%u idle=%u segments=%u cells=%u "
      "handoffs=%u locks-reset=%u reverse=%u peers={3:%u,4:%u} "
      "abi={args:%zu/%zu params:%zu/%zu work:%zu/%zu}\n",
      p.grid_blocks_, p.iters_per_block_, active, idle, segments, cells,
      handoffs, final_resets, reverse_witness, hist3, hist4,
      sizeof(Scheduler::Arguments), alignof(Scheduler::Arguments),
      sizeof(Params), alignof(Params), sizeof(Work), alignof(Work));
  return 0;
}
