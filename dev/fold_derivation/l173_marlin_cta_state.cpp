// L173 -- production-bound host oracle for standalone Marlin CTA address state.
//
// The production collective owns both init_cta_state() and rebase_segment().
// This oracle calls those functions directly.  Its independent anchors are:
//   * classic's closed-form A/B/scale pointer equations; and
//   * L168's independently reconstructed reverse-q segment sequence.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/marlin_scheduler_ppu.hpp"

using Tile = cute::Shape<cute::_16, cute::_128, cute::_128>;
using Warp = cute::Shape<cute::_16, cute::_64, cute::_32>;
using Main = cutlass::gemm::collective::MarlinCollectivePPU<
    Tile, Warp, 4, 128, cute::Stride<int64_t, cute::_1, int64_t>,
    cute::Stride<int64_t, cute::_1, int64_t>,
    cute::Stride<cute::_1, int64_t, int64_t>>;
using Scheduler = cutlass::gemm::kernel::marlin::MarlinSchedulerPPU<
    Tile, cute::Shape<cute::_1, cute::_1, cute::_1>>;

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
  std::fprintf(stderr, "[l173:red] plant=%s caught=1 reason=%s result=RED\n",
               plant, reason);
  return 1;
}

struct ClassicAddress {
  int a = 0;
  std::array<int, Main::BInnerIters> b{};
  int scale = 0;
};

ClassicAddress classic_address(int tid, uint32_t q, uint32_t k) {
  constexpr int problem_n = 4096;
  constexpr int problem_k = 4096;
  int const a_stride = problem_k / 8;
  int const b_stride = 16 * problem_n / 32;
  int const b_outer = b_stride * Main::KBlocks;
  int const b_inner = b_stride * (Main::Threads / Main::BSharedStride);
  ClassicAddress out;
  out.a = a_stride * (tid / Main::AGlobalOuter) +
          tid % Main::AGlobalOuter + Main::AGlobalOuter * int(k);
  int const b_base = b_stride * (tid / Main::BSharedStride) +
                     tid % Main::BSharedStride +
                     Main::BSharedStride * int(q) + b_outer * int(k);
  for (int i = 0; i < Main::BInnerIters; ++i) {
    out.b[i] = b_inner * i + b_base;
  }
  out.scale = (problem_n / 8) * int(k) +
              Main::ScaleSharedStride * int(q) + tid;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  char const* plant = plant_name(argc, argv);
  typename Main::Params main_params{
      reinterpret_cast<cutlass::half_t const*>(0x10000),
      reinterpret_cast<cutlass::int4b_t const*>(0x20000),
      reinterpret_cast<cutlass::half_t const*>(0x30000), 128};
  auto const sched = Scheduler::make_params_for_tiles(1, 32, 1, 32, 72);
  if (!sched.valid_ || sched.active_blocks_ != 69 ||
      sched.grid_blocks_ != 72 || sched.iters_per_block_ != 15) {
    return fail(plant, "fixed scheduler Params changed");
  }
  // Independent L168 anchor: block 2 visits q=1 first, then rebases to the
  // two final K tiles of q=0.  This is stronger than merely observing q fall.
  auto l168_first = Scheduler::get_work_for_block(sched, 2);
  auto l168_second =
      Scheduler::fetch_next_work_for_params(sched, l168_first);
  if (!(l168_first.output_tile_idx == 1 && l168_first.K_idx == 0 &&
        l168_first.k_tile_count == 13 &&
        l168_second.output_tile_idx == 0 && l168_second.K_idx == 30 &&
        l168_second.k_tile_count == 2)) {
    return fail(plant, "L168 block-2 reverse-segment anchor changed");
  }

  uint32_t init_count = 0;
  uint32_t rebase_count = 0;
  uint32_t idle_init = 0;
  uint32_t reverse_witness = 0;
  for (uint32_t block = 0; block < sched.grid_blocks_; ++block) {
    auto work = Scheduler::get_work_for_block(sched, block);
    if (!work.is_valid()) {
      // This mirrors the production kernel's early return: no init call.
      if (is_plant(plant, "init-before-valid")) {
        auto bad = Main::init_cta_state(main_params, 1, 4096, 4096, 0);
        idle_init += bad.valid;
      }
      continue;
    }

    auto cta = Main::init_cta_state(main_params, 1, 4096, 4096, 0);
    ++init_count;
    if (!cta.valid) return fail(plant, "production CTA init returned invalid");
    uint32_t previous_q = work.output_tile_idx;
    bool first = true;
    typename Main::SegmentState prior{};
    while (work.is_valid()) {
      auto segment = Main::rebase_segment(cta, work);
      ++rebase_count;
      if (!segment.valid) return fail(plant, "production segment rebase invalid");
      if (!first) {
        if (work.output_tile_idx >= previous_q) {
          return fail(plant, "reverse-q L168 anchor changed");
        }
        ++reverse_witness;
      }
      previous_q = work.output_tile_idx;
      first = false;

      auto expected = classic_address(0, work.output_tile_idx, work.K_idx);
      if (is_plant(plant, "init-per-segment")) ++init_count;
      if (is_plant(plant, "stale-rebase") && prior.valid) segment = prior;
      if (is_plant(plant, "drop-b-q")) {
        int const delta = Main::BSharedStride * int(work.output_tile_idx);
        for (int& offset : segment.b_global_read) offset -= delta;
      }
      if (is_plant(plant, "drop-scale-q")) {
        segment.scale_global_read -=
            Main::ScaleSharedStride * int(work.output_tile_idx);
      }
      if (is_plant(plant, "drop-a-k")) {
        segment.a_global_read -= Main::AGlobalOuter * int(work.K_idx);
      }
      if (is_plant(plant, "drop-b-k")) {
        int const delta = cta.b_global_outer * int(work.K_idx);
        for (int& offset : segment.b_global_read) offset -= delta;
      }
      if (is_plant(plant, "drop-scale-k")) {
        segment.scale_global_read -=
            cta.scale_global_stride * int(work.K_idx);
      }

      bool b_matches = true;
      for (int i = 0; i < Main::BInnerIters; ++i) {
        b_matches = b_matches && segment.b_global_read[i] == expected.b[i];
      }
      if (segment.a_global_read != expected.a || !b_matches ||
          segment.scale_global_read != expected.scale) {
        return fail(plant, "production rebase diverged from classic absolute q/K");
      }
      prior = segment;
      work = Scheduler::fetch_next_work_for_params(sched, work);
    }
  }

  // The source XOR swizzle is independently invertible over the full A stage.
  // Tightening its row divisor to 8 aliases the classic 16-wide row map.
  std::array<uint8_t, Main::ASharedStage> seen{};
  for (int index = 0; index < Main::ASharedStage; ++index) {
    int mapped = Main::transform_a_index(index);
    if (is_plant(plant, "tight-a-swizzle")) {
      int const row = index / 8;
      mapped = 8 * row + ((index % 8) ^ row);
    }
    if (mapped < 0 || mapped >= Main::ASharedStage || ++seen[mapped] != 1) {
      return fail(plant, "A XOR swizzle ceased to be a stage bijection");
    }
  }

  if (idle_init != 0 || init_count != 69 || rebase_count != 98 ||
      reverse_witness != 29) {
    return fail(plant, "CTA-state fixed-launch census changed");
  }
  if (!is_plant(plant, "none")) {
    return fail(plant, "named plant did not perturb its intended invariant");
  }
  std::printf(
      "[l173] PASS: init=%u rebase=%u idle-init=%u reverse=%u "
      "absolute-addresses=classic-exact A-swizzle=bijective\n",
      init_count, rebase_count, idle_init, reverse_witness);
  return 0;
}
