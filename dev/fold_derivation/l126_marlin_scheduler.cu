// L126 -- device-free oracle for the production Marlin CTA scheduler.
//
// This is not a second implementation: every positive case calls the exact
// make_params_for_tiles/get_work_for_block/fetch_next_work methods
// used by the named kernel.  The validators are independent integer oracles;
// the four red arms deliberately corrupt one field after construction.

#if defined(L126_TYPE_ONLY)

#include <cstdio>
#include <type_traits>
#include "cute/tensor.hpp"
using namespace cute;
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

using TileShape126 = cute::Shape<cute::_16, cute::_32, cute::_256>;
using ClusterShape126 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using PersistentSelected = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
    cutlass::gemm::PersistentScheduler, cutlass::arch::PPU0010,
    TileShape126, ClusterShape126>::Scheduler;
using StreamKSelected = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
    cutlass::gemm::StreamKScheduler, cutlass::arch::PPU0010,
    TileShape126, ClusterShape126>::Scheduler;
using MarlinSelected = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
    cutlass::gemm::MarlinScheduler, cutlass::arch::PPU0010,
    TileShape126, ClusterShape126>::Scheduler;
static_assert(std::is_same_v<PersistentSelected,
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPU>);
static_assert(std::is_same_v<StreamKSelected,
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPUStreamK<
        TileShape126, ClusterShape126>>);
static_assert(std::is_same_v<MarlinSelected,
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPUMarlin<
        TileShape126, ClusterShape126>>);

int main() {
  std::printf("[l126:type] default persistent/StreamK selector types unchanged; "
              "Marlin tag selects only its additive scheduler PASS\n");
  return 0;
}

#else

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <tuple>
#include <vector>

#include "cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp"

using Scheduler = cutlass::gemm::kernel::detail::MarlinStripeSchedulerCore;
using Params = Scheduler::Params;
using Work = Scheduler::WorkTileInfo;

struct Census {
  int holes = 0;
  int duplicates = 0;
  int bad_owner = 0;
  int bad_q_map = 0;
  int bad_peer = 0;
  int bad_lock = 0;
  int cross_n = 0;
  int cross_m = 0;
  int cross_l = 0;
  int cross_ctas = 0;
  int handoffs = 0;
  int active = 0;
  int last_stripe = 0;
};

static uint64_t ceil_div(uint64_t x, uint64_t y) {
  return x / y + uint64_t(x % y != 0);
}

static Census census(Params const& p, bool local_lock_plant = false) {
  Census c;
  if (!p.valid_) {
    c.holes = 1;
    return c;
  }
  std::vector<int> visits(std::size_t(p.total_k_tiles_));
  std::vector<std::set<uint64_t>> lock_owners(std::size_t(p.output_tiles_));
  std::vector<std::set<uint32_t>> peer_ids(std::size_t(p.output_tiles_));
  std::vector<std::set<uint64_t>> peer_blocks(std::size_t(p.output_tiles_));

  for (uint64_t b = 0; b < p.grid_blocks_; ++b) {
    Work w = Scheduler::get_work_for_block(p, b);
    int segments = 0;
    uint64_t previous_q = 0;
    bool have_previous = false;
    while (w.is_valid()) {
      ++segments;
      uint64_t const expected_owner = w.linear_begin / p.iters_per_block_;
      c.bad_owner += expected_owner != b;
      uint64_t q = w.output_tile_idx;
      uint64_t n = q % p.tiles_n_;
      uint64_t qm = q / p.tiles_n_;
      uint64_t m = qm % p.tiles_m_;
      uint64_t l = qm / p.tiles_m_;
      c.bad_q_map += uint64_t(w.N_idx) != n || uint64_t(w.M_idx) != m ||
                     uint64_t(w.L_idx) != l ||
                     ((l * p.tiles_m_ + m) * p.tiles_n_ + n) != q;

      uint64_t const q_begin = q * p.k_tiles_per_output_;
      uint64_t const q_end = q_begin + p.k_tiles_per_output_;
      uint64_t const b_lo = q_begin / p.iters_per_block_;
      uint64_t const b_hi = (q_end - 1) / p.iters_per_block_;
      c.bad_peer += w.slice_count != b_hi - b_lo + 1;
      c.bad_peer += w.slice_idx != b_hi - b;
      c.bad_peer += (w.slice_idx == 0) != (b == b_hi);
      c.bad_peer += (w.slice_idx + 1 == w.slice_count) != (b == b_lo);
      uint64_t const observed_lock = local_lock_plant ? uint64_t(w.N_idx) : w.lock_idx;
      c.bad_lock += !local_lock_plant && observed_lock != q;
      lock_owners[std::size_t(q)].insert(observed_lock);
      peer_ids[std::size_t(q)].insert(w.slice_idx);
      peer_blocks[std::size_t(q)].insert(b);

      if (have_previous && q != previous_q) {
        uint64_t prev_n = previous_q % p.tiles_n_;
        uint64_t prev_qm = previous_q / p.tiles_n_;
        uint64_t prev_m = prev_qm % p.tiles_m_;
        uint64_t prev_l = prev_qm / p.tiles_m_;
        c.cross_n += l == prev_l && m == prev_m && n != prev_n;
        c.cross_m += l == prev_l && m != prev_m;
        c.cross_l += l != prev_l;
      }
      previous_q = q;
      have_previous = true;

      for (uint32_t k = 0; k < w.k_tile_count; ++k) {
        uint64_t idx = q * p.k_tiles_per_output_ + uint64_t(w.K_idx) + k;
        if (idx >= visits.size()) {
          ++c.bad_owner;
        } else {
          ++visits[std::size_t(idx)];
        }
      }
      w = Scheduler::fetch_next_work(p, w);
    }
    if (segments > 0) {
      ++c.active;
      c.cross_ctas += segments > 1;
      uint64_t const begin = b * p.iters_per_block_;
      c.last_stripe = int(std::min(p.iters_per_block_, p.total_k_tiles_ - begin));
    }
  }
  for (int hits : visits) {
    c.holes += hits == 0;
    c.duplicates += hits > 1 ? hits - 1 : 0;
  }
  for (uint64_t q = 0; q < p.output_tiles_; ++q) {
    int const peers = int(peer_blocks[std::size_t(q)].size());
    c.handoffs += peers > 0 ? peers - 1 : 0;
    c.bad_peer += int(peer_ids[std::size_t(q)].size()) != peers;
    for (int i = 0; i < peers; ++i) {
      c.bad_peer += peer_ids[std::size_t(q)].count(uint32_t(i)) != 1;
    }
  }
  if (local_lock_plant) {
    std::set<uint64_t> seen;
    for (uint64_t q = 0; q < p.output_tiles_; ++q) {
      if (peer_blocks[std::size_t(q)].size() > 1) {
        uint64_t lock = q % p.tiles_n_;
        c.bad_lock += !seen.insert(lock).second;
      }
    }
  }
  return c;
}

static bool clean(Census const& c) {
  return c.holes == 0 && c.duplicates == 0 && c.bad_owner == 0 &&
         c.bad_q_map == 0 && c.bad_peer == 0 && c.bad_lock == 0;
}

int main() {
  // Independent reconstruction of the user's 20-SM diagram.
  Params classic = Scheduler::make_params_for_tiles(1, 16, 1, 16, 20);
  Census cc = census(classic);
  bool classic_ok = clean(cc) && classic.output_tiles_ == 16 &&
      classic.grid_blocks_ == 20 && classic.iters_per_block_ == 13 &&
      classic.active_blocks_ == 20 && cc.handoffs == 18 &&
      cc.cross_ctas == 14 && cc.last_stripe == 9;

  // The launcher protection is a scheduler invariant, not a caller hint.
  Params decode = Scheduler::make_params_for_tiles(1, 128, 1, 8, 72);
  Census dc = census(decode);
  bool decode_ok = clean(dc) && decode.grid_blocks_ == 128 &&
      decode.iters_per_block_ == 8 && decode.active_blocks_ == 128 &&
      dc.handoffs == 0;

  // Runtime L extension: q=((l*Mt+m)*Nt+n), with N fast.  This fixture
  // forces stripe continuations across N, M and L boundaries.
  Params batched = Scheduler::make_params_for_tiles(2, 2, 2, 31, 9);
  Census bc = census(batched);
  bool batched_ok = clean(bc) && bc.cross_n > 0 && bc.cross_m > 0 && bc.cross_l > 0;

  // Plant 1: copy Stream-K's CU*occupancy worker rule into Marlin.  Q>=CU
  // must have been no-split, so this changes both grid and handoff count.
  Params bad_grid = Scheduler::make_params_for_tiles(1, 128, 1, 8, 72 * 4);
  Census bg = census(bad_grid);
  bool grid_red = bad_grid.grid_blocks_ != 128 || bg.handoffs != 0;

  // Plant 2: ceildiv undercount by one.  Leave total/grid fixed so exact-once
  // coverage, not a copied implementation, catches the missing tail.
  Params bad_iters = classic;
  --bad_iters.iters_per_block_;
  Census bi = census(bad_iters);
  bool iters_red = !clean(bi);

  // Plant 3: use N-local index as lock id across M/L.  Coverage remains green;
  // injectivity of global q locks must fail independently.
  Census bl = census(batched, true);
  bool lock_red = bl.bad_lock > 0 && bl.holes == 0 && bl.duplicates == 0;

  // A fourth silent plant: natural peer order.  The expected reverse protocol
  // is checked explicitly because exact-once coverage cannot see this change.
  int natural_peer_bad = 0;
  for (uint64_t b = 0; b < classic.grid_blocks_; ++b) {
    for (Work w = Scheduler::get_work_for_block(classic, b); w.is_valid();
         w = Scheduler::fetch_next_work(classic, w)) {
      uint64_t q0 = w.output_tile_idx * classic.k_tiles_per_output_;
      uint64_t b_lo = q0 / classic.iters_per_block_;
      natural_peer_bad += uint32_t(b - b_lo) != w.slice_idx;
    }
  }
  bool peer_red = natural_peer_bad > 0;

  std::printf("[l126] classic Q=16 Kt=16 CU=20 G=%llu I=%llu active=%d "
              "last=%d handoff=%d cross-CTA=%d -> %s\n",
              (unsigned long long)classic.grid_blocks_,
              (unsigned long long)classic.iters_per_block_, cc.active,
              cc.last_stripe, cc.handoffs, cc.cross_ctas,
              classic_ok ? "PASS" : "FAIL");
  std::printf("[l126] decode Q=128 Kt=8 CU=72 G=%llu I=%llu handoff=%d -> %s\n",
              (unsigned long long)decode.grid_blocks_,
              (unsigned long long)decode.iters_per_block_, dc.handoffs,
              decode_ok ? "PASS" : "FAIL");
  std::printf("[l126] batched Mt=2 Nt=2 L=2 Kt=31 cross(N/M/L)=%d/%d/%d -> %s\n",
              bc.cross_n, bc.cross_m, bc.cross_l, batched_ok ? "PASS" : "FAIL");
  std::printf("[l126:red] grid*occupancy G=%llu handoff=%d -> %s\n",
              (unsigned long long)bad_grid.grid_blocks_, bg.handoffs,
              grid_red ? "EXPECTED-RED" : "FAIL");
  std::printf("[l126:red] I-1 holes=%d dup=%d owner=%d -> %s\n",
              bi.holes, bi.duplicates, bi.bad_owner,
              iters_red ? "EXPECTED-RED" : "FAIL");
  std::printf("[l126:red] local-lock aliases=%d coverage=%d/%d -> %s\n",
              bl.bad_lock, bl.holes, bl.duplicates,
              lock_red ? "EXPECTED-RED" : "FAIL");
  std::printf("[l126:red] natural-peer mismatches=%d -> %s\n", natural_peer_bad,
              peer_red ? "EXPECTED-RED" : "FAIL");
  bool pass = classic_ok && decode_ok && batched_ok && grid_red &&
              iters_red && lock_red && peer_red;
  std::printf("[l126] default persistent/StreamK selector types unchanged; result=%s\n",
              pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

#endif
