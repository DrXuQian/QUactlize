// Device-free checks of the exact scheduler instantiated by the SM90 kernel.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "quactlize/backends/ppu17/marlin_scheduler.cuh"

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "scheduler check failed line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
using Tile = cute::Shape<cute::_128, cute::_16, cute::_64>;
using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
using Scheduler = quactlize::ppu17::CooperativeMarlinScheduler<Tile, Cluster>;

int main() {
  unsigned cases = 0;
  bool saw_split = false, saw_cross_tile = false, saw_unequal_tail = false;
  for (unsigned m : {1, 2, 7, 115}) for (unsigned n : {1, 3})
  for (unsigned l : {1, 2}) for (unsigned k : {2, 4, 16, 64})
  for (unsigned sm : {1, 2, 7, 114}) {
    auto p = Scheduler::make_params(m, n, l, k, sm);
    CHECK(p.valid_);
    std::vector<unsigned> coverage(size_t(m) * n * l * k, 0);
    std::vector<std::vector<Scheduler::WorkTileInfo>> peers(size_t(m) * n * l);
    for (unsigned block = 0; block < p.grid_blocks_; ++block) {
      unsigned segments = 0;
      for (auto w = Scheduler::work_for_block(p, block); w.is_valid(); w = Scheduler::next_work(p, w)) {
        CHECK(++segments <= 2); // grid >= output tiles bounds each stripe to <= one tile length
        CHECK(w.K_idx % 2 == 0 && w.k_tile_count % 2 == 0 && w.K_idx + w.k_tile_count <= k);
        CHECK(w.M_idx >= 0 && unsigned(w.M_idx) < m && w.N_idx >= 0 && unsigned(w.N_idx) < n);
        CHECK(w.L_idx >= 0 && unsigned(w.L_idx) < l);
        auto q = (size_t(w.L_idx) * m + w.M_idx) * n + w.N_idx;
        CHECK(w.output_tile_idx == q);
        CHECK(Scheduler::lock_index(w, 0) == 2*q && Scheduler::lock_index(w, 1) == 2*q + 1);
        CHECK(w.block_idx < sm || w.slice_count == 1); // no cross-wave spin dependency
        for (unsigned kk = w.K_idx; kk < w.K_idx + w.k_tile_count; ++kk) ++coverage[q*k+kk];
        peers[q].push_back(w);
        saw_split |= w.slice_count > 1;
        saw_unequal_tail |= w.linear_end - w.linear_begin < p.iters_per_block_;
      }
      saw_cross_tile |= segments == 2;
    }
    for (unsigned count : coverage) CHECK(count == 1);
    for (auto& row : peers) {
      CHECK(!row.empty());
      std::sort(row.begin(), row.end(), [](auto a, auto b) { return a.slice_idx < b.slice_idx; });
      unsigned last_start = k, epilogues = 0;
      // Simulate the highest-K-initializes, lower-K-adds protocol, independently
      // checking ordered segments and exact once-only epilogue ownership.
      for (unsigned i = 0; i < row.size(); ++i) {
        auto w = row[i];
        CHECK(w.slice_idx == i && w.slice_count == row.size());
        CHECK(w.K_idx + w.k_tile_count == last_start);
        last_start = w.K_idx;
        epilogues += Scheduler::compute_epilogue(w, p);
      }
      CHECK(last_start == 0 && epilogues == 1);
    }
    ++cases;
  }
  CHECK(saw_split && saw_cross_tile && saw_unequal_tail);
  CHECK(!Scheduler::make_params(0,1,1,2,114).valid_);
  CHECK(!Scheduler::make_params(1,1,1,3,114).valid_);
  CHECK(!Scheduler::make_params(1,1,1,2,0).valid_);
  CHECK(!Scheduler::make_params(1,1,1,uint64_t(1)<<33,114).valid_);
  CHECK(!Scheduler::make_params(uint64_t(1)<<30,2,1,2,114).valid_);
  CHECK(!Scheduler::make_params(uint64_t(1)<<63,2,2,2,114).valid_);
  auto p = Scheduler::make_params(1,1,1,16,114);
  CHECK(!Scheduler::work_for_block(p, p.grid_blocks_).is_valid());
  std::printf("SM90_MARLIN_HOST PASS cases=%u coverage=EXACT peers=REVERSE groups=2 alignment=128 negatives=7\n", cases);
}
