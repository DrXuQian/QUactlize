// L133 -- exhaustive composition proof for the production Marlin stripe core.
//
// The manifest is generated from the three committed dense tactic tables and
// benchmarks/fixtures.py.  This program calls the exact CUTLASS_HOST_DEVICE
// core used by the kernel; it does not restate the dispatcher.  Raw shapes are
// first mapped to tile counts, then every distinct production Params tuple is
// exhausted.  Q>=CU is still checked q-by-q (not sampled); split cases are
// checked cell-by-cell in flattened (q,k-tile) space.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp"

using Core = cutlass::gemm::kernel::detail::MarlinStripeSchedulerCore;
using Params = Core::Params;
using Work = Core::WorkTileInfo;

struct Key {
  uint64_t mt, nt, l, kt, cu;
  bool operator<(Key const& x) const {
    return std::tie(mt, nt, l, kt, cu) <
           std::tie(x.mt, x.nt, x.l, x.kt, x.cu);
  }
};

struct Multiplicity {
  uint64_t deployment = 0;
  uint64_t cross_l = 0;
};

struct Totals {
  uint64_t raw_deployment = 0;
  uint64_t raw_cross_l = 0;
  uint64_t raw_protected = 0;
  uint64_t raw_stripe_regime = 0;
  uint64_t raw_actual_split = 0;
  uint64_t raw_ceil_unsplit = 0;
  uint64_t unique_protected = 0;
  uint64_t unique_stripe_regime = 0;
  uint64_t unique_actual_split = 0;
  uint64_t unique_ceil_unsplit = 0;
  uint64_t unique_checked = 0;
  uint64_t segments = 0;
  uint64_t logical_cells = 0;
  uint64_t output_checks = 0;
  uint64_t handoffs = 0;
  uint64_t cross_n = 0;
  uint64_t cross_m = 0;
  uint64_t cross_l = 0;
  uint64_t max_kt = 0;
  uint64_t errors = 0;
};

static uint64_t ceil_div(uint64_t x, uint64_t y) {
  return x / y + uint64_t(x % y != 0);
}

static bool q_lt_cu_ceil_unsplit(uint64_t q, uint64_t kt, uint64_t cu) {
  // (CU-Q)*Kt < CU, written without multiplication overflow. All scheduler
  // inputs are positive; equality belongs to the split side.
  return q < cu && kt != 0 && (cu - q) <= (cu - 1) / kt;
}

static int cell_value(uint64_t q, uint64_t k) {
  return int((q + k) % 3) - 1;
}

// The exact fixture is periodic, so a whole scheduler segment is checked in
// O(1). Every individual contribution is an integer in {-1,0,1}; even an
// adversarial reassociation of <=400 terms is exact in FP32, and the final
// result (absolute value <=1) is exact in FP16.
static int range_sum(uint64_t q, uint64_t k, uint64_t count) {
  int out = 0;
  for (uint64_t i = 0; i < count % 3; ++i) {
    out += cell_value(q, k + i);
  }
  return out;
}

static void fail(Totals& t, char const* what, Key const& x, uint64_t a = 0,
                 uint64_t b = 0) {
  if (t.errors < 8) {
    std::fprintf(stderr,
        "[l133] mismatch=%s tuple=(Mt=%llu,Nt=%llu,L=%llu,Kt=%llu,CU=%llu) got=%llu want=%llu\n",
        what, (unsigned long long)x.mt, (unsigned long long)x.nt,
        (unsigned long long)x.l, (unsigned long long)x.kt,
        (unsigned long long)x.cu, (unsigned long long)a,
        (unsigned long long)b);
  }
  ++t.errors;
}

static bool decode_ok(Work const& w, Key const& x) {
  uint64_t q = w.output_tile_idx;
  uint64_t n = q % x.nt;
  uint64_t qm = q / x.nt;
  uint64_t m = qm % x.mt;
  uint64_t l = qm / x.mt;
  return uint64_t(w.N_idx) == n && uint64_t(w.M_idx) == m &&
         uint64_t(w.L_idx) == l && ((l * x.mt + m) * x.nt + n) == q;
}

struct Peer {
  uint64_t block;
  uint32_t count;
  uint32_t index;
};

static void check_no_split(Key const& x, Params const& p, Totals& t) {
  uint64_t const q_count = x.mt * x.nt * x.l;
  if (p.grid_blocks_ != q_count || p.active_blocks_ != q_count ||
      p.iters_per_block_ != x.kt) {
    fail(t, "default-B1-G=max(Q,CU)-no-split", x, p.grid_blocks_, q_count);
    return;
  }
  for (uint64_t q = 0; q < q_count; ++q) {
    Work w = Core::get_work_for_block(p, q);
    bool ok = w.is_valid() && w.output_tile_idx == q && w.lock_idx == q &&
              w.K_idx == 0 && w.k_tile_count == x.kt &&
              w.slice_count == 1 && w.slice_idx == 0 &&
              w.linear_begin == q * x.kt &&
              w.linear_next == (q + 1) * x.kt &&
              w.linear_end == (q + 1) * x.kt && decode_ok(w, x);
    if (!ok) {
      fail(t, "no-split-work", x, q, w.output_tile_idx);
      continue;
    }
    if (Core::fetch_next_work(p, w).is_valid()) {
      fail(t, "no-split-fetch", x, q, 0);
    }
    int const scheduled = range_sum(w.output_tile_idx, w.K_idx,
                                    w.k_tile_count);
    int const dp = range_sum(q, 0, x.kt);
    if (scheduled != dp) fail(t, "exact-fixture-no-split", x, scheduled, dp);
    ++t.segments;
    t.logical_cells += x.kt;
    ++t.output_checks;
  }
}

static void check_stripe_regime(Key const& x, Params const& p,
                                bool require_cross_l, Totals& t) {
  uint64_t const q_count = x.mt * x.nt * x.l;
  uint64_t const cells = q_count * x.kt;
  std::vector<uint8_t> visits(std::size_t(cells), uint8_t(0));
  std::vector<int> sums(std::size_t(q_count), 0);
  std::vector<std::vector<Peer>> peers;
  peers.resize(std::size_t(q_count));
  uint64_t local_cross_n = 0, local_cross_m = 0, local_cross_l = 0;

  for (uint64_t block = 0; block < p.grid_blocks_; ++block) {
    uint64_t const stripe_begin = block * p.iters_per_block_;
    uint64_t const stripe_end = std::min(stripe_begin + p.iters_per_block_,
                                         p.total_k_tiles_);
    Work w = Core::get_work_for_block(p, block);
    if (stripe_begin >= p.total_k_tiles_) {
      if (w.is_valid()) fail(t, "idle-CTA-work", x, block, p.active_blocks_);
      continue;
    }
    uint64_t cursor = stripe_begin;
    uint64_t previous_q = 0;
    bool have_previous = false;
    while (w.is_valid()) {
      uint64_t const q = cursor / x.kt;
      uint64_t const k = cursor % x.kt;
      uint64_t const want_next = std::min(stripe_end, (q + 1) * x.kt);
      bool const shape_ok = w.block_idx == block &&
          w.linear_begin == cursor && w.linear_end == stripe_end &&
          w.linear_next == want_next && w.output_tile_idx == q &&
          w.lock_idx == q && w.K_idx == k &&
          uint64_t(w.k_tile_count) == want_next - cursor &&
          decode_ok(w, x);
      if (!shape_ok || q >= q_count || k + w.k_tile_count > x.kt) {
        fail(t, "split-segment", x, cursor, want_next);
        break;
      }
      if (have_previous && q != previous_q) {
        uint64_t pn = previous_q % x.nt;
        uint64_t pqm = previous_q / x.nt;
        uint64_t pm = pqm % x.mt;
        uint64_t pl = pqm / x.mt;
        uint64_t n = q % x.nt;
        uint64_t qm = q / x.nt;
        uint64_t m = qm % x.mt;
        uint64_t l = qm / x.mt;
        local_cross_n += l == pl && m == pm && n != pn;
        local_cross_m += l == pl && m != pm;
        local_cross_l += l != pl;
      }
      have_previous = true;
      previous_q = q;
      peers[std::size_t(q)].push_back({block, w.slice_count, w.slice_idx});
      sums[std::size_t(q)] += range_sum(q, k, w.k_tile_count);
      for (uint64_t kk = k; kk < k + w.k_tile_count; ++kk) {
        ++visits[std::size_t(q * x.kt + kk)];
      }
      ++t.segments;
      t.logical_cells += w.k_tile_count;
      cursor = w.linear_next;
      w = Core::fetch_next_work(p, w);
    }
    if (cursor != stripe_end) fail(t, "stripe-not-consumed", x, cursor, stripe_end);
  }

  for (uint64_t cell = 0; cell < cells; ++cell) {
    if (visits[std::size_t(cell)] != 1) {
      fail(t, "cell-exact-once", x, visits[std::size_t(cell)], 1);
    }
  }
  std::set<uint64_t> locks;
  for (uint64_t q = 0; q < q_count; ++q) {
    auto& ps = peers[std::size_t(q)];
    if (ps.empty()) {
      fail(t, "output-no-peer", x, q, 1);
      continue;
    }
    for (std::size_t i = 0; i < ps.size(); ++i) {
      uint32_t const reverse = uint32_t(ps.size() - 1 - i);
      if (ps[i].count != ps.size() || ps[i].index != reverse) {
        fail(t, "reverse-peer-protocol", x, ps[i].index, reverse);
      }
    }
    t.handoffs += ps.size() - 1;
    if (!locks.insert(q).second) fail(t, "global-q-lock", x, q, q);
    int const dp = range_sum(q, 0, x.kt);
    if (sums[std::size_t(q)] != dp) {
      fail(t, "exact-fixture-split", x, sums[std::size_t(q)], dp);
    }
    ++t.output_checks;
  }
  t.cross_n += local_cross_n;
  t.cross_m += local_cross_m;
  t.cross_l += local_cross_l;
  if (require_cross_l &&
      (local_cross_n == 0 || local_cross_m == 0 || local_cross_l == 0)) {
    fail(t, "cross-N/M/L-witness", x,
         (local_cross_n != 0) + 2 * (local_cross_m != 0) +
             4 * (local_cross_l != 0), 7);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s manifest.tsv expected-row-count\n", argv[0]);
    return 2;
  }
  unsigned long long expected_raw_ull = 0;
  if (std::sscanf(argv[2], "%llu", &expected_raw_ull) != 1) {
    std::fprintf(stderr, "[l133] invalid expected-row-count %s\n", argv[2]);
    return 2;
  }
  uint64_t const expected_raw = uint64_t(expected_raw_ull);
  std::ifstream in(argv[1]);
  if (!in) {
    std::fprintf(stderr, "[l133] cannot open manifest %s\n", argv[1]);
    return 2;
  }

  Totals t;
  std::map<Key, Multiplicity> unique;
  char kind = 0;
  uint64_t M, N, K, L, TM, TN, TK, CU;
  while (in >> kind >> M >> N >> K >> L >> TM >> TN >> TK >> CU) {
    if (kind != 'D' && kind != 'X') {
      std::fprintf(stderr, "[l133] invalid manifest kind %c\n", kind);
      return 2;
    }
    uint64_t const mt = ceil_div(M, TM);
    uint64_t const nt = ceil_div(N, TN);
    uint64_t const kt = ceil_div(K, TK);
    Key x{mt, nt, L, kt, CU};
    Params p = Core::make_params_for_tiles(mt, nt, L, kt, CU);
    uint64_t const q = mt * nt * L;
    uint64_t const cells = q * kt;
    uint64_t const G = std::max(q, CU);
    uint64_t const I = ceil_div(cells, G);
    uint64_t const active = ceil_div(cells, I);
    bool const params_ok = p.valid_ && p.tiles_m_ == mt && p.tiles_n_ == nt &&
        p.tiles_l_ == L && p.k_tiles_per_output_ == kt &&
        p.output_tiles_ == q && p.total_k_tiles_ == cells &&
        p.grid_blocks_ == G && p.iters_per_block_ == I &&
        p.active_blocks_ == active;
    if (!params_ok) fail(t, "raw-to-production-params", x, p.grid_blocks_, G);
    if (q >= CU) {
      ++t.raw_protected;
    }
    else {
      ++t.raw_stripe_regime;
      bool const ceil_unsplit = I == kt;
      // For Q<CU, ceil(Q*Kt/CU)==Kt iff the final missing-CTA gap is
      // smaller than one K-tile quantum: (CU-Q)*Kt < CU.  This is strict;
      // equality is already the split side. Use division to avoid overflow.
      bool const inequality = q_lt_cu_ceil_unsplit(q, kt, CU);
      if (ceil_unsplit != inequality)
        fail(t, "q<CU-ceil-unsplit-iff", x, ceil_unsplit, inequality);
      if (ceil_unsplit) {
        ++t.raw_ceil_unsplit;
        if (active != q) fail(t, "q<CU-ceil-unsplit-active", x, active, q);
      }
      else ++t.raw_actual_split;
    }
    t.max_kt = std::max(t.max_kt, kt);
    Multiplicity& mult = unique[x];
    if (kind == 'D') { ++mult.deployment; ++t.raw_deployment; }
    else { ++mult.cross_l; ++t.raw_cross_l; }
  }
  if (!in.eof()) {
    std::fprintf(stderr, "[l133] malformed manifest before EOF\n");
    ++t.errors;
  }

  // Fail before a planted Params bug can turn the deeper walk into OOB noise.
  if (t.errors == 0) {
    for (auto const& item : unique) {
      Key const& x = item.first;
      Params p = Core::make_params_for_tiles(x.mt, x.nt, x.l, x.kt, x.cu);
      uint64_t const q = x.mt * x.nt * x.l;
      if (q >= x.cu) {
        ++t.unique_protected;
        check_no_split(x, p, t);
      }
      else {
        ++t.unique_stripe_regime;
        if (p.iters_per_block_ < x.kt) ++t.unique_actual_split;
        else ++t.unique_ceil_unsplit;
        check_stripe_regime(x, p, item.second.cross_l != 0, t);
      }
      ++t.unique_checked;
      if (t.errors != 0) break;  // planted controls fail at their first causal tuple
    }
  }

  uint64_t const scanned = t.raw_deployment + t.raw_cross_l;
  uint64_t const raw_remaining = expected_raw >= scanned ? expected_raw - scanned : 0;
  uint64_t const unique_remaining = unique.size() >= t.unique_checked
      ? uint64_t(unique.size()) - t.unique_checked : 0;
  if (scanned != expected_raw) ++t.errors;
  std::printf("[l133] raw deployment=%llu cross-L=%llu scanned=%llu remaining=%llu\n",
      (unsigned long long)t.raw_deployment,
      (unsigned long long)t.raw_cross_l,
      (unsigned long long)scanned, (unsigned long long)raw_remaining);
  std::printf("[l133] equivalence unique=%llu checked=%llu remaining=%llu "
              "protected=%llu stripe-regime=%llu actual-split=%llu q<CU-ceil-unsplit=%llu "
              "raw-protected/stripe/actual/q<CU-ceil-unsplit=%llu/%llu/%llu/%llu\n",
      (unsigned long long)unique.size(),
      (unsigned long long)t.unique_checked,
      (unsigned long long)unique_remaining,
      (unsigned long long)t.unique_protected,
      (unsigned long long)t.unique_stripe_regime,
      (unsigned long long)t.unique_actual_split,
      (unsigned long long)t.unique_ceil_unsplit,
      (unsigned long long)t.raw_protected,
      (unsigned long long)t.raw_stripe_regime,
      (unsigned long long)t.raw_actual_split,
      (unsigned long long)t.raw_ceil_unsplit);
  uint64_t printed_ceil_unsplit = 0;
  for (auto const& item : unique) {
    Key const& x = item.first;
    uint64_t const q = x.mt * x.nt * x.l;
    Params const p = Core::make_params_for_tiles(x.mt, x.nt, x.l, x.kt, x.cu);
    if (q >= x.cu || p.iters_per_block_ != x.kt) continue;
    uint64_t const raw = item.second.deployment + item.second.cross_l;
    std::printf("[l133] q<CU-ceil-unsplit class Mt=%llu Nt=%llu L=%llu Kt=%llu CU=%llu "
                "Q=%llu G=%llu I=%llu active=%llu raw=%llu\n",
        (unsigned long long)x.mt, (unsigned long long)x.nt,
        (unsigned long long)x.l, (unsigned long long)x.kt,
        (unsigned long long)x.cu, (unsigned long long)q,
        (unsigned long long)p.grid_blocks_,
        (unsigned long long)p.iters_per_block_,
        (unsigned long long)p.active_blocks_, (unsigned long long)raw);
    ++printed_ceil_unsplit;
  }
  if (printed_ceil_unsplit != t.unique_ceil_unsplit)
    fail(t, "q<CU-ceil-unsplit-census", Key{}, printed_ceil_unsplit,
         t.unique_ceil_unsplit);
  bool const q64 = q_lt_cu_ceil_unsplit(64, 8, 72);
  bool const q63 = q_lt_cu_ceil_unsplit(63, 8, 72);
  if (!q64 || q63) fail(t, "q<CU-ceil-unsplit-strict-boundary", Key{}, q64, q63);
  std::printf("[l133] q<CU-ceil-unsplit iff (CU-Q)*Kt<CU; "
              "boundary Q64/Kt8/CU72=%s Q63/Kt8/CU72=%s %s\n",
      q64 ? "unsplit" : "split", q63 ? "unsplit" : "split",
      q64 && !q63 ? "PASS" : "FAIL");
  std::printf("[l133] production segments=%llu logical-(q,k)-cells=%llu outputs=%llu handoffs=%llu cross(N/M/L)=%llu/%llu/%llu\n",
      (unsigned long long)t.segments,
      (unsigned long long)t.logical_cells,
      (unsigned long long)t.output_checks,
      (unsigned long long)t.handoffs,
      (unsigned long long)t.cross_n,
      (unsigned long long)t.cross_m,
      (unsigned long long)t.cross_l);
  std::printf("[l133] fixture=marlin-cell-exact contribution={-1,0,1} max-terms=%llu < 2048 "
              "ORDER-INDEPENDENT+FP16-EXACT criterion=raw-integer-equality-before-run\n",
      (unsigned long long)t.max_kt);
  std::printf("[l133] proposition-A exact-once=>DP result=%s "
              "raw-remaining=%llu unique-remaining=%llu failures=%llu\n",
      t.errors == 0 ? "PASS" : "FAIL",
      (unsigned long long)raw_remaining,
      (unsigned long long)unique_remaining,
      (unsigned long long)t.errors);
  return t.errors == 0 ? 0 : 1;
}
