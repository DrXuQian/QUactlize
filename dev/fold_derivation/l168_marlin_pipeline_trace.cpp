// L168 -- device-free trace oracle for the fixed classic/Awesome-Cute Marlin
// decode pipeline.
//
// The scope is intentionally one exact launch:
//   M=1, N=K=4096, L=1, gs=128
//   CTA=16x128x128, warp=16x64x32, stages=4, CU=72
//
// Two independent walkers model the source-level scheduler order: classic
// visits the output-tile intersections in ascending q order, while
// Awesome-Cute starts from end_tile_idx and visits them in descending q order.
// They must nevertheless cover the same flattened (q,k-tile) cells and emit
// the same event ledger.  This is a host trace, not an instruction estimate for
// any other tactic.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using u64 = std::uint64_t;

constexpr u64 kM = 1;
constexpr u64 kN = 4096;
constexpr u64 kK = 4096;
constexpr u64 kL = 1;
constexpr u64 kGroupSize = 128;
constexpr u64 kTileM = 16;
constexpr u64 kTileN = 128;
constexpr u64 kTileK = 128;
constexpr u64 kWarpM = 16;
constexpr u64 kWarpN = 64;
constexpr u64 kWarpK = 32;
constexpr u64 kMmaN = 16;
constexpr u64 kMmaK = 16;
constexpr u64 kStages = 4;
constexpr u64 kCu = 72;

constexpr u64 ceil_div(u64 x, u64 y) { return x / y + u64(x % y != 0); }
constexpr u64 round_up(u64 x, u64 y) { return ceil_div(x, y) * y; }

constexpr u64 kTilesM = ceil_div(kM, kTileM);
constexpr u64 kTilesN = ceil_div(kN, kTileN);
constexpr u64 kTilesK = ceil_div(kK, kTileK);
constexpr u64 kOutputTiles = kTilesM * kTilesN * kL;
constexpr u64 kCells = kOutputTiles * kTilesK;
constexpr u64 kWarpGridM = kTileM / kWarpM;
constexpr u64 kWarpGridN = kTileN / kWarpN;
constexpr u64 kWarpGridK = kTileK / kWarpK;
constexpr u64 kWarps = kWarpGridM * kWarpGridN * kWarpGridK;
constexpr u64 kInner = kWarpK / kMmaK;
constexpr u64 kMmasPerWarpInner = kWarpN / kMmaN;
constexpr u64 kMmasPerCell = kWarps * kInner * kMmasPerWarpInner;

static_assert(kTilesM == 1 && kTilesN == 32 && kTilesK == 32);
static_assert(kOutputTiles == 32 && kCells == 1024);
static_assert(kGroupSize / kTileK == 1,
              "the fixed gs128 launch must not change stripe rounding");
static_assert(kWarpGridM == 1 && kWarpGridN == 2 && kWarpGridK == 4);
static_assert(kWarps == 8 && kInner == 2 && kMmasPerWarpInner == 4);
static_assert(kMmasPerCell == 64 && kCells * kMmasPerCell == 65536);

enum class Flavor { Classic, Awesome };
enum class Plant { None, OccupancyGrid, MissingStageAttempt, FlatReduction };
enum class Event { Attempt, WaitBarrier, S2R, Clear, Mma, Reduction };

char const* flavor_name(Flavor flavor) {
  return flavor == Flavor::Classic ? "classic" : "awesome-cute";
}

char const* plant_name(Plant plant) {
  switch (plant) {
    case Plant::None: return "none";
    case Plant::OccupancyGrid: return "occupancy-grid";
    case Plant::MissingStageAttempt: return "missing-stage-attempt";
    case Plant::FlatReduction: return "flat-reduction";
  }
  return "unknown";
}

struct Ledger {
  u64 valid_copies = 0;
  u64 attempts = 0;
  u64 pred_false = 0;
  u64 wait_barriers = 0;
  u64 s2r = 0;
  u64 clears = 0;
  u64 reductions = 0;
  u64 handoffs = 0;
  u64 final_stores = 0;
  u64 mma = 0;

  auto tie() const {
    return std::tie(valid_copies, attempts, pred_false, wait_barriers, s2r,
                    clears, reductions, handoffs, final_stores, mma);
  }
  bool operator==(Ledger const& other) const { return tie() == other.tie(); }
};

constexpr Ledger kExpected{
    1024, 1318, 294, 1122, 2146, 98, 98, 66, 32, 65536};

struct Segment {
  u64 block = 0;
  u64 q = 0;
  u64 k_begin = 0;
  u64 k_count = 0;
  u64 lock = 0;

  auto key() const {
    return std::make_tuple(block, q, k_begin, k_count, lock);
  }
};

struct SegmentTrace {
  Segment work;
  std::vector<Event> events;
  // For four K cohorts, both source trees execute offsets 2, then 1, then the
  // K0 final load.  The current flat fan-in plant is represented as 3,0.
  std::vector<int> reduction_cadence;
};

struct Trace {
  Flavor flavor = Flavor::Classic;
  Plant plant = Plant::None;
  u64 grid = 0;
  u64 iters_per_block = 0;
  u64 active_blocks = 0;
  u64 reduction_barriers = 0;
  Ledger ledger;
  std::vector<SegmentTrace> segments;
  std::vector<int> cell_visits;
  std::vector<int> copy_visits;
};

std::vector<Segment> make_classic_segments(Plant plant, u64& grid,
                                           u64& iters_per_block,
                                           u64& active_blocks) {
  u64 const occupancy = plant == Plant::OccupancyGrid ? 2 : 1;
  grid = std::max(kOutputTiles, kCu * occupancy);
  u64 const group_quantum = kGroupSize / kTileK;
  iters_per_block = round_up(ceil_div(kCells, grid), group_quantum);
  active_blocks = ceil_div(kCells, iters_per_block);

  std::vector<Segment> out;
  for (u64 block = 0; block < grid; ++block) {
    u64 const begin = block * iters_per_block;
    if (begin >= kCells) continue;
    u64 const end = std::min((block + 1) * iters_per_block, kCells);
    // Direct transcription of classic's slice_row/slice_col advancement.
    u64 q = begin / kTilesK;
    u64 k_begin = begin % kTilesK;
    u64 remaining = end - begin;
    while (remaining != 0) {
      u64 const count = std::min(remaining, kTilesK - k_begin);
      out.push_back(Segment{block, q, k_begin, count, q});
      remaining -= count;
      ++q;
      k_begin = 0;
    }
  }
  return out;
}

std::vector<Segment> make_awesome_segments(Plant plant, u64& grid,
                                           u64& iters_per_block,
                                           u64& active_blocks) {
  u64 const occupancy = plant == Plant::OccupancyGrid ? 2 : 1;
  grid = std::max(kOutputTiles, kCu * occupancy);
  u64 const group_quantum = kGroupSize / kTileK;
  iters_per_block = round_up(ceil_div(kCells, grid), group_quantum);
  active_blocks = ceil_div(kCells, iters_per_block);

  std::vector<Segment> out;
  for (u64 block = 0; block < grid; ++block) {
    u64 const block_begin = block * iters_per_block;
    if (block_begin >= kCells) continue;
    u64 const block_end = std::min((block + 1) * iters_per_block, kCells);
    u64 const begin_tile_idx = block_begin / kTilesK;
    u64 const end_tile_idx = (block_end - 1) / kTilesK;

    // Awesome-Cute initializes tile_idx=end_tile_idx, intersects that output
    // tile with the block interval, then decrements tile_idx.
    for (u64 q = end_tile_idx + 1; q-- > begin_tile_idx;) {
      u64 const q_begin = q * kTilesK;
      u64 const segment_begin = std::max(block_begin, q_begin);
      u64 const segment_end = std::min(block_end, q_begin + kTilesK);
      out.push_back(Segment{block, q, segment_begin - q_begin,
                            segment_end - segment_begin, q});
    }
  }
  return out;
}

void emit(Event event, SegmentTrace& segment, Ledger& ledger) {
  segment.events.push_back(event);
  switch (event) {
    case Event::WaitBarrier: ++ledger.wait_barriers; break;
    case Event::S2R: ++ledger.s2r; break;
    case Event::Clear: ++ledger.clears; break;
    case Event::Mma: ++ledger.mma; break;
    case Event::Reduction: ++ledger.reductions; break;
    case Event::Attempt: break;
  }
}

void emit_attempt(Trace& trace, SegmentTrace& segment, u64 relative_k) {
  emit(Event::Attempt, segment, trace.ledger);
  ++trace.ledger.attempts;
  if (relative_k < segment.work.k_count) {
    ++trace.ledger.valid_copies;
    u64 const cell = segment.work.q * kTilesK + segment.work.k_begin + relative_k;
    ++trace.copy_visits.at(std::size_t(cell));
  } else {
    ++trace.ledger.pred_false;
  }
}

void emit_mma_inner(SegmentTrace& segment, Ledger& ledger) {
  // One PPU n16 instruction fuses Awesome-Cute's pair of NVIDIA n8
  // instructions.  Each inner step therefore issues 8 warps * 4 n16 MMAs.
  for (u64 warp = 0; warp < kWarps; ++warp) {
    for (u64 n_atom = 0; n_atom < kMmasPerWarpInner; ++n_atom) {
      (void)warp;
      (void)n_atom;
      emit(Event::Mma, segment, ledger);
    }
  }
}

void emit_segment(Trace& trace, Segment const& work) {
  SegmentTrace segment;
  segment.work = work;

  u64 const prologue_attempts =
      trace.plant == Plant::MissingStageAttempt ? kStages - 2 : kStages - 1;
  for (u64 pipe = 0; pipe < prologue_attempts; ++pipe) {
    emit_attempt(trace, segment, pipe);
  }

  // The two sources have the same ledger but a deliberately recorded prologue
  // ordering difference.  Classic spells zero -> wait -> prime; Awesome-Cute
  // spells wait -> prime -> clear.
  if (trace.flavor == Flavor::Classic) {
    emit(Event::Clear, segment, trace.ledger);
    emit(Event::WaitBarrier, segment, trace.ledger);
    emit(Event::S2R, segment, trace.ledger);
  } else {
    emit(Event::WaitBarrier, segment, trace.ledger);
    emit(Event::S2R, segment, trace.ledger);
    emit(Event::Clear, segment, trace.ledger);
  }

  for (u64 tile = 0; tile < work.k_count; ++tile) {
    u64 const cell = work.q * kTilesK + work.k_begin + tile;
    ++trace.cell_visits.at(std::size_t(cell));

    // k_inner=0: load the other register buffer, issue the next-stage attempt,
    // wait/publish the stage, then consume the current registers.
    emit(Event::S2R, segment, trace.ledger);
    emit_attempt(trace, segment, (kStages - 1) + tile);
    emit(Event::WaitBarrier, segment, trace.ledger);
    emit_mma_inner(segment, trace.ledger);

    // k_inner=1: load the next stage's inner-0 buffer, then consume inner-1.
    emit(Event::S2R, segment, trace.ledger);
    emit_mma_inner(segment, trace.ledger);
  }

  emit(Event::Reduction, segment, trace.ledger);
  segment.reduction_cadence = trace.plant == Plant::FlatReduction
      ? std::vector<int>{3, 0}
      : std::vector<int>{2, 1, 0};
  trace.reduction_barriers += segment.reduction_cadence.size();
  trace.segments.push_back(std::move(segment));
}

Trace build_trace(Flavor flavor, Plant plant) {
  Trace trace;
  trace.flavor = flavor;
  trace.plant = plant;
  trace.cell_visits.assign(std::size_t(kCells), 0);
  trace.copy_visits.assign(std::size_t(kCells), 0);

  auto work = flavor == Flavor::Classic
      ? make_classic_segments(plant, trace.grid, trace.iters_per_block,
                              trace.active_blocks)
      : make_awesome_segments(plant, trace.grid, trace.iters_per_block,
                              trace.active_blocks);
  for (Segment const& segment : work) emit_segment(trace, segment);

  std::vector<u64> peers(std::size_t(kOutputTiles), 0);
  for (SegmentTrace const& segment : trace.segments) {
    ++peers.at(std::size_t(segment.work.q));
  }
  for (u64 q = 0; q < kOutputTiles; ++q) {
    if (peers[std::size_t(q)] != 0) {
      trace.ledger.handoffs += peers[std::size_t(q)] - 1;
      ++trace.ledger.final_stores;
    }
  }
  return trace;
}

struct Report {
  std::set<std::string> categories;
  std::vector<std::string> messages;

  void fail(std::string category, std::string message) {
    categories.insert(std::move(category));
    if (messages.size() < 8) messages.push_back(std::move(message));
  }
  bool ok() const { return categories.empty(); }
  bool has(char const* category) const {
    return categories.find(category) != categories.end();
  }
};

std::string count_message(char const* name, u64 got, u64 want) {
  std::ostringstream out;
  out << name << " got=" << got << " want=" << want;
  return out.str();
}

void check_count(Report& report, char const* category, char const* name,
                 u64 got, u64 want) {
  if (got != want) report.fail(category, count_message(name, got, want));
}

void expect_event(Report& report, SegmentTrace const& segment, std::size_t& at,
                  Event expected, char const* where) {
  if (at >= segment.events.size() || segment.events[at] != expected) {
    std::ostringstream out;
    out << where << " block=" << segment.work.block << " q=" << segment.work.q
        << " event_index=" << at;
    report.fail("dynamic-sequence", out.str());
    // Do not advance beyond end.  Advancing on an in-range mismatch lets the
    // verifier report the causal shift without looping on the same event.
    if (at < segment.events.size()) ++at;
    return;
  }
  ++at;
}

void validate_segment_sequence(Trace const& trace, SegmentTrace const& segment,
                               Report& report) {
  std::size_t at = 0;
  for (u64 pipe = 0; pipe < kStages - 1; ++pipe) {
    (void)pipe;
    expect_event(report, segment, at, Event::Attempt, "prologue-attempt");
  }
  if (trace.flavor == Flavor::Classic) {
    expect_event(report, segment, at, Event::Clear, "classic-clear");
    expect_event(report, segment, at, Event::WaitBarrier, "classic-wait");
    expect_event(report, segment, at, Event::S2R, "classic-prime");
  } else {
    expect_event(report, segment, at, Event::WaitBarrier, "awesome-wait");
    expect_event(report, segment, at, Event::S2R, "awesome-prime");
    expect_event(report, segment, at, Event::Clear, "awesome-clear");
  }

  for (u64 tile = 0; tile < segment.work.k_count; ++tile) {
    (void)tile;
    expect_event(report, segment, at, Event::S2R, "inner0-s2r");
    expect_event(report, segment, at, Event::Attempt, "steady-attempt");
    expect_event(report, segment, at, Event::WaitBarrier, "steady-wait");
    for (u64 mma = 0; mma < kWarps * kMmasPerWarpInner; ++mma) {
      expect_event(report, segment, at, Event::Mma, "inner0-mma");
    }
    expect_event(report, segment, at, Event::S2R, "inner1-s2r");
    for (u64 mma = 0; mma < kWarps * kMmasPerWarpInner; ++mma) {
      expect_event(report, segment, at, Event::Mma, "inner1-mma");
    }
  }
  expect_event(report, segment, at, Event::Reduction, "cta-reduction");
  if (at != segment.events.size()) {
    std::ostringstream out;
    out << "trailing-events block=" << segment.work.block << " q="
        << segment.work.q << " got=" << segment.events.size() - at;
    report.fail("dynamic-sequence", out.str());
  }
}

std::pair<u64, u64> holes_and_duplicates(std::vector<int> const& visits) {
  u64 holes = 0;
  u64 duplicates = 0;
  for (int count : visits) {
    holes += count == 0;
    duplicates += count > 1 ? u64(count - 1) : 0;
  }
  return {holes, duplicates};
}

Report validate_trace(Trace const& trace) {
  Report report;
  check_count(report, "grid", "grid", trace.grid, 72);
  check_count(report, "grid", "iters-per-block", trace.iters_per_block, 15);
  check_count(report, "grid", "active-blocks", trace.active_blocks, 69);
  check_count(report, "grid", "segments", trace.segments.size(), 98);

  auto ledger_check = [&](char const* field, u64 got, u64 want) {
    check_count(report, "ledger", field, got, want);
  };
  ledger_check("valid-copies", trace.ledger.valid_copies,
               kExpected.valid_copies);
  ledger_check("attempts", trace.ledger.attempts, kExpected.attempts);
  ledger_check("pred-false", trace.ledger.pred_false, kExpected.pred_false);
  ledger_check("wait-barriers", trace.ledger.wait_barriers,
               kExpected.wait_barriers);
  ledger_check("s2r", trace.ledger.s2r, kExpected.s2r);
  ledger_check("clears", trace.ledger.clears, kExpected.clears);
  ledger_check("reductions", trace.ledger.reductions, kExpected.reductions);
  ledger_check("handoffs", trace.ledger.handoffs, kExpected.handoffs);
  ledger_check("final-stores", trace.ledger.final_stores,
               kExpected.final_stores);
  ledger_check("mma", trace.ledger.mma, kExpected.mma);
  if (trace.ledger.attempts !=
      trace.ledger.valid_copies + trace.ledger.pred_false) {
    report.fail("ledger", "attempts != valid-copies + pred-false");
  }

  auto const [cell_holes, cell_duplicates] =
      holes_and_duplicates(trace.cell_visits);
  if (cell_holes || cell_duplicates) {
    std::ostringstream out;
    out << "consumed-cell holes=" << cell_holes
        << " duplicates=" << cell_duplicates;
    report.fail("cell-exact-once", out.str());
  }
  auto const [copy_holes, copy_duplicates] =
      holes_and_duplicates(trace.copy_visits);
  if (copy_holes || copy_duplicates) {
    std::ostringstream out;
    out << "valid-copy holes=" << copy_holes
        << " duplicates=" << copy_duplicates;
    report.fail("copy-exact-once", out.str());
  }

  std::map<u64, u64> lock_to_q;
  std::vector<std::set<u64>> locks_by_q{std::size_t(kOutputTiles)};
  std::vector<u64> peers_by_q(std::size_t(kOutputTiles), 0);
  for (SegmentTrace const& segment : trace.segments) {
    validate_segment_sequence(trace, segment, report);
    ++peers_by_q.at(std::size_t(segment.work.q));
    locks_by_q.at(std::size_t(segment.work.q)).insert(segment.work.lock);
    auto const [it, inserted] =
        lock_to_q.emplace(segment.work.lock, segment.work.q);
    if (!inserted && it->second != segment.work.q) {
      report.fail("global-q-lock", "one lock aliases two global q values");
    }
    if (segment.work.lock != segment.work.q) {
      report.fail("global-q-lock", "segment lock is not global q");
    }
    if (segment.reduction_cadence != std::vector<int>({2, 1, 0})) {
      std::ostringstream out;
      out << "non-tree cadence block=" << segment.work.block
          << " q=" << segment.work.q;
      report.fail("reduction-cadence", out.str());
    }
  }
  check_count(report, "reduction-cadence", "cta-reduction-barriers",
              trace.reduction_barriers, 294);

  u64 peers3 = 0;
  u64 peers4 = 0;
  for (u64 q = 0; q < kOutputTiles; ++q) {
    if (locks_by_q[std::size_t(q)] != std::set<u64>{q}) {
      report.fail("global-q-lock", "q does not own exactly its global lock");
    }
    peers3 += peers_by_q[std::size_t(q)] == 3;
    peers4 += peers_by_q[std::size_t(q)] == 4;
  }
  check_count(report, "global-q-lock", "unique-global-q-locks",
              lock_to_q.size(), kOutputTiles);
  check_count(report, "peer-census", "three-peer-output-tiles", peers3, 30);
  check_count(report, "peer-census", "four-peer-output-tiles", peers4, 2);
  return report;
}

std::multiset<decltype(Segment{}.key())> segment_keys(Trace const& trace) {
  std::multiset<decltype(Segment{}.key())> keys;
  for (SegmentTrace const& segment : trace.segments) {
    keys.insert(segment.work.key());
  }
  return keys;
}

Report compare_sources(Trace const& classic, Trace const& awesome) {
  Report report;
  if (!(classic.ledger == awesome.ledger)) {
    report.fail("source-ledger", "classic and Awesome-Cute ledgers differ");
  }
  if (segment_keys(classic) != segment_keys(awesome)) {
    report.fail("source-workset", "classic and Awesome-Cute segment sets differ");
  }

  auto check_block_order = [&](Trace const& trace, bool ascending) {
    std::map<u64, std::vector<u64>> q_by_block;
    for (SegmentTrace const& segment : trace.segments) {
      q_by_block[segment.work.block].push_back(segment.work.q);
    }
    u64 witnesses = 0;
    for (auto const& [block, q] : q_by_block) {
      (void)block;
      if (q.size() < 2) continue;
      ++witnesses;
      bool const sorted = ascending
          ? std::is_sorted(q.begin(), q.end())
          : std::is_sorted(q.rbegin(), q.rend());
      if (!sorted) {
        report.fail("source-order", "per-block q traversal has wrong direction");
      }
    }
    if (witnesses == 0) {
      report.fail("source-order", "fixture has no cross-q order witness");
    }
  };
  check_block_order(classic, true);
  check_block_order(awesome, false);
  return report;
}

u64 fnv1a(std::string const& text) {
  u64 hash = UINT64_C(14695981039346656037);
  for (unsigned char c : text) {
    hash ^= c;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

bool bind_source(std::string const& path,
                 std::vector<std::string> const& anchors, u64& hash) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "[l168:source] cannot open %s\n", path.c_str());
    return false;
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  std::string const text = buffer.str();
  bool ok = true;
  for (std::string const& anchor : anchors) {
    if (text.find(anchor) == std::string::npos) {
      std::fprintf(stderr, "[l168:source] %s missing anchor: %s\n",
                   path.c_str(), anchor.c_str());
      ok = false;
    }
  }
  hash = fnv1a(text);
  return ok;
}

bool bind_sources(std::string const& marlin_root) {
  std::string const classic = marlin_root + "/marlin_classic_ppu.cuh";
  std::string const awesome = marlin_root +
      "/ref/awesome-cute/gemm/marlin_gemm/marlin_cute_trait.h";
  u64 classic_hash = 0;
  u64 awesome_hash = 0;
  bool const classic_ok = bind_source(classic, {
      "int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);",
      "for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < slice_iters);",
      "zero_accums(); wait_for_stage(); fetch_to_registers(0, 0);",
      "for (int k = 0; k < b_sh_wr_iters; k++)",
      "fetch_to_registers(k + 1, pipe % stages);",
      "if (k == b_sh_wr_iters - 2)",
      "matmul(k);",
      "for (int i = red_off; i > 0; i /= 2)",
      "thread_block_reduce();",
      "if (last) write_result();",
  }, classic_hash);
  bool const awesome_ok = bind_source(awesome, {
      "ceil_div(m_tiles_parallel * k_iters * n_tiles, gridDim.x);",
      "min(end_tile_idx, m_tiles_parallel * n_tiles",
      "for (int pipe_idx = 0; pipe_idx < kStage - 1; pipe_idx++)",
      "launch_s2r(0, 0);",
      "clear(tCrC_mma);",
      "for (int k_inner_idx = 0; k_inner_idx < k_inner_cnt; k_inner_idx++)",
      "launch_s2r(pipe_idx % kStage, k_inner_idx + 1);",
      "if (k_inner_idx == k_inner_cnt - 2)",
      "launch_gemm(k_inner_idx);",
      "for (int warp_offset = kMmaThrLayoutK / 2; warp_offset > 0;",
      "launch_epilog_cta_reduce();",
      "launch_epilog_r2s2g();",
  }, awesome_hash);
  std::printf("[l168:source] classic_fnv1a64=%016llx "
              "awesome_fnv1a64=%016llx anchors=%s\n",
              static_cast<unsigned long long>(classic_hash),
              static_cast<unsigned long long>(awesome_hash),
              classic_ok && awesome_ok ? "BOUND" : "DRIFT");
  return classic_ok && awesome_ok;
}

void print_ledger(Trace const& trace, Report const& report) {
  Ledger const& x = trace.ledger;
  std::printf(
      "[l168:%s] G=%llu I=%llu active=%llu segments=%zu "
      "copies=%llu attempts=%llu pred_false=%llu waits=%llu s2r=%llu "
      "clears=%llu reductions=%llu reduce_barriers=%llu handoffs=%llu "
      "stores=%llu mma=%llu result=%s\n",
      flavor_name(trace.flavor),
      static_cast<unsigned long long>(trace.grid),
      static_cast<unsigned long long>(trace.iters_per_block),
      static_cast<unsigned long long>(trace.active_blocks), trace.segments.size(),
      static_cast<unsigned long long>(x.valid_copies),
      static_cast<unsigned long long>(x.attempts),
      static_cast<unsigned long long>(x.pred_false),
      static_cast<unsigned long long>(x.wait_barriers),
      static_cast<unsigned long long>(x.s2r),
      static_cast<unsigned long long>(x.clears),
      static_cast<unsigned long long>(x.reductions),
      static_cast<unsigned long long>(trace.reduction_barriers),
      static_cast<unsigned long long>(x.handoffs),
      static_cast<unsigned long long>(x.final_stores),
      static_cast<unsigned long long>(x.mma), report.ok() ? "PASS" : "FAIL");
}

void print_errors(char const* label, Report const& report) {
  for (std::string const& message : report.messages) {
    std::fprintf(stderr, "[l168:%s] %s\n", label, message.c_str());
  }
}

Plant parse_plant(std::string const& value) {
  if (value == "none") return Plant::None;
  if (value == "occupancy-grid") return Plant::OccupancyGrid;
  if (value == "missing-stage-attempt") return Plant::MissingStageAttempt;
  if (value == "flat-reduction") return Plant::FlatReduction;
  std::fprintf(stderr, "[l168] unknown plant %s\n", value.c_str());
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  std::string marlin_root;
  Plant plant = Plant::None;
  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    if (arg.rfind("--marlin-root=", 0) == 0) {
      marlin_root = arg.substr(std::string("--marlin-root=").size());
    } else if (arg.rfind("--plant=", 0) == 0) {
      plant = parse_plant(arg.substr(std::string("--plant=").size()));
    } else {
      std::fprintf(stderr,
                   "usage: %s --marlin-root=PATH "
                   "[--plant=occupancy-grid|missing-stage-attempt|flat-reduction]\n",
                   argv[0]);
      return 2;
    }
  }
  if (marlin_root.empty()) {
    std::fprintf(stderr, "[l168] --marlin-root is required\n");
    return 2;
  }
  if (!bind_sources(marlin_root)) return 2;

  Trace const classic = build_trace(Flavor::Classic, plant);
  Trace const awesome = build_trace(Flavor::Awesome, plant);
  Report const classic_report = validate_trace(classic);
  Report const awesome_report = validate_trace(awesome);
  Report const comparison = compare_sources(classic, awesome);

  print_ledger(classic, classic_report);
  print_ledger(awesome, awesome_report);
  print_errors("classic-error", classic_report);
  print_errors("awesome-error", awesome_report);
  print_errors("comparison-error", comparison);

  if (plant == Plant::None) {
    bool const pass = classic_report.ok() && awesome_report.ok() && comparison.ok();
    std::printf(
        "[l168] fixed=M1_N4096_K4096_L1_gs128_tile16x128x128_"
        "warp16x64x32_s4_CU72 exact_once=%s global_q_locks=%s "
        "source_ledgers=%s result=%s\n",
        classic_report.has("cell-exact-once") ||
                classic_report.has("copy-exact-once") ||
                awesome_report.has("cell-exact-once") ||
                awesome_report.has("copy-exact-once")
            ? "OPEN" : "CLOSED",
        classic_report.has("global-q-lock") ||
                awesome_report.has("global-q-lock")
            ? "OPEN" : "CLOSED",
        comparison.ok() ? "MATCH" : "MISMATCH", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
  }

  bool caught = false;
  switch (plant) {
    case Plant::OccupancyGrid:
      caught = classic_report.has("grid") && awesome_report.has("grid");
      break;
    case Plant::MissingStageAttempt:
      caught = classic_report.has("copy-exact-once") &&
               awesome_report.has("copy-exact-once") &&
               classic_report.has("dynamic-sequence") &&
               awesome_report.has("dynamic-sequence");
      break;
    case Plant::FlatReduction:
      caught = classic_report.has("reduction-cadence") &&
               awesome_report.has("reduction-cadence");
      break;
    case Plant::None: break;
  }
  std::printf("[l168:red] plant=%s caught=%d result=%s\n", plant_name(plant),
              int(caught), caught ? "RED" : "ESCAPED");
  // A caught plant deliberately exits 1.  The runner distinguishes this from
  // source/CLI failures (2) and rejects an escaped plant (0).
  return caught ? 1 : 0;
}
