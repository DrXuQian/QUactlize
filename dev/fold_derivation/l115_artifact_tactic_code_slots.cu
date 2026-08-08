// L115 -- DOES ONE ARTIFACT'S PHYSICAL CODE-SLOT MAP SURVIVE A LARGER TACTIC TILEK?
//
// This is the durable witness for #37's ArtifactTileK/TacticTileK split.  It fixes the artifact A, derives each
// plane's resident fold F from (bits,A), then asks the current consumer map for tactic T where every logical (n,k)
// lands.  Two independent summaries are printed because collapsing them hid two different failures:
//
//   slots    writes through place_from_map's exact physical address layouts; detects two deliveries owning one slot
//   logical  walks plane_map/tile_map_hi itself; detects a consumer map that never reaches part of the logical tile
//   owner_diff compares those physical slots with consecutive A-sized baseline tiles; zero is the actual "one
//              resident artifact serves larger T" contract (bijective-but-differently-permuted is still wrong)
//
// Every number below is asserted, including owner_diff=0 for all single-plane cross-T rows. The int4 row is the
// positive cross-T control.  The final row plants a duplicate
// logical owner into an otherwise complete int4 map; it MUST be classified INCOMPLETE even though every physical slot
// is still occupied.  A checker that only counts output slots would miss that control.
//
// EXECUTION TIER: MANUAL HOST PROBE; NOT REGISTERED IN ci/local_gates.py.
// The work is host-side, but instantiating the real CuTe MMA/copy objects is intentionally template-heavy.  Run this
// after changing xplane_offline.hpp, the mixed-input B copy/swizzle atom, ArtifactTileK plumbing, or either two-plane
// converter placement.  It does not prove device numerical correctness; that remains a BOX.md gate.
//
// BUILD/RUN (from repository root; no device required):
//   /usr/local/cuda/bin/nvcc -std=c++17 -O2 -x cu -arch=sm_80 -w \
//     -I dev/fold_derivation/stub_inc -I quactlize/include -I third_party/actlize/include \
//     dev/fold_derivation/l115_artifact_tactic_code_slots.cu -o /tmp/l115 && /tmp/l115
// Measured in the 0cd1abb development container: about 14 s compile, under 0.1 s run.  l104's larger matrix expands
// 36 heavy CuTe instantiations and can exceed two minutes on a loaded host; the slot walk itself is not the cost.
//
// The physical-address layouts below are the two layouts in xplane::place_from_map.  The map itself is NOT copied:
// it comes directly from xplane::plane_map/tile_map_hi, whose CubeTV is built from the production AIU copy traits.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "ppu_tactic_space.hpp"
#include "xplane_offline.hpp"

namespace {

using namespace cute;

constexpr int kFixtureN = 256;
constexpr int kFixtureK = 256;
constexpr int kFixtureCodes = kFixtureN * kFixtureK;

template <int Bits, int ArtifactTileK>
constexpr int artifact_fold() {
  constexpr int Fold = ppu_tactics::fold_for(Bits, ArtifactTileK);
  static_assert(Fold > 0, "the shared tactic rule must derive a legal artifact fold");
  return Fold;
}

struct LogicalStats {
  int unique = 0;
  int total = 0;
  int collisions = 0;
  int unset = 0;
  int oob = 0;
  int entries = 0;
};

LogicalStats logical_stats(std::vector<int> const& map, int total) {
  LogicalStats out;
  out.total = total;
  out.entries = int(map.size());
  std::vector<int> hits(size_t(total), 0);
  for (int owner : map) {
    if (owner == -1) {
      ++out.unset;
    } else if (owner < 0 || owner >= total) {
      ++out.oob;
    } else {
      out.collisions += hits[size_t(owner)] != 0;
      ++hits[size_t(owner)];
    }
  }
  for (int n : hits) out.unique += n != 0;
  return out;
}

struct SlotStats {
  std::vector<int> owner;
  std::vector<uint32_t> owner_or;
  int writes = 0;
  int unique = 0;
  int total = 0;
  int collisions = 0;
  int oob = 0;
  int logical_unique = 0;
  int logical_collisions = 0;
};

// Replay xplane::place_from_map's two destination layouts while retaining the logical owner instead of packed bits.
// N=K=256 makes the requested single-plane counts directly comparable.  A Q6 row contains two TN=128 N tiles;
// its `logical` report is one 128x256 local map while its `slots` report is the complete 256x256 fixture.
template <int Bits, int TN, int TacticTileK, int Fold, int ArtifactTileK = TacticTileK>
SlotStats materialize(std::vector<int> const& map, int N = kFixtureN, int K = kFixtureK) {
  constexpr int CPW = 32 / Bits;
  constexpr int PhysicalRows = TN / Fold;
  constexpr int Deliveries = (Fold * TacticTileK * Bits / 8) / 32;
  constexpr int ArtifactDeliveries = (Fold * ArtifactTileK * Bits / 8) / 32;
  constexpr int ArtifactTilesPerTactic = TacticTileK / ArtifactTileK;
  static_assert(TacticTileK % ArtifactTileK == 0,
                "artifact TileK must evenly tile tactic TileK");
  static_assert((Fold * ArtifactTileK * Bits) % 256 == 0,
                "an artifact row must contain whole 32 B deliveries");
  static_assert(Deliveries >= 1, "a physical row must contain a whole AIU delivery");
  static_assert(ArtifactDeliveries >= 1 &&
                    Deliveries == ArtifactTilesPerTactic * ArtifactDeliveries,
                "artifact deliveries must tile the full tactic row exactly");

  int const word_row_offset = 256 / CPW;
  int const runs = word_row_offset / 8;
  int const physical_n = N / Fold;
  constexpr int kInterleave = 256;
  constexpr int contig_bytes = Fold * ArtifactTileK * Bits / 8;
  constexpr int aiu_bytes = contig_bytes > 128 ? 128 : contig_bytes;
  constexpr int aiu_elements = aiu_bytes * 8 / Bits;
  constexpr int runs_per_super = kInterleave / aiu_elements;
  int const n_tiles = N / TN;
  int const k_tiles = K / TacticTileK;
  int const artifact_k_tiles = K / ArtifactTileK;

  auto dst_fold = make_layout(
      make_shape(Int<CPW>{}, _8{}, Int<PhysicalRows>{}, n_tiles, runs ? runs : 1,
                 artifact_k_tiles / (runs ? runs : 1)),
      make_stride(_1{}, Int<CPW>{}, word_row_offset * CPW,
                  PhysicalRows * word_row_offset * CPW, 8 * CPW,
                  physical_n * word_row_offset * CPW));
  auto dst_flat = make_layout(
      make_shape(Int<CPW>{}, _8{}, Int<ArtifactDeliveries>{}, runs_per_super ? runs_per_super : 1,
                 Int<PhysicalRows>{}, n_tiles,
                 artifact_k_tiles / (runs_per_super ? runs_per_super : 1)),
      make_stride(_1{}, Int<CPW>{}, 8 * CPW, aiu_elements, kInterleave,
                  TN * kInterleave, N * kInterleave));

  SlotStats out;
  out.total = N * K;
  out.owner.assign(size_t(out.total), -1);
  out.owner_or.assign(size_t(out.total), 0);
  std::vector<int> logical_hits(size_t(out.total), 0);
  for (int tn = 0; tn < n_tiles; ++tn)
    for (int ki = 0; ki < k_tiles; ++ki)
      for (int row = 0; row < PhysicalRows; ++row)
        for (int dl = 0; dl < Deliveries; ++dl)
          for (int wd = 0; wd < 8; ++wd)
            for (int j = 0; j < CPW; ++j) {
              size_t const map_index = (((size_t)row * Deliveries + dl) * 8 + wd) * CPW + j;
              if (map_index >= map.size()) continue;
              int const local = map[map_index];
              if (local < 0 || local >= TN * TacticTileK) continue;
              int const n = tn * TN + local / TacticTileK;
              int const k = ki * TacticTileK + local % TacticTileK;
              int const artifact_ki = ki * ArtifactTilesPerTactic + dl / ArtifactDeliveries;
              int const artifact_dl = dl % ArtifactDeliveries;
              size_t const code = Fold > 1
                  ? size_t(dst_fold(j, wd, row, tn, artifact_ki % (runs ? runs : 1),
                                    artifact_ki / (runs ? runs : 1)))
                  : size_t(dst_flat(j, wd, artifact_dl,
                                    artifact_ki % (runs_per_super ? runs_per_super : 1), row, tn,
                                    artifact_ki / (runs_per_super ? runs_per_super : 1)));
              if (code >= out.owner.size()) {
                ++out.oob;
                continue;
              }
              int const logical = k * N + n;
              ++out.writes;
              out.logical_collisions += logical_hits[size_t(logical)] != 0;
              ++logical_hits[size_t(logical)];
              int& prior = out.owner[code];
              out.owner_or[code] |= uint32_t(logical + 1);
              if (prior < 0) {
                prior = logical;
                ++out.unique;
              } else {
                ++out.collisions;
              }
            }
  for (int n : logical_hits) out.logical_unique += n != 0;
  return out;
}

// The slot walk above must not become a second, self-consistent writer.  Label every logical code by logical+1
// (17 bits for a 256x256 fixture), feed those bits through the REAL production place_from_map writer, reconstruct
// each physical slot's owner-OR, and compare it with the walk.  logical+1 distinguishes owner zero from unwritten.
template <int Bits, int TM, int TN, int TacticTileK, int WM, int WN, int Fold,
          int ArtifactTileK = TacticTileK>
int production_writer_diff(std::vector<int> const& map, SlotStats const& replay,
                           int N = kFixtureN, int K = kFixtureK) {
  constexpr int LabelBits = 17;
  constexpr int Passes = (LabelBits + Bits - 1) / Bits;
  constexpr int Mask = (1 << Bits) - 1;
  std::vector<uint32_t> writer_owner_or(size_t(N) * K, 0);
  std::vector<uint8_t> q(size_t(N) * K);
  std::vector<int8_t> bytes(size_t(N) * K * Bits / 8);
  for (int pass = 0; pass < Passes; ++pass) {
    int const shift = pass * Bits;
    for (int logical = 0; logical < N * K; ++logical)
      q[size_t(logical)] = uint8_t(((logical + 1) >> shift) & Mask);
    xplane::place_from_map<Bits, TM, TN, TacticTileK, WM, WN, Fold, ArtifactTileK>(
        bytes.data(), map, q, N, K);
    for (int code = 0; code < N * K; ++code) {
      size_t const bit0 = size_t(code) * Bits;
      uint32_t value = 0;
      for (int b = 0; b < Bits; ++b)
        value |= uint32_t((uint8_t(bytes[(bit0 + b) / 8]) >> ((bit0 + b) % 8)) & 1) << b;
      writer_owner_or[size_t(code)] |= value << shift;
    }
  }
  int diff = 0;
  for (size_t i = 0; i < replay.owner_or.size(); ++i)
    diff += replay.owner_or[i] != writer_owner_or[i];
  return diff;
}

int owner_diff(SlotStats const& a, SlotStats const& b) {
  if (a.owner.size() != b.owner.size()) return std::max(a.owner.size(), b.owner.size());
  int diff = 0;
  for (size_t i = 0; i < a.owner.size(); ++i) diff += a.owner[i] != b.owner[i];
  return diff;
}

bool complete(SlotStats const& s, LogicalStats const& l) {
  return s.unique == s.total && s.collisions == 0 && s.oob == 0 &&
         s.logical_unique == s.total && s.logical_collisions == 0 &&
         l.unique == l.total && l.entries == l.total &&
         l.collisions == 0 && l.unset == 0 && l.oob == 0;
}

void print_row(char const* format, char const* plane, int artifact_tile_k, int tactic_tile_k,
               int tm, int tn, int wm, int wn, int low_fold, int high_fold,
               SlotStats const& slots, LogicalStats const& logical, int diff, int writer_diff) {
  bool const ok = complete(slots, logical);
  std::printf(
      "ROW format=%-11s plane=%-4s A=%3d T=%3d tile=%dx%dx%d warp=%dx%d F=%d/%d "
      "slots=%d/%d collisions=%d oob=%d logical=%d/%d entries=%d duplicates=%d unset=%d map_oob=%d "
      "owner_diff=%d writer_diff=%d %s\n",
      format, plane, artifact_tile_k, tactic_tile_k, tm, tn, tactic_tile_k, wm, wn,
      low_fold, high_fold, slots.unique, slots.total, slots.collisions, slots.oob,
      logical.unique, logical.total, logical.entries, logical.collisions, logical.unset, logical.oob,
      diff, writer_diff, ok ? "COMPLETE" : "INCOMPLETE");
}

int failures = 0;

void expect(bool condition, char const* what) {
  if (!condition) {
    std::fprintf(stderr, "EXPECTATION FAILED: %s\n", what);
    ++failures;
  }
}

struct Observation {
  SlotStats slots;
  LogicalStats logical;
  int writer_diff = 0;
};

template <int Bits, int ArtifactTileK, int TacticTileK, int TM, int TN, int WM, int WN>
Observation low_map() {
  constexpr int Fold = artifact_fold<Bits, ArtifactTileK>();
  auto map = xplane::plane_map<Bits, TM, TN, TacticTileK, WM, WN, Fold, ArtifactTileK>();
  auto slots = materialize<Bits, TN, TacticTileK, Fold, ArtifactTileK>(map);
  return {slots, logical_stats(map, TN * TacticTileK),
          production_writer_diff<Bits, TM, TN, TacticTileK, WM, WN, Fold, ArtifactTileK>(map, slots)};
}

template <int LowBits, int HighBits, int ArtifactTileK, int TacticTileK,
          int TM, int TN, int WM, int WN>
Observation high_map() {
  constexpr int LowFold = artifact_fold<LowBits, ArtifactTileK>();
  constexpr int HighFold = artifact_fold<HighBits, ArtifactTileK>();
  auto map = xplane::tile_map_hi<LowBits, HighBits, TM, TN, TacticTileK, WM, WN,
                                  HighFold, LowFold>();
  auto slots = materialize<HighBits, TN, TacticTileK, HighFold>(map);
  return {slots, logical_stats(map, TN * TacticTileK),
          production_writer_diff<HighBits, TM, TN, TacticTileK, WM, WN, HighFold>(map, slots)};
}

void positive_and_single_plane_rows() {
  constexpr int TM = 64, TN = 64, WM = 32, WN = 32;

  auto i4_a64 = low_map<4, 64, 64, TM, TN, WM, WN>();
  auto i4_t128 = low_map<4, 64, 128, TM, TN, WM, WN>();
  int const i4_diff = owner_diff(i4_a64.slots, i4_t128.slots);
  print_row("int4+control", "low", 64, 128, TM, TN, WM, WN, 1, 1,
            i4_t128.slots, i4_t128.logical, i4_diff, i4_t128.writer_diff);
  expect(i4_t128.slots.unique == 65536 && i4_t128.slots.total == 65536,
         "int4 A64->T128 must fill 65536/65536 physical slots");
  expect(i4_t128.slots.collisions == 0 && i4_t128.slots.oob == 0 && i4_diff == 0,
         "int4 A64->T128 must have zero collisions/oob/owner diff");
  expect(i4_t128.writer_diff == 0 && complete(i4_t128.slots, i4_t128.logical),
         "int4 A64->T128 writer parity and positive control must be COMPLETE");

  auto i2_a64 = low_map<2, 64, 64, TM, TN, WM, WN>();
  auto i2_t128 = low_map<2, 64, 128, TM, TN, WM, WN>();
  int const i2_diff = owner_diff(i2_a64.slots, i2_t128.slots);
  print_row("int2", "low", 64, 128, TM, TN, WM, WN, 2, 1,
            i2_t128.slots, i2_t128.logical, i2_diff, i2_t128.writer_diff);
  expect(i2_t128.slots.unique == 65536 && i2_t128.slots.total == 65536 &&
             i2_t128.slots.collisions == 0,
         "int2 A64/F2->T128 must cover the resident map without collisions");
  expect(i2_diff == 0 && i2_t128.writer_diff == 0 && complete(i2_t128.slots, i2_t128.logical),
         "int2 A64/F2->T128 must preserve resident owners, writer parity and COMPLETE coverage");

  constexpr int I1TN = 128, I1WN = 64;
  auto i1_a64 = low_map<1, 64, 64, TM, I1TN, WM, I1WN>();
  auto i1_t128 = low_map<1, 64, 128, TM, I1TN, WM, I1WN>();
  auto i1_t256 = low_map<1, 64, 256, TM, I1TN, WM, I1WN>();
  int const i1_t128_diff = owner_diff(i1_a64.slots, i1_t128.slots);
  int const i1_t256_diff = owner_diff(i1_a64.slots, i1_t256.slots);
  print_row("int1", "low", 64, 128, TM, I1TN, WM, I1WN, 4, 1,
            i1_t128.slots, i1_t128.logical, i1_t128_diff,
            i1_t128.writer_diff);
  print_row("int1", "low", 64, 256, TM, I1TN, WM, I1WN, 4, 1,
            i1_t256.slots, i1_t256.logical, i1_t256_diff,
            i1_t256.writer_diff);
  expect(i1_t128.slots.unique == 65536 && i1_t128.slots.total == 65536 &&
             i1_t128.slots.collisions == 0,
         "int1 A64/F4->T128 must cover the resident map without collisions");
  expect(i1_t256.slots.unique == 65536 && i1_t256.slots.total == 65536 &&
             i1_t256.slots.collisions == 0,
         "int1 A64/F4->T256 must cover the resident map without collisions");
  expect(i1_t128_diff == 0 && i1_t256_diff == 0 &&
             i1_t128.writer_diff == 0 && i1_t256.writer_diff == 0 &&
             complete(i1_t128.slots, i1_t128.logical) && complete(i1_t256.slots, i1_t256.logical),
         "int1 A64/F4 cross-T must preserve resident owners, writer parity and COMPLETE coverage");
}

void q6_rows() {
  constexpr int TM = 64, TN = 128, T = 256, WM = 64, WN = 64;

  auto q6_a128 = high_map<4, 2, 128, T, TM, TN, WM, WN>();
  print_row("Q6_K", "high", 128, T, TM, TN, WM, WN, 1, 1,
            q6_a128.slots, q6_a128.logical, -1, q6_a128.writer_diff);
  expect(q6_a128.logical.unique == 16384 && q6_a128.logical.total == 32768,
         "Q6 A128 F1/F1->T256 must miss 16384/32768 logical high-plane slots");

  auto q6_a64 = high_map<4, 2, 64, T, TM, TN, WM, WN>();
  print_row("Q6_K", "high", 64, T, TM, TN, WM, WN, 1, 2,
            q6_a64.slots, q6_a64.logical, -1, q6_a64.writer_diff);
  expect(q6_a64.logical.unique == 8192 && q6_a64.logical.total == 32768,
         "Q6 A64 F1/F2->T256 must miss 24576/32768 logical high-plane slots");

  auto q6_a32 = high_map<4, 2, 32, T, TM, TN, WM, WN>();
  print_row("Q6_K", "high", 32, T, TM, TN, WM, WN, 2, 4,
            q6_a32.slots, q6_a32.logical, -1, q6_a32.writer_diff);
  expect(q6_a32.logical.unique == 4096 && q6_a32.logical.total == 32768,
         "Q6 A32 F2/F4->T256 must miss 28672/32768 logical high-plane slots");

  expect(q6_a128.logical.oob == 0 && q6_a64.logical.oob == 0 && q6_a32.logical.oob == 0,
         "all Q6 logical maps must distinguish missing slots from out-of-range owners");
  expect(q6_a128.writer_diff == 0 && q6_a64.writer_diff == 0 && q6_a32.writer_diff == 0,
         "all Q6 slot replays must match the production place_from_map writer");
}

void planted_negative() {
  constexpr int TM = 64, TN = 64, TK = 64, WM = 32, WN = 32;
  auto bad = xplane::plane_map<4, TM, TN, TK, WM, WN, 1>();
  expect(bad.size() >= 2 && bad[0] >= 0 && bad[1] >= 0,
         "negative control needs two valid owners to corrupt");
  if (bad.size() < 2) return;
  expect(bad[0] != bad[1], "negative control must replace a distinct owner");
  bad[1] = bad[0];
  // Keep the common 256x256 fixture.  The bad local owner repeats once in each of its 16 outer tiles, so physical
  // occupancy stays complete while logical coverage loses 16 elements -- exactly the blind spot this control attacks.
  auto slots = materialize<4, TN, TK, 1>(bad);
  auto logical = logical_stats(bad, TN * TK);
  int const writer_diff = production_writer_diff<4, TM, TN, TK, WM, WN, 1>(bad, slots);
  print_row("PLANTED_BAD", "low", 64, 64, TM, TN, WM, WN, 1, 1,
            slots, logical, -1, writer_diff);
  expect(logical.unique == 4095 && logical.total == 4096 && logical.collisions == 1,
         "planted duplicate must leave 4095/4096 unique logical owners and one duplicate");
  expect(slots.unique == 65536 && slots.collisions == 0 && slots.oob == 0,
         "planted logical duplicate must not manufacture a physical occupancy failure");
  expect(slots.logical_unique == 65520 && slots.logical_collisions == 16,
         "planted local duplicate must repeat exactly once in each of 16 outer tiles");
  expect(writer_diff == 0, "negative-control slot replay must match the production writer");
  expect(!complete(slots, logical), "planted duplicate must be classified INCOMPLETE");
}

}  // namespace

int main() {
  std::printf("L115 artifact/tactic physical code-slot witness (current consumer contract)\n");
  positive_and_single_plane_rows();
  q6_rows();
  planted_negative();
  std::printf("RESULT %s failures=%d\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
