// L183 -- device-free exhaustive ring oracle for the standalone Marlin
// pipeline-depth axis.
//
// Scope: the shipping decode target has K=4096 and TileK=128, hence every
// scheduler segment has k_tile_count in [1,32].  This oracle exhausts that
// complete interval for every declared pipeline depth in [2,6].  It models
// the exact source cadence in MarlinCollectivePPU::run_segment:
//
//   * Stages-1 prologue copy attempts;
//   * one register prime from slot zero;
//   * read-next / refill-behind / wait / consume for two K-inner fragments;
//   * one refill attempt for every consumed K tile.
//
// A physical next-slot load also occurs after the final tile.  Classic Marlin
// and the production collective both spell that dead speculative prefetch.
// The oracle reports it separately and proves that no such value reaches an
// MMA.  Calling it a live initialized read would hide the actual source
// cadence; calling it a correctness failure would reject classic itself.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kMinStages = 2;
constexpr int kMaxStages = 6;
constexpr int kProblemK = 4096;
constexpr int kTileK = 128;
constexpr int kMaxSegmentTiles = kProblemK / kTileK;
constexpr int kInnerIters = 2;

static_assert(kMaxSegmentTiles == 32,
              "the exhaustive segment interval belongs to K4096/TK128");

enum class Plant {
  None,
  PreloadShort,
  WrongRingSlot,
  WrongRefillPredicate,
};

char const* plant_name(Plant plant) {
  switch (plant) {
    case Plant::None: return "none";
    case Plant::PreloadShort: return "preload-short";
    case Plant::WrongRingSlot: return "wrong-ring-slot";
    case Plant::WrongRefillPredicate: return "wrong-refill-predicate";
  }
  return "unknown";
}

Plant parse_plant(char const* value) {
  if (value == nullptr || std::strcmp(value, "none") == 0) {
    return Plant::None;
  }
  if (std::strcmp(value, "preload-short") == 0) {
    return Plant::PreloadShort;
  }
  if (std::strcmp(value, "wrong-ring-slot") == 0) {
    return Plant::WrongRingSlot;
  }
  if (std::strcmp(value, "wrong-refill-predicate") == 0) {
    return Plant::WrongRefillPredicate;
  }
  std::fprintf(stderr, "[l183] FAIL: unknown plant=%s\n", value);
  std::exit(2);
}

char const* option(int argc, char** argv, char const* prefix) {
  std::size_t const n = std::strlen(prefix);
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], prefix, n) == 0) return argv[i] + n;
  }
  return nullptr;
}

struct Slot {
  bool initialized = false;
  int tile = -1;
  std::array<bool, kInnerIters> live_reads{{false, false}};
};

struct Register {
  bool initialized = false;
  int tile = -1;
  int inner = -1;
};

struct Totals {
  std::uint64_t cases = 0;
  std::uint64_t tiles = 0;
  std::uint64_t preload_attempts = 0;
  std::uint64_t refill_attempts = 0;
  std::uint64_t preload_copies = 0;
  std::uint64_t refill_copies = 0;
  std::uint64_t live_reads = 0;
  std::uint64_t dead_speculative_reads = 0;
  std::uint64_t mma_consumes = 0;
};

struct Failure {
  bool failed = false;
  std::string category;
  std::string detail;

  void set(char const* new_category, std::string new_detail) {
    if (failed) return;
    failed = true;
    category = new_category;
    detail = std::move(new_detail);
  }
};

std::string context(int stages, int count, int tile, int slot) {
  std::ostringstream out;
  out << "stages=" << stages << " count=" << count << " tile=" << tile
      << " slot=" << slot;
  return out.str();
}

void run_case(int stages, int count, Plant plant, Totals& totals,
              Failure& failure) {
  ++totals.cases;
  totals.tiles += std::uint64_t(count);

  std::vector<Slot> slots(static_cast<std::size_t>(stages));
  std::array<Register, kInnerIters> registers{};
  std::vector<int> copies(std::size_t(count), 0);
  std::vector<int> preload_copies(std::size_t(count), 0);
  std::vector<int> refill_copies(std::size_t(count), 0);
  std::vector<std::array<int, kInnerIters>> reads(
      static_cast<std::size_t>(count));
  std::vector<std::array<int, kInnerIters>> consumes(
      static_cast<std::size_t>(count));

  auto write_slot = [&](int slot, int tile, bool preload) {
    if (slot < 0 || slot >= stages) {
      failure.set("ring-slot-range", context(stages, count, tile, slot));
      return;
    }
    Slot& destination = slots[std::size_t(slot)];
    if (destination.initialized) {
      int const old = destination.tile;
      bool const consumed =
          old >= 0 && old < count &&
          consumes[std::size_t(old)][0] == 1 &&
          consumes[std::size_t(old)][1] == 1;
      if (!consumed) {
        std::ostringstream out;
        out << context(stages, count, tile, slot)
            << " overwrites_unconsumed_tile=" << old;
        failure.set("overwrite-before-consume", out.str());
      }
    }
    destination = Slot{true, tile, {{false, false}}};
    if (tile < 0 || tile >= count) {
      failure.set("copy-out-of-range", context(stages, count, tile, slot));
      return;
    }
    ++copies[std::size_t(tile)];
    if (preload) {
      ++preload_copies[std::size_t(tile)];
      ++totals.preload_copies;
    } else {
      ++refill_copies[std::size_t(tile)];
      ++totals.refill_copies;
    }
  };

  auto load_register = [&](int reg, int slot, int expected_tile, int inner,
                           bool live) {
    if (slot < 0 || slot >= stages) {
      failure.set("ring-slot-range",
                  context(stages, count, expected_tile, slot));
      return;
    }
    Slot& source = slots[std::size_t(slot)];
    if (!live) {
      ++totals.dead_speculative_reads;
      // Preserve what the machine would see without pretending it is an
      // initialized semantic value.  This register is never consumed.
      registers[std::size_t(reg)] =
          Register{source.initialized, source.tile, inner};
      return;
    }
    ++totals.live_reads;
    if (!source.initialized) {
      failure.set("live-uninitialized-read",
                  context(stages, count, expected_tile, slot));
      registers[std::size_t(reg)] = Register{};
      return;
    }
    if (source.tile != expected_tile) {
      std::ostringstream out;
      out << context(stages, count, expected_tile, slot)
          << " resident_tile=" << source.tile;
      failure.set("live-stale-read", out.str());
    }
    if (inner < 0 || inner >= kInnerIters) {
      failure.set("inner-range", context(stages, count, expected_tile, slot));
      return;
    }
    if (expected_tile >= 0 && expected_tile < count) {
      ++reads[std::size_t(expected_tile)][std::size_t(inner)];
      if (source.live_reads[std::size_t(inner)]) {
        failure.set("duplicate-live-read",
                    context(stages, count, expected_tile, slot));
      }
      source.live_reads[std::size_t(inner)] = true;
    }
    registers[std::size_t(reg)] =
        Register{source.initialized, source.tile, inner};
  };

  auto consume_register = [&](int reg, int expected_tile, int inner) {
    ++totals.mma_consumes;
    Register const& value = registers[std::size_t(reg)];
    if (!value.initialized) {
      failure.set("mma-consumes-uninitialized",
                  context(stages, count, expected_tile, -1));
      return;
    }
    if (value.tile != expected_tile || value.inner != inner) {
      std::ostringstream out;
      out << context(stages, count, expected_tile, -1)
          << " register={tile:" << value.tile << ",inner:" << value.inner
          << "}";
      failure.set("mma-consumes-wrong-fragment", out.str());
    }
    ++consumes[std::size_t(expected_tile)][std::size_t(inner)];
  };

  // Exact production prologue: Stages-1 attempts, with false predicates after
  // the segment is exhausted.  The preload-short plant deletes the final
  // attempt rather than merely changing its predicate.
  int const prologue_attempts =
      stages - 1 - (plant == Plant::PreloadShort ? 1 : 0);
  totals.preload_attempts += std::uint64_t(prologue_attempts);
  for (int pipe = 0; pipe < prologue_attempts; ++pipe) {
    if (pipe < count) write_slot(pipe, pipe, true);
  }

  // Every real scheduler segment is non-empty, so slot zero must be live.
  load_register(0, 0, 0, 0, true);

  int remaining = count;
  for (int tile = 0; tile < count; ++tile) {
    int const pipe = tile % stages;

    // Load the current tile's second inner fragment before the behind-slot is
    // refilled, matching load_registers(inner+1, pipe%Stages).
    load_register(1, pipe, tile, 1, true);

    ++totals.refill_attempts;
    int target = (pipe + stages - 1) % stages;
    if (plant == Plant::WrongRingSlot) target = pipe;
    bool predicate = remaining >= stages;
    if (plant == Plant::WrongRefillPredicate) predicate = remaining > stages;
    int const candidate_tile = (stages - 1) + tile;
    if (predicate) write_slot(target, candidate_tile, false);

    // wait_stage() makes the refill visible before this consume and the
    // next-slot prefetch.  The already primed inner-0 register is consumed.
    consume_register(0, tile, 0);

    bool const next_is_live = tile + 1 < count;
    load_register(0, (pipe + 1) % stages, tile + 1, 0, next_is_live);
    consume_register(1, tile, 1);
    --remaining;
  }

  if (remaining != 0) {
    failure.set("remaining-count", "main loop did not drain the segment");
  }
  if (prologue_attempts != stages - 1) {
    failure.set("preload-attempt-count", "prologue did not issue Stages-1 attempts");
  }
  if (totals.refill_attempts == 0) {
    failure.set("refill-attempt-count", "non-empty segment issued no refill attempt");
  }

  for (int tile = 0; tile < count; ++tile) {
    bool const should_preload = tile < stages - 1;
    if (copies[std::size_t(tile)] != 1) {
      std::ostringstream out;
      out << context(stages, count, tile, -1)
          << " copies=" << copies[std::size_t(tile)];
      failure.set("copy-exact-once", out.str());
    }
    if (preload_copies[std::size_t(tile)] != (should_preload ? 1 : 0) ||
        refill_copies[std::size_t(tile)] != (should_preload ? 0 : 1)) {
      std::ostringstream out;
      out << context(stages, count, tile, -1)
          << " preload=" << preload_copies[std::size_t(tile)]
          << " refill=" << refill_copies[std::size_t(tile)];
      failure.set("preload-refill-partition", out.str());
    }
    for (int inner = 0; inner < kInnerIters; ++inner) {
      if (reads[std::size_t(tile)][std::size_t(inner)] != 1) {
        std::ostringstream out;
        out << context(stages, count, tile, -1) << " inner=" << inner
            << " reads=" << reads[std::size_t(tile)][std::size_t(inner)];
        failure.set("live-read-exact-once", out.str());
      }
      if (consumes[std::size_t(tile)][std::size_t(inner)] != 1) {
        std::ostringstream out;
        out << context(stages, count, tile, -1) << " inner=" << inner
            << " consumes=" << consumes[std::size_t(tile)][std::size_t(inner)];
        failure.set("mma-consume-exact-once", out.str());
      }
    }
  }
}

std::string read_file(char const* path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  std::ostringstream out;
  out << input.rdbuf();
  return out.str();
}

std::uint64_t fnv1a(std::string const& text) {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (unsigned char c : text) {
    hash ^= c;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

bool bind_source(char const* path, std::vector<char const*> const& anchors,
                 char const* label) {
  std::string const source = read_file(path);
  if (source.empty()) {
    std::fprintf(stderr, "[l183:source] FAIL: cannot read %s=%s\n", label,
                 path == nullptr ? "<null>" : path);
    return false;
  }
  bool ok = true;
  for (char const* anchor : anchors) {
    if (source.find(anchor) == std::string::npos) {
      std::fprintf(stderr, "[l183:source] FAIL: %s missing anchor: %s\n",
                   label, anchor);
      ok = false;
    }
  }
  std::printf("[l183:source] %s_fnv1a64=%016llx anchors=%zu result=%s\n",
              label, static_cast<unsigned long long>(fnv1a(source)),
              anchors.size(), ok ? "BOUND" : "DRIFT");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  Plant const plant = parse_plant(option(argc, argv, "--plant="));
  char const* production = option(argc, argv, "--production=");
  char const* classic = option(argc, argv, "--classic=");
  if (production == nullptr || classic == nullptr) {
    std::fprintf(stderr,
                 "[l183] FAIL: --production and --classic are required\n");
    return 2;
  }

  bool const production_bound = bind_source(
      production,
      {"for (int i = 0; i < Stages - 1; ++i)",
       "copy_stage(i, i, i < k_tiles_remaining);",
       "load_registers(cute::Int<0>{}, 0);",
       "for (int pipe = 0; pipe < Stages;)",
       "load_registers(cute::Int<inner + 1>{}, pipe % Stages);",
       "(pipe + Stages - 1) % Stages, pipe,",
       "k_tiles_remaining >= Stages",
       "--k_tiles_remaining;"},
      "production");
  bool const classic_bound = bind_source(
      classic,
      {"for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < slice_iters);",
       "zero_accums(); wait_for_stage(); fetch_to_registers(0, 0);",
       "for (int pipe = 0; pipe < stages;)",
       "fetch_to_registers(k + 1, pipe % stages);",
       "fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages)",
       "slice_iters--;"},
      "classic");
  if (!production_bound || !classic_bound) return 1;

  Totals totals;
  Failure failure;
  for (int stages = kMinStages; stages <= kMaxStages; ++stages) {
    for (int count = 1; count <= kMaxSegmentTiles; ++count) {
      run_case(stages, count, plant, totals, failure);
    }
  }

  std::uint64_t const expected_cases =
      std::uint64_t(kMaxStages - kMinStages + 1) * kMaxSegmentTiles;
  std::uint64_t const sum_1_to_32 =
      std::uint64_t(kMaxSegmentTiles) * (kMaxSegmentTiles + 1) / 2;
  std::uint64_t const expected_tiles =
      std::uint64_t(kMaxStages - kMinStages + 1) * sum_1_to_32;
  if (!failure.failed &&
      (totals.cases != expected_cases || totals.tiles != expected_tiles ||
       totals.live_reads != 2 * expected_tiles ||
       totals.mma_consumes != 2 * expected_tiles ||
       totals.dead_speculative_reads != expected_cases)) {
    std::ostringstream out;
    out << "totals cases=" << totals.cases << "/" << expected_cases
        << " tiles=" << totals.tiles << "/" << expected_tiles
        << " live_reads=" << totals.live_reads << "/" << 2 * expected_tiles
        << " consumes=" << totals.mma_consumes << "/" << 2 * expected_tiles
        << " dead_prefetch=" << totals.dead_speculative_reads << "/"
        << expected_cases;
    failure.set("global-ledger", out.str());
  }

  if (plant != Plant::None) {
    if (!failure.failed) {
      std::fprintf(stderr,
                   "[l183:red] plant=%s caught=0 reason=escaped result=FAIL\n",
                   plant_name(plant));
      return 2;
    }
    std::fprintf(stderr,
                 "[l183:red] plant=%s caught=1 category=%s reason=%s result=RED\n",
                 plant_name(plant), failure.category.c_str(),
                 failure.detail.c_str());
    return 1;
  }

  if (failure.failed) {
    std::fprintf(stderr, "[l183] FAIL: category=%s reason=%s\n",
                 failure.category.c_str(), failure.detail.c_str());
    return 1;
  }

  std::printf(
      "[l183] PASS: stages=%d..%d segment_k_tiles=1..%d exhaustive_cases=%llu "
      "tiles=%llu preload={attempts:%llu,copies:%llu} "
      "refill={attempts:%llu,copies:%llu} live_reads=%llu "
      "mma_consumes=%llu dead_final_prefetch=%llu "
      "live_uninitialized=0 overwrite_before_consume=0\n",
      kMinStages, kMaxStages, kMaxSegmentTiles,
      static_cast<unsigned long long>(totals.cases),
      static_cast<unsigned long long>(totals.tiles),
      static_cast<unsigned long long>(totals.preload_attempts),
      static_cast<unsigned long long>(totals.preload_copies),
      static_cast<unsigned long long>(totals.refill_attempts),
      static_cast<unsigned long long>(totals.refill_copies),
      static_cast<unsigned long long>(totals.live_reads),
      static_cast<unsigned long long>(totals.mma_consumes),
      static_cast<unsigned long long>(totals.dead_speculative_reads));
  return 0;
}
