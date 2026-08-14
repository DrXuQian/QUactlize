// Exhaustive host oracle for the standalone Marlin tactic authority.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <tuple>

#include "marlin_tactic_space_ppu.hpp"

namespace mt = marlin_tactics_ppu;

namespace {

enum class Plant {
  None,
  DropLoadAxis,
  DropStage5,
  CollapseWarpK,
  BroadenClassicWarpK,
};

Plant parse_plant(char const* text) {
  if (text == nullptr || *text == '\0') return Plant::None;
  if (!std::strcmp(text, "drop-load-axis")) return Plant::DropLoadAxis;
  if (!std::strcmp(text, "drop-stage5")) return Plant::DropStage5;
  if (!std::strcmp(text, "collapse-warp-k")) return Plant::CollapseWarpK;
  if (!std::strcmp(text, "broaden-classic-warp-k"))
    return Plant::BroadenClassicWarpK;
  return static_cast<Plant>(-1);
}

char const* plant_name(Plant plant) {
  switch (plant) {
    case Plant::None: return "none";
    case Plant::DropLoadAxis: return "drop-load-axis";
    case Plant::DropStage5: return "drop-stage5";
    case Plant::CollapseWarpK: return "collapse-warp-k";
    case Plant::BroadenClassicWarpK: return "broaden-classic-warp-k";
  }
  return "unknown";
}

int fail(Plant plant, char const* reason) {
  if (plant == Plant::None) {
    std::fprintf(stderr, "[l172] FAIL: %s\n", reason);
  } else {
    std::fprintf(stderr,
                 "[l172:red] plant=%s caught=1 reason=%s result=RED\n",
                 plant_name(plant), reason);
  }
  return 1;
}

using Key = std::tuple<int, int, int, int, int, int, int, int>;

Key key(mt::MarlinTacticPPU c) {
  return {c.tm, c.tn, c.tk, c.wm, c.wn, c.warp_k, c.stages,
          static_cast<int>(c.load)};
}

template <class T, std::size_t N>
bool axis_matches(std::set<int> const& seen, std::array<T, N> const& axis) {
  if (seen.size() != axis.size()) return false;
  for (T value : axis)
    if (!seen.count(static_cast<int>(value))) return false;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Plant plant = Plant::None;
  if (argc == 2 && !std::strncmp(argv[1], "--plant=", 8)) {
    plant = parse_plant(argv[1] + 8);
  } else if (argc != 1) {
    std::fprintf(stderr, "usage: %s [--plant=NAME]\n", argv[0]);
    return 2;
  }
  if (plant == static_cast<Plant>(-1)) {
    std::fprintf(stderr, "[l172] FAIL: unknown plant\n");
    return 2;
  }

  constexpr std::size_t reason_count =
      static_cast<std::size_t>(mt::MarlinTacticExclusionPPU::Count);
  std::array<uint64_t, reason_count> reasons{};
  std::array<uint64_t, 4> kinds{};
  std::set<Key> keys;
  std::set<int> tm, tn, tk, wm, wn, warp_k, stages, loads;
  std::set<int> admitted_tm, admitted_tn, admitted_tk, admitted_wm,
      admitted_wn, admitted_warp_k, admitted_stages, admitted_loads;
  uint64_t declared = 0;
  uint64_t admitted = 0;
  uint64_t classic = 0;
  bool saw_m8 = false;
  bool saw_m16 = false;

  mt::for_each_declared([&](mt::MarlinTacticPPU original) {
    if (plant == Plant::DropLoadAxis &&
        original.load == mt::MarlinLoadKindPPU::Aiu) {
      return;
    }
    ++declared;
    keys.insert(key(original));
    tm.insert(original.tm);
    tn.insert(original.tn);
    tk.insert(original.tk);
    wm.insert(original.wm);
    wn.insert(original.wn);
    warp_k.insert(original.warp_k);
    stages.insert(original.stages);
    loads.insert(static_cast<int>(original.load));

    mt::MarlinTacticPPU classified = original;
    if (plant == Plant::CollapseWarpK) classified.warp_k = 32;
    auto exclusion = mt::classify(classified);
    if (plant == Plant::DropStage5 && original.stages == 5 &&
        exclusion == mt::MarlinTacticExclusionPPU::None) {
      exclusion = mt::MarlinTacticExclusionPPU::PipelineDepthUnproved;
    }
    ++reasons[static_cast<std::size_t>(exclusion)];
    ++kinds[static_cast<std::size_t>(mt::exclusion_kind(exclusion))];

    bool is_classic = mt::is_classic_subspace(original);
    if (plant == Plant::BroadenClassicWarpK && original.wm == original.tm &&
        original.wn == 64 && original.warp_k == 64 &&
        original.stages == 4 &&
        original.load == mt::MarlinLoadKindPPU::CpAsync) {
      is_classic = true;
    }
    classic += is_classic ? 1 : 0;

    if (exclusion == mt::MarlinTacticExclusionPPU::None) {
      ++admitted;
      saw_m8 |= original == mt::MarlinTacticPPU{
          8, 128, 128, 8, 64, 32, 4,
          mt::MarlinLoadKindPPU::CpAsync};
      saw_m16 |= original == mt::kMarlinClassicReferencePPU;
      admitted_tm.insert(original.tm);
      admitted_tn.insert(original.tn);
      admitted_tk.insert(original.tk);
      admitted_wm.insert(original.wm);
      admitted_wn.insert(original.wn);
      admitted_warp_k.insert(original.warp_k);
      admitted_stages.insert(original.stages);
      admitted_loads.insert(static_cast<int>(original.load));
    }
  });

  if (declared != mt::cartesian_size())
    return fail(plant, "declared Cartesian count changed");
  if (keys.size() != declared)
    return fail(plant, "declared Cartesian rows are not unique");
  if (!axis_matches(tm, mt::kMarlinTileM) ||
      !axis_matches(tn, mt::kMarlinTileN) ||
      !axis_matches(tk, mt::kMarlinTileK) ||
      !axis_matches(wm, mt::kMarlinWarpM) ||
      !axis_matches(wn, mt::kMarlinWarpN) ||
      !axis_matches(warp_k, mt::kMarlinWarpK) ||
      !axis_matches(stages, mt::kMarlinStages) ||
      !axis_matches(loads, mt::kMarlinLoadKinds))
    return fail(plant, "one or more declared axes were not enumerated exactly");

  uint64_t reason_sum = 0;
  uint64_t kind_sum = 0;
  for (uint64_t value : reasons) reason_sum += value;
  for (uint64_t value : kinds) kind_sum += value;
  if (reason_sum != declared || kind_sum != declared)
    return fail(plant, "first-failure census does not close");
  if (classic != 60)
    return fail(plant, "classic subspace is not the independent 5x3x4 relation");
  if (admitted != 70 || !saw_m8 || !saw_m16)
    return fail(plant,
                "admission is not exactly the proved 70-row m8/m16 TN/TK/WN/WarpK x s2..s6 family");
  if (admitted_tm.size() != 2 || admitted_tn.size() != 3 ||
      admitted_tk.size() != 2 || admitted_wm.size() != 2 ||
      admitted_wn.size() != 2 || admitted_warp_k.size() != 2 ||
      admitted_stages.size() != 5 || admitted_loads.size() != 1)
    return fail(plant, "an unproved standalone axis became active");

  // Both broad categories must be populated.  A zero bucket here would mean
  // the census had stopped distinguishing physical/resource impossibility from
  // work that is merely not implemented yet.
  if (kinds[static_cast<std::size_t>(mt::MarlinExclusionKindPPU::HardwareOrIsa)] == 0 ||
      kinds[static_cast<std::size_t>(mt::MarlinExclusionKindPPU::ResourceLimit)] == 0 ||
      kinds[static_cast<std::size_t>(mt::MarlinExclusionKindPPU::CurrentImplementation)] == 0)
    return fail(plant, "one exclusion category is accidentally empty");

  if (plant != Plant::None)
    return fail(plant, "plant escaped every oracle invariant");

  std::printf(
      "[l172] PASS: declared=%llu unique=%zu admitted=%llu "
      "classic_subspace=%llu categories={admitted:%llu,hardware:%llu,resource:%llu,current:%llu}\n",
      static_cast<unsigned long long>(declared), keys.size(),
      static_cast<unsigned long long>(admitted),
      static_cast<unsigned long long>(classic),
      static_cast<unsigned long long>(kinds[0]),
      static_cast<unsigned long long>(kinds[1]),
      static_cast<unsigned long long>(kinds[2]),
      static_cast<unsigned long long>(kinds[3]));
  std::puts(
      "[l172] axes: TM=5 TN=3 TK=4 WM=5 WN=4 WarpK=5 stages=5 load=2; "
      "active-cardinality=2/3/2/2/2/2/5/1 family=m8,m16 geometries=7 stages=s2..s6");
  return 0;
}
