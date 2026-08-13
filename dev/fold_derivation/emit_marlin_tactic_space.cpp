// Emit the standalone Marlin tactic domain from its one host-readable
// authority.  With no arguments this prints a compact census; --all prints
// every declared row and its first named exclusion.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "marlin_tactic_space_ppu.hpp"

namespace mt = marlin_tactics_ppu;

namespace {

template <class T, std::size_t N>
void print_int_axis(char const* name, std::array<T, N> const& values) {
  std::printf("axis.%s=", name);
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) std::putchar(',');
    std::printf("%d", int(values[i]));
  }
  std::putchar('\n');
}

void print_tactic(mt::MarlinTacticPPU c, mt::MarlinTacticExclusionPPU e) {
  std::printf(
      "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%d\t%d\t%lld\t%d\t%s\t%s\n",
      c.tm, c.tn, c.tk, c.wm, c.wn, c.warp_k, c.stages,
      mt::load_kind_name(c.load), mt::cta_warps(c),
      mt::warp_k_cohorts(c), static_cast<long long>(mt::shared_bytes(c)),
      mt::is_classic_subspace(c) ? 1 : 0, mt::exclusion_kind_name(mt::exclusion_kind(e)),
      mt::exclusion_name(e));
}

}  // namespace

int main(int argc, char** argv) {
  bool const all = argc == 2 && std::strcmp(argv[1], "--all") == 0;
  if (argc > 2 || (argc == 2 && !all)) {
    std::fprintf(stderr, "usage: %s [--all]\n", argv[0]);
    return 2;
  }

  constexpr auto reason_count =
      static_cast<std::size_t>(mt::MarlinTacticExclusionPPU::Count);
  std::array<uint64_t, reason_count> reasons{};
  std::array<uint64_t, 4> kinds{};
  uint64_t declared = 0;
  uint64_t classic = 0;

  if (all) {
    std::puts(
        "tm\ttn\ttk\twm\twn\twarp_k\tstages\tload\tcta_warps\t"
        "k_cohorts\tsmem_bytes\tclassic_subspace\tkind\texclusion");
  }
  mt::for_each_declared([&](mt::MarlinTacticPPU c) {
    auto const exclusion = mt::classify(c);
    ++declared;
    ++reasons[static_cast<std::size_t>(exclusion)];
    ++kinds[static_cast<std::size_t>(mt::exclusion_kind(exclusion))];
    classic += mt::is_classic_subspace(c) ? 1 : 0;
    if (all) print_tactic(c, exclusion);
  });

  if (all) return 0;

  std::printf(
      "schema=marlin-tactic-space-ppu-v1 declared=%llu admitted=%llu "
      "classic_subspace=%llu\n",
      static_cast<unsigned long long>(declared),
      static_cast<unsigned long long>(
          reasons[static_cast<std::size_t>(mt::MarlinTacticExclusionPPU::None)]),
      static_cast<unsigned long long>(classic));
  print_int_axis("tm", mt::kMarlinTileM);
  print_int_axis("tn", mt::kMarlinTileN);
  print_int_axis("tk", mt::kMarlinTileK);
  print_int_axis("wm", mt::kMarlinWarpM);
  print_int_axis("wn", mt::kMarlinWarpN);
  print_int_axis("warp_k", mt::kMarlinWarpK);
  print_int_axis("stages", mt::kMarlinStages);
  std::printf("axis.load=%s,%s\n",
              mt::load_kind_name(mt::kMarlinLoadKinds[0]),
              mt::load_kind_name(mt::kMarlinLoadKinds[1]));
  for (std::size_t i = 0; i < reasons.size(); ++i) {
    auto const exclusion = static_cast<mt::MarlinTacticExclusionPPU>(i);
    std::printf("reason.%s=%llu\n", mt::exclusion_name(exclusion),
                static_cast<unsigned long long>(reasons[i]));
  }
  std::printf(
      "kind.ADMITTED=%llu kind.HARDWARE_OR_ISA=%llu "
      "kind.RESOURCE_LIMIT=%llu kind.CURRENT_IMPLEMENTATION=%llu\n",
      static_cast<unsigned long long>(kinds[0]),
      static_cast<unsigned long long>(kinds[1]),
      static_cast<unsigned long long>(kinds[2]),
      static_cast<unsigned long long>(kinds[3]));
  std::puts("admitted=16,128,128,16,64,32,4,cp_async");
  return 0;
}
