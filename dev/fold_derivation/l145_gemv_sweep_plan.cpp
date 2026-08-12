#include <cstdio>
#include <cstring>
#include <exception>

#include "benchmarks/gemv_perf_plan.hpp"

int main(int argc, char** argv) {
  using C = gemv_perf_manifest::CompiledGroup;
  using F = ppu_gemv::tactic_space::Format;
  using L = ppu_gemv::tactic_space::Layout;
  static constexpr C kFull[] = {
      {"i4-native", F::Int4, L::Native, 0}, {"i4-tileK", F::Int4, L::TileK, 256},
      {"i2-native", F::Int2, L::Native, 0}, {"i2-tileK", F::Int2, L::TileK, 256},
      {"i1-native", F::Int1, L::Native, 0}, {"i1-tileK", F::Int1, L::TileK, 256},
      {"q3-native", F::Q3, L::Native, 0},   {"q3-tileK", F::Q3, L::TileK, 256},
      {"q6-native", F::Q6, L::Native, 0},   {"q6-tileK", F::Q6, L::TileK, 256},
  };
  C const* groups = kFull;
  std::size_t count = sizeof(kFull) / sizeof(kFull[0]);
  if (argc == 2 && std::strcmp(argv[1], "i4-native") == 0) count = 1;
  else if (argc != 1) return 2;
  try {
    std::fputs(gemv_perf_plan::manifest_json(groups, count).c_str(), stdout);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
