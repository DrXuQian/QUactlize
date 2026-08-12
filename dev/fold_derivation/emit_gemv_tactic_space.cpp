#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "gemv_lowbit/gemv_tactic_space.hpp"

namespace ts = ppu_gemv::tactic_space;

namespace {

template <class F>
void for_each_candidate(F&& f) {
  for (auto const& fmt : ts::kFormats)
    for (auto const& lt : ts::kLayoutTiles)
      for (int sk : ts::kStepKs)
        for (int th : ts::kThreads) {
          for (int cm : ts::kDenseCtaMs)
            for (int cn : ts::kCtaNs)
              for (int ch : ts::kChunks)
                f(ts::Candidate{fmt.format, lt.layout, lt.tile_size_k, sk, th,
                                ts::Route::Dense, cm, cn, ch});
          for (int cm : ts::kGroupedCtaMs)
            for (int cn : ts::kCtaNs)
              for (int ch : ts::kChunks)
                f(ts::Candidate{fmt.format, lt.layout, lt.tile_size_k, sk, th,
                                ts::Route::Grouped, cm, cn, ch});
        }
}

void print_row(ts::Candidate const& c, ts::Exclusion x) {
  std::printf("ROW,%s,%s,%d,%d,%d,%s,%d,%d,%d,%d,%s\n",
              ts::name_of(c.format), ts::name_of(c.layout), c.tile_size_k,
              c.step_k, c.threads, ts::name_of(c.route), c.cta_m, c.cta_n,
              c.chunk, x == ts::Exclusion::None, ts::name_of(x));
}

}  // namespace

int main(int argc, char** argv) {
  bool const rows = argc == 2 && std::strcmp(argv[1], "--rows") == 0;
  bool const legal_rows = argc == 2 && std::strcmp(argv[1], "--legal") == 0;
  bool const units = argc == 2 && std::strcmp(argv[1], "--units") == 0;
  if (argc > 2 || (argc == 2 && !rows && !legal_rows && !units)) {
    std::fprintf(stderr, "usage: %s [--rows|--legal|--units]\n", argv[0]);
    return 2;
  }

  // A compile unit owns one (format, artifact layout, StepK, Threads, CtaN,
  // Chunk) tuple and instantiates CtaM inside that TU.  Route and CtaM remain
  // result axes, but neither changes the generated source.  Derive this view
  // from the same Candidate/static_exclusion authority as the full census;
  // CMake consumes the checked output rather than maintaining a second tier.
  if (units) {
    std::uint64_t count = 0;
    for (auto const& fmt : ts::kFormats)
      for (auto const& lt : ts::kLayoutTiles)
        for (int sk : ts::kStepKs)
          for (int th : ts::kThreads)
            for (int cn : ts::kCtaNs)
              for (int ch : ts::kChunks) {
                ts::Candidate const c{fmt.format, lt.layout, lt.tile_size_k,
                                      sk, th, ts::Route::Dense, 1, cn, ch};
                if (ts::static_exclusion(c) != ts::Exclusion::None) continue;
                std::printf("UNIT,%s,%s,%d,%d,%d,%d,%d\n",
                            ts::name_of(c.format), ts::name_of(c.layout),
                            c.tile_size_k, c.step_k, c.threads, c.cta_n, c.chunk);
                ++count;
              }
    std::printf("UNIT_CENSUS,compile_units,%llu\n",
                static_cast<unsigned long long>(count));
    std::printf("RESULT,%s\n", count == 540 ? "PASS" : "FAIL");
    return count == 540 ? 0 : 1;
  }

  std::array<std::uint64_t, 14> excluded{};
  std::uint64_t total = 0, legal = 0;
  for_each_candidate([&](ts::Candidate const& c) {
    ++total;
    auto const x = ts::static_exclusion(c);
    if (x == ts::Exclusion::None) ++legal;
    else ++excluded[static_cast<unsigned>(x)];
    if (rows || (legal_rows && x == ts::Exclusion::None)) print_row(c, x);
  });

  std::printf("AXIS,formats,5\nAXIS,layout_tile,2\nAXIS,step_k,3\nAXIS,threads,3\n");
  std::printf("AXIS,dense_cta_m,15\nAXIS,grouped_cta_m,4\nAXIS,cta_n,4\nAXIS,chunk,4\n");
  std::printf("CENSUS,total,%llu\nCENSUS,legal,%llu\nCENSUS,rejected,%llu\n",
              static_cast<unsigned long long>(total), static_cast<unsigned long long>(legal),
              static_cast<unsigned long long>(total - legal));
  for (unsigned i = 1; i < excluded.size(); ++i)
    if (excluded[i])
      std::printf("EXCLUSION,%s,%llu\n", ts::name_of(static_cast<ts::Exclusion>(i)),
                  static_cast<unsigned long long>(excluded[i]));

  for (auto const& a : ts::kShippingAnchors)
    std::printf("ANCHOR,%s,%d,%d,native,0,%d,%d,8,2\n", ts::name_of(a.format), a.qtype,
                a.group_size, a.step_k, a.threads);
  std::printf("ANCHOR,int1,NOT_SHIPPED\n");

  // One shape census proves that applicability is not folded into static
  // generation: all statically legal rows remain candidates, while TileK256
  // rows reject a K=96 problem at the shape layer.
  ts::Problem const shape{ts::Route::Dense, 1, 64, 96, 32};
  std::uint64_t shape_legal = 0;
  std::array<std::uint64_t, 10> shape_excluded{};
  for_each_candidate([&](ts::Candidate const& c) {
    if (ts::static_exclusion(c) != ts::Exclusion::None) return;
    auto const x = ts::shape_exclusion(c, shape);
    if (x == ts::ShapeExclusion::None) ++shape_legal;
    else ++shape_excluded[static_cast<unsigned>(x)];
  });
  std::printf("SHAPE,dense-m1-n64-k96-gs32,legal,%llu\n",
              static_cast<unsigned long long>(shape_legal));
  for (unsigned i = 1; i < shape_excluded.size(); ++i)
    if (shape_excluded[i])
      std::printf("SHAPE_EXCLUSION,%s,%llu\n",
                  ts::name_of(static_cast<ts::ShapeExclusion>(i)),
                  static_cast<unsigned long long>(shape_excluded[i]));

  bool const conserved = total == ts::cartesian_size() && legal + (total - legal) == total;
  std::printf("RESULT,%s\n", conserved ? "PASS" : "FAIL");
  return conserved ? 0 : 1;
}
