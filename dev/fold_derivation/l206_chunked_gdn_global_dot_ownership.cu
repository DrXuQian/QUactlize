// L206 -- exhaust the exact production PPU global-dot accumulator map.
//
// This source is compiled locally with __HGGCCC__ defined so the shipping
// collective exposes the very same CollectiveBuilder::TiledMma that hgcc
// selects.  No copied lane formula participates in the proof: all 64x64
// coordinates come from get_thread_slice(t).partition_C(identity).

#include <array>
#include <cstdio>
#include <type_traits>

#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_collective.cuh"

#if !defined(__HGGCCC__)
#error "L206 must instantiate the production __HGGCCC__ CollectiveBuilder branch"
#endif

// nvcc is only the local two-pass front end.  The oracle itself is host-only;
// suppressing it in the synthetic device pass avoids asking nvcc to type-check
// PPU inline assembly while the host pass still instantiates the exact builder
// type and executes its constexpr layout algebra.
#if !defined(__CUDA_ARCH__)
namespace {

using Mainloop =
    cutlass::linear_attention::detail::PpuChunkedGdnGlobalDotMainloop;
using ProductionMma = typename Mainloop::TiledMma;

static_assert(cute::size(ProductionMma{}) == 128,
              "L206 production global-dot TiledMma must have 128 threads");
static_assert(cute::size<0>(typename ProductionMma::AtomShape_MNK{}) == 16 &&
                  cute::size<1>(typename ProductionMma::AtomShape_MNK{}) == 16 &&
                  cute::size<2>(typename ProductionMma::AtomShape_MNK{}) == 16,
              "L206 must enumerate the production m16n16k16 atom");
using ProductionFragment = decltype(cute::partition_fragment_C(
    ProductionMma{}, cute::Shape<cute::_64, cute::_64>{}));
static_assert(cute::size(ProductionFragment{}) == 32,
              "L206 production store loop must consume 32 accumulators/thread");
static_assert(cute::size(ProductionMma{}) * cute::size(ProductionFragment{}) ==
                  64 * 64,
              "L206 thread/fragment cardinality must equal the full output tile");

struct Ownership {
  int visits = 0;
  int out_of_bounds = 0;
  int holes = 0;
  int duplicate_coordinates = 0;
  int duplicate_visits = 0;
  int min_visits = 0;
  int max_visits = 0;

  constexpr bool clean() const {
    return visits == 64 * 64 && out_of_bounds == 0 && holes == 0 &&
           duplicate_coordinates == 0 && duplicate_visits == 0 &&
           min_visits == 1 && max_visits == 1;
  }
};

// CoordinateStride is deliberately applied after asking the real TiledMma
// for its coordinate.  The production case is 1.  A value of 2 is the planted
// "same shape, wrong destination stride" bug: even columns get two owners and
// odd columns get none.
template <int Threads, int CoordinateStride>
constexpr Ownership enumerate() {
  std::array<int, 64 * 64> owners{};
  Ownership result{};
  auto identity =
      cute::make_identity_tensor(cute::Shape<cute::_64, cute::_64>{});

  for (int thread = 0; thread < Threads; ++thread) {
    auto part = ProductionMma{}.get_thread_slice(thread).partition_C(identity);
    for (int slot = 0; slot < int(cute::size(part)); ++slot) {
      auto coordinate = part(slot);
      int const row = int(cute::get<0>(coordinate));
      int const column = int(cute::get<1>(coordinate));
      if (row < 0 || row >= 64 || column < 0 || column >= 64) {
        ++result.out_of_bounds;
        continue;
      }
      int const mapped_column = (column * CoordinateStride) % 64;
      ++owners[std::size_t(row * 64 + mapped_column)];
      ++result.visits;
    }
  }

  result.min_visits = result.visits == 0 ? 0 : result.visits;
  for (int count : owners) {
    result.holes += count == 0;
    result.duplicate_coordinates += count > 1;
    result.duplicate_visits += count > 1 ? count - 1 : 0;
    result.min_visits = count < result.min_visits ? count : result.min_visits;
    result.max_visits = count > result.max_visits ? count : result.max_visits;
  }
  return result;
}

#if defined(L206_PLANT_THREAD_COUNT)
Ownership selected() { return enumerate<64, 1>(); }
constexpr int kSelectedThreads = 64;
constexpr int kSelectedCoordinateStride = 1;
#elif defined(L206_PLANT_COORDINATE_STRIDE)
Ownership selected() { return enumerate<128, 2>(); }
constexpr int kSelectedThreads = 128;
constexpr int kSelectedCoordinateStride = 2;
#else
Ownership selected() { return enumerate<128, 1>(); }
constexpr int kSelectedThreads = 128;
constexpr int kSelectedCoordinateStride = 1;
#endif

}  // namespace

int main() {
  Ownership const chosen = selected();
  Ownership const wrong_threads = enumerate<64, 1>();
  Ownership const wrong_stride = enumerate<128, 2>();
  bool const negative_signatures =
      wrong_threads.visits == 2048 && wrong_threads.holes == 2048 &&
      wrong_threads.duplicate_coordinates == 0 &&
      wrong_stride.visits == 4096 && wrong_stride.holes == 2048 &&
      wrong_stride.duplicate_coordinates == 2048 &&
      wrong_stride.duplicate_visits == 2048;
  bool const ok = chosen.clean() && negative_signatures;
  std::printf(
      "[l206] %s: source=__HGGCCC__-CollectiveBuilder::TiledMma "
      "threads=%d coordinate_stride=%d tile=64x64 visits=%d holes=%d "
      "duplicate_coordinates=%d "
      "duplicate_visits=%d oob=%d min=%d max=%d\n",
      ok ? "PASS" : "FAIL", kSelectedThreads,
      kSelectedCoordinateStride, chosen.visits, chosen.holes,
      chosen.duplicate_coordinates, chosen.duplicate_visits,
      chosen.out_of_bounds, chosen.min_visits, chosen.max_visits);
  std::printf(
      "[l206 negatives] threads64=(visits=%d holes=%d duplicates=%d) "
      "stride2=(visits=%d holes=%d duplicates=%d duplicate_visits=%d)\n",
      wrong_threads.visits, wrong_threads.holes,
      wrong_threads.duplicate_coordinates, wrong_stride.visits,
      wrong_stride.holes, wrong_stride.duplicate_coordinates,
      wrong_stride.duplicate_visits);
  return ok ? 0 : 1;
}
#endif
