// L244 -- compose the complete direct Q4 reader semantic chain at the
// production-sized TN64 x TK256 tile for WN16/WN32/WN64.
//
// Unlike the atom, offline and adapter gates in isolation, this oracle maps
// every source nibble through the actual multi-warp TiledMMA coordinate, the
// Q4N16K64UniversalReader source/copy views, MixGemmEmit and the adapter's
// compute-owned converter destination.  The resulting physical address must
// equal q4_n16k64_direct::physical_nibble for all 16384 logical codes.

#if defined(L244_COMPILER_PROBE)

#include <cuda_fp16.h>

__global__ void l244_compiler_probe(half const* x, half* y) {
  int const i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i == 0) y[0] = __hadd(x[0], x[0]);
}

int main() { return 0; }

#else

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "actlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_b_s2r_adapter.hpp"
#include "q4_n16k64_direct_offline.hpp"

namespace {
using namespace cute;
namespace s2r =
    cutlass::gemm::collective::detail::quactlize_b_s2r;

constexpr int kTN = 64;
constexpr int kTK = 256;
constexpr int kCodes = kTN * kTK;

struct Metrics {
  int map_exact = 0;
  int total = 0;
  int source_bad = 0;
  int destination_bad = 0;
  int logical_bad = 0;
  int physical_bad = 0;
  int warp_n_bad = 0;
};

template <int WN>
Metrics prove_geometry() {
  using Adapter = s2r::Q4N16K64UniversalReader<kTN, WN, kTK>;
  using WarpOnN = Int<kTN / WN>;
  using Compute = TiledMMA<
      MMA_Atom<PPU0010_8x16x16_F32F16F16F32_TN>,
      Layout<Shape<_1, WarpOnN, _1>>,
      Tile<_8, Int<(kTN / WN) * 16>, Int<kTK>>>;

  Metrics m{};
  std::array<int, kCodes> logical_hits{};
  std::array<int, kCodes> physical_hits{};
  std::array<std::uint32_t, kCodes / 8> stage{};

  auto source_identity =
      make_identity_tensor(typename Adapter::SharedSourceShape{});
  typename Adapter::SharedSourceLayout source_layout{};
  auto logical_identity = make_identity_tensor(Shape<Int<kTN>, Int<kTK>>{});
  auto logical_smem = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<kTN>, Int<kTK>>{},
                  Stride<Int<kTK>, _1>{}));
  auto warp_layout = Compute{}.get_thr_layout_vmnk();

  for (int warp_n = 0; warp_n < Adapter::warp_n_tiles; ++warp_n) {
    int const warp_thread0 =
        int(warp_layout(make_coord(0, 0, warp_n, 0)));
    m.warp_n_bad += warp_thread0 % 32 != 0;
    m.warp_n_bad += warp_thread0 / 32 != warp_n;

    auto source = Adapter::make_shared_source(
        make_smem_ptr(stage.data()), warp_n);
    for (int lane = 0; lane < Adapter::reader_threads; ++lane) {
      int const thread = warp_thread0 + lane;
      auto source_partition =
          Adapter::make_source_partition(source_identity, lane);
      auto source_copy_view =
          Adapter::make_copy_source_view(source_partition);

      auto source_data_partition =
          Adapter::make_source_partition(source, lane);
      auto register_owner =
          Adapter::make_register_owner(source_data_partition);
      auto copy_view = Adapter::make_copy_view(register_owner, lane);
      CUTE_STATIC_ASSERT_V(shape(source_copy_view) == shape(copy_view));

      auto thr_mma = Compute{}.get_thread_slice(thread);
      auto logical_owner = thr_mma.partition_fragment_B(logical_smem);
      auto logical_partition = thr_mma.partition_B(logical_identity);
      auto logical_inverse = right_inverse(logical_owner.layout());
      CUTE_STATIC_ASSERT_V(size<2>(logical_owner) ==
                           Int<Adapter::k_blocks *
                               Adapter::k_atoms_per_copy>{});

      cute::for_each(cute::make_int_sequence<Adapter::k_blocks>{},
          [&] (auto k_block) {
        auto converter_input =
            Adapter::make_converter_input(copy_view, k_block);
        auto destination = Adapter::make_converter_destination(
            logical_owner, converter_input, k_block,
            Int<Adapter::k_atoms_per_copy>{});
        int const destination_base = int(logical_owner.layout()(
            0, 0, k_block * Int<Adapter::k_atoms_per_copy>{}));
        int const iterations = int(size<1>(converter_input));

        for (int ii = 0; ii < iterations; ++ii) {
          int const converter_word_base =
              int(copy_view.layout()(0, ii, k_block));
          std::array<int, 4> vreg_hits{};
          for (int source_v = 0; source_v < 4; ++source_v) {
            int const register_word =
                int(copy_view.layout()(source_v, ii, k_block));
            int const vreg = register_word - converter_word_base;
            if (vreg < 0 || vreg >= 4) {
              ++m.source_bad;
              continue;
            }
            ++vreg_hits[std::size_t(vreg)];

            auto const source_coord =
                source_copy_view(source_v, ii, k_block);
            int source_word =
                warp_n * Adapter::warp_n_base_words +
                int(source_layout(source_coord));
#if defined(L244_PLANT_WN_PITCH) && L244_PLANT_WN_PITCH
            int const local_n_word = int(get<0>(get<0>(source_coord)));
            int const local_k_row = int(get<1>(get<0>(source_coord)));
            int const n_cohort = int(get<1>(source_coord));
            int const kb = int(get<2>(source_coord));
            source_word =
                warp_n * Adapter::warp_n_base_words +
                local_k_row * (2 * WN) + local_n_word +
                n_cohort * Adapter::n_cohort_stride_words +
                kb * (4 * 2 * WN);
#elif defined(L244_PLANT_WARP_BASE) && L244_PLANT_WARP_BASE
            source_word -= warp_n * Adapter::warp_n_base_words;
#endif

            for (int code = 0; code < 8; ++code) {
              int const physical = 8 * source_word + code;
              int const emitted =
                  cutlass::MixGemmEmit<4>::index(code, vreg);
              // convert_tensor reinterprets destination(_, ii).data() as a
              // contiguous 32-half DstArray.  MixGemmEmit is that raw array
              // offset; the CuTe mode-0 stride describes logical ownership,
              // but is deliberately not applied by the converter store.
              int const raw = destination_base +
                  int(destination.layout()(0, ii)) + emitted;
              if (raw < 0 || raw >= int(size(logical_owner))) {
                ++m.destination_bad;
                continue;
              }
              auto const nk = logical_partition(logical_inverse(raw));
              int const n = int(get<0>(nk));
              int const k = int(get<1>(nk));
              if (n < 0 || n >= kTN || k < 0 || k >= kTK) {
                ++m.destination_bad;
                continue;
              }
              int const logical = n * kTK + k;
              std::size_t const wanted =
                  q4_n16k64_direct::physical_nibble(n, k, kTN);
              m.map_exact +=
                  physical == static_cast<int>(wanted);
              ++m.total;
              ++logical_hits[std::size_t(logical)];
              if (physical < 0 || physical >= kCodes)
                ++m.source_bad;
              else
                ++physical_hits[std::size_t(physical)];
            }
          }
          for (int hit : vreg_hits) m.source_bad += hit != 1;
        }
      });
    }
  }

  for (int hit : logical_hits) m.logical_bad += hit != 1;
  for (int hit : physical_hits) m.physical_bad += hit != 1;
  return m;
}

bool exact(Metrics const& m) {
  return m.total == kCodes && m.map_exact == kCodes &&
         m.source_bad == 0 && m.destination_bad == 0 &&
         m.logical_bad == 0 && m.physical_bad == 0 &&
         m.warp_n_bad == 0;
}

template <int WN>
bool report() {
  auto const m = prove_geometry<WN>();
  bool const ok = exact(m);
  std::printf(
      "L244 CROSS WN=%d map=%d/%d source_bad=%d destination_bad=%d "
      "logical_bad=%d physical_bad=%d warp_n_bad=%d result=%s\n",
      WN, m.map_exact, m.total, m.source_bad, m.destination_bad,
      m.logical_bad, m.physical_bad, m.warp_n_bad,
      ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  bool ok = report<16>();
  ok &= report<32>();
  ok &= report<64>();
  std::printf(
      "L244 Q4_N16K64_CROSS_SEMANTIC %s shape=64x256 "
      "WN=16,32,64 coverage=16384/16384 reds=2\n",
      ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

#endif
