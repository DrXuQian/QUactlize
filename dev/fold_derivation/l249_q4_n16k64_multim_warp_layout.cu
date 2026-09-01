// L249 -- close the WarpOnM>1 coordinate seam for the experimental Q4
// N16xK64 AIU-plain + UniversalCopy delivery path.
//
// L242/L244 prove the adapter and the complete raw-word-to-MMA map when the
// CTA has one warp on M.  This oracle deliberately uses two M warps and walks
// every physical thread through the real TiledMMA ThrLayoutVMNK.  The B map is
// expected to repeat exactly once per M warp while each warp selects shared B
// through the semantic VMNK N coordinate.  The physical shared row remains
// the complete CTA pitch, 2*TileN u32, never the local 2*WarpN pitch.

#if defined(L249_COMPILER_PROBE)

#include <cuda_fp16.h>

__global__ void l249_compiler_probe(half const* x, half* y) {
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

constexpr int kWarpOnM = 2;
constexpr int kTM = 8 * kWarpOnM;
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
  int vmnk_bad = 0;
  int threads = 0;
  int warps = 0;
  int warp_n_tiles = 0;
  int row_pitch_words = 0;
};

template <int WN>
Metrics prove_geometry() {
  using Adapter = s2r::Q4N16K64UniversalReader<kTN, WN, kTK>;
  using Physical = typename Adapter::Physical;
  using WarpOnN = Int<kTN / WN>;
  using Compute = TiledMMA<
      MMA_Atom<PPU0010_8x16x16_F32F16F16F32_TN>,
      Layout<Shape<Int<kWarpOnM>, WarpOnN, _1>>,
      Tile<Int<kTM>, Int<(kTN / WN) * 16>, Int<kTK>>>;
  using ThrLayoutVMNK = typename Compute::ThrLayoutVMNK;

  static_assert(int(size<1>(ThrLayoutVMNK{})) == kWarpOnM);
  static_assert(int(size<2>(ThrLayoutVMNK{})) == Adapter::warp_n_tiles);
  static_assert(Physical::physical_n_words == 2 * kTN,
                "direct Q4 shared rows use the complete CTA TileN pitch");

  constexpr int kThreads = int(size(ThrLayoutVMNK{}));
  constexpr int kWarps = kThreads / Adapter::reader_threads;
  Metrics m{};
  m.threads = kThreads;
  m.warps = kWarps;
  m.warp_n_tiles = Adapter::warp_n_tiles;
  m.row_pitch_words = Physical::physical_n_words;

  std::array<int, kCodes> logical_hits{};
  std::array<int, kCodes> physical_hits{};
  std::array<std::uint32_t, Physical::physical_words> stage{};

  auto source_identity =
      make_identity_tensor(typename Adapter::SharedSourceShape{});
  typename Adapter::SharedSourceLayout source_layout{};
  auto logical_identity = make_identity_tensor(Shape<Int<kTN>, Int<kTK>>{});
  auto logical_smem = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<kTN>, Int<kTK>>{},
                  Stride<Int<kTK>, _1>{}));
  auto compute = Compute{};
  auto warp_layout = compute.get_thr_layout_vmnk();

  for (int thread = 0; thread < kThreads; ++thread) {
    auto const vmnk = warp_layout.get_flat_coord(thread);
    int const lane = int(get<0>(vmnk));
    int const warp_m = int(get<1>(vmnk));
    int const warp_n = int(get<2>(vmnk));
    int const warp_k = int(get<3>(vmnk));
    int const physical_warp = thread / Adapter::reader_threads;
    int const roundtrip = int(warp_layout(make_coord(
        get<0>(vmnk), get<1>(vmnk), get<2>(vmnk), get<3>(vmnk))));

    m.vmnk_bad += roundtrip != thread;
    m.vmnk_bad += lane != thread % Adapter::reader_threads;
    m.vmnk_bad += warp_m < 0 || warp_m >= kWarpOnM;
    m.vmnk_bad += warp_n < 0 || warp_n >= Adapter::warp_n_tiles;
    m.vmnk_bad += warp_k != 0;

    int source_warp_n = warp_n;
#if defined(L249_PLANT_PHYSICAL_WARP_N) && L249_PLANT_PHYSICAL_WARP_N
    // RED: with WarpOnM>1 the physical warp id is not the semantic N
    // coordinate.  Modulo keeps the address in range, making this a silent
    // coordinate corruption instead of an obvious out-of-bounds access.
    source_warp_n = physical_warp % Adapter::warp_n_tiles;
#endif
    m.vmnk_bad += source_warp_n != warp_n;

    auto source = Adapter::make_shared_source(
        make_smem_ptr(stage.data()), source_warp_n);
    auto source_partition =
        Adapter::make_source_partition(source_identity, lane);
    auto source_copy_view =
        Adapter::make_copy_source_view(source_partition);
    auto source_data_partition =
        Adapter::make_source_partition(source, lane);

#if defined(L249_PLANT_RVALUE_OWNER) && L249_PLANT_RVALUE_OWNER
    // RED: copy_view is an alias and must not escape from an unnamed owner.
    auto dangling_copy_view = Adapter::make_copy_view(
        Adapter::make_register_owner(source_data_partition), lane);
    (void)dangling_copy_view;
#endif

    auto register_owner = Adapter::make_register_owner(source_data_partition);
    auto copy_view = Adapter::make_copy_view(register_owner, lane);
    CUTE_STATIC_ASSERT_V(shape(source_copy_view) == shape(copy_view));

    auto thr_mma = compute.get_thread_slice(thread);
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

      // Make the physical-input rest stride observably different.  The good
      // adapter borrows only converter mode 0 and roots all rest strides in
      // logical_owner.  A destination that borrows the input layout turns red.
      auto adversarial_n_stride = compact_col_major(
          shape<1>(converter_input.layout()), Int<777>{});
      auto adversarial_layout = make_layout(
          shape(converter_input.layout()),
          make_stride(stride<0>(converter_input.layout()),
                      adversarial_n_stride));
      auto adversarial_input = make_tensor(
          converter_input.data(), adversarial_layout);

#if defined(L249_PLANT_INPUT_OWNED_DESTINATION) && \
    L249_PLANT_INPUT_OWNED_DESTINATION
      auto destination = make_tensor(
          logical_owner(_, _,
                        k_block * Int<Adapter::k_atoms_per_copy>{}).data(),
          adversarial_input.layout());
#else
      auto destination = Adapter::make_converter_destination(
          logical_owner, adversarial_input, k_block,
          Int<Adapter::k_atoms_per_copy>{});
#endif

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
              source_warp_n * Adapter::warp_n_base_words +
              int(source_layout(source_coord));
#if defined(L249_PLANT_WARP_N_PITCH) && L249_PLANT_WARP_N_PITCH
          int const local_n_word = int(get<0>(get<0>(source_coord)));
          int const local_k_row = int(get<1>(get<0>(source_coord)));
          int const n_cohort = int(get<1>(source_coord));
          int const kb = int(get<2>(source_coord));
          source_word =
              source_warp_n * Adapter::warp_n_base_words +
              local_k_row * (2 * WN) + local_n_word +
              n_cohort * Adapter::n_cohort_stride_words +
              kb * (4 * 2 * WN);
#endif

          for (int code = 0; code < 8; ++code) {
            int const physical = 8 * source_word + code;
            int const emitted =
                cutlass::MixGemmEmit<4>::index(code, vreg);
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
            m.map_exact += physical == static_cast<int>(wanted);
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

  for (int hit : logical_hits) m.logical_bad += hit != kWarpOnM;
  for (int hit : physical_hits) m.physical_bad += hit != kWarpOnM;
  return m;
}

bool exact(Metrics const& m) {
  return m.total == kWarpOnM * kCodes &&
         m.map_exact == kWarpOnM * kCodes &&
         m.source_bad == 0 && m.destination_bad == 0 &&
         m.logical_bad == 0 && m.physical_bad == 0 &&
         m.vmnk_bad == 0;
}

template <int WN>
bool report() {
  auto const m = prove_geometry<WN>();
  bool const ok = exact(m);
  std::printf(
      "L249 MULTIM WOM=%d WN=%d threads=%d warps=%d warp_n_tiles=%d "
      "row_pitch_words=%d map=%d/%d source_bad=%d destination_bad=%d "
      "logical_bad=%d physical_bad=%d vmnk_bad=%d result=%s\n",
      kWarpOnM, WN, m.threads, m.warps, m.warp_n_tiles,
      m.row_pitch_words, m.map_exact, m.total, m.source_bad,
      m.destination_bad, m.logical_bad, m.physical_bad, m.vmnk_bad,
      ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  bool ok = report<16>();
  ok &= report<32>();
  std::printf(
      "L249 Q4_N16K64_MULTIM_WARP %s shape=%dx%dx%d WOM=2 WN=16,32 "
      "physical_row_words=%d coverage=2x%d reds=4\n",
      ok ? "PASS" : "FAIL", kTM, kTN, kTK, 2 * kTN, kCodes);
  return ok ? 0 : 1;
}

#endif
