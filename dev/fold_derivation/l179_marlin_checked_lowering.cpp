// L179 -- host-side arithmetic and output-domain proof for the standalone
// Marlin assume-valid device path.
//
// Arguments are rejected before lowering if any int address expression used
// by CtaState/SegmentState can overflow.  Once accepted, the scheduler's
// global q and the fixed 64-thread output cohort make col<N redundant.  This
// oracle exhausts the production output-map helpers and exact arithmetic
// boundaries; it does not execute a device kernel.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_fp16.h>
#include "cutlass/util/packed_stride.hpp"

__half2 l179_unreachable_hfma2(__half2, __half2, __half2);
unsigned int l179_unreachable_cvta(void const*);
struct L179UnreachableThreadIdx { int x = 0, y = 0, z = 0; };
inline constexpr L179UnreachableThreadIdx l179_unreachable_thread_idx{};
void l179_unreachable_syncthreads();
#define __hfma2 l179_unreachable_hfma2
#define __cvta_generic_to_shared l179_unreachable_cvta
#define threadIdx l179_unreachable_thread_idx
#define __syncthreads l179_unreachable_syncthreads
#include "quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/marlin_output_map_ppu.hpp"
#undef __syncthreads
#undef threadIdx
#undef __cvta_generic_to_shared
#undef __hfma2

using Main = cutlass::gemm::collective::MarlinCollectivePPU<
    cute::Shape<cute::_16, cute::_128, cute::_128>,
    cute::Shape<cute::_16, cute::_64, cute::_32>, 4, 128,
    cute::Stride<int64_t, cute::_1, int64_t>,
    cute::Stride<int64_t, cute::_1, int64_t>,
    cute::Stride<cute::_1, int64_t, int64_t>>;
using Vector128 = cutlass::gemm::collective::marlin_ppu_detail::Vector128;

namespace {

char const* plant_name(int argc, char** argv) {
  constexpr char prefix[] = "--plant=";
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], prefix, sizeof(prefix) - 1) == 0) {
      return argv[i] + sizeof(prefix) - 1;
    }
  }
  return "none";
}

bool is_plant(char const* plant, char const* name) {
  return std::strcmp(plant, name) == 0;
}

int fail(char const* plant, char const* reason) {
  std::fprintf(stderr,
               "[l179:red] plant=%s caught=1 reason=%s result=RED\n",
               plant, reason);
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  char const* plant = plant_name(argc, argv);
  std::vector<Vector128> a(8192), b(524288), scale(16624);
  Main::Arguments args{
      reinterpret_cast<cutlass::half_t const*>(a.data()),
      reinterpret_cast<cutlass::int4b_t const*>(b.data()),
      reinterpret_cast<cutlass::half_t const*>(scale.data()), 128};

  auto const fixed = cute::make_shape(1, 4096, 4096, 1);
  using PackedD = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
  auto const packed_d = cutlass::make_cute_packed_stride(
      PackedD{}, cute::make_shape(1, 4096, 1));
  if (int64_t(cute::get<0>(packed_d)) != 4096 ||
      int64_t(cute::get<1>(packed_d)) != 1 ||
      int64_t(cute::get<2>(packed_d)) != 0) {
    return fail(plant, "L=1 packed D stride is not (N,1,0)");
  }
  if (!Main::can_implement(fixed, args) ||
      !Main::address_arithmetic_supported(fixed)) {
    return fail(plant, "fixed production shape was rejected");
  }
  // These are the exact adjacent admitted/rejected multiples for the two int
  // products that dominate B rebasing.  They call production can_implement;
  // L179's overlay controls delete one real guard at a time and require the
  // corresponding first-invalid value to survive.
  auto const n_product_pass = cute::make_shape(1, 17318400, 4096, 1);
  auto const n_product_fail = cute::make_shape(1, 17318656, 4096, 1);
  auto const k_product_pass = cute::make_shape(1, 4096, 16777216, 1);
  auto const k_product_fail = cute::make_shape(1, 4096, 16777344, 1);
  auto const b_delta_pass = cute::make_shape(1, 536870656, 128, 1);
  auto const b_delta_fail = cute::make_shape(1, 536870912, 128, 1);
  bool const n_pass = Main::can_implement(n_product_pass, args);
  bool const n_fail = Main::can_implement(n_product_fail, args);
  bool const k_pass = Main::can_implement(k_product_pass, args);
  bool const k_fail = Main::can_implement(k_product_fail, args);
  bool const delta_pass = Main::can_implement(b_delta_pass, args);
  bool const delta_fail = Main::can_implement(b_delta_fail, args);
  if (is_plant(plant, "drop-b-k-product")) {
    if (n_pass && n_fail && k_pass && k_fail) {
      return fail(plant, "production b_k_delta*K boundary was removed");
    }
    return fail(plant, "production b_k_delta*K plant missed its boundary");
  }
  if (is_plant(plant, "drop-b-delta-materialization")) {
    if (delta_pass && delta_fail) {
      return fail(plant, "production b_k_delta materialization boundary was removed");
    }
    return fail(plant, "production b_k_delta materialization plant missed its boundary");
  }
  if (!n_pass || n_fail || !k_pass || k_fail || !delta_pass || delta_fail) {
    return fail(plant, "production overflow boundary pair drifted");
  }

  constexpr int kQ = 32;
  constexpr int kN = kQ * Main::TileN;
  uint64_t coordinates = 0;
  for (int q = 0; q < kQ; ++q) {
    std::array<uint8_t, Main::TileM * Main::TileN> seen{};
    for (int tid = 0; tid < 64; ++tid) {
      int const lane = tid % 32;
      for (int n_block = 0; n_block < 4; ++n_block) {
        int const n_base =
            cutlass::gemm::kernel::marlin_ppu_detail::output_n_base(
                q, tid, n_block);
        for (int value = 0; value < 8; ++value) {
          int const row =
              cutlass::gemm::kernel::marlin_ppu_detail::output_row(
                  lane, value);
          int col = n_base +
                    cutlass::gemm::kernel::marlin_ppu_detail::output_col_offset(
                        lane, value);
          if (is_plant(plant, "col-plus-one") && q == kQ - 1 &&
              tid == 63 && n_block == 3 && value == 7) {
            ++col;
          }
          int const local_col = col - q * Main::TileN;
          if (row < 0 || row >= Main::TileM || local_col < 0 ||
              local_col >= Main::TileN || col < 0 || col >= kN) {
            return fail(plant, "output cohort escaped its global q tile");
          }
          int const cell = row * Main::TileN + local_col;
          if (++seen[cell] != 1) {
            return fail(plant, "output cohort duplicated a tile coordinate");
          }
          ++coordinates;
        }
      }
    }
    for (uint8_t visits : seen) {
      if (visits != 1) {
        return fail(plant, "output cohort left a tile-coordinate hole");
      }
    }
  }
  if (coordinates != uint64_t(kQ) * Main::TileM * Main::TileN) {
    return fail(plant, "output-coordinate census drifted");
  }
  if (!is_plant(plant, "none")) {
    return fail(plant, "named coordinate plant missed its invariant");
  }
  std::printf(
      "[l179] PASS: fixed-shape=accepted overflow-boundaries=3/3 "
      "packed-D=(4096,1,0) output-coordinates=%llu q=32 "
      "range=[0,4095] exact-once=1\n",
      static_cast<unsigned long long>(coordinates));
  return 0;
}
