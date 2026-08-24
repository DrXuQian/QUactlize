// L222 -- exact CuTe ownership oracle for the diagnostic direct accumulator store.
//
// The device bisection bypasses EpilogueParallel after the shipping mainloop
// and writes its register fragment through TiledMma::partition_C.  This host
// oracle binds that store to the exact TM8/TN64/WM8/WN16 m8 tactic, proves
// exact-once ownership for full and M=1 residue tiles, and uses coordinate tags
// so a wrong register-to-coordinate relation cannot pass by writing the same
// constant everywhere.

#include <array>
#include <cstdint>
#include <cstdio>

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_direct_accumulator_store.hpp"

namespace {
using namespace cute;

constexpr int kTM = 8;
constexpr int kTN = 64;
constexpr int kWM = 8;
constexpr int kWN = 16;

using Atom = PPU0010_8x16x16_F32F16F16F32_TN;
using Mma = TiledMMA<
    MMA_Atom<Atom>,
    Layout<Shape<Int<kTM / kWM>, Int<kTN / kWN>, _1>>,
    Tile<Int<(kTM / kWM) * 8>, Int<(kTN / kWN) * 16>, _16>>;
using Fragment = decltype(partition_fragment_C(
    Mma{}, Shape<Int<kTM>, Int<kTN>>{}));

static_assert(size(Mma{}) == 128,
              "L222 must remain bound to the exact CTA128 tactic");
static_assert(size(Fragment{}) == 4,
              "L222 must remain bound to four FP32 accumulators per thread");
static_assert(size(Mma{}) * size(Fragment{}) == kTM * kTN,
              "thread fragments must span the complete CTA tile");

// Instantiate the exact production helper in this host oracle.  This CUTLASS
// fork exposes layout-only code to host compilation while hgcc gives the same
// function its device qualifier in the real PPU build.
struct DirectParams {
  float* ptr_D;
  Stride<int, _1, int> dD;
};

#ifndef L222_BAD_THREAD_MODULO
#define L222_BAD_THREAD_MODULO 0
#endif
#ifndef L222_BAD_FRAGMENT_ROTATE
#define L222_BAD_FRAGMENT_ROTATE 0
#endif

constexpr std::uint32_t kPoison = UINT32_C(0x7fc22222);

float marker(int m, int n) {
  std::uint32_t bits = UINT32_C(0x3f000000) +
                       std::uint32_t(1 + m * kTN + n);
  float value;
  __builtin_memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint32_t bits(float value) {
  std::uint32_t result;
  __builtin_memcpy(&result, &value, sizeof(result));
  return result;
}

struct Result {
  int visits = 0;
  int holes = 0;
  int duplicates = 0;
  int value_bad = 0;
  int invalid_touched = 0;
};

Result verify(int residue_m, int residue_n) {
  std::array<float, kTM * kTN> output;
  std::array<int, kTM * kTN> coverage{};
  float poison;
  __builtin_memcpy(&poison, &kPoison, sizeof(poison));
  output.fill(poison);

  auto destination = make_tensor(
      output.data(),
      make_layout(Shape<Int<kTM>, Int<kTN>>{},
                  Stride<Int<kTN>, _1>{}));
  auto identity = make_identity_tensor(Shape<Int<kTM>, Int<kTN>>{});

  Result result;
  DirectParams params{output.data(),
                      make_stride(kTN, _1{}, residue_m * residue_n)};
  for (int physical_thread = 0; physical_thread < int(size(Mma{}));
       ++physical_thread) {
#if L222_BAD_THREAD_MODULO
    int const store_thread = physical_thread % 32;
#else
    int const store_thread = physical_thread;
#endif
    auto thread_mma = Mma{}.get_thread_slice(store_thread);
    auto coordinates = thread_mma.partition_C(identity);
    (void)destination;
    Fragment accumulators;
    for (int i = 0; i < int(size(accumulators)); ++i) {
      int const source_i =
#if L222_BAD_FRAGMENT_ROTATE
          (i + 1) % int(size(accumulators));
#else
          i;
#endif
      auto coord = coordinates(source_i);
      int const m = int(get<0>(coord));
      int const n = int(get<1>(coord));
      accumulators(i) = marker(m, n);
    }
    for (int i = 0; i < int(size(accumulators)); ++i) {
      auto coord = coordinates(i);
      int const m = int(get<0>(coord));
      int const n = int(get<1>(coord));
      if (m < residue_m && n < residue_n) {
        ++coverage[std::size_t(m * kTN + n)];
        ++result.visits;
      }
    }
    cutlass::gemm::kernel::detail::store_splitk_accumulators_direct(
        params, make_shape(residue_m, residue_n, 5120, 1),
        Shape<Int<kTM>, Int<kTN>, Int<256>>{},
        make_coord(0, 0, _, Int<0>{}), accumulators, Mma{},
        make_tuple(residue_m, residue_n, 0), 0, store_thread);
  }

  for (int m = 0; m < kTM; ++m) {
    for (int n = 0; n < kTN; ++n) {
      int const index = m * kTN + n;
      bool const valid = m < residue_m && n < residue_n;
      if (valid) {
        result.holes += coverage[std::size_t(index)] == 0;
        result.duplicates += coverage[std::size_t(index)] > 1;
        result.value_bad +=
            bits(output[std::size_t(index)]) != bits(marker(m, n));
      } else {
        result.invalid_touched +=
            bits(output[std::size_t(index)]) != kPoison;
      }
    }
  }
  return result;
}

}  // namespace

int main() {
  Result full = verify(kTM, kTN);
  Result decode = verify(1, kTN);
  int const expected = kTM * kTN + kTN;
  bool const pass = full.visits + decode.visits == expected &&
                    full.holes + decode.holes == 0 &&
                    full.duplicates + decode.duplicates == 0 &&
                    full.value_bad + decode.value_bad == 0 &&
                    full.invalid_touched + decode.invalid_touched == 0;
  std::printf(
      "L222_DIRECT_ACCUMULATOR_STORE visits=%d expected=%d holes=%d "
      "duplicates=%d value_bad=%d invalid_touched=%d verdict=%s\n",
      full.visits + decode.visits, expected,
      full.holes + decode.holes, full.duplicates + decode.duplicates,
      full.value_bad + decode.value_bad,
      full.invalid_touched + decode.invalid_touched,
      pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
