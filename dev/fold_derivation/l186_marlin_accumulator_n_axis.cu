// L186 -- compile-only standalone Marlin accumulator-N axis.
//
// WN64 owns four native n16 fragments.  That shipping specialization must
// retain its exact named accumulator type, while WN128 owns eight fragments
// of the same native m8/m16 register ABI.  This oracle binds both facts to the
// production header without instantiating a second accumulator model.

#include <cstddef>
#include <cstdio>
#include <type_traits>

#include <cuda_fp16.h>

// Stock nvcc is only a host compiler for this type oracle.  Production helper
// bodies mention PPU-only device intrinsics even though no helper is invoked;
// keep them parseable without pretending NVIDIA executes the path.
__half2 l186_unreachable_hfma2(__half2, __half2, __half2);
unsigned int l186_unreachable_cvta(void const*);
#define __hfma2 l186_unreachable_hfma2
#define __cvta_generic_to_shared l186_unreachable_cvta
#include "actlize_extensions/cutlass/gemm/collective/marlin_mma_ppu.hpp"
#undef __cvta_generic_to_shared
#undef __hfma2

namespace md = cutlass::gemm::collective::marlin_ppu_detail;

namespace {

using ShippingM8 = md::MarlinAccumulatorForN<8, 4>;
using ShippingM16 = md::MarlinAccumulatorForN<16, 4>;
using WideM8 = md::MarlinAccumulatorForN<8, 8>;
using WideM16 = md::MarlinAccumulatorForN<16, 8>;

template <class Accumulator>
using Fragments = std::remove_reference_t<
    decltype((std::declval<Accumulator&>().fragments))>;

static_assert(std::is_same_v<ShippingM8, md::MarlinAccumulatorM8PPU> &&
                  std::is_same_v<ShippingM16, md::MarlinAccumulatorPPU>,
              "L186_WN64_EXACT_TYPE_IDENTITY");
static_assert(std::extent_v<Fragments<ShippingM8>> == 4 &&
                  std::extent_v<Fragments<ShippingM16>> == 4,
              "L186_WN64_FOUR_NATIVE_N16_FRAGMENTS");
static_assert(std::extent_v<Fragments<WideM8>> == 8 &&
                  std::extent_v<Fragments<WideM16>> == 8,
              "L186_WN128_EIGHT_NATIVE_N16_FRAGMENTS");
static_assert(std::is_same_v<
                  std::remove_extent_t<Fragments<WideM8>>, md::FragmentC8> &&
                  std::is_same_v<
                      std::remove_extent_t<Fragments<WideM16>>, md::FragmentC>,
              "L186_WIDE_N_REUSES_NATIVE_PPU_FRAGMENTS");
static_assert(sizeof(WideM8) == 8 * sizeof(md::FragmentC8) &&
                  sizeof(WideM16) == 8 * sizeof(md::FragmentC),
              "L186_WIDE_N_HAS_NO_HIDDEN_STATE");
static_assert(offsetof(WideM8, fragments) == 0 &&
                  offsetof(WideM16, fragments) == 0 &&
                  std::is_standard_layout_v<WideM8> &&
                  std::is_standard_layout_v<WideM16> &&
                  std::is_trivially_copyable_v<WideM8> &&
                  std::is_trivially_copyable_v<WideM16>,
              "L186_WIDE_N_IS_A_PLAIN_REGISTER_AGGREGATE");

}  // namespace

int main() {
  std::printf(
      "L186 shipping=m8:%zuB/4,m16:%zuB/4 "
      "wide=m8:%zuB/8,m16:%zuB/8 "
      "WN64-type-identity=1 native-fragments=1 result=PASS\n",
      sizeof(ShippingM8), sizeof(ShippingM16),
      sizeof(WideM8), sizeof(WideM16));
  return 0;
}
