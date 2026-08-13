// L175 -- compile-only native PPU Marlin accumulator contract.
//
// The standalone kernel executes one native m16n16 instruction per N block.
// Its register ABI is therefore four independent eight-float fragments, not a
// CuTe/NVIDIA two-n8 logical C tensor.  Keep these assertions on the public
// collective aliases so the test follows the exact type used by production.

#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <utility>

#include <cuda_fp16.h>

// Stock nvcc is only a host compiler for this oracle.  The production header
// contains uninstantiated PPU device helpers; these declarations make their
// bodies parseable without pretending that NVIDIA can assemble them.
__half2 l175_unreachable_hfma2(__half2, __half2, __half2);
unsigned int l175_unreachable_cvta(void const*);
struct L175UnreachableThreadIdx { int x = 0, y = 0, z = 0; };
inline constexpr L175UnreachableThreadIdx l175_unreachable_thread_idx{};
void l175_unreachable_syncthreads();
#define __hfma2 l175_unreachable_hfma2
#define __cvta_generic_to_shared l175_unreachable_cvta
#define threadIdx l175_unreachable_thread_idx
#define __syncthreads l175_unreachable_syncthreads
#include "cute/tensor.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
#undef __syncthreads
#undef threadIdx
#undef __cvta_generic_to_shared
#undef __hfma2

namespace {

using namespace cute;
using ProductionCollective = cutlass::gemm::collective::MarlinCollectivePPU<
    Shape<_16, _128, _128>, Shape<_16, _64, _32>, 4, 128,
    Stride<int64_t, _1, int64_t>,
    Stride<int64_t, _1, int64_t>,
    Stride<_1, int64_t, int64_t>>;
using FragmentC = typename ProductionCollective::FragmentC;
using Accumulator = typename ProductionCollective::Accumulator;
using FragmentValues = std::remove_reference_t<
    decltype((std::declval<FragmentC&>().value))>;
using AccumulatorFragments = std::remove_reference_t<
    decltype((std::declval<Accumulator&>().fragments))>;

static_assert(std::is_same_v<
                  FragmentC,
                  cutlass::gemm::collective::marlin_ppu_detail::FragmentC>,
              "L175_COLLECTIVE_FRAGMENT_ALIAS_IS_NATIVE");
static_assert(std::is_same_v<
                  Accumulator,
                  cutlass::gemm::collective::marlin_ppu_detail::MarlinAccumulatorPPU>,
              "L175_COLLECTIVE_ACCUMULATOR_ALIAS_IS_NATIVE");
static_assert(std::is_array_v<FragmentValues> &&
                  std::extent_v<FragmentValues> == 8 &&
                  std::is_same_v<std::remove_extent_t<FragmentValues>, float>,
              "L175_FRAGMENT_C_EIGHT_FLOATS");
static_assert(std::is_array_v<AccumulatorFragments> &&
                  std::extent_v<AccumulatorFragments> == 4 &&
                  std::is_same_v<
                      std::remove_extent_t<AccumulatorFragments>, FragmentC>,
              "L175_ACCUMULATOR_FOUR_NATIVE_FRAGMENTS");
static_assert(sizeof(FragmentC) == 8 * sizeof(float),
              "L175_FRAGMENT_C_SIZE_32");
static_assert(sizeof(Accumulator) == 4 * 8 * sizeof(float),
              "L175_ACCUMULATOR_SIZE_128");
static_assert(offsetof(Accumulator, fragments) == 0,
              "L175_ACCUMULATOR_HAS_NO_PREFIX_STATE");
static_assert(std::is_standard_layout_v<FragmentC> &&
                  std::is_trivially_copyable_v<FragmentC>,
              "L175_FRAGMENT_C_PLAIN_REGISTER_AGGREGATE");
static_assert(std::is_standard_layout_v<Accumulator> &&
                  std::is_trivially_copyable_v<Accumulator>,
              "L175_ACCUMULATOR_PLAIN_REGISTER_AGGREGATE");

// Unevaluated overload resolution pins the native MMA interface without
// asking NVIDIA's assembler to consume the PPU instruction.
using MmaResult = decltype(
    cutlass::gemm::collective::marlin_ppu_detail::mma_n16<0>(
        std::declval<cutlass::gemm::collective::marlin_ppu_detail::FragmentA const&>(),
        std::declval<cutlass::gemm::collective::marlin_ppu_detail::FragmentB const&>(),
        std::declval<cutlass::gemm::collective::marlin_ppu_detail::FragmentB const&>(),
        std::declval<FragmentC&>()));
static_assert(std::is_same_v<MmaResult, void>,
              "L175_MMA_ACCEPTS_ONE_NATIVE_FRAGMENT");

}  // namespace

int main() {
  std::printf(
      "L175 fragment_bytes=%zu fragment_values=%zu accumulator_bytes=%zu "
      "accumulator_fragments=%zu standard_layout=1 trivial=1 result=PASS\n",
      sizeof(FragmentC), std::extent_v<FragmentValues>, sizeof(Accumulator),
      std::extent_v<AccumulatorFragments>);
  return 0;
}
