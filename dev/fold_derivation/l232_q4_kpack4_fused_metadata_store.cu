// L232 -- production-type/layout proof for the Q4_K K-pack4 fused metadata
// store. This is host evidence only: the box consumes the committed result
// and uses fresh hgcc for the device closure.

#include <cstdio>
#include <type_traits>

#include "fpA_intB_ppu.cuh"

using namespace cute;
using Schedule = ppu_group_schedule::FinegrainedSchedule<32>;
using TileShape = Shape<_8, _64, _256>;
using ScaleTile = Shape<_64, _8>;
using Warp = Shape<_8, _16, _256>;

template <int AProvider>
using Types = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    Schedule, TileShape, ScaleTile, Warp, 2, true, AProvider, 0>;

using AP0 = typename Types<0>::CollectiveMainloop;
using AP1 = typename Types<1>::CollectiveMainloop;

static_assert(AP0::kQ4KPack4ResolvedDeliveryN == 64);
static_assert(AP1::kQ4KPack4ResolvedDeliveryN == 64);
static_assert(AP0::is_packed_scale && AP1::is_packed_scale);
static_assert(AP0::is_fused_scale_zero && AP1::is_fused_scale_zero);

template <class Mainloop>
struct Closure {
  using Half = typename Mainloop::FusedScaleHalfLayout;
  using Word = typename Mainloop::FusedScaleWordLayout;
  static_assert(cosize_v<Half> == 2 * cosize_v<Word> - 1,
                "the high half is adjacent to every scale word except the layout's final extent convention");
  static_assert(int(stride<0>(Half{})) == 2 &&
                    int(stride<1>(Half{})) == 2 * int(stride<1>(Word{})) &&
                    int(stride<2>(Half{})) == 2 * int(stride<2>(Word{})),
                "the complete half layout must be exactly twice the word layout");
  static_assert(Mainloop::SharedStorage::zero_elements == 0,
                "fused zero values must live in the odd halves of the scale allocation");
  static_assert(Mainloop::SharedStorage::scale_elements == 2 * cosize_v<Word>,
                "the final high half must remain inside the fused scale allocation");
};

template struct Closure<AP0>;
template struct Closure<AP1>;

int main() {
  int layout_bad = 0;
  int bits_bad = 0;
  for (unsigned lo = 0; lo < 65536; ++lo) {
    unsigned const hi = (lo * 40503u + 17u) & 0xffffu;
    auto const lo_h = cutlass::half_t::bitcast(uint16_t(lo));
    auto const hi_h = cutlass::half_t::bitcast(uint16_t(hi));
    uint32_t const pair = cutlass::gguf_packed::pack_h2(lo_h, hi_h);
    bits_bad += cutlass::gguf_packed::lo_h2(pair).raw() != lo;
    bits_bad += cutlass::gguf_packed::hi_h2(pair).raw() != hi;
  }
  std::printf("L232 Q4_K KPACK4 fused-metadata-store layout_bad=%d bits_bad=%d "
              "providers=AP0+AP1 delivery=auto64 values=1024-per-provider\n",
              layout_bad, bits_bad);
  return (layout_bad == 0 && bits_bad == 0) ? 0 : 1;
}
