// Representative nonpersistent/persistent grouped type-only closure for the
// canonical K-pack FullyQuantized packed-metadata route.

#include <cstdio>
#include <type_traits>

#ifndef L253_QTYPE
#error "L253_QTYPE must be one of 10,11,12,13,14"
#endif
#define PPU_PACKED_SCALE 1
#include "fully_quantized_grouped_kpack_discovery.hpp"

#if L253_QTYPE == 10
static constexpr int kLayout = 2, kTK = 128;
#elif L253_QTYPE == 11
static constexpr int kLayout = 2, kTK = 256;
#elif L253_QTYPE == 12
static constexpr int kLayout = 1, kTK = 64;
#elif L253_QTYPE == 13
static constexpr int kLayout = 2, kTK = 256;
#elif L253_QTYPE == 14
static constexpr int kLayout = 2, kTK = 128;
#else
#error "L253_QTYPE must be one of 10,11,12,13,14"
#endif

template <bool Persistent>
using Types = fully_quantized_grouped_kpack::RowTypes<
    L253_QTYPE, kLayout, 64, 64, kTK, 64, 32, 2, 32, Persistent>;

template <class T>
constexpr bool closes() {
  return T::Descriptor::quant_mode ==
             ppu_mixed_policy::QuantMode::FinegrainedScaleZero &&
         T::Descriptor::kpack_transpose &&
         T::Descriptor::packed_metadata &&
         !T::Descriptor::interleaved_metadata &&
         T::Descriptor::artifact_tile_k == 0 &&
         std::is_same_v<typename T::Kernel::CollectiveMainloop,
                        typename T::Mainloop> &&
         T::Kernel::SharedStorageSize > 0;
}

static_assert(closes<Types<false>>());
static_assert(closes<Types<true>>());

int main() {
  std::printf("L253 FullyQuantized grouped K-pack type PASS "
              "qtype=%d algorithms=nonpersistent+persistent\n", L253_QTYPE);
}
