// Type-only grouped closure for all canonical K-pack ScaleFirst formats.

#include <cstdio>
#include <type_traits>

#include "scalefirst_grouped_kpack_discovery.hpp"

template <int Q, int Layout, int TK>
using Types = scalefirst_grouped_kpack::RowTypes<
    Q, Layout, 64, 64, TK, 64, 32, 2, 32>;

using Q2 = Types<10, 2, 128>;
using Q3 = Types<11, 2, 256>;
using Q4 = Types<12, 1, 64>;
using Q5 = Types<13, 2, 256>;
using Q6 = Types<14, 2, 128>;

template <class T>
constexpr bool grouped_type_closes() {
  return T::Descriptor::quant_mode ==
             ppu_mixed_policy::QuantMode::FinegrainedScaleZero &&
         T::Descriptor::kpack_transpose &&
         !T::Descriptor::packed_metadata &&
         std::is_same_v<typename T::Kernel::CollectiveMainloop,
                        typename T::Mainloop> &&
         std::is_same_v<typename T::PersistentKernel::CollectiveMainloop,
                        typename T::Mainloop> &&
         T::Kernel::SharedStorageSize > 0 &&
         T::PersistentKernel::SharedStorageSize == T::Kernel::SharedStorageSize;
}

static_assert(grouped_type_closes<Q2>());
static_assert(grouped_type_closes<Q3>());
static_assert(grouped_type_closes<Q4>());
static_assert(grouped_type_closes<Q5>());
static_assert(grouped_type_closes<Q6>());
static_assert(Q3::Descriptor::quant_mode ==
                  ppu_mixed_policy::QuantMode::FinegrainedScaleZero &&
              Q6::Descriptor::quant_mode ==
                  ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
              "Q3/Q6 grouped K-pack must retain the center-correction plane");

int main() {
  std::printf("L251 grouped K-pack ScaleFirst types PASS "
              "formats=Q2/Q3/Q4/Q5/Q6 persistent=EXACT-OCCUPANCY "
              "splitk=STRUCTURAL_UNAVAILABLE\n");
}
