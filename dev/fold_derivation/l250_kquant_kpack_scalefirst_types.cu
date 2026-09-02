// Compile-time closure for the canonical Q2/Q3/Q5/Q6 K-pack readers with
// resident fp16 ScaleFirst metadata.  This deliberately instantiates the
// same RowTypes used by generated dense sweep units, not a reduced mock.

#include <cstdio>
#include <type_traits>

#include "scalefirst_internal_sweep_bench.hpp"

template <int Q, int TK>
using KPackTypes = scalefirst_internal_sweep::RowTypes<
    Q, 0, 64, 64, TK, 64, 32, 2, 0, 2, 0, 32>;

using Q2 = KPackTypes<10, 128>;
using Q3 = KPackTypes<11, 256>;
using Q5 = KPackTypes<13, 256>;
using Q6 = KPackTypes<14, 128>;

template <class Types, int LowPack, int HighPack, int TransportK>
constexpr bool closes_kpack_type() {
  using Descriptor = typename Types::MainloopDescriptor;
  using Mainloop = typename Types::Mainloop;
  return Types::use_generic_kpack && !Types::use_kpack4 &&
      Descriptor::kpack_transpose &&
      Descriptor::kpack_low == LowPack &&
      Descriptor::kpack_high == HighPack &&
      Descriptor::transport_tile_k == TransportK &&
      Descriptor::artifact_tile_k == 0 &&
      Mainloop::kKPackTranspose && !Mainloop::is_packed_scale &&
      std::is_same_v<
          typename Types::PersistentKernel::CollectiveMainloop, Mainloop> &&
      std::is_same_v<
          typename Types::SplitKernel::CollectiveMainloop, Mainloop> &&
      Types::Shipping::SharedStorageSize > 0 &&
      Types::PersistentKernel::SharedStorageSize > 0 &&
      Types::SplitKernel::SharedStorageSize > 0;
}

static_assert(closes_kpack_type<Q2, 8, 0, 128>());
static_assert(closes_kpack_type<Q3, 8, 16, 256>());
static_assert(closes_kpack_type<Q5, 4, 16, 256>());
static_assert(closes_kpack_type<Q6, 4, 8, 128>());
static_assert(Q3::MainloopDescriptor::quant_mode ==
                  ppu_mixed_policy::QuantMode::FinegrainedScaleZero &&
              Q6::MainloopDescriptor::quant_mode ==
                  ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
              "Q3/Q6 converter centre correction requires the zero plane");

int main() {
  std::printf("L250 K-quant K-pack ScaleFirst types PASS "
              "formats=Q2/Q3/Q5/Q6 metadata=FP16-scale-zero\n");
}
