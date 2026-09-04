// Type closure for canonical K-pack4 weights with historical ScaleFirst FP16
// metadata and the exact persistent prefill driver.

#include <cstdio>
#include <type_traits>

#include "scalefirst_internal_sweep_bench.hpp"

using Types = scalefirst_internal_sweep::RowTypes<
    12, 0, 64, 64, 64, 64, 32, 3, 0, 1>;
using LargeK64Types = scalefirst_internal_sweep::RowTypes<
    12, 0, 64, 128, 64, 64, 16, 6, 0, 1>;
using LargeK256Types = scalefirst_internal_sweep::RowTypes<
    12, 0, 64, 128, 256, 64, 16, 2, 0, 1>;
using PackedA1Types = scalefirst_internal_sweep::RowTypes<
    12, 0, 8, 64, 64, 8, 16, 2, 0, 1, 1, 16>;
using XplaneTypes = scalefirst_internal_sweep::RowTypes<
    12, 64, 64, 64, 64, 64, 32, 3, 0, 0>;
using XplaneExpected = fpa_intb_ppu::DenseKernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    ppu_group_schedule::FinegrainedSchedule<32>,
    cute::Shape<cute::_64, cute::_64, cute::_64>,
    cute::Shape<cute::_64, cute::_2>,
    cute::Shape<cute::_64, cute::_32, cute::_64>,
    3, true, cutlass::int4b_t, void, 64>;
using Mainloop = typename Types::Shipping::CollectiveMainloop;
using Descriptor = typename Types::Shipping::MainloopPolicy::Descriptor;
using PersistentMainloop = typename Types::PersistentKernel::CollectiveMainloop;

static_assert(Types::use_kpack4);
static_assert(!XplaneTypes::use_kpack4);
static_assert(std::is_same_v<typename XplaneTypes::Shipping, XplaneExpected>,
              "the explicit Xplane arm must retain the historical type");
static_assert(Descriptor::q4_kpack4_transpose);
static_assert(Descriptor::transport_tile_k == 64);
static_assert(Mainloop::kQ4KPack4Transpose);
static_assert(!Mainloop::is_packed_scale,
              "ScaleFirst K-pack4 must consume FP16 scale/zero planes");
static_assert(std::is_same_v<Mainloop, PersistentMainloop>,
              "K-pack4 prefill must reuse the exact ScaleFirst persistent mainloop");
static_assert(Types::PersistentKernel::SharedStorageSize > 0);
static_assert(ppu_tactics::fits_block_smem(
                  Types::PersistentKernel::SharedStorageSize));
static_assert(ppu_tactics::fits_block_smem(
                  LargeK64Types::PersistentKernel::SharedStorageSize));
static_assert(ppu_tactics::fits_block_smem(
                  LargeK256Types::PersistentKernel::SharedStorageSize));
static_assert(PackedA1Types::MainloopPolicy::PackedARows == 1,
              "the AP1 control must bind the real Rows=1 policy");
static_assert(scalefirst_internal_sweep::packed_a_shape_admissible<
                  PackedA1Types::MainloopPolicy::PackedARows>(1));
static_assert(!scalefirst_internal_sweep::packed_a_shape_admissible<
                  PackedA1Types::MainloopPolicy::PackedARows>(2));
static_assert(scalefirst_internal_sweep::packed_a_shape_admissible<
                  Types::MainloopPolicy::PackedARows>(2),
              "ordinary AP0 remains admissible above M=1");

int main() {
  std::printf("L234 Q4 KPACK4 ScaleFirst persistent type PASS "
              "metadata=FP16 layout=transpose-v1 tactics=3 "
              "packed_a=Rows1-only\n");
}
