// L113 -- THE SHARED MIXED-METADATA POLICY OWNS COARSE/FINE GROUPING.
//
// This is deliberately a type-level gate. The three collectives have different B providers and two scale-storage
// views (flat and explicit group/stage), but their boundary predicate and group index must remain the same object.
#include <cstdio>
#include <type_traits>

#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"

namespace md = cutlass::gemm::collective::detail;

using Flat2Zero = md::MixedMetadataPolicy<float, 2, true, md::FlatMetadataAddress>;
using Flat2Scale = md::MixedMetadataPolicy<float, 2, false, md::FlatMetadataAddress>;
using Split4Zero = md::MixedMetadataPolicy<float, 4, true, md::SplitMetadataAddress>;

using Coarse2x4 = Flat2Zero::Coarse<4>;
static_assert(Coarse2x4::active);
static_assert(Coarse2x4::steps_per_group == 2);
static_assert(Coarse2x4::starts_group(0) && !Coarse2x4::starts_group(1));
static_assert(Coarse2x4::starts_group(2) && !Coarse2x4::starts_group(3));
static_assert(Coarse2x4::group(0) == 0 && Coarse2x4::group(1) == 0);
static_assert(Coarse2x4::group(2) == 1 && Coarse2x4::group(3) == 1);

using Fine2x1x8 = Flat2Zero::Fine<1, 8>;
static_assert(Fine2x1x8::active);
static_assert(Fine2x1x8::atoms_per_group == 4);
static_assert(Fine2x1x8::starts_group(0) && !Fine2x1x8::starts_group(3));
static_assert(Fine2x1x8::starts_group(4) && !Fine2x1x8::starts_group(7));
static_assert(Fine2x1x8::group(0) == 0 && Fine2x1x8::group(3) == 0);
static_assert(Fine2x1x8::group(4) == 1 && Fine2x1x8::group(7) == 1);

using Fine4x2x8 = Split4Zero::Fine<2, 8>;
static_assert(Fine4x2x8::active && Fine4x2x8::atoms_per_group == 2);
static_assert(Fine4x2x8::starts_group(6) && Fine4x2x8::group(6) == 3);

static_assert(Flat2Zero::scale_groups == 2 && Flat2Zero::has_zero);
static_assert(!Flat2Scale::has_zero);
static_assert(std::is_same_v<typename Flat2Zero::AddressPolicy, md::FlatMetadataAddress>);
static_assert(std::is_same_v<typename Split4Zero::AddressPolicy, md::SplitMetadataAddress>);
using FlatCap = Flat2Zero::ScaleCopy<128, 64>;
using SplitCap = Split4Zero::ScaleCopy<128, 64>;
using Q3Cap = md::ScaleCopyPlan<128, 8, 64>;
static_assert(FlatCap::Coverage::value && SplitCap::Coverage::value);
static_assert(Q3Cap::thread_layout_h == 8 && Q3Cap::thread_layout_w == 8);
static_assert(Q3Cap::values_per_thread == 16 && Q3Cap::thread_slots == 64);
static_assert(Q3Cap::Coverage::within_cta && Q3Cap::Coverage::atom_aligned &&
              Q3Cap::Coverage::full_tile);

int main() {
  std::printf("[l113] shared mixed metadata policy: COARSE/FINE grouping and storage policies agree\n");
  return 0;
}
