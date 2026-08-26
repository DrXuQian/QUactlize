// Exact type witness for native Q4_K A32/F2 decode with packed metadata.
//
// B is not converted to A64/F1: the resident artifact remains logical
// N64 x K32 and physical N32 x K64.  The folded collective restores logical
// N x K for MMA, while its independent packed channel owns one native 16-byte
// Q4_K metadata unit per logical N column.

#include <cstdio>
#include <type_traits>

#include "fully_quantized_splitk_producer_bench.hpp"

using Types = fq_internal_sweep::TcRowTypes<
    12, 32, 8, 64, 256, 8, 32, 2, 0, 0>;
using Policy = typename Types::Shipping::MainloopPolicy;
using Mainloop = typename Types::Mainloop;

static_assert(Policy::ArtifactTileK == 32);
static_assert(Policy::TacticTileK == 256);
static_assert(Policy::ArtifactLowFold == 2);
static_assert(std::is_same_v<typename Policy::Descriptor::BProviderType,
                             ppu_mixed_policy::FoldedBProvider<2>>);
static_assert(Policy::Descriptor::packed_metadata);
static_assert(Mainloop::is_packed_scale);
static_assert(Mainloop::packed_scale_copy_threads == 64);
static_assert(Mainloop::packed_scale_columns_per_thread == 1);
static_assert(std::is_same_v<typename Types::SplitKernel::CollectiveMainloop,
                             Mainloop>);

int main() {
  std::printf("L227 Q4/A32 native-F2 packed-metadata "
              "physical=N/2x2K logical=NxK owners=64x1 PASS\n");
}
