// L140 -- the minimal aligned WarpK tactic seam, bound to the real dense Cfg.
//
// The old seven-parameter Cfg must remain exactly the same type stack as an
// explicit WarpK=TK spelling.  The sole new positive is ordinary single-plane
// int4 F1 at 16x128x128/w16x64x32: its real builder must expose eight warps
// (256 threads).  This probe deliberately stops before the Marlin cooperative;
// CTA-local reduction and scheduler wiring are separate seams.

#define LOWBIT_DENSE_UNIT_BUILD 1
#define BENCH_GS 128
#define BENCH_TSK 128
#define TILE_M 16
#define TILE_N 128
#define WARP_M 16
#define WARP_N 64
#define STAGES 4
#include "test_lowbit_dense_bench.cu"

#include <type_traits>

using L140Default = Cfg<128, 16, 128, 128, 16, 64, 4>;
using L140ExplicitDefault = Cfg<128, 16, 128, 128, 16, 64, 4, 128>;
using L140Aligned = Cfg<128, 16, 128, 128, 16, 64, 4, 32>;

static_assert(std::is_same_v<L140Default, L140ExplicitDefault>,
              "the default template argument must name the exact old Cfg specialization");
static_assert(std::is_same_v<typename L140Default::CfgWarp,
                             typename L140ExplicitDefault::CfgWarp>);
static_assert(std::is_same_v<typename L140Default::Policy,
                             typename L140ExplicitDefault::Policy>);
static_assert(std::is_same_v<typename L140Default::Main,
                             typename L140ExplicitDefault::Main>,
              "omitting WarpK must preserve the shipping collective type exactly");
static_assert(cute::size(typename L140Default::Main::TiledMma{}) == 64,
              "the old 2N x 1K spelling must remain a two-warp CTA");
static_assert(cute::size(typename L140Aligned::Main::TiledMma{}) == 256,
              "the aligned 2N x 4K spelling must build an eight-warp CTA");
static_assert(ppu_tactics::cta_warps(
                  ppu_tactics::Candidate{ppu_tactics::kWarpKControlI4,
                      16, 128, 128, 16, 64, 128, 0, 32}) == 8);

#if defined(L140_DROP_K_COHORT_PLANT)
static_assert(cute::size(typename L140Aligned::Main::TiledMma{}) == 64,
              "L140 deliberate regression: K cohorts may not disappear from the launch size");
#endif

#if defined(L140_ACCEPT_FOLDED_WK_PLANT)
static_assert(ppu_tactics::common_kernel_exclusion(
                  ppu_tactics::Candidate{ppu_tactics::kArtifactFoldControlI2,
                      16, 128, 128, 16, 64, 64, 0, 32}) ==
                  ppu_tactics::Exclusion::None,
              "L140 deliberate regression: folded WarpK must remain fail-closed");
#endif

int main() { return 0; }
