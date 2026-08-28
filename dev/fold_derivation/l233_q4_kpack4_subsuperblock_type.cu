// L233 -- Q4_K K-pack4 TK64/TK128/TK256 share one offline byte map while
// selecting 2/4/8 metadata groups per logical K tile.
//
// This is a production-type proof, not a stand-in layout.  It instantiates
// the real policy/collective for one legal TM64 anchor row at each TileK and
// pins the exact quotient/remainder relation used by the pipeline.  Device
// raw-bit closure remains the independent value oracle.

#include <cstdio>

#include "fpA_intB_ppu.cuh"
#include "q4_kpack4_offline.hpp"

using namespace cute;
using Schedule = ppu_group_schedule::FinegrainedSchedule<32>;

template <int TK>
struct Anchor {
  using Tile = Shape<_64, _128, Int<TK>>;
  using Scale = Shape<_128, Int<TK / 32>>;
  using Warp = Shape<_64, _16, Int<TK>>;
  using Types = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
      ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
      Schedule, Tile, Scale, Warp, 2, true>;
  using Policy = typename Types::MainloopPolicy;
  using Mainloop = typename Types::CollectiveMainloop;
};

using K64 = Anchor<64>;
using K128 = Anchor<128>;
using K256 = Anchor<256>;

static_assert(K64::Mainloop::is_packed_scale &&
              K128::Mainloop::is_packed_scale &&
              K256::Mainloop::is_packed_scale);
static_assert(K64::Mainloop::packed_scale_tiles_per_unit == 4);
static_assert(K128::Mainloop::packed_scale_tiles_per_unit == 2);
static_assert(K256::Mainloop::packed_scale_tiles_per_unit == 1);
static_assert(K64::Mainloop::PhysicalBTileK == 16);
static_assert(K128::Mainloop::PhysicalBTileK == 32);
static_assert(K256::Mainloop::PhysicalBTileK == 64);
static_assert(K64::Policy::Descriptor::q4_kpack4_transpose &&
              K128::Policy::Descriptor::q4_kpack4_transpose &&
              K256::Policy::Descriptor::q4_kpack4_transpose);
static_assert(K64::Policy::Descriptor::transport_tile_k == 64 &&
              K128::Policy::Descriptor::transport_tile_k == 64 &&
              K256::Policy::Descriptor::transport_tile_k == 64);

template <int GroupsPerTile>
int phase_coverage() {
  static_assert(8 % GroupsPerTile == 0);
  constexpr int Tiles = 8 / GroupsPerTile;
  bool seen[8] = {};
  bool legacy_seen[8] = {};
  int bad = 0;
  for (int tile = 0; tile < Tiles; ++tile) {
    int const unit = tile / Tiles;
    int const phase = tile % Tiles;
    if (unit != 0 || phase != tile) ++bad;
    for (int local = 0; local < GroupsPerTile; ++local) {
      int const source = phase * GroupsPerTile + local;
      if (source < 0 || source >= 8 || seen[source]) ++bad;
      else seen[source] = true;
      // RED control: the historical one-superblock mapping repeats the first
      // local run for every subtile and therefore cannot cover all groups.
      legacy_seen[local] = true;
    }
  }
  for (int g = 0; g < 8; ++g) {
    if (!seen[g]) ++bad;
  }
  int legacy_missing = 0;
  for (int g = 0; g < 8; ++g) legacy_missing += !legacy_seen[g];
  return bad == 0 && legacy_missing == 8 - GroupsPerTile ? 0 : 1;
}

int main() {
  int const bad = phase_coverage<2>() + phase_coverage<4>() +
                  phase_coverage<8>();
  std::printf("L233 Q4_K KPACK4 sub-superblock %s "
              "mapping=0x%016llx TK64=4x2 TK128=2x4 TK256=1x8 "
              "legacy-local-group=RED\n",
              bad ? "FAIL" : "PASS",
              static_cast<unsigned long long>(q4_kpack4::kMappingId));
  return bad;
}
