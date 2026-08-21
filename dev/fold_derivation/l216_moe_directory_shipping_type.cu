// Force the exact grouped mixed-input collective through the new directory
// persistent driver.  The local CUDA compiler cannot encode PPU instructions;
// the companion script admits only those known vendor-asm diagnostics and
// rejects every scheduler/adapter/type error.

#include "moe_grouped_ppu.cuh"

using namespace cute;

using Q = moe_grouped_ppu::QuantMode;
using L216Tile = Shape<Int<64>, Int<64>, Int<64>>;
using L216ScaleTile = Shape<Int<64>, Int<2>>;
using L216Warp = Shape<Int<64>, Int<32>, Int<64>>;

void force_l216(
    cutlass::half_t const* a, cutlass::int4b_t const* b,
    cutlass::half_t const* scale, cutlass::half_t** d,
    moe_grouped_ppu::DStride* stride_d, int const* rows,
    moe_grouped_ppu::GroupShape* shapes, int const* offsets,
    char* workspace) {
  (void)moe_grouped_ppu::launch<
      Q::FinegrainedScaleOnly,
      ppu_group_schedule::FinegrainedSchedule<32>,
      L216Tile, L216ScaleTile, L216Warp, 3, true,
      cutlass::int4b_t, void, false,
      false, false, 64, true>(
          a, b, scale, nullptr, d, stride_d, rows,
          128, 4096, 4096, 64, 32,
          shapes, nullptr, offsets,
          workspace, 1u << 20, nullptr);
}

int main() { return 0; }
