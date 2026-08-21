#pragma once

#include "cute/config.hpp"

namespace cutlass::gemm::collective::detail {

// l84/l85 replay ppu_tsm_ld_swzl_sim at their fixed 16-row cube height and search all pitches for eight cubes/stages.
// These are the smallest COLLISION-FREE pitches in that model that are themselves 64-half/128-B aligned. Searching
// aligned candidates matters: for R=2 the tempting 128-half candidate collides; the first clean aligned pitch is 320.
// The collective owns a lower public R bound: entries above it are derivation records, not a claim that one pitch works
// across every compiled TileM cube height and cube/stage count.
constexpr int aPackPitchForRows(int rows) {
  return rows <= 0  ? 0
       : rows == 1 ? 64
       : rows <= 4 ? 320
       : rows <= 8 ? 640
       : rows <= 12 ? 960
       : rows <= 16 ? 1024
       : 0;
}

// One production authority for the first-R-row writer's physical run start.  The inputs and result are deliberately
// named in physical units: cube_height is the PPU0010 TSM cube height, slice is one of the four logical 16-half K
// runs, and the result is a half-element offset from that cube's shared-memory base.  The independent L186 oracle
// derives the same address through the verbatim PPU0010_TSM_LD_SWZL_M8 lane/vreg model; it does not call this helper
// on its expected/read side.
CUTE_HOST_DEVICE constexpr int aPackRunOffsetHalfs(int cube_height, int row, int slice) {
  int const slice_start_vec = (((slice & 1) << 1) + ((slice & 2) >> 1)) * 2;  // 0, 4, 2, 6
  int const line = row / 4;
  int const run_vec = (2 * (row % 4) + slice_start_vec) % 8;
  return 2 * (cube_height * 8 * slice + line * 32 + run_vec * 4);
}

static_assert(aPackPitchForRows(1) == 64 && aPackPitchForRows(2) == 320 &&
              aPackPitchForRows(5) == 640 && aPackPitchForRows(9) == 960 &&
              aPackPitchForRows(13) == 1024 && aPackPitchForRows(16) == 1024,
              "packed-A pitch table must match fold_derivation/l85");
static_assert(aPackRunOffsetHalfs(16, 0, 0) == 0 &&
              aPackRunOffsetHalfs(16, 0, 1) == 288 &&
              aPackRunOffsetHalfs(16, 0, 2) == 528 &&
              aPackRunOffsetHalfs(16, 0, 3) == 816,
              "packed-A row-0 run authority must match the calibrated PPU0010 swizzle order");

}  // namespace cutlass::gemm::collective::detail
