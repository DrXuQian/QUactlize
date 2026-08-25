#pragma once

#include "cute/config.hpp"

namespace cutlass::gemm::collective::detail {

// l84/l85 replay ppu_tsm_ld_swzl_sim at their fixed 16-row cube height and search the live-row writer addresses.
// These are the smallest WRITER-COLLISION-FREE cube pitches that are themselves 64-half/128-B aligned. Searching
// aligned candidates matters: for R=2 the tempting 128-half candidate collides; the first clean aligned pitch is 320.
// They are not pipeline-stage pitches: the projected m8 atom's physical x4 read needs the separate authority below.
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

// The packed cube pitch is intentionally smaller than one physical m16x64
// swizzle footprint.  That is safe between cubes in one already-published
// stage: values read into masked M rows may alias another cube's live row.
// It is not safe between pipeline stages, because the x4 hardware load reads
// all four registers even when the m8 CuTe atom publishes only v0/v1.  A
// next-stage cp.async must therefore start after the complete physical read
// footprint of the preceding stage, not merely after its live row writes.
CUTE_HOST_DEVICE constexpr int aPackStagePitchHalfs(
    int cube_pitch, int cubes_per_stage, int physical_cube_span) {
  return cube_pitch * (cubes_per_stage - 1) + physical_cube_span;
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
static_assert(aPackStagePitchHalfs(64, 4, 16 * 64) == 1216,
              "TM8/TK256 packed-A stages must not overlap the physical x4 read footprint");
static_assert(aPackRunOffsetHalfs(16, 0, 0) == 0 &&
              aPackRunOffsetHalfs(16, 0, 1) == 288 &&
              aPackRunOffsetHalfs(16, 0, 2) == 528 &&
              aPackRunOffsetHalfs(16, 0, 3) == 816,
              "packed-A row-0 run authority must match the calibrated PPU0010 swizzle order");

}  // namespace cutlass::gemm::collective::detail
