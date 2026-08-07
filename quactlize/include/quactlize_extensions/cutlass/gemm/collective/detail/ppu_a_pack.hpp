#pragma once

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

static_assert(aPackPitchForRows(1) == 64 && aPackPitchForRows(2) == 320 &&
              aPackPitchForRows(5) == 640 && aPackPitchForRows(9) == 960 &&
              aPackPitchForRows(13) == 1024 && aPackPitchForRows(16) == 1024,
              "packed-A pitch table must match fold_derivation/l85");

}  // namespace cutlass::gemm::collective::detail
