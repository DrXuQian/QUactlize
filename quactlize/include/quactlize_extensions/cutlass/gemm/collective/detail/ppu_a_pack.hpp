#pragma once

namespace cutlass::gemm::collective::detail {

// l84/l85 replay ppu_tsm_ld_swzl_sim for the first R rows and search all pitches for the largest current packed-A
// footprint (eight cubes/stages). These are the smallest COLLISION-FREE pitches that are themselves 64-half/128-B
// aligned. Searching aligned candidates matters: for R=2 the tempting 128-half candidate collides; the first clean
// aligned pitch is 320. R=13 reaches the natural 16x64 cube pitch, so packing has no remaining allocation benefit.
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
