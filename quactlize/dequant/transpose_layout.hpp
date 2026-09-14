#pragma once
#include "cute/swizzle.hpp"

namespace quactlize::dequant {

// Physical uint32 cells, not bytes. Keep each logical K4 vector contiguous
// and 16-byte aligned. K[7:5] folds into bank bits [4:2], while N[4:3]
// separates the four N8 vectors used by the producer warp. Host ownership
// tests use this exact CuTe address function, not a duplicate formula.
template<int StageK>
struct FullTransposeLayout {
    static_assert(StageK == 128 || StageK == 256);
    static constexpr int kCells = 32 * StageK;
    CUTE_HOST_DEVICE static constexpr int offset(int row, int k) {
        return int(cute::Swizzle<3, 2, 3>{}(row * StageK + k)) ^ (row & 24);
    }
};

} // namespace quactlize::dequant
