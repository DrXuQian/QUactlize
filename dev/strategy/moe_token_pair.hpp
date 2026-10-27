#pragma once
#include <stdint.h>
#include "cutlass/cutlass.h"

namespace quactlize::execution::simt::moe_pair {

struct Rows {
    bool leader;
    int peer;
};

// matches includes row itself; all callers validate 0 <= row < 64. Pair
// adjacent occurrences, not adjacent router slots or equal slot numbers.
// Odd occurrence counts leave one singleton. No uniqueness assumption on IDs.
CUTLASS_HOST_DEVICE Rows locate(uint64_t matches, unsigned row) {
    uint64_t self=uint64_t(1)<<row;
    if(__builtin_popcountll(matches&(self-1))&1)return {false,-1};
    uint64_t after=matches&~(self|(self-1));
    return {true,after ? __builtin_ctzll(after) : -1};
}

CUTLASS_HOST_DEVICE int64_t activation_row(int row,int topk,int channels,
                                                int64_t token_stride,int64_t row_stride) {
    return int64_t(row/topk)*token_stride+(row%topk%channels)*row_stride;
}
} // namespace quactlize::execution::simt::moe_pair
