#include "quactlize/prefill/dense_io_layout.hpp"
#include <cassert>

int main() {
    using quactlize::prefill::dense_cont_eligible;
    auto valid=[](int rows,int width,int64_t stride) {
        return dense_cont_eligible(0x1000,0x2000,rows,width,stride);
    };
    assert(valid(2048,6144,6144));
    assert(valid(1,8,8));
    assert(valid(1,2056,2056)); // Last CTA may be partial, not the eight-value chunk.
    assert(valid(8,3,3));     // Whole tensor is contiguous even across odd rows.
    assert(!valid(1,7,7));
    assert(!valid(3,17,23));
    assert(!valid(0,8,8));
    assert(!valid(-1,8,8));
    assert(!valid(1,-8,-8));
    assert(!dense_cont_eligible(0x1004,0x2000,1,8,8));
    assert(!dense_cont_eligible(0x1000,0x2002,1,8,8));
    assert(valid(8,INT_MAX/2,INT_MAX/2));
    assert(!valid(8,INT_MAX/2+1,int64_t(INT_MAX/2)+1));
    assert(!valid(65535,INT_MAX,INT_MAX));
}
