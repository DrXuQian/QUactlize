#include "quactlize/ffn/dense_swiglu.hpp"
#include <cassert>
#include <vector>

int main() {
    using quactlize::ffn::dense_swiglu_vectorized;
    alignas(16) float storage[64]{};
    assert(dense_swiglu_vectorized(storage,storage+16,storage+32,16,32,32,16));
    assert(!dense_swiglu_vectorized(storage,storage+16,storage+32,15,32,32,16));
    for(int shift=1;shift<4;++shift) {
        assert(!dense_swiglu_vectorized(storage+shift,storage+16,storage+32,16,32,32,16));
        assert(!dense_swiglu_vectorized(storage,storage+16+shift,storage+32,16,32,32,16));
        assert(!dense_swiglu_vectorized(storage,storage+16,storage+32+shift,16,32,32,16));
        assert(!dense_swiglu_vectorized(storage,storage+16,storage+32,16,32+shift,32,16));
        assert(!dense_swiglu_vectorized(storage,storage+16,storage+32,16,32,32+shift,16));
        assert(!dense_swiglu_vectorized(storage,storage+16,storage+32,16,32,32,16+shift));
    }
    for(int hidden:{1,3,4,33,512,516,1024,17408,25600,25601}) {
        for(bool vector:{false,true}) {
            if(vector && hidden%4) continue;
            int width=vector?4:1,threads=vector?128:256;
            std::vector<int> visits(hidden);
            for(int block=0;block<(hidden+threads*width-1)/(threads*width);++block)
                for(int lane=0;lane<threads;++lane) {
                    int col=(block*threads+lane)*width;if(col>=hidden) continue;
                    for(int j=0;j<width;++j) ++visits.at(col+j);
                }
            for(int count:visits) assert(count==1);
        }
    }
    for(int rows:{0,1,8,17,2048,5120,65535,65536,131071}) {
        std::vector<int> visits(rows);int gy=rows>65535?65535:rows;
        for(int y=0;y<gy;++y) for(int64_t row=y;row<rows;row+=gy) ++visits.at(row);
        for(int count:visits) assert(count==1);
    }
}
