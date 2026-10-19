#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// Independent host model of llama quantize_q8_1: F32 max/division, round-away
// integer codes, then an F16 stored scale. Never change the actual GPU inputs.
template<class RoundHalf>
static std::vector<float> native_q8_1_input(const std::vector<float> & input, RoundHalf half) {
    if (input.size()%32) throw std::runtime_error("Q8_1 oracle requires complete 32-value blocks");
    std::vector<float> result(input.size());
    for (size_t first=0;first<input.size();first+=32) {
        float amax=0;
        for (size_t k=0;k<32;++k) {
            if (!std::isfinite(input[first+k])) throw std::runtime_error("nonfinite Q8_1 oracle input");
            amax=std::max(amax,std::abs(input[first+k]));
        }
        const float d=amax/127.0f, stored=half(d);
        for (size_t k=0;k<32;++k)
            result[first+k]=amax==0 ? 0 : std::round(input[first+k]/d)*stored;
    }
    return result;
}
