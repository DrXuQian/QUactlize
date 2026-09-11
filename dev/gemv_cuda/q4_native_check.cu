// Exact metadata/code gates for the CUDA-only Q4 instruction candidate.
#include "q4_native.cuh"
#include "reader.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}

__global__ void decode(uint8_t const* units, uint32_t const* words,
                       uint16_t* output, int count) {
    using namespace quactlize::dev::q4_native;
    int const i = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint4 const unit = load_unit(units + 16*i);
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        auto const s = scale_zero(unit, g);
        output[24*i + 2*g] = __half_as_ushort(s.scale);
        output[24*i + 2*g+1] = __half_as_ushort(s.zero);
    }
    __half2 const c[4] = {codes<0>(words[i]), codes<1>(words[i]),
                         codes<2>(words[i]), codes<3>(words[i])};
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        output[24*i+16+2*s] = __half_as_ushort(__low2half(c[s]));
        output[24*i+17+2*s] = __half_as_ushort(__high2half(c[s]));
    }
}
}

int main() {
    try {
        using namespace gguf_scale;
        using H = cutlass::half_t;
        constexpr int count = 32768;
        constexpr uint16_t headers[] = {0x0000, 0x8000, 0x0001, 0x0400,
                                        0x2401, 0x3000, 0xb001, 0x3c00};
        std::vector<uint8_t> units(count*16);
        std::vector<uint32_t> words(count);
        std::vector<uint16_t> gold(count*24), got(gold.size());
        uint32_t rng = 0x13198a2e;
        for (int i = 0; i < count; ++i) {
            auto p = units.data() + 16*i;
            uint16_t const d = headers[i/4096], dm = headers[(i/512)%8];
            p[0] = uint8_t(d); p[1] = uint8_t(d>>8);
            p[2] = uint8_t(dm); p[3] = uint8_t(dm>>8);
            for (int g = 0; g < 8; ++g) {
                packed_unit::put_code<KType::Q4_K>(p, g, 0, (i+7*g)&63);
                packed_unit::put_code<KType::Q4_K>(p, g, 1, ((i/64)+11*g)&63);
            }
            for (int g = 0; g < 8; ++g) {
                auto const s = packed_unit::unit_group<KType::Q4_K,8>(p, g);
                gold[24*i+2*g] = s.scale.raw(); gold[24*i+2*g+1] = s.zero.raw();
            }
            rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5;
            words[i] = rng;
            for (int s = 0; s < 4; ++s) for (int n = 0; n < 2; ++n) {
                int const q = (rng>>(4*s+16*n))&15;
                gold[24*i+16+2*s+n] = H(float(q-8)).raw();
            }
        }
        uint8_t* du; uint32_t* dw; uint16_t* dout;
        check(cudaMalloc(&du, units.size()+16));
        check(cudaMalloc(&dw, words.size()*4));
        check(cudaMalloc(&dout, got.size()*2));
        check(cudaMemcpy(dw, words.data(), words.size()*4, cudaMemcpyHostToDevice));
        for (int offset : {0, 1, 2, 4, 8, 15}) {
            check(cudaMemcpy(du+offset, units.data(), units.size(), cudaMemcpyHostToDevice));
            check(cudaMemset(dout, 0x7b, got.size()*2));
            decode<<<(count+127)/128,128>>>(du+offset,dw,dout,count);
            check(cudaGetLastError()); check(cudaDeviceSynchronize());
            check(cudaMemcpy(got.data(),dout,got.size()*2,cudaMemcpyDeviceToHost));
            size_t bad = 0;
            for (size_t i = 0; i < got.size(); ++i) if (gold[i] != got[i]) {
                if (bad++ == 0) std::printf("Q4_NATIVE_FIRST offset=%d index=%zu want=0x%04x got=0x%04x\n",
                                            offset,i,gold[i],got[i]);
            }
            std::printf("Q4_NATIVE_MATH offset=%d cells=%zu bad=%zu\n",offset,got.size(),bad);
            if (bad) throw std::runtime_error("metadata/code raw-bit mismatch");
        }
        check(cudaFree(dout)); check(cudaFree(dw)); check(cudaFree(du));
        std::puts("Q4_NATIVE_MATH PASS metadata=CANONICAL_ZMUL8 codes=SIGNED_Q4 dot=UNCHANGED_FP32");
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr,"Q4_NATIVE_MATH FAIL: %s\n",e.what()); return 1;
    }
}
