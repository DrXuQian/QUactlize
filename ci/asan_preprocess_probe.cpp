// MEMORY SAFETY OF THE PREPROCESSING CHAIN, UNDER ASAN, WITH NO TORCH AND NO DEVICE.
//
// Why this is a separate program rather than a pytest: the bug it was written for -- mem_cacheline_col_tile_interleave
// writing a full buffer's length past the end of its destination -- did not fail where it happened. It corrupted the
// heap and surfaced as an intermittent Bus error or SIGSEGV several tests later, in a test that had nothing to do
// with it, about four runs in five. No amount of assertion in python would have located that; ASAN named the line on
// the first run. Keeping the chain callable without torch is what makes ASAN usable at all, since instrumenting the
// whole torch process is far more work than compiling these two translation units.
//
// It sweeps SHAPES rather than testing one, because the overflow was shape-dependent: it needed a matrix narrower
// than the interleave stride, and the square 256x256 case that was tried first is exactly the one that hides it.
//
//   g++ -std=c++17 -O1 -g -fsanitize=address -DUSE_AIU=1 \
//       -I quactlize/csrc/preprocess -I third_party/cutlass/include -I /usr/local/cuda/include \
//       -o asan_probe ci/asan_preprocess_probe.cpp quactlize/csrc/preprocess/cutlass_kernels/cutlass_preprocessors.cpp
//
// Exit status is ASAN's: it aborts on the first violation, so a zero exit is the whole verdict.
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cutlass_kernels/cutlass_preprocessors.h>

using namespace acext::kernels::cutlass_kernels;

namespace {

struct Shape {
    size_t experts, rows, cols;
};

// Deliberately includes shapes that are NOT multiples of the interleave stride, of the AIU column tile, or of each
// other. A sweep made only of round numbers tests the case that cannot fail.
const Shape kShapes[] = {
    {1, 64, 64},     {1, 64, 128},   {1, 128, 64},  {1, 128, 128},
    {1, 128, 192},   {1, 256, 64},   {1, 256, 256}, {1, 512, 128},
    {1, 512, 256},   {1, 1024, 256}, {3, 64, 128},  {2, 256, 256},
    {2, 512, 256},
};

const char* name(QuantType q)
{
    switch (q) {
        case QuantType::INT8_WEIGHT_ONLY: return "int8";
        case QuantType::PACKED_INT4_WEIGHT_ONLY: return "int4";
        case QuantType::PACKED_INT2_WEIGHT_ONLY: return "int2";
        default: return "other";
    }
}

// One (shape, quant type) through quantise -> permute -> transpose -> interleave -> bias, which is every buffer the
// preprocessing writes. The output buffers are sized the way the torch op sizes them, so an overflow here is an
// overflow there.
void probe_quantize(Shape s, QuantType q)
{
    const size_t bpc = s.cols * get_bits_in_quant_type(q) / 8;
    std::vector<half> w(s.experts * s.rows * s.cols);
    for (size_t i = 0; i < w.size(); ++i) w[i] = half(float(int(i % 17) - 8) * 0.3f);

    std::vector<int8_t> processed(s.experts * s.rows * bpc), unprocessed(s.experts * s.rows * bpc);
    std::vector<half> scales(s.experts * s.cols);

    printf("  quantize   e=%zu k=%zu n=%zu %s\n", s.experts, s.rows, s.cols, name(q));
    fflush(stdout);
    symmetric_quantize<half, half>(processed.data(), unprocessed.data(), scales.data(), w.data(),
                                   {s.experts, s.rows, s.cols}, q, 80);
}

// The layout transform on its own, over both flags. is_int8_mma and use_aiu_interleaved each take a different branch
// through preprocess_weights_for_mixed_gemm, and the AIU one is compiled in only under USE_AIU.
void probe_preprocess(Shape s, QuantType q, bool int8_mma, bool aiu)
{
    const size_t bpc = s.cols * get_bits_in_quant_type(q) / 8;
    std::vector<int8_t> in(s.experts * s.rows * bpc), out(s.experts * s.rows * bpc);
    for (size_t i = 0; i < in.size(); ++i) in[i] = int8_t(i * 7 + 3);

    std::vector<size_t> shape;
    if (s.experts == 1) shape = {s.rows, s.cols};
    else shape = {s.experts, s.rows, s.cols};

    // Printed and FLUSHED before the call, not after: ASAN aborts the process, so the last line on stdout is the
    // case that failed. Anything reported afterwards would never be written.
    printf("  preprocess e=%zu k=%zu n=%zu %s int8_mma=%d aiu=%d\n", s.experts, s.rows, s.cols, name(q), int8_mma, int(aiu));
    fflush(stdout);
    preprocess_weights_for_mixed_gemm(out.data(), in.data(), shape, q, int8_mma, aiu, 80);
}

}  // namespace

int main()
{
    // int2 IS IN THE SWEEP because the sweep is the reason to trust the fix: its MMA tile is 64 rows, and that row
    // count was wrong in the layout table until the tile height was forced into the token. A memory-safety sweep
    // that skipped the width whose row arithmetic had just changed would have proved nothing about it.
    const QuantType kTypes[] = {QuantType::PACKED_INT4_WEIGHT_ONLY, QuantType::INT8_WEIGHT_ONLY,
                                QuantType::PACKED_INT2_WEIGHT_ONLY};
    int n = 0;
    for (Shape s : kShapes) {
        for (QuantType q : kTypes) {
            // symmetric_quantize covers int4 and int8 only -- it throws "Unsupported quantization type" for int2,
            // whose codes are produced by the offline packers rather than by this quantiser. The LAYOUT transform
            // does support int2, and that is the half this sweep needs it for: int2's 64-row MMA tile is the row
            // arithmetic that was recently wrong in the layout table.
            if (q != QuantType::PACKED_INT2_WEIGHT_ONLY) {
                probe_quantize(s, q);
                ++n;
            }
            for (bool mma : {false, true}) {
                // is_int8_mma is the W4A8 path and its row permutation only fits a 32-row MMA tile, which 4-bit
                // weights give and 8-bit weights do not. The combination is refused at the permutation's own
                // precondition; running it here would test the refusal, not the chain, so it is skipped and the
                // reason is stated rather than left as an unexplained gap in the sweep.
                if (mma && q != QuantType::PACKED_INT4_WEIGHT_ONLY) continue;
                for (bool aiu : {false, true}) {
                    // The AIU branch is only reachable for packed int4 and only when both k and n are multiples of
                    // 256; asking for it elsewhere is a no-op downstream, not a fault, so it is still worth running.
                    probe_preprocess(s, q, mma, aiu);
                    ++n;
                }
            }
        }
    }
    printf("asan_preprocess_probe: %d calls over %zu shapes, no violation\n", n, sizeof(kShapes) / sizeof(kShapes[0]));
    return 0;
}
