"""Precision-only adaptation of the supplied raw-GGUF GEMV reference."""
from dev.gemv_cuda.build import replace_once
from dev.gemv_ppu.build import ppu_api

REFERENCE_SHA256 = "6d575665df5f9fe16d9259178d28dc625867bdcba9967e49ad8985b0af528e65"
RECIPES = ((256, 4, 1), (512, 8, 1), (1024, 8, 1))


def reference_fp32(original, *, ppu=True):
    # Keep the input as raw GGUF and retain the reference's HALF weight
    # reconstruction. Only dot accumulators/folding/output are changed.
    src = original
    for i in range(4):
        src = replace_once(src, f"half2& dq{i}", f"float2& dq{i}")
        src = replace_once(src, f"half2 dq{i} = __float2half2_rn(0.f);",
                           f"float2 dq{i} = make_float2(0.f, 0.f);")
        group, a = ("lo", "a_lo0") if i == 0 else ("hi", "a_hi0") if i == 1 else (
                    ("lo", "a_lo1") if i == 2 else ("hi", "a_hi1"))
        old = f"dq{i} = __hfma2(__hfma2(qm.q{i}, scale_{group}2, zero_{group}2), {a}, dq{i});"
        new = f"dq{i} = fp32_dot(__hfma2(qm.q{i}, scale_{group}2, zero_{group}2), {a}, dq{i});"
        src = replace_once(src, old, new)
    helper = """__device__ __forceinline__
float2 fp32_dot(half2 w, half2 a, float2 sum) {
    float2 const wf = __half22float2(w), af = __half22float2(a);
    return make_float2(fmaf(wf.x, af.x, sum.x), fmaf(wf.y, af.y, sum.y));
}

"""
    src = replace_once(src, "__device__ __forceinline__\nvoid q4k_dot_word", helper + "__device__ __forceinline__\nvoid q4k_dot_word")
    src = replace_once(src,
        """            half2 const s = __hadd2(__hadd2(dq0, dq1), __hadd2(dq2, dq3));
            acc[ii] += __half2float(__low2half(s)) + __half2float(__high2half(s));""",
        """            float2 const s = make_float2((dq0.x+dq1.x)+(dq2.x+dq3.x),
                                         (dq0.y+dq1.y)+(dq2.y+dq3.y));
            acc[ii] += s.x + s.y;""")
    src = replace_once(src, "half*              out,", "float*             out,")
    src = replace_once(src, "const block_q4_K* w, half* out,", "const block_q4_K* w, float* out,")
    src = replace_once(src, "= __float2half(acc[col]);", "= acc[col];")
    src = replace_once(src, "= __float2half(sum);", "= sum;")
    # This comment is arithmetically wrong in the supplied file. Do not alter
    # the actual LUT expression, which evaluates to 0xea.
    src = src.replace("// 0xe4 = (A & B) | C", "// 0xea = (A & B) | C")
    # Strip the original prose header: it still describes the HALF-dot
    # original. Its exact bytes remain in reference/gemv_ref.cuh for auditing.
    src = src[src.index("#pragma once"):]
    src = src.replace("namespace q4k_gemv", "namespace q4k_gemv_fp32")
    if not ppu:
        # The supplied PPU spelling uses immediates for packed-half arithmetic.
        # NVIDIA ptxas requires register operands here (same constants/math).
        for constant in ("FP16_TOP_MAGIC", "ONE_SIXTEENTH", "NEG_72"):
            src = src.replace(f'"n"({constant})', f'"r"({constant})')
    return ppu_api(src) if ppu else src


def reference_wrapper(*, ppu=True):
    text = '''#include "gemv_ref_fp32.cuh"
#include "bload_contract.hpp"
extern "C" int q4_ref_fp32_run(int c, int w, int n, int k,
        void const* a, void const* raw, void* out, void* stream) {
    if (!q4_bload::shape(n,k) || w!=8 || (c!=1 && c!=2 && c!=4)) return -1;
    if (!a || !raw || !out || (uintptr_t(a)&15) || (uintptr_t(raw)&15) || (uintptr_t(out)&3)) return -2;
#define RAW_ARM(C) if (c==C) { q4k_gemv_fp32::launch_q4k_gemv<C,8,1>( \\
    static_cast<half const*>(a),static_cast<q4k_gemv_fp32::block_q4_K const*>(raw), \\
    static_cast<float*>(out),1,n,k,static_cast<hggcStream_t>(stream)); return int(hggcGetLastError()); }
    RAW_ARM(1)
    RAW_ARM(2)
    RAW_ARM(4)
#undef RAW_ARM
    return -1;
}
'''
    return text if ppu else text.replace("hggc", "cuda")
