"""One-seam A-staging experiment over the measured Q4 N4/S1 bodies."""
from pathlib import Path

from dev.gemv_cuda.build import q4_large_static_source, replace_once
from dev.gemv_ppu.build import candidate_validation, ppu_api, q4_dispatch

ROOT = Path(__file__).resolve().parents[2]

# Fixed winners from q4-simt-ppu.0uCEmO.results.tgz, SHA256
# 94eab9aa135647ce40b00f6d733139ac1fc379bac9db7657858ead7f883ac8db.
# This is experiment replay, not a new production tactic table.
CASES = (
    (512, 2048, "warm", (4, 8, 1), (1, 8, 1)),
    (512, 2048, "rotating", (2, 4, 1), (1, 8, 1)),
    (1024, 5120, "warm", (2, 10, 1), (1, 8, 1)),
    (1024, 5120, "rotating", (2, 10, 1), (1, 8, 1)),
    (4096, 2048, "warm", (8, 16, 1), (2, 8, 1)),
    (4096, 2048, "rotating", (8, 16, 1), (2, 8, 1)),
    (4096, 4096, "warm", (8, 16, 1), (2, 8, 1)),
    (4096, 4096, "rotating", (8, 16, 1), (2, 8, 1)),
    (5120, 8192, "warm", (4, 8, 1), (2, 8, 1)),
    (5120, 8192, "rotating", (4, 8, 1), (2, 8, 1)),
    (8192, 5120, "warm", (8, 10, 1), (4, 8, 1)),
    (8192, 5120, "rotating", (8, 10, 1), (4, 8, 1)),
)

STAGE = """    // Cooperative 16-byte copies, one owner per A vector. All threads
    // reach this barrier, including workers with no final K group.
    extern __shared__ float4 staged_a_vectors[];
    auto staged_a = reinterpret_cast<__half*>(staged_a_vectors);
    #pragma unroll
    for (int pass = 0; pass < (K / 8 + Warps * 32 - 1) / (Warps * 32); ++pass) {
        int const i = pass * Warps * 32 + tid;
        if (i < K / 8)
            reinterpret_cast<float4*>(staged_a)[i] =
                reinterpret_cast<float4 const*>(c.a)[i];
    }
    __syncthreads();
"""


def static_body(source, name):
    start = source.index("template<int Columns,int Warps,int N,int K>\n__global__ void " + name)
    end = source.index("\n}\n", start) + 3
    return start, end, source[start:end]


def stage_body(body):
    body = replace_once(body, "    auto low=", STAGE + "    auto low=")
    return replace_once(body, "aligned_activation<Type>(c.a,", "aligned_activation<Type>(staged_a,")


def restore_body(body):
    """Removing exactly the A seam must recover the original entire body."""
    body = replace_once(body, STAGE, "")
    return replace_once(body, "aligned_activation<Type>(staged_a,", "aligned_activation<Type>(c.a,")


def source(original):
    result = q4_large_static_source(original)
    for name in ("kpack_q4_small_static", "kpack_q4_large_static"):
        begin, end, before = static_body(result, name)
        after = stage_body(before)
        if restore_body(after) != before:
            raise ValueError("A staging changed an unrelated compute seam")
        result = result[:begin] + after + result[end:]
        result = result.replace(name, name + "_astage")

    # Match Xplane's dynamically allocated A staging; the existing static
    # reduction scratch remains a separate allocation with unchanged lifetime.
    import re
    result, count = re.subn(
        r"(kpack_q4_(?:small|large)_static_astage<[^\n]+><<<n4_grid,Warps\*32,)0(,stream>>>\(c\);)",
        r"\g<1>size_t(c.k)*sizeof(__half)\2", result)
    if count != 6:
        raise ValueError("static A staging launch extents changed")

    # Instantiate only the five replay geometries, with the original launch
    # template. Exact shape/geometry admission below prevents generic fallback.
    begin = result.index('extern "C" int QKG_CONCAT(qkg_launch_,QKG_QTYPE)')
    recipes = sorted({row[3][:2] for row in CASES})
    result = result[:begin] + """extern "C" int qkg_launch_12(qkg_call_v1 const&, qkg_config_v1 const&) {
    return QKG_INVALID;
}
extern "C" int qkg_pair_launch_12(qkg_call_v1 const& c, qkg_config_v1 const& f) {
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
    if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
"""
    for columns, warps in recipes:
        result += (f"    if(f.columns=={columns} && f.warps=={warps}) "
                   f"return launch<{columns},{warps},true>(c,f.split);\n")
    result += "    return QKG_INVALID;\n}\n"
    return ppu_api(result.replace("QKG_CONCAT(kpack_q,QKG_QTYPE)", "kpack_ppu_astage_q12"))


def dispatch(original):
    result = q4_dispatch(original)
    guard = "if(c->qtype!=12) return QKG_FORMAT;"
    allowed = sorted({(n, k, *recipe[:2]) for n, k, _, recipe, _ in CASES})
    shape = " ||\n        ".join(
        f"(c->n=={n} && c->k=={k} && f->columns=={col} && f->warps=={warps})"
        for n, k, col, warps in allowed)
    gate = """
    if(c->mode!=QKG_DENSE || c->rows!=1 || c->experts!=1 ||
       c->input_type!=QKG_F16 || f->split!=1 ||
       (uintptr_t(c->a)&15) || (uintptr_t(c->low)&7) || (uintptr_t(c->units)&15))
        return QKG_INVALID;
    if(!(""" + shape + ")) return QKG_SHAPE;"
    if result.count(guard) != 2:
        raise ValueError("Q4 query boundary changed")
    return result.replace(guard, guard + gate)


def validation(original):
    return candidate_validation(original)
