"""WHAT IS IMPLEMENTED, AS A MATRIX OF (execution scheme x GEMM shape x format).

THE AXIS THIS FILE ADDS. formats.py answers "what does a format cost to store" and "which band routes where". It
does not answer the question that actually decides what to build next: for a given format, HOW MUCH IS EXPANDED
BEFORE THE GEMM RUNS. There are three answers and they are different kernels, not tuning knobs:

    DEQUANT_FIRST      expand the weights to fp16, then a dense fp16 GEMM. Nothing quantised reaches the math.
    SCALE_FIRST        expand only the SCALES into fp16 planes. Weights stay packed and the GEMM converts them
                       in its mainloop. This is the mixed-input GEMM and it is what everything here runs today.
    FULLY_QUANTIZED    expand nothing. The GEMM reads the format's own packed scale bytes.

Crossed with the GEMM shape -- dense, grouped (MoE), and the CUDA-core GEMV used at decode -- that is the grid a
reader needs to see before asking "is format X supported". "Supported" alone is not a fact: Q4_K is validated on
SCALE_FIRST/grouped, exists behind a build flag on FULLY_QUANTIZED/grouped, and has nothing at all on GEMV.

THE FOUR CAPABILITY SETS IN formats.py ARE CROSS-CHECKED AGAINST THIS TABLE, not derived from it -- formats.py
cannot import this module, because this one imports formats.py for QuantType. The distinction matters because the
header used to claim derivation and there was none: the sets were literals, capability() below had no callers at
all, and the two places drifted exactly as the warning said they would. tests/test_schemes_consistency.py is what
makes the claim true; capability() is what it compares against.

WHAT THE STATUSES MEAN, and why PARTIAL exists. A path can have its pieces without having the path: DEQUANT_FIRST
has a working dequantiser -- unfused_weight_dequantize.hpp, which produces the fp16 reference every other harness
compares against -- and no GEMM that consumes it in production. Calling that "absent" hides work already done;
calling it "implemented" claims a path nobody can run.

    python -m quactlize.schemes           the matrix
    python -m quactlize.schemes --gaps    only what is missing, with what blocks each one
"""
import sys
from enum import Enum, IntEnum
from typing import Dict, FrozenSet, NamedTuple, Tuple

from .formats import QuantType


class Scheme(Enum):
    DEQUANT_FIRST = "dequant_first"
    SCALE_FIRST = "scale_first"
    FULLY_QUANTIZED = "fully_quantized"


class Shape(Enum):
    DENSE = "dense"
    GROUPED = "grouped"          # MoE prefill/mid band: many experts, ragged rows, one tensor-core launch
    GEMV = "gemv"                # DENSE decode: one token, CUDA cores, no tensor-core tile
    # MoE AT DECODE, AND IT IS NOT EITHER OF THE OTHER TWO. GROUPED is the tensor-core mixed-input GEMM and
    # assumes enough rows per expert to fill a tile; GEMV assumes ONE weight. MoE decode is top-k experts, one
    # token each, gathered rows -- a different launch (experts on a grid dimension) and a different kernel.
    # It had no cell, which hid two facts at once: that gemv_lowbit already serves it through its `Grouped`
    # template parameter, and that the native-scale route does NOT. And it is the band the whole splitk bench
    # pins (L=64, top-k 8, N=K=2048, gs=32), so it is not a corner -- it is where an MoE model spends decode.
    GEMV_MOE = "gemv_moe"


class Status(IntEnum):
    ABSENT = 0        # no code path
    PARTIAL = 1       # a necessary piece exists; the path does not
    IMPLEMENTED = 2   # the path runs, but no INDEPENDENT oracle covers it (a self-comparison is not one)
    VALIDATED = 3     # an independent oracle covers it -- a CPU golden, synthetic or over real checkpoint bytes


class Impl(NamedTuple):
    status: Status
    launcher: str        # the file that would run it, or what exists instead
    flag: str = ""       # build flag it sits behind; "" means it is in the default build
    note: str = ""


# The weight CODE plane, which is not the same axis as the format. Q4_K and GPTQ-int4 both carry 4-bit codes and
# differ only in their scale channel; Q3_K/Q5_K/Q6_K carry TWO planes because their code widths are not powers of
# two that swzl can deliver. The two-plane formats run through a different collective, which is why they are absent
# from FULLY_QUANTIZED for a structural reason and not an unfinished one.
CODE_PLANE: Dict[QuantType, str] = {
    QuantType.Q2_K: "i2",
    QuantType.Q3_K: "i2+i1",
    QuantType.Q4_K: "i4",
    QuantType.Q5_K: "i4+i1",
    QuantType.Q6_K: "i4+i2",
    QuantType.GPTQ_INT4_SYM: "i4",
    QuantType.GPTQ_INT4_ASYM: "i4",
    QuantType.AWQ_INT4: "i4",
}
TWO_PLANE = frozenset({QuantType.Q3_K, QuantType.Q5_K, QuantType.Q6_K})

_GROUPED = "quactlize/include/moe_grouped_ppu.cuh"
_DENSE = "quactlize/include/fpA_intB_ppu.cuh"
_GEMV = "quactlize/include/gemv_lowbit/"
_DEQ = "quactlize/include/unfused_weight_dequantize.hpp"

_ROUTES = "quactlize/routes.py"

_DEQUANT_DENSE = Impl(
    Status.VALIDATED, _ROUTES, note=(
        "RUNS END TO END: raw GGUF blocks -> fp16 weight -> torch's cuBLAS -> (m, n), against the official gguf "
        "package as oracle for all five k-quants at m in (1, 7, 64), worst relative error 1.05e-3 which is the fp16 "
        "weight's own floor. Exercised on a CUDA tensor as well, so the GEMM half is the library and not a CPU "
        "matmul. This sat at PARTIAL for longer than it deserved on a note saying the remainder was 'host-side "
        "wiring' -- the remainder was one line, and the cost of not writing it was that no two routes had ever "
        "produced the same number in the same process. Cost is why it stays a FALLBACK, not why it stayed unwired: "
        "a dequantised fp16 weight is 4x the int4 codes and ~3.6x the native block, so it pays only where the "
        "result is reused across many rows"))

_DEQUANT_GROUPED = Impl(
    Status.VALIDATED, _ROUTES, note=(
        "the same route with a per-expert weight and ragged rows, validated with DIFFERENT bytes per expert (so "
        "reading expert 0 for everyone fails) and a zero-row expert (so the skip branch runs). The GEMM is one "
        "library call per expert; substituting DeepGemm changes the GEMM, not the route, and the arrangement it "
        "would need is the same expert-ordered gather the grouped kernels already assume"))

_NO_PACKED_2PLANE = Impl(
    Status.ABSENT, "", note=(
        "two-plane formats run through the SEPARATE two-plane collective, which has no packed-scale plumbing at "
        "all. This is structural, not unfinished, and generalising the multistage collective does NOT reach it: "
        "that work unlocks Q2_K only. The scale UNIT is format-general now (gguf_packed_unit.hpp, byte-neutral at "
        "20/14/16/16/18 against GGUF's own, bit-exact round trip for all five in CI) -- the unit was the "
        "prerequisite, not the feature"))

_NO_PACKED_TRAITS = Impl(
    Status.ABSENT, "", note=(
        "single-plane, so the shape fits. The scale unit is generalised already -- gguf_packed_unit.hpp derives "
        "every number from a trait and reproduces the shipped Q4_K bit positions exactly -- but the COLLECTIVE "
        "still hardcodes 16 bytes per unit, one 128-bit cp.async, Scale_TileK==8 and four 32-bit words, and the "
        "unit size genuinely differs per format (20 for Q2_K), which is what that single cp.async assumes away. "
        "The staging tile and its copy have to come from the trait"))


# (scheme, shape, format) -> Impl. Absent entries mean ABSENT with no note worth writing.
IMPL: Dict[Tuple[Scheme, Shape, QuantType], Impl] = {}


def _add(scheme, shape, fmts, impl):
    for f in fmts:
        IMPL[(scheme, shape, f)] = impl


_KQUANTS = (QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K)
_ALL = tuple(QuantType)

# --- DEQUANT_FIRST: a dequantiser without a GEMM ----------------------------------------------------------------
_add(Scheme.DEQUANT_FIRST, Shape.DENSE, _KQUANTS, _DEQUANT_DENSE)
_add(Scheme.DEQUANT_FIRST, Shape.GROUPED, _KQUANTS, _DEQUANT_GROUPED)
# GPTQ is NOT covered by the above: routes.py reads k-quant blocks, and the symmetric packed forms go through
# unfused_weight_dequantize.hpp, which has no host binding. Left PARTIAL rather than folded in, because a table
# that generalises from five formats to six is how a claim outruns its test.
_add(Scheme.DEQUANT_FIRST, Shape.DENSE, (QuantType.GPTQ_INT4_SYM,), Impl(
    Status.PARTIAL, _DEQ, note="dequantiser exists in C++; no host binding, so the route cannot be called"))
_add(Scheme.DEQUANT_FIRST, Shape.GROUPED, (QuantType.GPTQ_INT4_SYM,), Impl(
    Status.PARTIAL, _DEQ, note="dequantiser exists in C++; no host binding, so the route cannot be called"))

# --- SCALE_FIRST: the workhorse ---------------------------------------------------------------------------------
# Grouped, validated against an INDEPENDENT oracle. The oracle strength per format is in ci/registry.py and is
# cross-checked against this table by check_against_registry() below; the distinction that matters here is only
# independent-or-not, which is why a self-comparison leaves a path at IMPLEMENTED.
_add(Scheme.SCALE_FIRST, Shape.GROUPED,
     (QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K,
      QuantType.GPTQ_INT4_SYM),
     Impl(Status.VALIDATED, _GROUPED, note="two-plane formats reach it through the PlaneB2 template parameter"))

_add(Scheme.SCALE_FIRST, Shape.DENSE, (QuantType.GPTQ_INT4_SYM,),
     Impl(Status.IMPLEMENTED, _DENSE, note=(
         "test_fpA_intB_ppu compares the launcher against ANOTHER CONFIGURATION OF ITSELF. That catches plumbing "
         "and is structurally blind to a wrong shared constant, since both sides move together")))

# --- MoE AT DECODE. The column that did not exist, and what it exposes ---------------------------------------
# gemv_lowbit ALREADY serves this: gemv_exec launches dim3(grid_m, n/CtaN, Grouped ? num_experts : 1), so experts
# are a grid dimension, and test_gemv_lowbit drives it with num_experts/row_offsets/max_rows. That was invisible
# while MoE decode shared a cell with dense decode.
_add(Scheme.SCALE_FIRST, Shape.GEMV_MOE, (QuantType.GPTQ_INT4_SYM,), Impl(
    Status.IMPLEMENTED, _GEMV, note=(
        "the Grouped arm: experts on the grid's z dimension, ragged rows through row_offsets. IMPLEMENTED rather "
        "than VALIDATED because ci/registry.py has ONE path name, 'gemv', for both MoE and dense decode -- so the "
        "synthetic oracle that covers the dense arm would silently approve this one. schemes._CELL_PATH_IS_COARSE "
        "names that, and a consistency test refuses to let this cell be called VALIDATED until the vocabulary is "
        "split")))
_add(Scheme.SCALE_FIRST, Shape.GEMV_MOE, _KQUANTS, Impl(
    Status.PARTIAL, _GEMV, note=(
        "same launcher, same blocker as the dense GEMV cell: it reads fp16 scale planes, which at decode must be "
        "resident, and that is the stored-byte increase the constraint forbids")))

# THE GAP THIS COLUMN EXISTS TO MAKE VISIBLE. The native-scale GEMV is DENSE-ONLY: vecdot_rows_kernel takes one
# weight, one row range, no expert gather and no routing. Measured at MoE-sized WORK (16384 rows = 8 experts x
# 2048) the dense kernel reaches 51.6% of peak cold and 68.1% warm against the tensor-core collective's 29.87%
# Memory SoL at the same band -- but that number has no gather in it, so it is an UPPER BOUND on what a real MoE
# GEMV would reach, not a result. This is the cell A3B's decode actually needs, and it is empty.
_add(Scheme.FULLY_QUANTIZED, Shape.GEMV_MOE, _KQUANTS, Impl(
    Status.ABSENT, "", note=(
        "vecdot_rows_kernel is dense-only: one weight, no expert gather, no routing, no ragged rows. The band an "
        "MoE model spends decode in -- top-k experts, one token each -- has no native-scale kernel. Recorded "
        "elsewhere as 'A3B decode bulk unaccelerated, needs a separate MoE GEMV'")))

_add(Scheme.DEQUANT_FIRST, Shape.GEMV_MOE, _KQUANTS, Impl(
    Status.VALIDATED, _ROUTES, note=(
        "routes.matmul_dequant_first_grouped is shape-agnostic in rows, so the decode band is one instance of what "
        "the grouped test already covers with ragged rows including an empty expert. What it is NOT is efficient: "
        "it materialises every selected expert's weight in fp16, which at decode is the whole point of not doing")))

_add(Scheme.SCALE_FIRST, Shape.GEMV, (QuantType.GPTQ_INT4_SYM,),
     Impl(Status.VALIDATED, _GEMV, note=(
         "test_gemv_lowbit's scale-only int4 sweep IS the GPTQ symmetric representation -- same packing, same fp16 "
         "scale dtype, zero folded into the code range. Synthetic: no real checkpoint reaches the GEMV")))

# Every k-quant is absent from GEMV, and for one reason: these GEMV kernels read fp16 scale planes, and at decode
# those planes have to be resident, which is exactly the stored-byte increase the product constraint forbids.
_add(Scheme.SCALE_FIRST, Shape.GEMV, _KQUANTS, Impl(
    Status.PARTIAL, _GEMV, note=(
        "the GEMV kernels read fp16 scale planes, which at decode must be resident -- the stored-byte increase the "
        "constraint forbids. The PRE-PASS does not fix that: it removes the storage cost by building the planes in "
        "a workspace, and at decode they would be rebuilt every token. What changed is that the offline chain now "
        "reaches this launcher -- gguf_unpack then pack_int4 then preprocess_weights_to_layout, all validated ops -- "
        "and the row permutation was MEASURED to stay inside its own 32-block for both int4 and int8, so a scale "
        "plane indexed by k//gs is unaffected by the reorder. So this is wiring, not a kernel. The native-scale "
        "answer for this band is the FULLY_QUANTIZED entry")))

# --- FULLY_QUANTIZED: one format, behind a flag -----------------------------------------------------------------
_add(Scheme.FULLY_QUANTIZED, Shape.GROUPED, (QuantType.Q4_K,), Impl(
    Status.VALIDATED, _GROUPED, flag="PPU_PACKED_SCALE=1", note=(
        "numerically validated on real Q4_K checkpoint bytes, but ONLY test_q4k_packed_gemm's rowC exercises the "
        "packed decoder -- rowA and rowB are fp16-path controls. Consumes native scale CODES in a reordered 16-byte "
        "unit, not GGUF's on-disk 12-byte packing, which is not half-separable; the reorder is byte-neutral. "
        "PERFORMANCE IS UNMEASURED: every figure recorded before commit 80dfeec came from a bench whose two paths "
        "computed different numbers")))
_add(Scheme.FULLY_QUANTIZED, Shape.GROUPED, (QuantType.Q2_K,), _NO_PACKED_TRAITS)
_add(Scheme.FULLY_QUANTIZED, Shape.GROUPED, tuple(TWO_PLANE), _NO_PACKED_2PLANE)
_add(Scheme.FULLY_QUANTIZED, Shape.GEMV, _KQUANTS, Impl(
    Status.PARTIAL, "quactlize/include/gguf_vecdot.hpp", note=(
        "the band with no M reuse, so the mid band's shared-memory publication buys nothing and the scale pair has "
        "to be consumed where it is produced. The DECODE exists for all five and is checked as a DOT PRODUCT "
        "against the official gguf package -- which the per-group scale tests cannot do, since a decoder with the "
        "right scales and the wrong element order passes those. dp4a is ruled out by decision: it quantises the "
        "activation and this band has no second chance at the error, and PPU is confirmed to have an int8 TENSOR "
        "core but not a cuda-core four-way int8 dot. THE HOST SIDE IS WIRED NOW -- routes.matmul_native_gemv "
        "assembles a full (1, n) product from the per-block op. Its CPU fp32 witness agrees with the official gguf "
        "package to 1.4e-7; the direct CUDA fp16/half2 kernel has a separate conditioned-error gate at 2^-11 and "
        "observes 1.05e-4..2.02e-4. It stays PARTIAL because that assembly tiles the ACTIVATION k/256-fold on the "
        "host: correct, and useless as a timing. The native kernel is subgroup-cooperative and CUDA-validated for "
        "all five; at rows=131072, bpr=8 on a 5090 its cold element rate is 1659..3361 Gelem/s and it reaches "
        "55.4%..76.1% of peak HBM bandwidth. A runtime rows/bpr policy separately covers the 2048-row decode shape. "
        "Adjacent Q4/Q5 groups share one 32-byte run per owner lane, and every format extracts four codes from each "
        "aligned word with checked fallbacks and actlize's shared byte4-to-half converter. What "
        "keeps this PARTIAL is device-library wiring: "
        "libquactlize_ppu.so still needs to export a launcher for this kernel before the Python route is a PPU "
        "inference path")))


def get(scheme: Scheme, shape: Shape, fmt: QuantType) -> Impl:
    return IMPL.get((scheme, shape, fmt), Impl(Status.ABSENT, ""))


def capability(scheme: Scheme, *shapes: Shape, minimum: Status = Status.IMPLEMENTED) -> FrozenSet[QuantType]:
    """Formats reaching `minimum` on this scheme, over the given shapes (all shapes if none are named).

    This is what formats.py's capability sets are built from. They used to be written out beside this information;
    deriving them means a format cannot be claimed in one place and missing in the other."""
    want = shapes or tuple(Shape)
    return frozenset(f for f in QuantType
                     if any(get(scheme, s, f).status >= minimum for s in want))


# (scheme, shape) -> the registry's path name. The registry has carried per-(format, path) evidence all along --
# coverage_by_path() -- and this file was calling the WEAKER coverage(), which answers "does this format have an
# oracle anywhere". Under that question a format validated on one cell approved every other cell for free: ten new
# VALIDATED cells were added and the check reported no problems, and a planted claim that Q2_K was validated on
# fully_quantized/dense -- a path with no harness in existence -- also reported no problems. The strong function
# was already written; it was simply not the one wired in.
_CELL_PATH = {
    (Scheme.DEQUANT_FIRST, Shape.DENSE): "dequant_then_dense",
    (Scheme.DEQUANT_FIRST, Shape.GROUPED): "dequant_then_dense",
    (Scheme.SCALE_FIRST, Shape.DENSE): "fused_fp16_scale",
    (Scheme.SCALE_FIRST, Shape.GROUPED): "fused_fp16_scale",
    (Scheme.SCALE_FIRST, Shape.GEMV): "gemv",
    (Scheme.FULLY_QUANTIZED, Shape.DENSE): "fused_native_scale",
    (Scheme.FULLY_QUANTIZED, Shape.GROUPED): "fused_native_scale",
    (Scheme.FULLY_QUANTIZED, Shape.GEMV): "gemv",
    (Scheme.DEQUANT_FIRST, Shape.GEMV): "dequant_then_dense",
    (Scheme.SCALE_FIRST, Shape.GEMV_MOE): "gemv",
    (Scheme.FULLY_QUANTIZED, Shape.GEMV_MOE): "gemv",
    (Scheme.DEQUANT_FIRST, Shape.GEMV_MOE): "dequant_then_dense",
}

# WHERE THE PATH VOCABULARY IS COARSER THAN THE CELL. ci/registry.py has four path names and this table has nine
# cells, so some cells share a name -- and sharing a name means sharing EVIDENCE. Two of those shares are wrong
# enough to matter:
#
#   fully_quantized/gemv  borrows "gemv", whose only harness is the fp16-PLANE GEMV. Native-scale and fp16-plane
#                         GEMV are precisely the two routes the decode band must choose between.
#   dequant_first/gemv    borrows "dequant_then_dense", whose harness runs a GEMM at m in (1, 7, 64). A materialise-
#                         then-GEMV route shares the materialisation and nothing else.
#
# Left mapped rather than unmapped, because an UNMAPPED cell is approved by default: the lookup that should refuse
# it finds nothing to check and returns no problem. Mapped-but-listed-here is refused instead -- a consistency test
# asserts no cell in this set is VALIDATED, so the day one is, the vocabulary has to be split first.
_CELL_PATH_IS_COARSE = frozenset({
    (Scheme.FULLY_QUANTIZED, Shape.GEMV),
    # The registry's single "gemv" cannot tell MoE decode from dense decode either. gemv_lowbit's grouped arm is
    # a different launch -- experts on the grid's z dimension, rows gathered through row_offsets -- and while
    # test_gemv_lowbit does drive it, the registry records ONE path name for both, so evidence for either arm
    # would silently approve the other.
    (Scheme.SCALE_FIRST, Shape.GEMV_MOE),
    (Scheme.FULLY_QUANTIZED, Shape.GEMV_MOE),
})
# THE DEQUANT_FIRST GEMV CELLS ARE NOT COARSE, and I had them here until this test refused a VALIDATED claim on
# one. Under this scheme a GEMV is not a different kernel: the weight is materialised to fp16 and handed to the
# same library call, so "GEMV" is that GEMM at m=1 and MoE decode is the grouped route at one row per expert.
# test_gguf_routes.py runs m in (1, 7, 64) and drives the grouped route with ragged rows including an empty
# expert, so the evidence is the same CODE, not a neighbouring one. Coarseness is about the kernel differing,
# not about the cell label differing -- putting them here was a category error, and the gate caught it.


def check_against_registry():
    """Every VALIDATED cell must have a harness for THAT (format, path). Returns a list of problems.

    One direction only, deliberately: a format may have a real oracle for one scheme and nothing for another, so a
    registry entry does not imply a cell here. What must not happen is the reverse -- this table calling a path
    validated with nothing behind it."""
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ci"))
    try:
        import registry
    except Exception as e:
        return [f"cannot import ci/registry.py to check VALIDATED claims: {e}"]

    have = registry.coverage_by_path()
    quant_to_format = {v: k for k, v in registry.FORMAT_TO_QUANT_TYPE.items()}
    bad = []
    for (scheme, shape, fmt), impl in sorted(IMPL.items(), key=lambda kv: (kv[0][0].value, kv[0][1].value, kv[0][2].name)):
        if impl.status < Status.VALIDATED:
            continue
        cell = f"{scheme.value}/{shape.value}/{fmt.name}"
        name = quant_to_format.get(fmt.name)
        path = _CELL_PATH.get((scheme, shape))
        if name is None:
            bad.append(f"{cell} is VALIDATED but maps to no registry format")
        elif path is None:
            bad.append(f"{cell} is VALIDATED but ({scheme.value}, {shape.value}) maps to no registry path")
        elif not have.get((name, path)):
            bad.append(f"{cell} is VALIDATED, but no harness runs {name!r} through {path!r}")
    return bad


_MARK = {Status.ABSENT: "  -  ", Status.PARTIAL: " ~~~ ", Status.IMPLEMENTED: " impl", Status.VALIDATED: " OK  "}


def report(gaps_only: bool = False) -> str:
    out = ["== quactlize: how much is expanded before the GEMM runs ==",
           "   dequant_first    weights -> fp16, then a dense fp16 GEMM",
           "   scale_first      scales -> fp16 planes; weights stay packed (the mixed-input GEMM)",
           "   fully_quantized  nothing expanded; the GEMM reads the format's own packed scale bytes",
           "",
           "   OK = an independent oracle covers it   impl = runs, no independent oracle",
           "   ~~~ = a piece exists, the path does not   -  = absent",
           ""]
    if not gaps_only:
        hdr = f"   {'format':<15} {'codes':<7}"
        for sc in Scheme:
            for sh in Shape:
                hdr += f" {sc.value[:5]}/{sh.value[:5]:<5}"
        out += [hdr, "   " + "-" * (len(hdr) - 3)]
        for f in QuantType:
            if f not in CODE_PLANE:
                continue
            row = f"   {f.name:<15} {CODE_PLANE[f]:<7}"
            for sc in Scheme:
                for sh in Shape:
                    row += f" {_MARK[get(sc, sh, f).status]:<11}"
            out.append(row)
        out.append("")

    out.append("   what blocks each missing or unproven cell:")
    seen = set()
    for (sc, sh, f), impl in IMPL.items():
        if impl.status >= Status.VALIDATED or not impl.note or impl.note in seen:
            continue
        seen.add(impl.note)
        # The cells sharing THIS Impl object, listed as scheme/shape: formats. One object is deliberately shared by
        # several cells when one sentence explains all of them; listing formats alone printed each name once per
        # cell, so a note covering dense and grouped looked like a duplicated format list.
        cells = {}
        for (a, b, x), i in IMPL.items():
            if i is impl:
                cells.setdefault(f"{a.value}/{b.value}", set()).add(x.name)
        out.append(f"     [{_MARK[impl.status].strip()}] "
                   + "; ".join(f"{k}: {', '.join(sorted(v))}" for k, v in sorted(cells.items())))
        for line in _wrap(impl.note, 104):
            out.append(f"           {line}")
    flagged = [(sc, sh, f, i) for (sc, sh, f), i in IMPL.items() if i.flag]
    if flagged:
        out.append("")
        out.append("   behind a build flag (present, not in the default build):")
        for sc, sh, f, i in flagged:
            out.append(f"     {sc.value}/{sh.value}/{f.name}: {i.flag}")
    return "\n".join(out)


def _wrap(text, width):
    words, line, lines = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines


if __name__ == "__main__":
    print(report(gaps_only="--gaps" in sys.argv))
    problems = check_against_registry()
    print()
    for p in problems:
        print(f"   PROBLEM  {p}")
    print(f"   {len(problems)} problem(s) cross-checking VALIDATED claims against ci/registry.py")
    sys.exit(1 if problems else 0)
