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

THE FOUR CAPABILITY SETS IN formats.py ARE NOW DERIVED FROM THIS TABLE rather than written beside it. They were a
second place where the same facts lived, and a second place is where drift starts.

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
    GROUPED = "grouped"          # MoE: many experts, ragged rows, one launch
    GEMV = "gemv"                # decode: one token, CUDA cores, no tensor-core tile


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

_NO_DEQUANT_GEMM = Impl(
    Status.PARTIAL, _DEQ, note=(
        "dequantize_weight exists and is correct -- every harness's fp16 reference comes from it -- but nothing "
        "consumes it as a production path. There is no dense fp16 GEMM launcher here and no harness with an "
        "independent oracle, which is why formats.DEQUANT_THEN_DENSE is empty"))

_NO_PACKED_2PLANE = Impl(
    Status.ABSENT, "", note=(
        "two-plane formats run through the SEPARATE two-plane collective, which has no packed-scale plumbing at "
        "all. This is structural, not unfinished: the packed path's staging, unit size and converter all assume a "
        "single plane"))

_NO_PACKED_TRAITS = Impl(
    Status.ABSENT, "", note=(
        "single-plane, so the shape fits, but the packed path hardcodes Q4_K: kPackedScaleBias=0, kPackedHasMin, "
        "kPackedZMul=8, 16 bytes per unit, one 128-bit cp.async, Scale_TileK==8, four words and 6-bit extraction. "
        "Generalising is a trait-driven change to the staging skeleton, not three constants"))


# (scheme, shape, format) -> Impl. Absent entries mean ABSENT with no note worth writing.
IMPL: Dict[Tuple[Scheme, Shape, QuantType], Impl] = {}


def _add(scheme, shape, fmts, impl):
    for f in fmts:
        IMPL[(scheme, shape, f)] = impl


_KQUANTS = (QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K)
_ALL = tuple(QuantType)

# --- DEQUANT_FIRST: a dequantiser without a GEMM ----------------------------------------------------------------
_add(Scheme.DEQUANT_FIRST, Shape.DENSE, _KQUANTS + (QuantType.GPTQ_INT4_SYM,), _NO_DEQUANT_GEMM)
_add(Scheme.DEQUANT_FIRST, Shape.GROUPED, _KQUANTS + (QuantType.GPTQ_INT4_SYM,), _NO_DEQUANT_GEMM)

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

_add(Scheme.SCALE_FIRST, Shape.GEMV, (QuantType.GPTQ_INT4_SYM,),
     Impl(Status.VALIDATED, _GEMV, note=(
         "test_gemv_lowbit's scale-only int4 sweep IS the GPTQ symmetric representation -- same packing, same fp16 "
         "scale dtype, zero folded into the code range. Synthetic: no real checkpoint reaches the GEMV")))

# Every k-quant is absent from GEMV, and for one reason: these GEMV kernels read fp16 scale planes, and at decode
# those planes have to be resident, which is exactly the stored-byte increase the product constraint forbids.
_add(Scheme.SCALE_FIRST, Shape.GEMV, _KQUANTS, Impl(
    Status.ABSENT, "", note=(
        "the GEMV kernels read fp16 scale planes; at decode those must be resident, and for a k-quant that is the "
        "stored-byte increase the constraint forbids. A k-quant GEMV needs the NATIVE scale, i.e. an entry under "
        "FULLY_QUANTIZED, not a harness here")))

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
    Status.ABSENT, "", note=(
        "the band with no M reuse at all, so the mid band's shared-memory publication buys nothing. llama.cpp uses "
        "MMVQ here, extracting only the scale pair each vec-dot fragment consumes, straight into registers")))


def get(scheme: Scheme, shape: Shape, fmt: QuantType) -> Impl:
    return IMPL.get((scheme, shape, fmt), Impl(Status.ABSENT, ""))


def capability(scheme: Scheme, *shapes: Shape, minimum: Status = Status.IMPLEMENTED) -> FrozenSet[QuantType]:
    """Formats reaching `minimum` on this scheme, over the given shapes (all shapes if none are named).

    This is what formats.py's capability sets are built from. They used to be written out beside this information;
    deriving them means a format cannot be claimed in one place and missing in the other."""
    want = shapes or tuple(Shape)
    return frozenset(f for f in QuantType
                     if any(get(scheme, s, f).status >= minimum for s in want))


def check_against_registry():
    """Every VALIDATED cell must have an oracle in ci/registry.py. Returns a list of problems.

    One direction only, deliberately: a format may have a real oracle for one scheme and nothing for another, so a
    registry entry does not imply a cell here. What must not happen is the reverse -- this table calling a path
    validated with nothing behind it."""
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ci"))
    try:
        import registry
    except Exception as e:
        return [f"cannot import ci/registry.py to check VALIDATED claims: {e}"]

    have = registry.coverage()
    quant_to_format = {v: k for k, v in registry.FORMAT_TO_QUANT_TYPE.items()}
    bad = []
    for (scheme, shape, fmt), impl in sorted(IMPL.items(), key=lambda kv: (kv[0][0].value, kv[0][1].value, kv[0][2].name)):
        if impl.status < Status.VALIDATED:
            continue
        name = quant_to_format.get(fmt.name)
        if name is None:
            bad.append(f"{scheme.value}/{shape.value}/{fmt.name} is VALIDATED but maps to no registry format")
        elif not have.get(name):
            bad.append(f"{scheme.value}/{shape.value}/{fmt.name} is VALIDATED, but no harness validates {name!r}")
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
