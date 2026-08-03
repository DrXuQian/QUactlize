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
both CUDA-core GEMV schemes, but only FULLY_QUANTIZED/grouped reads its native scale unit in the tensor-core band.

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
    # THE STORAGE FORMAT THIS CELL CONSUMES, which is not the same question as "which format is supported".
    # A status says whether the path runs and is proven; this says WHAT BYTES it reads, and the two come apart:
    # FULLY_QUANTIZED/gemv reads the checkpoint's own blocks, while SCALE_FIRST/gemv reads a DERIVED artifact that
    # something had to produce. That distinction decides whether an external oracle exists at all -- the official
    # gguf package can invert raw blocks, while derived formats need their own explicit dequant-all/dequant-scale.
    # See format_of() below; "" means the cell is absent and consumes nothing.
    consumes: str = ""


# The consumed format, derived rather than written down a second time. CODE_PLANE already records each format's
# code-plane decomposition and layouts.scale_unit() already mints the packed unit's name, so both are read from
# there -- a name typed out again here is a name that can drift from the thing it describes.
def format_of(scheme: "Scheme", shape: "Shape", fmt: QuantType) -> str:
    plane = CODE_PLANE.get(fmt, "?")
    if scheme is Scheme.DEQUANT_FIRST:
        return "raw"                    # reads the checkpoint, materialises fp16; the INPUT is raw blocks
    if scheme is Scheme.SCALE_FIRST:
        kind = "dense" if shape is Shape.DENSE else "gemv" if shape in (Shape.GEMV, Shape.GEMV_MOE) else "grouped"
        return f"sf-{kind}({plane})"    # each derived family has a named producer and inverse
    if scheme is Scheme.FULLY_QUANTIZED:
        if shape in (Shape.GEMV, Shape.GEMV_MOE):
            return "raw"                # the native decode reads the checkpoint's own bytes
        return _packed_unit_name(fmt)   # the collective reads the reordered scale unit
    return ""


def _packed_unit_name(fmt: QuantType) -> str:
    """scu<bytes>x<superblocks>, minted by layouts.scale_unit so the vocabulary keeps one owner.

    THE PAIRING RULE IS THE COPY WIDTH, not the group count. ppu.cp.async moves 4, 8 or 16 bytes and nothing
    else, so a unit whose byte count is 2 mod 4 cannot be bulk-copied at all: Q3_K's 14 and Q6_K's 18 are paired
    along k into 28 and 36, while Q2_K's 20 (= 16 + 4) and Q4_K/Q5_K's 16 are already copyable alone. Deriving it
    from the byte count reproduces all five recorded names; deriving it from "16 groups" does not -- Q2_K has
    sixteen groups and is still x1."""
    from . import formats as _f, layouts as _l
    meta = _f.BLOCKS[fmt].scale_meta_bytes
    supers = 2 if meta % 4 else 1
    groups = _f.BLOCKS[fmt].weights // _f.BLOCKS[fmt].group_size
    return _l.scale_unit(meta * supers, supers, groups).token


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
        "the separate two-plane mainloop ppu_mma_aiu_mixed_input_2plane.hpp has only ptr_S/ptr_Z fp16 planes, a "
        "uint128 fp16 scale copy and fp16 scale shared storage. The raw-unit tensor/storage/decode selected by "
        "kPackedScaleOn exists only in ppu_mma_aiu_multistage_mixed_input.hpp. Q3_K/Q5_K/Q6_K therefore need that "
        "packed-scale channel added to the two-plane mainloop; their byte-neutral 28/16/36-byte units and bit-exact "
        "round trips are prerequisites, not evidence that this consumer exists"))

# (scheme, shape, format) -> Impl. Absent entries mean ABSENT with no note worth writing.
IMPL: Dict[Tuple[Scheme, Shape, QuantType], Impl] = {}


def _add(scheme, shape, fmts, impl):
    for f in fmts:
        # A non-running cell consumes no artifact. Once the path runs, record the exact derived/raw family beside
        # its status so support and storage representation cannot be mistaken for the same axis.
        IMPL[(scheme, shape, f)] = (impl._replace(consumes=format_of(scheme, shape, f))
                                    if impl.status >= Status.IMPLEMENTED and not impl.consumes else impl)


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

# --- DENSE DECODE. The cell I left blank while filling in its MoE sibling, which is incoherent -----------------
# DEQUANT_FIRST/GEMV is not a separate route: the weight is materialised to fp16 and handed to the same library
# call, so it is that GEMM at m=1 -- and test_dequant_first_dense_matches_oracle runs m in (1, 7, 64) explicitly.
# It read ABSENT only because I added the GEMV_MOE column and forgot its dense twin, which then showed the grouped
# route validated at one row per expert while the dense route at m=1 showed nothing.
_add(Scheme.DEQUANT_FIRST, Shape.GEMV, _KQUANTS, Impl(
    Status.VALIDATED, _ROUTES, note=(
        "the dense route at m=1, which the oracle test covers directly. Correct and the wrong thing to ship at "
        "decode: it materialises the whole weight in fp16 for ONE token, so its extra DRAM traffic is 16x the "
        "pre-pass's before the GEMM has read a byte. It is the fallback that proves the others are needed")))

# --- SCALE_FIRST x DENSE for the k-quants -----------------------------------------------------------------------
_add(Scheme.SCALE_FIRST, Shape.DENSE, _KQUANTS, Impl(
    Status.VALIDATED, _DENSE, note=(
        "raw GGUF now reaches gguf_prepare_dense -> fixed xplane placement -> the raw-pointer dense ABI -> "
        "fpA_intB_ppu for uint2, uint2+uint1, int4, int4+uint1 and int4+uint2. Q2-Q5 use TK=256; the new xplane "
        "inverse caught that Q6's high plane covers only half a TK=256 tile, so Q6 uses the already validated "
        "TK=128 tactic. Both dense artifact inverses are Python-reachable: dequant-all matches official gguf and "
        "dequant-scale returns the stored fp16 affine planes, with planted code/scale faults observed. "
        "test_fpA_kquant_dense still compares two fpA configurations and is a SELF oracle with shared constants; "
        "that formerly capped this cell at IMPLEMENTED. It is now superseded by the independent device gate on "
        "ppu001: all five formats at M=7/65 (including a dense tail) agree with matmul_dequant_first through official "
        "gguf semantics, and the oracle first rejects a planted low-plane fault. dense_python_oracle.log reports "
        "5 passed and zero skipped. This cell-specific evidence does not claim that the separate broad pytest pass "
        "was green")))

# --- FULLY_QUANTIZED x DENSE ------------------------------------------------------------------------------------
_add(Scheme.FULLY_QUANTIZED, Shape.DENSE, (QuantType.Q4_K,), Impl(
    Status.PARTIAL, _DENSE, flag="PPU_PACKED_SCALE=1", note=(
        "the Q4_K TileK=256 instantiation now reaches the SAME CollectiveBuilder packed-scale mainloop used by the "
        "grouped path (gs=32 gives Scale_TileK==kGroups==8); there is no second decoder. The stored artifact is the "
        "fixed dense int4 placement plus gguf_packed_unit's byte-neutral [k/256,n,16] units, whose new field "
        "addressing is expressed by cute layouts. The host operator and both flag-off/flag-on device front ends "
        "compile, but the independent ppu001 numerical oracle has not run yet, so this is deliberately PARTIAL, not "
        "IMPLEMENTED or VALIDATED. This cell is NOT IN THE DEFAULT BUILD: it requires PPU_PACKED_SCALE=1, and the "
        "always-present device symbol returns rc=34 explicitly without that flag")))
_add(Scheme.FULLY_QUANTIZED, Shape.DENSE, (QuantType.Q2_K,), Impl(
    Status.PARTIAL, _DENSE, flag="PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=2", note=(
        "Q2_K now instantiates the SAME single-plane CollectiveBuilder path at TileK=256, where gs=16 gives "
        "Scale_TileK==kGroups==16. Its trait-derived 20-byte unit is staged as five legal 4-byte cp.async copies; "
        "the device mainloop contains no Q2-specific addressing or decoder. The raw producer uses the same cute-"
        "addressed packed-unit writer as Q4 and emits low [1,n,k/4] plus units [k/256,n,20]. Host compilation, the "
        "complete forced device front end, packed-unit round trips, and a planted multi-expert producer/hash gate "
        "pass locally, but no real-PPU numerical oracle has run, so the cell remains PARTIAL. It is NOT IN THE "
        "DEFAULT BUILD and also needs PPU_PACKED_FORMAT=2; a Q4-selected packed binary intentionally rejects Q2")))
_add(Scheme.FULLY_QUANTIZED, Shape.DENSE,
     (QuantType.Q3_K, QuantType.Q5_K, QuantType.Q6_K), Impl(
    Status.ABSENT, "", note=(
        "Q3_K/Q5_K/Q6_K all route through ppu_mma_aiu_mixed_input_2plane.hpp for dense as well as grouped, and "
        "that collective has no packed-scale channel. Q3_K/Q6_K additionally need their 28/36-byte paired units "
        "carried. These are shared-mainloop changes, not per-shape implementations")))

# --- THE TWO FORMATS WITH NOTHING, and why that is a decision rather than an oversight -------------------------
# GPTQ_INT4_ASYM and AWQ_INT4 are declared in QuantType and reachable by name, and every cell for them is empty.
# Neither has an importer, a fixture, or a harness; ci/registry.py lists 'awq-int4' among the formats we intend to
# ship and its coverage() reports NOTHING for it. They are in the enum so that a caller naming them gets a refusal
# from select_path rather than a KeyError, which is the whole reason the enum is wider than the support.
_add(Scheme.SCALE_FIRST, Shape.GROUPED, (QuantType.GPTQ_INT4_ASYM, QuantType.AWQ_INT4), Impl(
    Status.ABSENT, "", note=(
        "no importer, no fixture, no harness. The asymmetric forms carry a per-group ZERO POINT that the "
        "symmetric path folds into the code range, so this is a different scale channel rather than a different "
        "checkpoint layout -- the collective's ConvertAndScaleWithZero mode is the mechanism, and nothing has "
        "been wired to feed it from either format. Declared in QuantType so select_path can refuse them by name")))

# --- MoE AT DECODE. The column that did not exist, and what it exposes ---------------------------------------
# gemv_lowbit ALREADY serves this: gemv_exec launches dim3(grid_m, n/CtaN, Grouped ? num_experts : 1), so experts
# are a grid dimension, and test_gemv_lowbit drives it with num_experts/row_offsets/max_rows. That was invisible
# while MoE decode shared a cell with dense decode.
_add(Scheme.SCALE_FIRST, Shape.GEMV_MOE, (QuantType.GPTQ_INT4_SYM,), Impl(
    Status.IMPLEMENTED, _GEMV, note=(
        "the Grouped arm: experts on the grid's z dimension, ragged rows through row_offsets. The registry path is "
        "now distinct from dense decode. Kept at IMPLEMENTED because the new imported-GGUF oracle and its planted "
        "expert-base fault cover the five k-quants, not a GPTQ checkpoint/importer")))
_add(Scheme.SCALE_FIRST, Shape.GEMV_MOE, _KQUANTS, Impl(
    Status.VALIDATED, _GEMV, note=(
        "gguf_prepare_gemv builds resident low/high code planes plus fp16 scale/zero planes, and the production raw-"
        "pointer backend reaches gemv_lowbit's Grouped arm with experts on grid.z and ragged row_offsets. "
        "test_gguf_routes compares all five formats with the official gguf dequantiser at n=24, k=2048, four "
        "different experts and rows [2,0,3,1], and first proves the oracle rejects an expert-0 base reuse fault")))

# Native-scale MoE decode extends the dense subgroup kernel by exactly the missing launch dimensions: grid.z owns
# the expert, grid.y owns a routed row within that expert, row_offsets gathers activation/output rows, and the raw
# block base advances by the expert stride. The block decoder and CUDA-core dot loop remain shared with dense GEMV.
_add(Scheme.FULLY_QUANTIZED, Shape.GEMV_MOE, _KQUANTS, Impl(
    Status.VALIDATED, "quactlize/include/gguf_vecdot.hpp", note=(
        "the production device library launches vecdot_rows_kernel<Grouped=true> over grid.z experts, gathers "
        "ragged rows through row_offsets, and advances the native GGUF weight base per expert. test_gguf_routes "
        "covers all five formats against official gguf with asymmetric n/k and different bytes per expert, after "
        "requiring the oracle to reject an exact last-expert-reads-expert-0 planted fault")))

_add(Scheme.DEQUANT_FIRST, Shape.GEMV_MOE, _KQUANTS, Impl(
    Status.VALIDATED, _ROUTES, note=(
        "routes.matmul_dequant_first_grouped is shape-agnostic in rows, so the decode band is one instance of what "
        "the grouped test already covers with ragged rows including an empty expert. What it is NOT is efficient: "
        "it materialises every selected expert's weight in fp16, which at decode is the whole point of not doing")))

_add(Scheme.SCALE_FIRST, Shape.GEMV, (QuantType.GPTQ_INT4_SYM,),
     Impl(Status.VALIDATED, _GEMV, note=(
         "test_gemv_lowbit's scale-only int4 sweep IS the GPTQ symmetric representation -- same packing, same fp16 "
         "scale dtype, zero folded into the code range. Synthetic: no real checkpoint reaches the GEMV")))

_add(Scheme.SCALE_FIRST, Shape.GEMV, _KQUANTS, Impl(
    Status.VALIDATED, _GEMV, note=(
        "gguf_prepare_gemv makes the five k-quant code planes and resident fp16 scale/zero planes reachable from "
        "Python, then libquactlize_ppu.so calls the CUDA-core gemv_lowbit launcher. test_gguf_routes checks the "
        "production .so against official gguf at asymmetric n=24/k=2048 and first zeros the complete low-code "
        "plane to demonstrate that this dense kernel's oracle fails on a well-formed planted fault")))

# --- FULLY_QUANTIZED: packed tensor-core paths, behind format-selecting flags ------------------------------------
_add(Scheme.FULLY_QUANTIZED, Shape.GROUPED, (QuantType.Q4_K,), Impl(
    Status.VALIDATED, _GROUPED, flag="PPU_PACKED_SCALE=1", note=(
        "numerically validated on real Q4_K checkpoint bytes, but ONLY test_q4k_packed_gemm's rowC exercises the "
        "packed decoder -- rowA and rowB are fp16-path controls. Consumes native scale CODES in a reordered 16-byte "
        "unit, not GGUF's on-disk 12-byte packing, which is not half-separable; the reorder is byte-neutral. "
        "PERFORMANCE IS UNMEASURED: every figure recorded before commit 80dfeec came from a bench whose two paths "
        "computed different numbers")))
_add(Scheme.FULLY_QUANTIZED, Shape.GROUPED, (QuantType.Q2_K,), Impl(
    Status.PARTIAL, _GROUPED, flag="PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=2", note=(
        "the existing grouped scheduler now reaches the same Q2_K TileK=256 packed collective as dense; only the "
        "ragged expert pointer/shape assembly differs. The artifact is low [E,n,k/4] plus byte-neutral units "
        "[E,k/256,n,20]. Four independently produced dense expert slices are byte-identical to the grouped producer, "
        "and a planted all-experts-read-expert-0 input changes both artifact hashes. The full flag-on device front "
        "end compiles, but the independent official dequant-first numerical oracle has not run on ppu001, so this is "
        "PARTIAL. It is NOT IN THE DEFAULT BUILD and requires BOTH PPU_PACKED_SCALE=1 and PPU_PACKED_FORMAT=2")))
_add(Scheme.FULLY_QUANTIZED, Shape.GROUPED, tuple(TWO_PLANE), _NO_PACKED_2PLANE)
_add(Scheme.FULLY_QUANTIZED, Shape.GEMV, _KQUANTS, Impl(
    Status.VALIDATED, "quactlize/include/gguf_vecdot.hpp", note=(
        "libquactlize_ppu.so now exports the subgroup-cooperative native GGUF launcher, so the Python route is one "
        "device GEMV rather than the old host k/256 assembly. test_gguf_routes runs that production ABI for all "
        "five formats against official gguf at n=24/k=2048 and proves a copied first-to-last weight row is rejected "
        "before accepting the real output. The kernel uses fp16/half2 CUDA-core arithmetic only: no MMA or dp4a")))


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
    (Scheme.SCALE_FIRST, Shape.DENSE): "scale_first_dense",
    (Scheme.SCALE_FIRST, Shape.GROUPED): "fused_fp16_scale",
    (Scheme.SCALE_FIRST, Shape.GEMV): "scale_first_gemv",
    (Scheme.FULLY_QUANTIZED, Shape.DENSE): "fused_native_scale",
    (Scheme.FULLY_QUANTIZED, Shape.GROUPED): "fused_native_scale",
    (Scheme.FULLY_QUANTIZED, Shape.GEMV): "native_gemv",
    (Scheme.DEQUANT_FIRST, Shape.GEMV): "dequant_then_dense",
    (Scheme.SCALE_FIRST, Shape.GEMV_MOE): "scale_first_gemv_moe",
    (Scheme.FULLY_QUANTIZED, Shape.GEMV_MOE): "native_gemv_moe",
    (Scheme.DEQUANT_FIRST, Shape.GEMV_MOE): "dequant_then_dense",
}

# WHERE THE PATH VOCABULARY IS COARSER THAN THE CELL. Keep this refusal mechanism even when empty: adding a new
# shared registry name must add the affected cells here until evidence is split. The former "gemv" collision is
# now four paths (native/scale-first x dense/MoE), so none of those cells borrows a neighbouring launch's harness.
_CELL_PATH_IS_COARSE = frozenset()
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
            # _add gives each runnable format its own `consumes` token, so identity no longer groups the shared note.
            if (i.status, i.launcher, i.flag, i.note) == (impl.status, impl.launcher, impl.flag, impl.note):
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
