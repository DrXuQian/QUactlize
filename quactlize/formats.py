"""WHICH QUANTISED FORMATS EXIST, WHAT THEY COST TO STORE, AND WHICH PATH CAN RUN THEM.

Structure follows vLLM's GGUF plugin, which solved two problems this tree also has:

  * THE FORMAT IS DATA, NOT A TYPE. vLLM stores the raw GGUF blocks as an untouched uint8 tensor and carries a
    separate small `qweight_type` tensor beside it. Nothing about a packed tensor's dtype or shape says which
    physical arrangement it is in -- a fact this tree learned the expensive way, when a request for the AIU
    interleave silently returned the ordinary layout and the two were indistinguishable afterwards. A format id
    travelling with the buffer is the fix.
  * A FORMAT IS NOT SIMPLY "SUPPORTED". vLLM keeps per-path capability sets -- MMVQ_QUANT_TYPES, MMQ_QUANT_TYPES,
    DEQUANT_TYPES -- and dispatches on (format, token count), falling back to dequantise-then-dense when the fused
    path does not have that format. A format can be usable and slow. One boolean cannot say that.

Both are why this file is executable rather than a table in a document: the readiness question is asked often enough
that the answer has to be checkable, and the last two answers written down by hand were wrong.

THE STORAGE CONSTRAINT is the sharpest of the questions, so it is COMPUTED from each format's block layout rather
than recorded. An offline reorder is allowed; an increase in stored bytes is not. Materialising a k-quant's native
6-bit scale/min pairs as fp16 planes is exactly such an increase, and the size of it decides whether a format can
ship on the fp16-scale path or must have a native decode channel.
"""
from enum import IntEnum
from typing import Dict, FrozenSet, NamedTuple, Optional


class QuantType(IntEnum):
    """ggml's own type numbers, so a GGUF file's type field maps straight onto this with no table.

    Values come from ggml_type in ggml.h. Non-GGUF formats get values above the ggml range, where they cannot
    collide with a future ggml addition.
    """
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14

    GPTQ_INT4_SYM = 1000
    GPTQ_INT4_ASYM = 1001
    AWQ_INT4 = 1002


class BlockLayout(NamedTuple):
    """The bytes of one quantisation block, split into the part that holds codes and the part that holds scales.

    weights          weights per block (256 for every k-quant)
    block_bytes      total stored bytes for those weights
    scale_meta_bytes the part of the block that encodes scales/mins, including the fp16 super-block factors
    group_size       weights sharing one scale
    has_min          the format carries an additive min (affine), not just a multiplicative scale
    """
    weights: int
    block_bytes: int
    scale_meta_bytes: int
    group_size: int
    has_min: bool

    @property
    def fp16_plane_bytes(self) -> int:
        """What the same scale information costs once expanded to fp16 planes -- one fp16 per group, twice if the
        format is affine."""
        return (self.weights // self.group_size) * 2 * (2 if self.has_min else 1)

    @property
    def bytes_on_fp16_scale_path(self) -> int:
        return self.block_bytes - self.scale_meta_bytes + self.fp16_plane_bytes

    @property
    def fp16_scale_growth(self) -> float:
        """Fractional storage increase from running this format on the fp16-scale path. Zero or less means the
        native scale channel buys nothing in storage terms and only matters for speed."""
        return self.bytes_on_fp16_scale_path / self.block_bytes - 1.0


# The k-quant block layouts, from ggml-common.h. Written as the FIELDS of each struct so they can be checked against
# the format's documented block size rather than trusted.
#
#   Q2_K  uint8 scales[16]; uint8 qs[64];                       half d, dmin      gs=16
#   Q3_K  uint8 hmask[32];  uint8 qs[64];  uint8 scales[12];    half d            gs=16
#   Q4_K  half d, dmin;     uint8 scales[12]; uint8 qs[128];                      gs=32
#   Q5_K  half d, dmin;     uint8 scales[12]; uint8 qh[32]; uint8 qs[128];        gs=32
#   Q6_K  uint8 ql[128];    uint8 qh[64];  int8 scales[16];     half d            gs=16
# The legacy (pre-k-quant) formats are one fp16 scale, and optionally one fp16 min, per 32 weights:
#
#   Q4_0  half d;          uint8 qs[16]                        Q4_1  half d, m;  uint8 qs[16]
#   Q5_0  half d;          uint8 qh[4]; uint8 qs[16]           Q5_1  half d, m;  uint8 qh[4]; uint8 qs[16]
#   Q8_0  half d;          int8  qs[32]
#
# Their scale meta IS an fp16 plane already, so the fp16-scale path costs them nothing and the whole native-scale
# question does not arise. That is worth stating rather than omitting: the storage problem belongs to the k-quants
# specifically, and a table that left these out would make it look like a property of GGUF.
BLOCKS: Dict[QuantType, BlockLayout] = {
    QuantType.Q4_0: BlockLayout(32, 2 + 16, 2, 32, False),
    QuantType.Q4_1: BlockLayout(32, 2 + 2 + 16, 2 + 2, 32, True),
    QuantType.Q5_0: BlockLayout(32, 2 + 4 + 16, 2, 32, False),
    QuantType.Q5_1: BlockLayout(32, 2 + 2 + 4 + 16, 2 + 2, 32, True),
    QuantType.Q8_0: BlockLayout(32, 2 + 32, 2, 32, False),
    QuantType.Q2_K: BlockLayout(256, 16 + 64 + 2 + 2, 16 + 2 + 2, 16, True),
    QuantType.Q3_K: BlockLayout(256, 32 + 64 + 12 + 2, 12 + 2, 16, False),
    QuantType.Q4_K: BlockLayout(256, 2 + 2 + 12 + 128, 2 + 2 + 12, 32, True),
    QuantType.Q5_K: BlockLayout(256, 2 + 2 + 12 + 32 + 128, 2 + 2 + 12, 32, True),
    QuantType.Q6_K: BlockLayout(256, 128 + 64 + 16 + 2, 16 + 2, 16, False),
}

# GPTQ and AWQ are not block formats: scales are separate tensors. GPTQ's are ALREADY fp16, so the fp16-scale path
# costs nothing extra -- which is the whole reason GPTQ symmetric is the first format that can ship. AWQ's zeros are
# packed int4, so materialising them as fp16 is a 4x increase on that tensor and it must be decoded on device.
NON_BLOCK_SCALE_GROWTH: Dict[QuantType, Optional[float]] = {
    QuantType.GPTQ_INT4_SYM: 0.0,
    # ASYMMETRIC GPTQ IS NOT FREE, though its scales are fp16 like the symmetric case. It also carries qzeros, packed
    # 4-bit inside int32 (vLLM models them exactly that way), and expanding those to fp16 is a 4x increase on that
    # tensor. This entry read 0.0 because "GPTQ scales are already fp16" was applied to the whole format instead of to
    # the scale tensor alone -- the same shape of error as reading a numerator for a quotient. None, not a number:
    # the ratio depends on group size and shape, so there is no single figure to quote.
    QuantType.GPTQ_INT4_ASYM: None,
    QuantType.AWQ_INT4: None,      # same reason: only the zero-point tensor grows, and it grows 4x
}


# ---------------------------------------------------------------------------------------------------------------
# Path capability. Mirrors vLLM's MMVQ / MMQ / DEQUANT sets.
#
# NATIVE_SCALE is this tree's extra axis and is the one the k-quants turn on: whether the GEMM can read the
# format's own packed scale bytes, as opposed to fp16 planes prepared beforehand. Without it a k-quant can still
# run -- at the storage cost computed above, which for Q2_K is +52% and therefore not shippable.
# ---------------------------------------------------------------------------------------------------------------

#: single-token scale-first CUDA-core GEMV. The k-quants are technically reachable through a resident artifact;
#: select_path still applies `fp16_planes` before choosing it, so capability does not silently waive storage policy.
GEMV: FrozenSet[QuantType] = frozenset({
    QuantType.GPTQ_INT4_SYM, QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K,
})

#: the fused mixed-input tensor-core GEMM, fp16 scale planes
FUSED_FP16_SCALE: FrozenSet[QuantType] = frozenset({
    QuantType.GPTQ_INT4_SYM, QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K,
})

#: the fused GEMM reading the format's OWN scale bytes, no fp16 planes
# ALL FIVE SINCE 2026-08-04. This was Q4_K alone while the packed collective had only a Q4_K unit; the other
# four reached VALIDATED on ppu001 for dense AND grouped, so the DISPATCHER may route them. The set says what
# may be routed, not what is proven -- ci/registry.py answers the second question, and conflating them would
# either bar a working path from use or let an unproven one look proven.
FUSED_NATIVE_SCALE: FrozenSet[QuantType] = frozenset({
    QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K})

#: dequantise to fp16 then a dense GEMM. Wins above roughly M=512, where the weight read stops dominating.
#:
#: The five k-quants, and the evidence is tests/test_gguf_routes.py: raw blocks -> fp16 weight -> torch's cuBLAS,
#: dense and per-expert, against the official gguf package as an independent oracle, worst relative error 1.05e-3.
#:
#: THE EMPTY SET IT REPLACES WAS CORRECT AT THE TIME, and the reason it stayed empty is worth keeping. It listed all
#: six formats until evidence was checked per (format, path) instead of per format, at which point every entry
#: turned out to rest on a harness for a DIFFERENT path; the note then said populating it needs a dense-path
#: harness, not an edit here. That is what happened -- the harness came first. GPTQ is deliberately still absent:
#: routes.py reads k-quant blocks, and the symmetric packed forms have no host binding to reach the route at all.
DEQUANT_THEN_DENSE: FrozenSet[QuantType] = frozenset({
    QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K,
})

#: Above this many rows the dense path's extra pass over the weights is repaid. Measured, not assumed: at 2048 it
#: was 2.1x faster than the fused path. The crossover itself has not been swept, so the constant is a boundary
#: someone chose, and it is named rather than buried in an inequality.
DENSE_CROSSOVER_ROWS = 512

#: Below this the GEMV path wins outright -- decode. vLLM calls its equivalent mmvq_safe.
GEMV_MAX_ROWS = 1


def select_path(qtype: QuantType, num_rows: int, native_scale_available: bool = True,
                fp16_planes: str = "auto") -> str:
    """Which implementation should run this (format, shape). Returns a path name, or raises if none can.

    THE STORAGE CONSTRAINT IS PART OF THE SELECTION, not a separate check someone remembers to make. A path that
    consumes fp16 scale planes is only admissible if those planes are allowed to exist, and for a format whose
    native scale meta is smaller than fp16 planes -- every k-quant -- materialising them is exactly the increase the
    constraint forbids. An earlier version routed Q3_K to fused_fp16_scale while needs_native_scale(Q3_K) was True,
    which is a contradiction the file itself could have caught.

    But "forbidden" is about STORED bytes, not about bytes that exist for the duration of one call. Prefill can
    convert scales into a transient workspace and the weight on disk and in HBM is unchanged; decode cannot, because
    the planes would have to be resident for every token. So the caller says which situation it is in:

      fp16_planes="auto"       planes are free for formats that do not need a native channel (their scales already
                               ARE fp16), and forbidden for the ones that do. The conservative reading, and the
                               right default for a resident weight.
      fp16_planes="workspace"  the caller will build the planes transiently and discard them; permitted for any
                               format. This is the prefill pre-pass.
      fp16_planes="never"      do not consider a plane-consuming path at all.
    """
    if fp16_planes not in ("auto", "workspace", "never"):
        raise ValueError(f"fp16_planes must be auto, workspace or never; got {fp16_planes!r}")
    planes_ok = {"never": False,
                 "workspace": True,
                 "auto": not needs_native_scale(qtype)}[fp16_planes]

    # The order is the same as vLLM's: try the fastest path the format is actually in, then fall back, then fail
    # loudly. Falling back is a real outcome; silently producing a wrong answer is not an option anywhere here.
    if num_rows <= GEMV_MAX_ROWS and qtype in GEMV and planes_ok:
        return "gemv"
    if num_rows >= DENSE_CROSSOVER_ROWS and qtype in DEQUANT_THEN_DENSE and planes_ok:
        return "dequant_then_dense"
    if native_scale_available and qtype in FUSED_NATIVE_SCALE:
        return "fused_native_scale"
    if qtype in FUSED_FP16_SCALE and planes_ok:
        return "fused_fp16_scale"
    if qtype in DEQUANT_THEN_DENSE and planes_ok:
        return "dequant_then_dense"
    if not planes_ok and (qtype in FUSED_FP16_SCALE or qtype in DEQUANT_THEN_DENSE):
        raise NotImplementedError(
            f"{qtype.name} at {num_rows} rows has only fp16-scale-plane paths, and materialising those planes grows "
            f"the stored weight by {storage_growth(qtype) if storage_growth(qtype) is not None else float('nan'):.1%}"
            f". Pass fp16_planes='workspace' if the planes are transient, or give the format a native scale channel.")
    raise NotImplementedError(f"{qtype.name} has no path at {num_rows} rows")


def storage_growth(qtype: QuantType) -> Optional[float]:
    """Fractional increase in stored bytes from running this format on the fp16-scale path, or None when it is not a
    single ratio. Zero means the format's scales are already fp16."""
    if qtype in BLOCKS:
        return BLOCKS[qtype].fp16_scale_growth
    return NON_BLOCK_SCALE_GROWTH.get(qtype)


# THE PLACED ARRANGEMENT IS A PROPERTY OF THE TENSOR, NOT OF THE FORMAT. This started as a search for one
# optimal F per k-quant, which would have made the layout a function of the format and left the artifact header
# with nothing to say. Measurement killed that: dense and MoE want different folds --
#
#     dense int1  TK64/F4  215.23 us / 63.9%      MoE int4  TK32/F2  317.26 us   vs TK64/F1 362.14  (-12.4%)
#     dense int2  TK64/F2  233.76 us / 58.8%      MoE int2  TK32/F4  295.08 us   vs TK64/F2 300.26
#
# -- and requiring F=1 everywhere would delete measured winners. What dissolves the conflict is that a dense
# layer's weight and an expert's weight are DIFFERENT TENSORS: no tensor is read by both paths, so each can be
# arranged for the operator that reads it. The header therefore carries three small integers per tensor, which
# GGUF is already shaped for, instead of the format implying them.
#
# WHAT A RECORDED ARRANGEMENT BINDS. The online tactic search is bounded by the tensor's LAYOUT CLASS, not by its
# F. At F=1 the class absorbs every TK <= 256 and every tile and warp shape measured, so the search is wide. At
# F>1 the same F at a different TK is a DIFFERENT class, so TK is pinned too and the search is narrower. Treating
# "same F" as "same layout" is the specific mistake to avoid; l105 has a regression case for that distinction.
#
# THE TK <= 256 BOUND IS REAL AND WAS FOUND BY LOOKING FOR IT: int4 at TM64/TN64/TK512 w32x32 F=1 compiles and
# produces DIFFERENT bytes, so "F=1 is tile-invariant" holds within the unfolded interleave-256 domain and not
# beyond it. An unbounded claim would have been wrong in exactly one corner nobody was sweeping.
def fold_for(bits: int, tile_k: int) -> int:
    """F, DERIVED -- never passed. The one expression, transcribed from the consumer that already computes it.

    moe_grouped_ppu.cuh:363 is the original:

        MOEG_RUN_B = TK * MOEG_BITS / 8                        # contiguous bytes at this TK
        MOEG_FOLD  = MOEG_RUN_B >= 32 ? 1 : (32 / MOEG_RUN_B)  # fold factor needed
        // "Requires the weight to have been preprocessed with the matching FoldTK=TK."

    THE REASON THIS IS A FUNCTION AND NOT A FIELD. The consumer derives F from (bits, TK) at compile time and
    cannot be told otherwise. A producer that ACCEPTS an F therefore introduces a second source for one value,
    and the two disagreeing is not a crash -- the weight is placed for one fold and read at another, which
    returns finite wrong numbers. So there is one expression and everything asks it.

    The arrangement contract is what this returns: the folds are determined by the 32-byte AIU floor rather than
    selected independently by a caller.

        int4 TK=256 -> run 128 B -> F=1     int2 TK=64 -> 16 B -> F=2     int1 TK=64 -> 8 B -> F=4
    """
    run_bytes = tile_k * bits // 8
    if run_bytes >= 32:
        return 1
    if run_bytes == 0:
        # A sub-byte run has no fold that fixes it: folding multiplies whole rows, so it can never reach 32 B
        # from nothing. Caught here because the // 8 above makes this a ZeroDivisionError two lines down, and a
        # ZeroDivisionError names the arithmetic instead of the configuration.
        raise ValueError(f"bits={bits} tile_k={tile_k}: the k-run is under one byte")
    if 32 % run_bytes:
        raise ValueError(f"bits={bits} tile_k={tile_k}: a {run_bytes}B run does not divide the 32B floor")
    return 32 // run_bytes


class PlacedArrangement(NamedTuple):
    """What an artifact must record so a reader can decode it without guessing what the packer did.

    fold is NOT stored. It is a function of (bits, tile_k) -- see fold_for -- and storing a derived value is
    how a manifest comes to disagree with the kernel that reads it.
    """
    bits: int          # the low plane's width; the format supplies the high plane's
    tile_k: int        # the whole of the arrangement: F follows from it and the width
    high_bits: int = 0  # the high plane's width for a two-plane format; 0 for single-plane

    @property
    def fold(self) -> int:
        return fold_for(self.bits, self.tile_k)

    @property
    def high_fold(self) -> int:
        return fold_for(self.high_bits, self.tile_k) if self.high_bits else 1

    def layout_is_tile_free(self) -> bool:
        """True when this arrangement's bytes do not depend on the tile, so any TK <= 256 reads it.

        SUFFICIENT, NOT THE BOUNDARY, and this used to be one sentence with no measurement under it.
        dev/fold_derivation/l115_artifact_tactic_code_slots.cu is the durable witness -- it walks
        xplane::place_from_map's exact physical address layouts and reports owner_diff, where 0 is the actual
        "one resident artifact serves a larger tactic T" contract (bijective-but-differently-permuted is still
        wrong). Run at HEAD on 2026-08-11, every cross-T row is owner_diff=0:

            int4  low  A=64  T=128   F=1/1        int1  low  A=64  T=128,256  F=4/1
            int2  low  A=64  T=128   F=2/1        Q6_K  high A=128 T=256      F=1/1   <- two-plane, F=1 both
            Q6_K  high A=64  T=256   F=1/2        Q6_K  high A=32  T=256      F=2/4
            Q3_K  high A=64  T=256   F=2/4        Q5_K  high A=64  T=256      F=1/4

        So F>1 artifacts survive a larger T as well, PROVIDED the fold travels with the artifact rather than
        being re-derived from (bits, T) -- which is exactly what task #37 is about and is not this predicate's
        business. Returning True only at F=1 is therefore CONSERVATIVE: it is the subset that needs no ABI
        promise, which is the right subset for a planner that must not assume plumbing exists.

        The two-plane case is covered by the Q6_K A=128 T=256 F=1/1 row above; an earlier note of mine claimed
        it was not, and claimed int2 F=2 and int1 F=4 fail cross-T. Both were true of an older tree and are not
        true of this one. l115 takes about 15 s to build and needs no device -- run it rather than citing me.
        """
        return self.fold == 1 and self.high_fold == 1


# Physical placement ABI constants.  Keep these numeric values identical to quactlize_ppu_config.h: they are
# serialized with the tensor and cross the host-extension/device-library boundary as data, not as Python types.
PLACED_ARRANGEMENT_VERSION_V1 = 1
PLACED_ARRANGEMENT_VERSION_V2 = 2
PLACED_LAYOUT_XPLANE_V1 = 0
PLACED_LAYOUT_Q4_KPACK4_TRANSPOSE_V1 = 1
PLACED_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1 = 2
PLACED_LAYOUT_Q4_N16K64_DIRECT_V1 = 3
Q4_KPACK4_MAPPING_ID = 0x51344B5034540001
KQUANT_KPACK_MAPPING_ID = 0x514B504B54000001
Q4_N16K64_DIRECT_MAPPING_ID = 0x51344E3136440001
Q4_KPACK4_TRANSPORT_TILE_K = 64
Q4_KPACK4_GROUP_SIZE = 32
Q4_N16K64_DIRECT_TRANSPORT_TILE_K = 64
Q4_N16K64_DIRECT_GROUP_SIZE = 32

# One production offline-layout decision per k-quant format.  K-pack4 is the
# Q4-specific byte map; the other formats use the generic descriptor whose
# per-plane pack factors are derived from their exact code widths.  Xplane is
# retained only for explicit development comparisons and is never automatic.
CANONICAL_FULLY_QUANTIZED_LAYOUT: Dict[QuantType, str] = {
    QuantType.Q2_K: "kquant-kpack",
    QuantType.Q3_K: "kquant-kpack",
    QuantType.Q4_K: "q4-kpack4",
    QuantType.Q5_K: "kquant-kpack",
    QuantType.Q6_K: "kquant-kpack",
}

# Archived means absent from automatic/whole-model production selection.  The
# low-level explicit Xplane path remains available for development A/B runs.
ARCHIVED_FULLY_QUANTIZED_LAYOUTS: Dict[QuantType, FrozenSet[str]] = {
    qtype: frozenset({"xplane"}) for qtype in CANONICAL_FULLY_QUANTIZED_LAYOUT
}


def canonical_fully_quantized_layout(qtype) -> str:
    """Return the sole automatic offline layout for a shipping k-quant."""
    q = QuantType(qtype)
    try:
        return CANONICAL_FULLY_QUANTIZED_LAYOUT[q]
    except KeyError as error:
        raise NotImplementedError(
            f"{q.name} has no canonical fully-quantized offline layout") from error


def archived_fully_quantized_layouts(qtype) -> FrozenSet[str]:
    """Layouts retained for diagnostics but forbidden to automatic packers."""
    return ARCHIVED_FULLY_QUANTIZED_LAYOUTS.get(QuantType(qtype), frozenset())


class PlacedArrangementV2(NamedTuple):
    """A byte-map identity, not a tactic hint.

    Xplane v1 could be identified by code widths plus ArtifactTileK.  K-pack4 transposes and packs the physical
    axes, has no ArtifactTileK axis, and therefore needs an explicit layout and mapping id.  Every field mirrors
    ``quactlize_ppu_placed_arrangement_v2`` in order so a Python artifact cannot lose or reorder an ABI field.
    """
    layout: int
    bits: int
    high_bits: int
    artifact_tile_k: int
    transport_tile_k: int
    group_size: int
    reserved: int
    mapping_id: int

    def validate(self) -> None:
        values = (self.layout, self.bits, self.high_bits, self.artifact_tile_k,
                  self.transport_tile_k, self.group_size, self.reserved, self.mapping_id)
        if any(not isinstance(x, int) for x in values):
            raise TypeError("placed arrangement v2 fields must all be integers")
        if self.layout == PLACED_LAYOUT_Q4_KPACK4_TRANSPOSE_V1:
            canonical = q4_kpack4_arrangement()
            if self != canonical:
                raise ValueError(
                    f"noncanonical Q4 K-pack4 descriptor {self}; expected {canonical}. The mapping id and "
                    "physical quanta identify bytes and are not tunable reader parameters")
        elif self.layout == PLACED_LAYOUT_Q4_N16K64_DIRECT_V1:
            canonical = q4_n16k64_direct_arrangement()
            if self != canonical:
                raise ValueError(
                    f"noncanonical Q4 N16xK64 direct descriptor {self}; expected {canonical}. The layout is an "
                    "explicit non-default offline ABI and none of its physical identity fields are tunable")
        elif self.layout == PLACED_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1:
            matches = [q for q in (QuantType.Q2_K, QuantType.Q3_K,
                                    QuantType.Q5_K, QuantType.Q6_K)
                       if self == kquant_kpack_arrangement(q)]
            if len(matches) != 1:
                raise ValueError(
                    f"noncanonical k-quant K-pack descriptor {self}; expected one exact Q2/Q3/Q5/Q6 mapping")
        elif self.layout == PLACED_LAYOUT_XPLANE_V1:
            if (self.bits <= 0 or self.high_bits < 0 or self.artifact_tile_k <= 0 or
                    self.transport_tile_k != 0 or self.group_size != 0 or self.reserved != 0 or
                    self.mapping_id != 0):
                raise ValueError(f"invalid Xplane arrangement-v2 descriptor {self}")
            _ = fold_for(self.bits, self.artifact_tile_k)
            if self.high_bits:
                _ = fold_for(self.high_bits, self.artifact_tile_k)
        else:
            raise ValueError(f"unknown placed arrangement-v2 layout {self.layout}")


def q4_kpack4_arrangement() -> PlacedArrangementV2:
    """The sole shipping K-pack4 byte class.

    Returning a canonical value rather than accepting knobs is deliberate: changing any field changes the bytes,
    so such a change is a new layout/version, not another kernel configuration.
    """
    return PlacedArrangementV2(
        PLACED_LAYOUT_Q4_KPACK4_TRANSPOSE_V1, 4, 0, 0,
        Q4_KPACK4_TRANSPORT_TILE_K, Q4_KPACK4_GROUP_SIZE, 0,
        Q4_KPACK4_MAPPING_ID)


def q4_n16k64_direct_arrangement() -> PlacedArrangementV2:
    """Explicit Q4 N16xK64 direct-reader byte map; never selected by ``auto``.

    The artifact is a compact ``[K/16][2*N]`` uint32 tensor.  Keeping this as
    layout 3 instead of overloading the shipping layout-1 descriptor makes a
    checkpoint opt in to the new mapping while canonical Q4 production remains
    q4-kpack4.
    """
    return PlacedArrangementV2(
        PLACED_LAYOUT_Q4_N16K64_DIRECT_V1, 4, 0, 0,
        Q4_N16K64_DIRECT_TRANSPORT_TILE_K,
        Q4_N16K64_DIRECT_GROUP_SIZE, 0,
        Q4_N16K64_DIRECT_MAPPING_ID)


def kquant_kpack_arrangement(qtype) -> PlacedArrangementV2:
    """Canonical per-plane b16 K-pack map for Q2/Q3/Q5/Q6.

    Each plane's pack factor is 16/bits; ``transport_tile_k`` is the smallest
    logical K quantum that lets every plane deliver one physical K16 reader
    tile. Q4 keeps its already-shipped layout-1 descriptor.
    """
    q = QuantType(qtype)
    if q == QuantType.Q4_K:
        raise ValueError("Q4_K retains the shipping q4-kpack4 descriptor")
    low, high = placed_code_planes(q)
    group = BLOCKS[q].group_size
    packs = [16 // low] + ([16 // high] if high else [])
    return PlacedArrangementV2(
        PLACED_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1, low, high, 0,
        16 * max(packs), group, 0, KQUANT_KPACK_MAPPING_ID)


# LOGICAL CODE PLANES ARE PART OF THE GGUF FORMAT, NOT A TACTIC CHOICE. Keeping this beside BLOCKS makes the
# Python artifact header derive from the same format identity that validates the raw blocks. TileK is deliberately
# absent: the producer records the TileK it actually used, and fold_for derives the physical fold from that value.
#
# This table is not a second copy of a tunable registry. Q2/Q3 split their 2/3 bits as 2+0/2+1, Q4/Q5 as 4+0/4+1,
# and Q6 as 4+2 -- the official format definition. quactlize/include/ppu_format_config.inc has a build-time copy
# for C++; tests cross-check it so a drift is a hard failure rather than two plausible artifact descriptors.
PLACED_CODE_PLANES = {
    QuantType.Q2_K: (2, 0),
    QuantType.Q3_K: (2, 1),
    QuantType.Q4_K: (4, 0),
    QuantType.Q5_K: (4, 1),
    QuantType.Q6_K: (4, 2),
}


def placed_code_planes(qtype) -> tuple[int, int]:
    """Return the low/high code-plane widths for a shipping k-quant, refusing every other qtype."""
    q = QuantType(qtype)
    if q not in PLACED_CODE_PLANES:
        raise NotImplementedError(
            f"{q.name} has no placed fully-quantized artifact; accepting it would manufacture a descriptor for "
            f"bytes no producer in this tree can build")
    return PLACED_CODE_PLANES[q]


def validate_fully_quantized_resident_geometry(qtype, n: int, k: int) -> None:
    """Validate the tensor-core resident boundary shared by packers and producers.

    This is deliberately stricter than an offline plane map's smallest physical
    quantum.  Product kernels consume 256-column N tiles and complete GGUF
    superblocks; Q3_K and Q6_K pair two superblocks in one packed metadata unit.
    Keeping this policy in one host-side function makes unsupported tails fail
    before a placement op or device library is called.
    """
    q = QuantType(qtype)
    if q not in PLACED_CODE_PLANES:
        raise NotImplementedError(
            f"{q.name} has no fully-quantized resident artifact")
    n, k = int(n), int(k)
    if n <= 0 or k <= 0:
        raise ValueError(
            f"fully-quantized resident dimensions must be positive, got N={n} K={k}")
    if n % 256:
        raise ValueError(
            f"fully-quantized resident artifact requires N multiple of 256, got N={n}")
    k_quantum = 512 if q in (QuantType.Q3_K, QuantType.Q6_K) else 256
    if k % k_quantum:
        raise ValueError(
            f"{q.name} fully-quantized resident artifact requires K multiple of {k_quantum}, got K={k}")


def placed_arrangement(qtype, tile_k: Optional[int] = None) -> PlacedArrangement:
    """The descriptor a dense producer must attach to its bytes.

    `tile_k=None` means the old/fixed producer. Its C++ placement uses the minimum K span that gives the narrowest
    plane a 32-byte delivery, which is 256/min(low_bits, high_bits). That is a FORMAT consequence, not a hidden
    reader default. An explicit TileK is recorded verbatim because it is the argument sent to *_for_tile.
    """
    low_bits, high_bits = placed_code_planes(qtype)
    if tile_k is None:
        tile_k = 256 // (high_bits or low_bits)
    tile_k = int(tile_k)
    if tile_k <= 0:
        raise ValueError(f"artifact TileK must be positive, got {tile_k}")
    arrangement = PlacedArrangement(low_bits, tile_k, high_bits)
    # Evaluate both derived folds now. A descriptor whose run cannot satisfy the 32-byte floor must fail at the
    # producer seam, not survive until a reader sees finite wrong values.
    _ = arrangement.fold, arrangement.high_fold
    return arrangement


# THE CODE CORRECTION OF THE PLACED DENSE ARRANGEMENT, per format. NOT a free parameter and NOT zero by default:
# a placed weight's codes carry a per-format offset, and reading them back with the wrong one produces plausible
# scales rather than an error. These values were measured against the stored planes --
#
#     Q2_K 0    Q3_K -4    Q4_K 8    Q5_K 8    Q6_K -24
#
# -- and the prepass accepts {-32, -24, -4, 0, 8}. The first version of the derivation route defaulted to 0, which
# is correct for exactly one of the five; four formats would have come back wrong-but-finite. Deriving it from the
# qtype removes the chance to pass the wrong one, which is the only reason it is a table here rather than an
# argument the caller remembers.
PLACED_CODE_ZMUL = {
    QuantType.Q2_K: 0,
    QuantType.Q3_K: -4,
    QuantType.Q4_K: 8,
    QuantType.Q5_K: 8,
    QuantType.Q6_K: -24,
}


def placed_code_zmul(qtype) -> int:
    """The placed arrangement's code correction for this format. Raises rather than assuming zero."""
    q = QuantType(qtype)
    if q not in PLACED_CODE_ZMUL:
        raise NotImplementedError(
            f"{q.name} has no recorded placed-code correction. Defaulting to 0 would be right for Q2_K alone and "
            f"silently wrong for the rest, so this refuses instead.")
    return PLACED_CODE_ZMUL[q]


def needs_native_scale(qtype: QuantType, budget: float = 0.0) -> bool:
    """Whether this format REQUIRES a native scale channel to be shippable under the storage constraint.

    `budget` is the tolerated increase; the default of zero is the constraint as stated ("can reorder offline, but
    must not increase storage"). Raising it is a decision someone has to make explicitly."""
    g = storage_growth(qtype)
    return True if g is None else g > budget


def report() -> str:
    """One table: the storage arithmetic and the paths each format has. This is the answer to 'which formats can
    ship', regenerated from the definitions above rather than maintained alongside them."""
    rows = []
    for q in QuantType:
        g = storage_growth(q)
        growth = "n/a" if g is None else f"{g * 100:+5.1f}%"
        paths = [n for n, s in (("gemv", GEMV), ("native", FUSED_NATIVE_SCALE),
                                ("fp16", FUSED_FP16_SCALE), ("dense", DEQUANT_THEN_DENSE)) if q in s]
        blk = BLOCKS.get(q)
        size = f"{blk.block_bytes:>3}B/{blk.weights}w gs={blk.group_size}" if blk else "not a block format"
        need = "yes" if needs_native_scale(q) else "no"
        rows.append(f"  {q.name:<15} {size:<20} fp16-scale growth {growth:>7}   native required: {need:<3} "
                    f"paths: {','.join(paths) or 'NONE'}")
    return ("== quactlize formats ==\n"
            "  growth = stored bytes if the format's own scale meta is replaced by fp16 planes\n"
            "  'native required' = that growth breaks the no-extra-storage constraint\n" + "\n".join(rows))


if __name__ == "__main__":
    print(report())
