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
    QuantType.GPTQ_INT4_ASYM: 0.0,
    QuantType.AWQ_INT4: None,      # not a single ratio: only the zero-point tensor grows, and it grows 4x
}


# ---------------------------------------------------------------------------------------------------------------
# Path capability. Mirrors vLLM's MMVQ / MMQ / DEQUANT sets.
#
# NATIVE_SCALE is this tree's extra axis and is the one the k-quants turn on: whether the GEMM can read the
# format's own packed scale bytes, as opposed to fp16 planes prepared beforehand. Without it a k-quant can still
# run -- at the storage cost computed above, which for Q2_K is +52% and therefore not shippable.
# ---------------------------------------------------------------------------------------------------------------

#: single-token decode, CUDA-core GEMV
GEMV: FrozenSet[QuantType] = frozenset({QuantType.GPTQ_INT4_SYM, QuantType.Q4_K})

#: the fused mixed-input tensor-core GEMM, fp16 scale planes
FUSED_FP16_SCALE: FrozenSet[QuantType] = frozenset({
    QuantType.GPTQ_INT4_SYM, QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K,
})

#: the fused GEMM reading the format's OWN scale bytes, no fp16 planes
FUSED_NATIVE_SCALE: FrozenSet[QuantType] = frozenset({QuantType.Q4_K})

#: dequantise to fp16 then a dense GEMM. Wins above roughly M=512, where the weight read stops dominating.
DEQUANT_THEN_DENSE: FrozenSet[QuantType] = frozenset({
    QuantType.GPTQ_INT4_SYM, QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K,
})

#: Above this many rows the dense path's extra pass over the weights is repaid. Measured, not assumed: at 2048 it
#: was 2.1x faster than the fused path. The crossover itself has not been swept, so the constant is a boundary
#: someone chose, and it is named rather than buried in an inequality.
DENSE_CROSSOVER_ROWS = 512

#: Below this the GEMV path wins outright -- decode. vLLM calls its equivalent mmvq_safe.
GEMV_MAX_ROWS = 1


def select_path(qtype: QuantType, num_rows: int, native_scale_available: bool = True) -> str:
    """Which implementation should run this (format, shape). Returns a path name, or raises if none can.

    The order matters and is the same as vLLM's: try the fastest path the format is actually in, then fall back,
    then fail loudly. Falling back is a real outcome, not an error -- but silently producing a wrong answer is not
    an option anywhere in this file.
    """
    if num_rows <= GEMV_MAX_ROWS and qtype in GEMV:
        return "gemv"
    if num_rows >= DENSE_CROSSOVER_ROWS and qtype in DEQUANT_THEN_DENSE:
        return "dequant_then_dense"
    if native_scale_available and qtype in FUSED_NATIVE_SCALE:
        return "fused_native_scale"
    if qtype in FUSED_FP16_SCALE:
        return "fused_fp16_scale"
    if qtype in DEQUANT_THEN_DENSE:
        return "dequant_then_dense"
    raise NotImplementedError(f"{qtype.name} has no path at {num_rows} rows")


def storage_growth(qtype: QuantType) -> Optional[float]:
    """Fractional increase in stored bytes from running this format on the fp16-scale path, or None when it is not a
    single ratio. Zero means the format's scales are already fp16."""
    if qtype in BLOCKS:
        return BLOCKS[qtype].fp16_scale_growth
    return NON_BLOCK_SCALE_GROWTH.get(qtype)


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
