"""quactlize -- low-bit weight-only GEMM for the T-Head PPU.

The Python layer OWNS THE PHYSICAL LAYOUT and the kernel takes tensors that are already arranged. That boundary is
taken from TensorRT-LLM (preprocess_weights_for_mixed_gemm) and from IST-DASLab/qutlass, whose utils.py holds the
scale-swizzle and blocking for the same reason: a format's byte arrangement is host knowledge, and putting it in the
kernel is what makes "one kernel per format" happen.

TWO HALVES, JOINED BY A RAW-POINTER ABI. The preprocessing half is a torch operator library built by setup.py; it is
pure host C++ and works anywhere torch is installed, including a machine with no PPU. The GEMV device half is built
as libquactlize_ppu.so by nvcc or hgcc and dlopened by that extension without introducing torch into device code.
The tensor-core GEMM family remains reached through its C++ harnesses; matmul_w4a16 raises with that narrower reason.

WHY torch.ops.load_library AND NOT `import quactlize._C`. The .so registers operators and defines no python module
initialiser, which is the shape torch documents for custom ops with no python-visible surface. Importing it as a
module fails with "dynamic module does not define module export function (PyInit__C)" -- a message that reads like a
build failure and is not one.
"""
import glob
import os
from typing import Literal

try:
    import torch
except ImportError as e:                                   # a host without torch can still use quactlize.formats
    torch = None
    _TORCH_ERR = e

_LOADED = False


def _ops():
    """Load the operator library on first use, with a message that says what to do rather than an ImportError trace."""
    global _LOADED
    if torch is None:
        raise ImportError(f"quactlize needs torch for the kernel ops: {_TORCH_ERR}")
    if not _LOADED:
        here = os.path.dirname(os.path.abspath(__file__))
        found = sorted(glob.glob(os.path.join(here, "_C*.so")))
        if not found:
            raise ImportError(
                "quactlize's operator library is not built. For the host half (preprocessing), which needs no PPU:\n"
                "    pip install --no-build-isolation -e .\n"
                "or, in place:\n"
                "    python setup.py build_ext --inplace")
        torch.ops.load_library(found[0])
        _LOADED = True
    return torch.ops.quactlize


# --------------------------------------------------------------------------------------------------------------
# Weight preprocessing -- the offline reorder.
#
# THE STORAGE CONSTRAINT: every function here is byte-neutral. The output holds as many bytes as the input, and for
# the layout transforms it holds the same VALUES, only rearranged (preprocess also applies the +8 / +128 bias that
# maps the signed code range to unsigned). That is what lets the product satisfy "an offline reorder is allowed, an
# increase in stored bytes is not", and it is asserted in tests/test_preprocess_ops.py rather than assumed here.

def preprocess_weights_to_layout(w_row_major, quant_type, layout: str = "mixed_gemm"):
    """Rearrange a packed weight into a NAMED layout. This is the supported entry point.

    `layout` is a canonical name or an alias from quactlize.layouts -- "mixed_gemm", "mixed_gemm_aiu", "w4a8",
    "logical", or the full token join such as "mmarow32_tr_cl4_aiu256_cvtword_bias". The name is the only thing that
    distinguishes one arrangement from another afterwards: two layouts of the same weight have identical dtype,
    shape and byte count, so a caller that stores a weight must record which layout it is in.

    The op validates the element width against the name and the shape against any step that constrains it, and
    refuses rather than returning a different arrangement than the name promises.

    The two-boolean form this replaced is still reachable as _preprocess_weights_for_mixed_gemm, for the step-level
    tests. It is underscored on both sides now; it was public here while the C++ had already demoted it, which left
    the package exporting exactly the interface the implementation had moved away from.
    """
    return _ops().preprocess_weights_to_layout(w_row_major, quant_type, layout)


def _preprocess_weights_for_mixed_gemm(w_row_major, quant_type, is_int8_mma: bool = False,
                                       use_aiu_interleaved: bool = False):
    """The flag form. Prefer preprocess_weights_to_layout: two booleans cannot name what they produced."""
    return _ops()._preprocess_weights_for_mixed_gemm(w_row_major, quant_type, is_int8_mma, use_aiu_interleaved)


def gguf_scale_prepass(scale_blocks, d, dmin, qtype: int, zmul: int):
    """GGUF k-quant scale metadata -> the fp16 (scale, zero) planes the collectives consume. THE ONLINE PRE-PASS.

    `scale_blocks` is uint8 [rows, block_bytes] holding the format's SCALE block only -- not the whole GGUF block,
    whose layout differs per format (Q2_K, Q3_K and Q6_K put d at the END). Slicing it out is the caller's job, and
    tests/test_gguf_golden.py verifies those ranges against the official gguf package rather than trusting them.

    `zmul` IS NOT A PROPERTY OF THE FORMAT. It is the consuming converter's centre correction: the int4 converter
    emits q-8 where a k-quant means q, so a consumer built on it needs zmul=8 and one without a shift needs 0. There
    is no default because a missing correction is off by 8*scale everywhere and still produces plausible-looking
    weights. Accepted values are 0 and 8, the two that exist in this tree.

    Returns (scale, zero), each fp16 [rows, groups].
    """
    return _ops().gguf_scale_prepass(scale_blocks, d, dmin, qtype, zmul)


def gguf_vecdot(blocks, x, qtype: int):
    """PURE CUDA-CORE DECODE: one dot product per RAW GGUF block, scales taken straight into registers.

    This is the decode-band answer the pre-pass cannot give. The pre-pass removes the STORAGE cost of a k-quant's
    fp16 planes by building them in a workspace; at decode they would have to be rebuilt per token or kept resident,
    and keeping them is the forbidden storage again. Here nothing is materialised at all.

    `blocks` is uint8 [rows, type_size] holding RAW GGUF blocks -- the checkpoint's own bytes, no repacking. `x` is
    fp16 or fp32 [rows, 256]; the device path consumes fp16 and the CPU oracle accepts both. Returns fp32 [rows].
    """
    return _ops().gguf_vecdot(blocks, x, qtype)


def gguf_vecdot_dense(blocks, x, n: int, k: int, qtype: int):
    """FULLY_QUANTIZED dense GEMV: one complete activation, one production CUDA-core launch."""
    return _ops().gguf_vecdot_dense(blocks, x, n, k, qtype)


def gguf_vecdot_moe(blocks, x, row_offsets, qtype: int):
    """FULLY_QUANTIZED MoE decode through the device library.

    `blocks` is uint8 `[experts,n*k/256,type_size]`, `x` is gathered fp16/fp32 `[total_rows,k]`, and
    `row_offsets` is int32 `[experts+1]`. Returns fp32 `[total_rows,n]`. There is deliberately no CPU fallback:
    expert/ragged launch plumbing is the operation being requested.
    """
    return _ops().gguf_vecdot_moe(blocks, x, row_offsets, qtype)


def gguf_gemv_bc(a, low, high, units, qtype: int):
    """SIMT decode over placed code planes plus packed scale units.

    Dense ``a`` has shape ``[M,K]`` with ``1 <= M < 8``.  All rows share the
    resident weight artifact and execute in one native grid-y batched launch;
    the result is fp32 ``[M,N]``.
    """
    return _ops().gguf_gemv_bc(a, low, high, units, qtype)


def gguf_gemv_bc_moe(a, low, high, units, row_offsets, qtype: int):
    """The MoE arm; row_offsets is the cumulative form, so an empty expert is representable. -> fp32 [rows, N]."""
    return _ops().gguf_gemv_bc_moe(a, low, high, units, row_offsets, qtype)


def gguf_packed_scale_prepass(units, qtype: int, zmul: int = 0):
    """Packed scale units -> fp16 (scale, zero) planes. Dense [k_unit,n,ub] or grouped [E,k_unit,n,ub] in,
    always [E, k/group_size, n] out."""
    return _ops().gguf_packed_scale_prepass(units, qtype, zmul)


def gguf_prepare_fully_quantized_dense(blocks, n: int, k: int, qtype: int):
    """FULLY_QUANTIZED/DENSE offline artifact: (low, high, units); high is empty unless the format is 2-plane."""
    return _ops().gguf_prepare_fully_quantized_dense(blocks, n, k, qtype)


def gguf_dense_fully_quantized(a, low, high, units, qtype: int):
    """Dense GEMM reading the format's own packed scale bytes. `high` is empty for a single-plane format.
    rc=34 without PPU_PACKED_SCALE=1."""
    return _ops().gguf_dense_fully_quantized(a, low, high, units, qtype)


def gguf_prepare_fully_quantized_grouped(blocks, n: int, k: int, qtype: int, experts: int):
    """FULLY_QUANTIZED/GROUPED offline artifact: (low, high, units) per expert; high empty unless 2-plane."""
    return _ops().gguf_prepare_fully_quantized_grouped(blocks, n, k, qtype, experts)


def gguf_grouped_fully_quantized(a, low, high, units, rows_per_expert, qtype: int):
    """Grouped (MoE) GEMM on packed scale bytes. Ragged rows; an expert may have none."""
    return _ops().gguf_grouped_fully_quantized(a, low, high, units, rows_per_expert, qtype)


def gguf_dequantize(blocks, qtype: int):
    """RAW GGUF blocks -> full fp16 weights. THE FALLBACK PATH'S MISSING LINK.

    The fallback is dequantise-then-library-GEMM: cuBLAS for dense, DeepGemm for MoE. dequantize_weight in
    unfused_weight_dequantize.hpp already handles the symmetric packed forms, but cannot read a k-quant block, so a
    GGUF checkpoint had no route to those GEMMs. Uses the same traversal as gguf_vecdot, so each format's bit layout
    is transcribed once.

    Costs what it says: a dequantised fp16 weight is 4x the int4 codes and ~3.6x the native k-quant block, so this
    path only pays where the result is reused across many rows. `blocks` is uint8 [rows, type_size]; returns fp16
    [rows, 256].
    """
    return _ops().gguf_dequantize(blocks, qtype)


def gguf_unpack(blocks, qtype: int):
    """RAW GGUF blocks -> (codes int8 [rows,256], scale fp16 [rows,groups], zero fp16 [rows,groups]).

    THE PIECE THAT LETS A CHECKPOINT REACH THE EXISTING KERNELS. They consume a packed low-bit weight plus fp16
    planes; nothing turned a k-quant block into that triple, so a GGUF file had no route to them. With this the chain
    is raw GGUF -> unpack -> pack_int4 -> preprocess_weights_to_layout, entirely through already-validated ops rather
    than a new packer with new ways to be wrong.

    The split obeys W = code * scale + zero exactly, so reconstructing and comparing against the official reference
    is a test that can fail. Codes are SIGNED for Q3_K (-4..3) and Q6_K (-32..31); int8 covers every format.
    """
    return _ops().gguf_unpack(blocks, qtype)


def gguf_prepare_gemv(blocks, n: int, k: int, qtype: int):
    """Build gemv_lowbit's resident SCALE_FIRST artifact from raw GGUF blocks.

    `blocks` is `[n*k/256,type_size]` for dense or `[experts,n*k/256,type_size]` for MoE. Returns
    `(low, high, scale, zero)`: native packed code planes `[experts,n,...]` and fp16 planes
    `[experts,k/group_size,n]`. `high` is empty for single-plane formats. The extra expert dimension is retained
    for dense (size one) so the artifact has one shape at the device-library boundary.
    """
    return _ops().gguf_prepare_gemv(blocks, n, k, qtype)


def gguf_prepare_dense(blocks, n: int, k: int, qtype: int):
    """Build fpA_intB_ppu's resident SCALE_FIRST dense artifact.

    The output is `(low, high, scale, zero)` with the same shapes as `gguf_prepare_gemv`, but the code planes are in
    the format-selected fpA xplane layout (TK=256 for Q2-Q5, TK=128 for Q6). Preparation is offline/one-time.
    """
    return _ops().gguf_prepare_dense(blocks, n, k, qtype)


def gguf_gemv_artifact_dequantize(low, high, scale, zero, qtype: int):
    """Inverse of `gguf_prepare_gemv`: resident artifact -> fp16 `[experts,n,k]` weight."""
    return _ops().gguf_gemv_artifact_dequantize(low, high, scale, zero, qtype)


def gguf_gemv_artifact_dequantize_scale(scale, zero, qtype: int):
    """Read GEMV artifact affine channels as stored fp16 `[experts,n,k/group_size]` planes."""
    return _ops().gguf_gemv_artifact_dequantize_scale(scale, zero, qtype)


def gguf_dense_artifact_dequantize(low, high, scale, zero, qtype: int):
    """Inverse of `gguf_prepare_dense`: fpA xplane artifact -> fp16 `[1,n,k]` weight."""
    return _ops().gguf_dense_artifact_dequantize(low, high, scale, zero, qtype)


def gguf_dense_artifact_dequantize_scale(scale, zero, qtype: int):
    """Read dense artifact affine channels as stored fp16 `[1,n,k/group_size]` planes."""
    return _ops().gguf_dense_artifact_dequantize_scale(scale, zero, qtype)


def gguf_gemv_scale_first(a, low, high, scale, zero, qtype: int):
    """Dense CUDA-core GEMV over a resident `gguf_prepare_gemv` artifact."""
    return _ops().gguf_gemv_scale_first(a, low, high, scale, zero, qtype)


def gguf_gemv_scale_first_moe(a, low, high, scale, zero, row_offsets, qtype: int):
    """Ragged MoE CUDA-core GEMV over a resident `gguf_prepare_gemv` artifact."""
    return _ops().gguf_gemv_scale_first_moe(a, low, high, scale, zero, row_offsets, qtype)


def gguf_dense_scale_first(a, low, high, scale, zero, qtype: int):
    """Dense mixed-input GEMM over a resident `gguf_prepare_dense` artifact."""
    return _ops().gguf_dense_scale_first(a, low, high, scale, zero, qtype)


def gguf_backend():
    """Which backend the gguf ops resolve to, as a string, plus why.

    PPU device code is built by build.sh with hgcc and cannot live in this extension, which setup.py builds with gcc
    and which has to keep running on machines with no SDK -- that is what makes the official gguf package usable as
    an oracle at all. The two halves share a PROCESS instead: build.sh emits libquactlize_ppu.so with C entry points,
    this extension dlopens it, and the op forwards. Set QUACTLIZE_PPU_LIB to point at a specific one.

    Returns "ppu (...)" or "cpu (...)". It is a value rather than something to infer from a timing because a silent
    fallback produces correct numbers slowly and reports nothing, which is indistinguishable from the device path
    working.
    """
    return _ops().gguf_backend()


def gguf_pack_unit(scale_blocks, d, dmin, qtype: int):
    """GGUF scale block -> the REORDERED packed unit the in-kernel path reads. Byte-neutral by construction.

    GGUF's own scale packing is not half-separable -- Q4_K's get_scale_min_k4 takes groups 4..7 from bytes 8-11 and
    the top two bits of bytes 0-3 -- so a k-tile covering half a superblock cannot read half a block. The unit makes
    each run of groups self-contained at no cost in stored bytes: 20, 14, 16, 16 and 18 bytes for Q2/Q3/Q4/Q5/Q6
    against GGUF's own 20, 14, 16, 16 and 18.

    A 2-D input [rows,scale_bytes] returns the row-wise reference form. A real offline artifact is 3-D:
    scale_blocks [N,superblocks,scale_bytes] and headers [N,superblocks] return [unit_superblocks,N,unit_bytes],
    changing GGUF's column-major outer order into the collective's k-major copy order as well as rearranging bits.
    """
    return _ops().gguf_pack_unit(scale_blocks, d, dmin, qtype)[0]


def gguf_unit_decode(units, qtype: int, zmul: int):
    """The packed unit -> (scale, zero), the same decode the collective runs in the kernel."""
    return _ops().gguf_unit_decode(units, qtype, zmul)


def gguf_q4_artifact_dequantize(weight, units, layout: str = "mixed_gemm"):
    """Consume the two OFFLINE-SHUFFLED Q4_K artifacts, with no raw GGUF input in this arm.

    `weight` is [K,N/2] after pack_int4 + preprocess_weights_to_layout. `units` is [K/256,N,16] from the 3-D
    gguf_pack_unit form. Returns float32 [K,N], decoding the stored converter-word order and the packed unit with
    ZMul=8, exactly the two forms the collective consumes. This is a correctness/reference consumer, not a GEMM.
    """
    return _ops().gguf_q4_artifact_dequantize(weight, units, layout)


def gguf_scale_block_shape(qtype: int):
    """(block_bytes, groups, group_size, has_min, scale_bias, signed) for a k-quant's SCALE block.

    Read from the C++ Traits so Python cannot carry a second, drifting copy. Note block_bytes is the SCALE block's,
    which is not the GGUF block size quactlize.formats uses for storage arithmetic -- confusing the two makes a test
    slice the wrong bytes and fail as if the decode were wrong.
    """
    return _ops().gguf_scale_block_shape(qtype)


def symmetric_quantize(w, quant_type, arch: int = 80):
    """fp16/bf16/fp32 weights -> (preprocessed codes, per-column scales). The reference path for GPTQ-style symmetric.

    Scale is max|w| over the K axis divided by 2**(bits-1) -- 8 for int4, not 7. The codes are computed with the fp32
    column max while the scale returned is that value in fp16, so dequantising with the returned scale is not exactly
    the inverse; see tests/test_preprocess_ops.py, which pins both.
    """
    return _ops().symmetric_quantize_last_axis_of_batched_matrix(w, quant_type, arch)


def symmetric_quantize_unprocessed(w, quant_type, arch: int = 80):
    """As symmetric_quantize, but also returns the codes BEFORE the layout transform, as (unprocessed, processed,
    scales). The unprocessed tensor is what an oracle should compare against -- it is still in logical row-major
    order, so a reference dequantisation can be written without knowing the physical layout."""
    return _ops()._symmetric_quantize_last_axis_of_batched_matrix(w, quant_type, arch)


def pack_int4(w):
    """int8 tensor of values in [-8, 7] -> packed int4, two per byte, low nibble first. Halves the last dimension."""
    return _ops().pack_int8_tensor_to_packed_int4(w)


def unpack_int4(w_packed):
    """Inverse of pack_int4."""
    return _ops().unpack_int4_packed_tensor_to_int8(w_packed)


def pack_uint4(w):
    return _ops().pack_uint8_tensor_to_packed_uint4(w)


def unpack_uint4(w_packed):
    return _ops().unpack_uint4_packed_tensor_to_uint8(w_packed)


# The remaining ops are the individual STEPS of preprocess_weights_for_mixed_gemm. They are registered with a leading
# underscore ("exposed purely for unit tests" in the C++), and are re-exported under the same convention so that
# nobody reaches for one thinking it is the supported entry point.

def _permute_B_rows_for_mixed_gemm(w, quant_type, arch_version: int = 80):
    return _ops()._permute_B_rows_for_mixed_gemm(w, quant_type, arch_version)


def _subbyte_transpose(w, quant_type):
    """CAUTION: the returned tensor's SHAPE is the input's, not the transposed one -- the bytes are transposed and the
    metadata is not. Only square inputs hide it. Carry the logical shape yourself."""
    return _ops()._subbyte_transpose(w, quant_type)


def _add_bias_and_interleave_int4s(w):
    return _ops()._add_bias_and_interleave_int4s(w)


def _add_bias_and_interleave_int8s(w):
    return _ops()._add_bias_and_interleave_int8s(w)


# --------------------------------------------------------------------------------------------------------------
# Legacy GPTQ convenience GEMM. The GGUF SCALE_FIRST dense path is bound separately above because its artifact
# includes format-specific low/high planes and affine corrections this three-tensor signature cannot express.

def matmul_w4a16(a, b_packed, scales, zeros=None, group_size: int = 128,
                 fmt: Literal["gptq-sym", "gguf-q4k"] = "gptq-sym"):
    raise NotImplementedError(
        "the GEMM is not bound as a torch op yet -- it is reached through the C++ harnesses in tests/ and "
        "benchmarks/. Binding it needs csrc/ to build under the PPU toolchain, which the overlay-based build.sh "
        "does not currently support. See docs/CHECKPOINT.md.")


__all__ = [
    "preprocess_weights_to_layout", "symmetric_quantize", "symmetric_quantize_unprocessed",
    "gguf_scale_prepass", "gguf_scale_block_shape", "gguf_vecdot", "gguf_vecdot_dense", "gguf_vecdot_moe", "gguf_dequantize", "gguf_unpack", "gguf_prepare_gemv", "gguf_prepare_dense", "gguf_gemv_artifact_dequantize", "gguf_gemv_artifact_dequantize_scale", "gguf_dense_artifact_dequantize", "gguf_dense_artifact_dequantize_scale", "gguf_gemv_scale_first", "gguf_gemv_scale_first_moe", "gguf_dense_scale_first", "gguf_backend", "gguf_pack_unit", "gguf_unit_decode", "gguf_q4_artifact_dequantize",
    "pack_int4", "unpack_int4", "pack_uint4", "unpack_uint4", "matmul_w4a16",
]
