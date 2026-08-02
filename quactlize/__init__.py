"""quactlize -- low-bit weight-only GEMM for the T-Head PPU.

The Python layer OWNS THE PHYSICAL LAYOUT and the kernel takes tensors that are already arranged. That boundary is
taken from TensorRT-LLM (preprocess_weights_for_mixed_gemm) and from IST-DASLab/qutlass, whose utils.py holds the
scale-swizzle and blocking for the same reason: a format's byte arrangement is host knowledge, and putting it in the
kernel is what makes "one kernel per format" happen.

TWO HALVES, AND ONLY ONE IS BOUND. The preprocessing half is a torch operator library built by setup.py; it is pure
host C++ and works anywhere torch is installed, including a machine with no PPU. The GEMM half is device code that
only hgcc compiles, and it is still reached through the C++ harnesses in tests/ and benchmarks/. Functions below that
call torch.ops work; matmul_w4a16 raises with the reason.

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
# The GEMM. NOT BOUND YET.

def matmul_w4a16(a, b_packed, scales, zeros=None, group_size: int = 128,
                 fmt: Literal["gptq-sym", "gguf-q4k"] = "gptq-sym"):
    raise NotImplementedError(
        "the GEMM is not bound as a torch op yet -- it is reached through the C++ harnesses in tests/ and "
        "benchmarks/. Binding it needs csrc/ to build under the PPU toolchain, which the overlay-based build.sh "
        "does not currently support. See docs/CHECKPOINT.md.")


__all__ = [
    "preprocess_weights_to_layout", "symmetric_quantize", "symmetric_quantize_unprocessed",
    "gguf_scale_prepass", "gguf_scale_block_shape",
    "pack_int4", "unpack_int4", "pack_uint4", "unpack_uint4", "matmul_w4a16",
]
