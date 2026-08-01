"""quactlize -- low-bit weight-only GEMM for the T-Head PPU.

The Python layer OWNS THE PHYSICAL LAYOUT and the kernel takes tensors that are already arranged. That boundary is
taken from TensorRT-LLM (preprocess_weights_for_mixed_gemm) and from IST-DASLab/qutlass, whose utils.py holds the
scale-swizzle and blocking for the same reason: a format's byte arrangement is host knowledge, and putting it in the
kernel is what makes "one kernel per format" happen.

The preprocessing half is already a torch op library in this tree (csrc/preprocess/thop), registered under `acext::`.
The GEMM half is not bound yet -- it is reached through the C++ harnesses in tests/ and benchmarks/. So the functions
below that call torch.ops are usable and the ones marked NOT BOUND are the work in progress; they raise rather than
returning something plausible.
"""
from typing import Literal, Optional

try:
    import torch
except ImportError as e:                                   # a host without torch can still use quactlize.formats
    torch = None
    _TORCH_ERR = e

_OPS_LOADED = False


def _ops():
    """Load the extension on first use, with a message that says what to do rather than an ImportError trace."""
    global _OPS_LOADED
    if torch is None:
        raise ImportError(f"quactlize needs torch for the kernel ops: {_TORCH_ERR}")
    if not _OPS_LOADED:
        try:
            import quactlize._C  # noqa: F401  -- registers torch.ops.acext / torch.ops.quactlize
        except ImportError as e:
            raise ImportError(
                "quactlize's extension is not built. It needs the PPU SDK (hgcc) and a torch that runs on the "
                f"device; see the README. Original error: {e}") from e
        _OPS_LOADED = True
    return torch.ops


# --------------------------------------------------------------------------------------------------------------
# Weight preprocessing -- the offline reorder. These are already registered (csrc/preprocess/thop).
# Every one of them is a PERMUTATION: the output holds the same bytes as the input, rearranged. That property is what
# lets the product satisfy "an offline reorder is allowed, an increase in stored bytes is not".

def preprocess_weights_for_mixed_gemm(w_row_major, quant_type, arch: int = 80):
    """Row permutation, sub-byte transpose and tile interleave, as one same-byte transform."""
    return _ops().acext.preprocess_weights_for_mixed_gemm(w_row_major, quant_type, arch)


def permute_B_rows_for_mixed_gemm(w, quant_type, arch: int = 80):
    return _ops().acext.permute_B_rows_for_mixed_gemm(w, quant_type, arch)


def subbyte_transpose(w, quant_type):
    return _ops().acext._subbyte_transpose(w, quant_type)


def add_bias_and_interleave_int4s(w):
    return _ops().acext.add_bias_and_interleave_int4s(w)


def symmetric_quantize(w, quant_type, arch: int = 80):
    """fp16 weights -> (processed int4/int8, per-column scales). The reference path for GPTQ-style symmetric."""
    return _ops().acext.symmetric_quantize_last_axis_of_batched_matrix(w, quant_type, arch)


# --------------------------------------------------------------------------------------------------------------
# The GEMM. NOT BOUND YET.

def matmul_w4a16(a, b_packed, scales, zeros=None, group_size: int = 128,
                 fmt: Literal["gptq-sym", "gguf-q4k"] = "gptq-sym"):
    raise NotImplementedError(
        "the GEMM is not bound as a torch op yet -- it is reached through the C++ harnesses in tests/ and "
        "benchmarks/. Binding it needs csrc/ to build under the PPU toolchain, which the overlay-based build.sh "
        "does not currently support. See docs/CHECKPOINT.md.")


__all__ = [
    "preprocess_weights_for_mixed_gemm", "permute_B_rows_for_mixed_gemm", "subbyte_transpose",
    "add_bias_and_interleave_int4s", "symmetric_quantize", "matmul_w4a16",
]
