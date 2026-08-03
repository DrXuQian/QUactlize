"""THE HOST SIDE OF EACH ROUTE, so a route can be RUN and TIMED rather than described.

WHY THIS FILE EXISTS. schemes.py had six cells reading "a piece exists, the path does not", with notes saying the
remainder was host-side wiring rather than arithmetic. That note is true and it was also treated as a blocker for
longer than it deserved: for DEQUANT_FIRST the remaining step is `a @ w.T` against torch's own cuBLAS, and no part
of it needs the PPU toolchain. A path nobody can call is a path nobody can time, and the project's measurement gap
was never in the kernels -- it was that no two routes had ever produced the same number in the same process.

WHAT EACH ROUTE ACTUALLY COSTS, since that is the only reason to have more than one:

    dequant_first   materialise the whole weight as fp16, then one library GEMM. 3.6x the native block in DRAM
                    traffic and a full extra write+read, but the GEMM itself is cuBLAS-grade. Pays when M is large
                    enough that reuse amortises the materialisation.
    scale_first     materialise only the scale planes. ~1/16 of the traffic above, and the GEMM is mixed-input.
                    Pays in the middle band. At decode the planes would be rebuilt every token, which is why the
                    pre-pass does not answer the M=1 case.
    fully_quantized materialise nothing. The only route whose DRAM traffic is the checkpoint's own bytes.

THE SHAPE CONVENTION, which is the thing most likely to be got wrong silently. A GGUF weight of logical shape
(n, k) is stored as n*(k/256) blocks in ROW-MAJOR order: block b of row i is at flat index i*(k/256) + b. So
`gguf_dequantize` returning [n*k/256, 256] reshapes to (n, k) directly. Getting this backwards produces a weight
that is a valid permutation of the right one, and a test with a symmetric fixture will not notice -- which is why
the golden test drives these through an ASYMMETRIC activation.
"""
from typing import Optional

import torch

from . import gguf_dequantize, gguf_vecdot
from .formats import QuantType


def _check_shape(blocks: torch.Tensor, n: int, k: int, qtype: int) -> None:
    if k % 256:
        raise ValueError(f"k={k} is not a multiple of the k-quant superblock (256)")
    want = n * (k // 256)
    if blocks.dim() != 2 or blocks.shape[0] != want:
        raise ValueError(
            f"blocks should be [{want}, type_size] for an ({n}, {k}) weight in {QuantType(qtype).name}, "
            f"got {tuple(blocks.shape)}. Row-major: block b of row i sits at i*(k/256)+b")


def dequantize_weight(blocks: torch.Tensor, n: int, k: int, qtype: int) -> torch.Tensor:
    """RAW GGUF blocks -> the fp16 weight (n, k). The materialisation dequant_first pays for."""
    _check_shape(blocks, n, k, qtype)
    return gguf_dequantize(blocks, int(qtype)).view(n, k)


def matmul_dequant_first(a: torch.Tensor, blocks: torch.Tensor, n: int, k: int, qtype: int,
                         weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """DEQUANT_FIRST, dense: expand to fp16, then torch's cuBLAS. a is (m, k); returns (m, n).

    `weight` lets a caller hoist the materialisation out of a timing loop, which is the honest way to measure the
    two halves separately -- amortised over many GEMMs this route is exactly cuBLAS, and the whole argument for the
    other routes is the part that is NOT amortised."""
    if weight is None:
        weight = dequantize_weight(blocks, n, k, qtype)
    if a.shape[-1] != k:
        raise ValueError(f"activation has k={a.shape[-1]}, weight has k={k}")
    return a.to(weight.dtype) @ weight.t()


def matmul_dequant_first_grouped(a: torch.Tensor, blocks: torch.Tensor, n: int, k: int, qtype: int,
                                 num_experts: int, rows_per_expert: torch.Tensor) -> torch.Tensor:
    """DEQUANT_FIRST, grouped (MoE): the same route with a per-expert weight and ragged rows.

    `blocks` holds num_experts weights back to back, each (n, k). `rows_per_expert` is int64 [num_experts] and must
    sum to a.shape[0]; rows are assumed already gathered into expert order, which is what the grouped kernels also
    assume. The loop is deliberately a loop: this is the FALLBACK, and its cost model is one library GEMM per
    expert. Replacing it with DeepGemm changes the GEMM, not the route."""
    if rows_per_expert.numel() != num_experts:
        raise ValueError(f"rows_per_expert has {rows_per_expert.numel()} entries, expected {num_experts}")
    total = int(rows_per_expert.sum().item())
    if total != a.shape[0]:
        raise ValueError(f"rows_per_expert sums to {total}, activation has {a.shape[0]} rows")
    per_expert = n * (k // 256)
    _check_shape(blocks, n * num_experts, k, qtype)
    out = a.new_empty((a.shape[0], n), dtype=torch.float16)
    start = 0
    for e in range(num_experts):
        rows = int(rows_per_expert[e].item())
        if rows == 0:
            continue
        w = dequantize_weight(blocks[e * per_expert:(e + 1) * per_expert], n, k, qtype)
        out[start:start + rows] = a[start:start + rows].to(torch.float16) @ w.t()
        start += rows
    return out


def matmul_native_gemv(a: torch.Tensor, blocks: torch.Tensor, n: int, k: int, qtype: int) -> torch.Tensor:
    """FULLY_QUANTIZED, GEMV: nothing materialised. a is (1, k) fp32; returns (1, n) fp32.

    REFERENCE SPEED, NOT A COMPETITOR. The bound op computes one dot product per (block, 256-element activation
    slice) pair, so a full GEMV is assembled here by tiling the activation across a row's blocks and summing. That
    tiling is a host-side k/256-fold expansion of the ACTIVATION -- irrelevant to correctness, fatal to a timing.
    The kernel that does this properly is vecdot_rows_kernel, reached through the device library.

    It will REFUSE rather than crawl on a CPU-resolved backend above the reference row limit. That refusal is the
    point: a silent CPU fallback produces correct numbers slowly and reports nothing, which is indistinguishable
    from the device path working."""
    _check_shape(blocks, n, k, qtype)
    if a.shape[0] != 1:
        raise ValueError(f"the GEMV band is m=1; got m={a.shape[0]}")
    bpr = k // 256
    x = a.reshape(bpr, 256).to(torch.float32)                  # one 256-slice per block position
    x = x.unsqueeze(0).expand(n, bpr, 256).reshape(n * bpr, 256).contiguous()
    return gguf_vecdot(blocks, x, int(qtype)).view(n, bpr).sum(dim=1).view(1, n)


ROUTES = {
    "dequant_first": matmul_dequant_first,
    "native_gemv": matmul_native_gemv,
}
