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

from . import (gguf_dequantize, gguf_vecdot_dense, gguf_vecdot_moe, gguf_prepare_gemv,
               gguf_gemv_scale_first, gguf_gemv_scale_first_moe)
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
    """FULLY_QUANTIZED, GEMV: nothing materialised. a is (1, k) fp16/fp32; returns (1, n) fp32.

    The dedicated dense host op forwards one complete activation to vecdot_rows_kernel, which shares it across n
    output columns and accumulates k/256 raw GGUF blocks inside one launch. Without a device library it retains a
    bounded scalar witness and refuses large rows rather than silently impersonating an inference path."""
    _check_shape(blocks, n, k, qtype)
    if a.shape[0] != 1:
        raise ValueError(f"the GEMV band is m=1; got m={a.shape[0]}")
    # The device ABI is fp16. Apply that contract before the backend branch so the CPU witness and production launch
    # differ only in accumulation order, not in which activation values they received.
    return gguf_vecdot_dense(blocks, a.to(torch.float16).contiguous(), n, k, int(qtype))


def _row_offsets(rows_per_expert: torch.Tensor, experts: int, total_rows: int) -> torch.Tensor:
    if rows_per_expert.numel() != experts:
        raise ValueError(f"rows_per_expert has {rows_per_expert.numel()} entries, expected {experts}")
    rows = rows_per_expert.to(dtype=torch.int64, device="cpu")
    if bool((rows < 0).any()):
        raise ValueError("rows_per_expert must be nonnegative")
    if int(rows.sum().item()) != total_rows:
        raise ValueError(f"rows_per_expert sums to {int(rows.sum().item())}, activation has {total_rows} rows")
    return torch.cat((torch.zeros(1, dtype=torch.int64), rows.cumsum(0))).to(torch.int32).contiguous()


def matmul_native_gemv_moe(a: torch.Tensor, blocks: torch.Tensor, n: int, k: int, qtype: int,
                            num_experts: int, rows_per_expert: torch.Tensor) -> torch.Tensor:
    """FULLY_QUANTIZED/GEMV_MOE: native GGUF bytes, gathered rows, one CUDA-core launch."""
    if a.dim() != 2 or a.shape[1] != k:
        raise ValueError(f"activation must be [total_rows,{k}], got {tuple(a.shape)}")
    bpr = k // 256
    want = n * bpr
    if blocks.dim() == 2:
        if blocks.shape[0] != num_experts * want:
            raise ValueError(f"flat blocks need {num_experts * want} rows, got {blocks.shape[0]}")
        blocks = blocks.view(num_experts, want, blocks.shape[1])
    if blocks.dim() != 3 or blocks.shape[:2] != (num_experts, want):
        raise ValueError(f"blocks must be [{num_experts},{want},type_size], got {tuple(blocks.shape)}")
    offsets = _row_offsets(rows_per_expert, num_experts, a.shape[0])
    return gguf_vecdot_moe(blocks.contiguous(), a, offsets, int(qtype))


def prepare_scale_first(blocks: torch.Tensor, n: int, k: int, qtype: int):
    """Offline/resident artifact for both SCALE_FIRST decode shapes."""
    return gguf_prepare_gemv(blocks, n, k, int(qtype))


def matmul_scale_first_gemv(a: torch.Tensor, artifact, qtype: int) -> torch.Tensor:
    """SCALE_FIRST/GEMV over prebuilt `(low, high, scale, zero)` planes."""
    low, high, scale, zero = artifact
    return gguf_gemv_scale_first(a, low, high, scale, zero, int(qtype))


def matmul_scale_first_gemv_moe(a: torch.Tensor, artifact, qtype: int,
                                 rows_per_expert: torch.Tensor) -> torch.Tensor:
    """SCALE_FIRST/GEMV_MOE over prebuilt planes and gathered/ragged rows."""
    low, high, scale, zero = artifact
    experts = int(low.shape[0])
    offsets = _row_offsets(rows_per_expert, experts, a.shape[0])
    return gguf_gemv_scale_first_moe(a, low, high, scale, zero, offsets, int(qtype))


ROUTES = {
    "dequant_first": matmul_dequant_first,
    "native_gemv": matmul_native_gemv,
    "native_gemv_moe": matmul_native_gemv_moe,
    "scale_first_gemv": matmul_scale_first_gemv,
    "scale_first_gemv_moe": matmul_scale_first_gemv_moe,
}
