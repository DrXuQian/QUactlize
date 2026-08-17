#!/usr/bin/env python3
"""Fail-closed source contract for the standalone chunked-GDN ABI harness."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/test_ppu_chunked_gdn_abi.cpp"
RUNNER = ROOT / "dev/fold_derivation/run_l205_ppu_chunked_gdn_abi.sh"
CUDA_ADAPTER = ROOT / "dev/fold_derivation/l205_chunked_gdn_cuda_adapter.cu"


def require(text: str, needle: str, label: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"missing {label}: {needle!r}")


def main() -> int:
    failures: list[str] = []
    source = SOURCE.read_text()
    runner = RUNNER.read_text()
    cuda_adapter = CUDA_ADAPTER.read_text()

    require(source, '#include "quactlize_ppu_linear_attention.h"', "public ABI include", failures)
    require(source, "constexpr int kTokens = 65;", "tail fixture", failures)
    require(source, "constexpr int kQkHeads = 1;", "QK-head fixture", failures)
    require(source, "constexpr int kVHeads = 2;", "GVA value-head fixture", failures)
    require(source, "constexpr int kHeadK = 128;", "K head size", failures)
    require(source, "constexpr int kHeadV = 128;", "V head size", failures)
    require(source, "constexpr int kChunk = 64;", "chunk size", failures)
    require(source, "gamma - previous_gamma", "chunk-local cumulative gamma difference", failures)
    for state in ("false", "true"):
        for pattern in ("kDistinct", "kPaired"):
            require(
                source,
                f"check_device_case({state}, KeyPattern::{pattern})",
                f"{pattern} {'nonzero' if state == 'true' else 'zero'}-state device arm",
                failures,
            )
    for plant in ("schema", "chunk", "head", "head-map", "grid-overflow", "extent-overflow"):
        require(source, f'{{"{plant}",', f"{plant} admission negative", failures)
    require(source, "plant=misaligned-q", "misaligned-q admission negative", failures)
    require(source, "output_raw_bad", "raw BF16 output verdict", failures)
    require(source, "state_raw_bad", "raw FP32 state verdict", failures)
    require(source, "check_bf16_boundaries", "constructive BF16-boundary proof", failures)
    require(source, "BF16-BOUNDARIES-EXACT/PASS", "exactness completion witness", failures)
    require(source, "[GDN fixture exactness plant] non-bf16-H", "BF16 proof negative control", failures)
    for boundary in ("exact.a", "exact.w", "exact.u", "exact.p", "exact.vnew", "exact.h_cast"):
        require(source, boundary, f"{boundary} materialization count", failures)
    require(source, "((t % kChunk) / 2) % kHeadK", "paired-key coordinate", failures)
    require(source, "std::size_t(kChunk / 2) * kVHeads", "paired-edge closed form", failures)
    require(source, "[GDN WY coverage]", "nontrivial WY coverage witness", failures)
    for witness in (
        "exact.strict_lower_nonzero == kExpectedPairedEdges",
        "exact.inverse_offdiag_nonzero == kExpectedPairedEdges",
        "exact.causal_offdiag_nonzero == kExpectedPairedEdges",
    ):
        require(source, witness, "paired-WY exact-count admission", failures)

    forbidden = (
        "ppu_chunked_gdn_kernel.cuh",
        "ppu_chunked_gdn_collective.cuh",
        "ppu_chunked_gdn_types.hpp",
        "triton",
        "cutedsl",
    )
    lowered = source.lower()
    for spelling in forbidden:
        if spelling.lower() in lowered:
            failures.append(f"harness crosses private/forbidden boundary: {spelling}")

    require(runner, "/workspace/quactlize-l205-ppu-chunked-gdn", "workspace artifact root", failures)
    require(runner, "TARGET=quactlize_ppu", "shipping shared-library build", failures)
    require(runner, "run_l203_chunked_gdn_oracle.sh", "L203 semantic authority", failures)
    require(runner, "l205_chunked_gdn_cuda_adapter.cu", "local CUDA adapter compile", failures)
    require(runner, '--box selects shipping PPU execution', "box/local device-authority split", failures)
    require(runner, "QZ_GDN_BOX_PREFLIGHT_ONLY", "box-mode negative-control seam", failures)
    require(cuda_adapter, "PpuChunkedGdnKernel<Arguments, Traits>", "exact scalar kernel type", failures)
    require(cuda_adapter, "scratch[blockIdx.x]", "global test scratch seam", failures)
    require(cuda_adapter, "Kernel::MaxThreadsPerBlock", "shipping block geometry", failures)
    if "/tmp" in runner or "mktemp" in runner:
        failures.append("runner must not use /tmp or mktemp")
    box_branch = runner.find('if [[ "$MODE" == "--box" ]]')
    cuda_invocation = runner.find('"$NVCC" -std=c++17')
    if box_branch < 0 or cuda_invocation < 0 or box_branch > cuda_invocation:
        failures.append("box mode must exclude the NVIDIA CUDA invocation before it can be reached")

    if failures:
        for failure in failures:
            print(f"[L205 contract] FAIL: {failure}")
        return 1
    print(
        "[L205 contract] PASS: public ABI only; T65/C64/KV128/GVA1:2; "
        "distinct+paired WY; zero+nonzero state; A/W/U/P/vnew/H exact; "
        "64/64/64 nontrivial-WY witness; seven admission negatives; raw verdicts"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
