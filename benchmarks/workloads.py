#!/usr/bin/env python3
"""THE THREE TARGET MODELS AND THE TOKEN COUNTS -- the sweep's problem shapes, not 4096 squared.

The user fixed these on 2026-08-04:

    Qwen3-32B              dense
    Qwen3.5-35B-A3B        MoE
    Qwen3.5-122B-A10B      MoE, TENSOR-PARALLEL ACROSS 2 CARDS
    n-token in {1, 2, 4, 64, 2048, 4096}

SHAPES ARE NOT WRITTEN HERE YET, ON PURPOSE. They must come from the checkpoints (tools/inspect_models.py reads
hidden_size / intermediate_size / moe_intermediate_size / num_experts / num_experts_per_tok), and those live on
the box. Every shape this project has measured so far was N=K=4096, which is not any of these three. Writing a
plausible 5120 here would be the same mistake as recording an arrangement the producer could not build: a number
nobody checked, in a file that reads like a record.

    python3 benchmarks/workloads.py --check      # says what is missing and how to obtain it
    python3 benchmarks/workloads.py              # the M axis and the derived per-expert rows, which ARE known

THREE STRUCTURAL FACTS THAT DO NOT NEED THE SHAPES, and that change what the sweep should even ask:

1.  SMALL M IS A TILE-AND-SMEM SETTING, NOT A HOLE. An earlier version of this file said M in {2,4} was served
    by neither the tensor-core path nor the vector one, on the grounds that TileM=16 discards 14 of 16 rows. That
    reasoning quietly equated DISCARDED ROWS with WASTED TIME, and the measurement says otherwise: in the decode
    profile `v.mma` is 1.5% of the kernel, against unpacking 42%, s.wait 15% and affine 13%. Eighty-eight percent
    of a 1.5% component is not a hole. The tensor-core path runs at M=2; what it needs is a small M-block and a
    smaller A footprint, not a different kernel.

    THE SMEM OPTION ALREADY EXISTS AND IS NOT YET GENERAL. `PPU_A_CPASYNC` keeps A in smem occupying ONE row --
    measured on 64x64x128 s3 as A 49152 B -> 768 B (64x), block 61840 -> 13456 B, blocks/CU 4 -> 19, bit-exact
    against a separate build. It works by giving SmemLayoutA a stride-0 M mode, which ALIASES every row onto the
    real one, so it is valid only at Mmax==1 and `launch()` rejects anything else. Covering M in {2,4} means
    allocating M rows instead of aliasing to one -- a generalisation of that layout, not a flag flip. It also
    requires bypassing the AIU/swzl read for A, whose instruction has a hard 16-row minimum with no stride
    operand; that part is already done and must not be re-litigated (four failed routes are recorded).

    AND THE PAYOFF MAY NOT BE WHERE IT LOOKS. The same measurement found that at DECODE the saving buys no
    occupancy: the warp count is pinned by the problem size (total warp-tiles / CU was unchanged when smem went
    57344 -> 40960), so the 64x has its value at PREFILL. Whether that carries to dense M=2 -- where the
    warp-tile arithmetic differs from the MoE decode it was measured on -- is not established. The sweep should
    therefore report blocks/CU alongside time at small M, or the result will not say which of the two is acting.

2.  FOR THE TWO MoE MODELS THE GLOBAL M IS NOT THE TILE'S M. Rows per expert is roughly

        rows_per_expert ~= M * num_experts_per_tok / num_experts

    so a 2048-token batch through a model with 128 experts and top-8 lands near 128 rows per expert -- exactly
    the band #17b recorded, where masked rows dominate. The MoE sweep must be swept on ROWS PER EXPERT, and the
    n-token values map onto it through that ratio. Sweeping MoE at M=2048 as if it were a dense GEMM measures a
    shape that never occurs.

    AND IT MUST BE RAGGED, NOT UNIFORM (user, 2026-08-04). That average is the mean of a distribution the router
    produces, and the thing the MoE kernel is actually fighting -- masked rows -- is a property of the SPREAD, not
    of the mean. A uniform M/expert fixture makes every expert's tile exactly full and so deletes the cost the
    sweep exists to measure; it would rank tactics by how well they do the one case that never happens. This is
    also a question already open in the repo rather than a new one: cutlass measured 33% MFU on ragged against
    the hand-written kernel on uniform, and the two were never put on the same fixture. The sweep is the place to
    close that, so the ragged distribution must be part of the fixture definition and stated with it -- an
    unnamed distribution is not reproducible, and "ragged" alone does not say how ragged.

3.  THE 122B IS TENSOR-PARALLEL OVER 2 CARDS -- confirmed by the user, not inferred from the card count. Its
    per-card shapes are therefore NOT the checkpoint's: column-parallel projections (q/k/v, gate, up) have N
    halved and row-parallel ones (o, down) have K halved. A sweep against the unsplit shape tunes for a GEMM
    that never runs. TP=2 is settled; what is still open is which projections the serving stack splits which
    way, and that has to be read off its configuration rather than assumed -- both conventions exist, and for an
    MoE the expert FFN may be split differently from the attention block.

    NOTE FOR THE MoE MODELS: TP interacts with fact 2. Halving the expert FFN's N does not change rows per
    expert, but halving K does change the K-loop count, and both change which (TileN, TileK) are even legal.
    So the ragged rows-per-expert distribution and the TP split have to be applied together, not in sequence.
"""
import argparse
import sys

# THE ONLY NUMBERS HERE THAT ARE NOT PENDING. The user gave these directly.
N_TOKENS = [1, 2, 4, 64, 2048, 4096]

MODELS = {
    "Qwen3-32B":         dict(kind="dense", tp=1, shapes=None),
    "Qwen3.5-35B-A3B":   dict(kind="moe",   tp=1, shapes=None),
    "Qwen3.5-122B-A10B": dict(kind="moe",   tp=2, shapes=None),
}

# The box paths inspect_models.py already documents. Named here so the request is copy-pasteable rather than
# reconstructed each time.
CHECKPOINTS = {
    "Qwen3.5-35B-A3B": ("gguf", "/sim/eec/shared/AI_workspace/llm-models/Qwen3.5-35B-A3B-Q4_K_M-GGUF/"
                                "Qwen3.5-35B-A3B-Q4_K_M.gguf"),
    "Qwen3.5-122B-A10B": ("gptq", "/sim/eec/shared/AI_workspace/llm-models/Qwen3.5-122B-A10B-GPTQ-Int4/ow7_224_ca"),
    "Qwen3-32B": ("?", "not yet located on the box -- find it before the sweep is scoped"),
}


def rows_per_expert(m: int, experts: int, topk: int) -> float:
    """The MoE tile's real batch. Not a tunable: it is what the router produces on average, and the reason a
    2048-token MoE batch is a ~128-row GEMM per expert rather than a 2048-row one."""
    return m * topk / experts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report what is missing and the command that supplies it")
    a = ap.parse_args()

    print(f"n-token axis: {N_TOKENS}")
    print("  M=1 is the GEMV kernel, not this sweep. M=2 and M=4 are the hole described in the docstring:")
    for m in (2, 4, 64):
        print(f"    M={m:<5} smallest tile is TileM=16 -> {100*(1-m/16):.0f}% of the tile's rows are discarded"
              if m < 16 else f"    M={m:<5} fills at least one 16-row tile")

    missing = [n for n, v in MODELS.items() if v["shapes"] is None]
    if not a.check:
        print(f"\nshapes pending for: {', '.join(missing)}")
        return 0

    print("\n== what is missing ==")
    for name in missing:
        kind, path = CHECKPOINTS[name]
        print(f"  {name}  ({MODELS[name]['kind']}, tp={MODELS[name]['tp']})")
        print(f"      python3 tools/inspect_models.py {kind} {path}")
    print("\n  WANTED per model: hidden_size, intermediate_size, moe_intermediate_size, num_experts,\n"
          "  num_experts_per_tok, and the per-layer (n, k) of each distinct projection. For the 122B also the\n"
          "  serving TP convention -- which axis the 2-card split halves -- because both exist and the sweep\n"
          "  tunes the wrong GEMM if it is assumed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
