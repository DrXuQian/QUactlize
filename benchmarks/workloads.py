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

1.  M IN {2, 4} FALLS IN A HOLE. The smallest TileM in the tactic space is 16, so a batch of 2 discards 14 of 16
    rows and a batch of 4 discards 12 -- before any tactic is chosen. The GEMV kernel that covers the other end
    is single-token: quactlize's mmvf path is 2D/bf16/one-token only. So M=2 and M=4 are served by neither a
    tensor-core tile nor the vector kernel, and the sweep will faithfully report that every tactic is bad there.
    That is a real answer, but the useful follow-up is "extend the GEMV to M<=4", not "which 16-row tile wins".
    Flagged before the sweep runs so the result is not mistaken for a tuning failure.

2.  FOR THE TWO MoE MODELS THE GLOBAL M IS NOT THE TILE'S M. Rows per expert is roughly

        rows_per_expert ~= M * num_experts_per_tok / num_experts

    so a 2048-token batch through a model with 128 experts and top-8 lands near 128 rows per expert -- exactly
    the band #17b recorded, where masked rows dominate. The MoE sweep must be swept on ROWS PER EXPERT, and the
    n-token values map onto it through that ratio. Sweeping MoE at M=2048 as if it were a dense GEMM measures a
    shape that never occurs.

3.  THE 122B IS TENSOR-PARALLEL OVER 2 CARDS, so its per-card shapes are NOT the checkpoint's. Column-parallel
    projections (q/k/v, gate, up) have N halved; row-parallel ones (o, down) have K halved. A sweep against the
    unsplit shape would tune for a GEMM that never runs. Which axis is halved has to be read off the serving
    configuration, not assumed -- both conventions exist.
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
