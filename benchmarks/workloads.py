#!/usr/bin/env python3
"""THE THREE TARGET MODELS AND THE TOKEN COUNTS -- the sweep's problem shapes, not 4096 squared.

The user fixed these on 2026-08-04:

    Qwen3-32B              dense
    Qwen3.5-35B-A3B        MoE
    Qwen3.5-122B-A10B      MoE, TENSOR-PARALLEL ACROSS 2 CARDS
    n-token in {1, 2, 4, 64, 2048, 4096}

SHAPES ARE NOW IN, read off huggingface config.json on 2026-08-04 rather than remembered. The box checkpoints
should still confirm them -- what runs is a QUANTISED repo and a GGUF's tensor shapes outrank a config.json for
the file actually loaded -- but nothing here is a plausible-looking guess.

AND THEY ARE NOTHING LIKE 4096 SQUARED, which is the only shape this project has ever measured. The 35B's expert
GEMM is n=512 k=2048; the 122B's, after TP, is n=512 k=3072 and its down-projection is k=512. A TileN of 128
covers a quarter of that N in one tile, and TileK=256 leaves the down-projection with TWO k-steps. Tactics tuned
on 4096 squared were tuned on a shape with two orders of magnitude more k-loop to amortise over.

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

    and with the real numbers -- 256 experts, top-8, for BOTH MoE models -- that is exactly M/32:

        M=1     0.03 rows/expert     only 8 experts are touched at all, one row each
        M=2     0.06                 16 experts, one row each
        M=4     0.12                 32 experts, one row each
        M=64    2.00
        M=2048 64.00
        M=4096 128.0                 the band #17b recorded, where masked rows dominate

    SO THE MoE DECODE POINTS ARE NOT SMALL-M GEMMs, THEY ARE ONE-ROW GEMMs OVER A SUBSET OF EXPERTS. At M<=4 no
    expert receives two rows; the kernel's job is to visit 8-32 experts and do a single row in each. That is the
    Mmax==1 case the PPU_A_CPASYNC stride-0 path was built for and measured on -- it is the shipping shape here,
    not a corner. It also means the "ragged" distribution at decode is degenerate: every touched expert has
    exactly one row, and the raggedness is in WHICH experts, not how many rows. The MoE sweep must be swept on ROWS PER EXPERT, and the
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
    that never runs. The user specified the STANDARD split, so projections() applies it rather than leaving it
    open: q/k/v/gate/up column-parallel, o/down row-parallel.

    IT BITES HARDEST ON THE SMALLEST TENSORS. k and v are n=512 before TP and n=256 after -- two TileN=128 tiles
    on a whole projection. The expert FFN goes n=1024 -> 512. Halving an already-narrow N is where a tactic tuned
    on the unsplit shape would be most wrong.

    NOTE FOR THE MoE MODELS: TP interacts with fact 2. Halving the expert FFN's N does not change rows per
    expert, but halving K does change the K-loop count, and both change which (TileN, TileK) are even legal.
    So the ragged rows-per-expert distribution and the TP split have to be applied together, not in sequence.
"""
import argparse
import sys

# THE ONLY NUMBERS HERE THAT ARE NOT PENDING. The user gave these directly.
N_TOKENS = [1, 2, 4, 64, 2048, 4096]

MODELS = {
    # READ FROM huggingface.co/Qwen/<name>/raw/main/config.json ON 2026-08-04, not remembered. The box checkpoints
    # must still confirm: what runs is a QUANTISED repo, and a GGUF's tensor shapes are the authority over a
    # config.json for the file we actually load. Both MoE configs also carry a vision tower, so these are the
    # text_config values -- the vision blocks are not in this sweep.
    "Qwen3-32B": dict(
        kind="dense", tp=1,
        hidden=5120, inter=25600, layers=64, heads=64, kv_heads=8, head_dim=128,
        experts=0, topk=0),
    "Qwen3.5-35B-A3B": dict(
        kind="moe", tp=1,
        hidden=2048, inter=None, moe_inter=512, shared_inter=512, layers=40,
        heads=16, kv_heads=2, head_dim=256, experts=256, topk=8),
    "Qwen3.5-122B-A10B": dict(
        kind="moe", tp=2,
        hidden=3072, inter=None, moe_inter=1024, shared_inter=1024, layers=48,
        heads=32, kv_heads=2, head_dim=256, experts=256, topk=8),
}


def projections(cfg: dict, tp: int = None):
    """-> [(name, n, k, split)] per card. split is 'col' (N halved by TP) or 'row' (K halved) or None.

    STANDARD TP, as the user specified: q/k/v and the FFN gate/up are COLUMN-parallel, o and down are
    ROW-parallel. Applied here rather than left as a note, because the earlier version of this file said the
    convention had to be read off the serving stack and that made a settled thing look open.
    """
    tp = cfg["tp"] if tp is None else tp
    h, hd = cfg["hidden"], cfg["head_dim"]
    out = [("q", cfg["heads"] * hd, h, "col"),
           ("k", cfg["kv_heads"] * hd, h, "col"),
           ("v", cfg["kv_heads"] * hd, h, "col"),
           ("o", h, cfg["heads"] * hd, "row")]
    if cfg["kind"] == "dense":
        out += [("gate", cfg["inter"], h, "col"), ("up", cfg["inter"], h, "col"),
                ("down", h, cfg["inter"], "row")]
    else:
        mi = cfg["moe_inter"]
        out += [("expert_gate", mi, h, "col"), ("expert_up", mi, h, "col"),
                ("expert_down", h, mi, "row")]
    return [(nm, n // tp if s == "col" else n, k // tp if s == "row" else k, s) for nm, n, k, s in out]

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
    ap.add_argument("--confirm", action="store_true",
                    help="print the box commands that confirm these shapes against the actual checkpoints")
    a = ap.parse_args()

    print(f"n-token axis: {N_TOKENS}\n")
    for name, cfg in MODELS.items():
        print(f"== {name}  ({cfg['kind']}, tp={cfg['tp']}) ==")
        for nm, n, k, split in projections(cfg):
            print(f"     {nm:<12} n={n:<6} k={k:<6} {split}-parallel")
        if cfg["kind"] == "moe":
            # THE LINE THAT REFRAMES THE SWEEP. At M<=4 no expert gets two rows: the kernel visits a SUBSET of
            # experts and does one row in each. That is not a small-M GEMM, it is the Mmax==1 shape.
            print(f"     rows/expert = M*{cfg['topk']}/{cfg['experts']} = M/{cfg['experts']//cfg['topk']}:")
            for m in N_TOKENS:
                r = rows_per_expert(m, cfg["experts"], cfg["topk"])
                note = f"  <- only {m*cfg['topk']} experts touched, ONE row each" if r < 1 else ""
                print(f"        M={m:<5} {r:>7.2f}{note}")
        print()

    if a.confirm:
        print("== confirm against the checkpoints (config.json is not what we load) ==")
        for name, (kind, path) in CHECKPOINTS.items():
            print(f"  {name}\n      python3 tools/inspect_models.py {kind} {path}")
        print("\n  A disagreement between these and the config.json values above is the checkpoint's win:\n"
              "  a quantised repo can differ, and the GGUF's tensor shapes are what the kernel sees.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
