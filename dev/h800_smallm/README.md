# H800 multi-row Q4 and MoE preparation experiments

Development-only CUDA experiments. They neither change canonical weight
bytes nor install a production selector. See
[measured results and open cases](../../docs/H800_SMALLM_20260911.md).

## GEMV

Use a fresh output directory for each build/run. Dependencies: CUDA toolkit,
Python, NumPy and the official `gguf` Python package; no Torch is required.
`FIXTURES` contains the six `q12-nN-kK-e1-c1.bin` standalone fixtures from
the existing Q4 H800 replay. These contain source GGUF plus canonical planes;
the runner independently decodes source weights, then creates random A.

```bash
python dev/h800_smallm/build_gemv.py --output "$BUILD" --jobs 5
python dev/h800_smallm/run_gemv.py \
  --bundle "$BUILD" --fixtures "$FIXTURES" --output "$SCREEN" \
  --experts 256 --rounds 1 --samples 7
```

The default comparison covers dense M1–7 and indexed M1 shared/independent
A, six families and two cache regimes. It does not scan a production
Cartesian product. `--scopes dense --m 2 --shapes 512x2048 --modes warm`
restricts a diagnostic. `--small-reader` and `--medium-reader` select
explicit alternative kernel bodies; `--arms` limits compilation to named
bodies plus required `xplane,reference` controls. Failed cells are recorded
and the other cells continue; an output directory is never overwritten.

Freeze screen winners and confirm, retaining control challenges seen in
other screens:

```bash
python dev/h800_smallm/select.py \
  --screens "$SCREEN" --bundles "$BUILD" --output "$FROZEN"
python dev/h800_smallm/run_gemv.py \
  --bundle "$FROZEN" --selection "$FROZEN/selection.json" \
  --fixtures "$FIXTURES" --output "$CONFIRM" \
  --experts 256 --rounds 6 --samples 15
```

Screening all alternative bodies requires separate `--small-reader` /
`--medium-reader` runs; merely compiling a body does not measure it.
`select.py` accepts multiple screen and bundle directories. The committed
measurement shortlist records the actual confirmation, including open cells.
It is not an online-tuning or production dispatch policy.

## MoE preparation

```bash
python dev/h800_smallm/build_moe.py --output "$MOE_BUILD"
"$MOE_BUILD/moe" --multi-token --model-splits --benchmark
"$MOE_BUILD/moe" --multi-token --model-splits --selected --benchmark
```

`--selected` uses vector M1 gathering and bounded compact M2–4 metadata;
other cases use the original path. Explicit rejected experiments remain
available as `--once`, `--once-wide` and `--fast-router`. The model-splits
mode sets S2/S2/S1. The benchmark has a fixed 15 samples, with 64 launches
per sample and five warmups. Without model-splits, the S4/S2/S8 descriptor stress
fixture is retained. Run the two arms sequentially on an idle GPU, not in
parallel. For final timing use four interleaved A/B rounds; the retained
report expects logs named `model-<0..3>-<baseline|selected>.log`.

`report.py` validates the full 108 GEMV and 96 MoE context denominators,
round/sample counts and reported winners, then writes compact JSON plus
compressed raw samples. It is CPU-only:

```bash
python dev/h800_smallm/report.py --results "$CONFIRM" \
  --moe "$MOE_BUILD" --bundle "$FROZEN" --output "$REPORT"
```

The top-8 preparation specialization covers tokens1–4 only. Dense M1–7
and indexed M1 are different tests, not evidence for a >32-row MoE router.
