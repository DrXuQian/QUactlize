# Q4 cold-weight six-shape comparison

This extends the C8 result to **shapes, not qtypes**. Only Q4_K dense M=1
is measured, with `(N,K)` from the existing six-shape registry:
`512x2048`, `1024x5120`, `4096x2048`, `4096x4096`, `5120x8192`, `8192x5120`.
It is not grouped, M2–M8, a new format, or an exhaustive config sweep.

Each shape has four arms:

- Xplane with its previously PPU-selected cold recipe;
- the supplied raw-GGUF FP32-accumulating reference with its selected cold recipe;
- the current per-shape best K-pack implementation, unchanged;
- C8/W8/P4 using the exact group-affine body from the single-shape experiment.

The old small/medium winners are retained, not replaced by global C4/W8.
Only six C8 specializations and a native marker were newly compiled; the
existing control DSOs are reused. There is no JIT or box compilation.
All timings are newly collected; old receipts supply configurations only.

Warm cache is not measured. Weights rotate through a ring exceeding 2.25
times verified L2 capacity. There are six alternating-order rounds of
fifteen graph samples per arm: **144 timing cells**, plus 24 forced-cold
ACU reports by default. Each cell checks independent GGUF FP64-dot accuracy,
zero-code/zero-A controls, guards, and post-replay repeatability. Setup,
graph upload and first-use warmups are excluded from event timings.
Controls reconstruct per-weight FP16; affine K-pack uses FP32 group-affine
arithmetic. All accumulate/output FP32, but rounding is not identical.

## Run

On an otherwise idle card, from the repository:

```bash
(
  git switch develop &&
  GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin develop &&
  git lfs pull --include="prebuilt/ppu0010/q4-simt-ab-v1/*.so,prebuilt/ppu0010/q4-h800-port-v1/*.so,prebuilt/ppu0010/q4-cold-shapes-v1/*.so" --exclude="" &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
  bash tools/run_q4_cold_shapes_ppu_box.sh
)
```

`FIXTURES=/path/to/existing/fixtures` reuses all six compatible fixtures.
Otherwise missing fixtures are generated before the run. `ACU=0` skips
only profiling. Successful cells are hash-bound and resumable with
`RESUME_RUN=/workspace/q4-cold-shapes-ppu.XXXXXX`; failed children do not
discard the other shapes/rounds. The calling Docker shell remains open.

The single-shape run took 83.2 seconds including four profiles, implying
about 8.3 minutes for six equal-cost shapes, **before fixture generation**.
Small shapes require more ring copies and launches; allow roughly 10–20
minutes on a similarly idle box. This is an estimate, not a runtime guarantee.

Upload the printed `q4-cold-shapes-ppu.XXXXXX.results.tgz`.
`summary.tsv` lists all four times, the faster K-pack arm and its deltas.
`status=PASS` means complete validated measurements, not performance parity.
`parity_verdict=WITHIN_5_PERCENT` requires the selected K-pack arm to be at
most 1.05 times each contemporaneous control. The separate C8 verdict makes
losses by C8 visible even if the older K-pack arm wins. ACU timings are never
substituted for rotating event timings. Production selection is unchanged.
