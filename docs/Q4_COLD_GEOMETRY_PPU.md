# Q4 cold-weight C4/C8 geometry experiment

The only target is **M=1, N=8192, K=5120**. Warm-cache performance is not an
admission criterion. All four arms rotate a weight ring larger than 2.25 times
the verified L2 capacity. Six forward/reverse rounds each collect 15 graph
samples after upload/first-launch warmups, for 24 timing cells. This is not a
new sweep or a production selector change.

| Arm | Body | Tile N | CTAs | Threads/CTA | K workers | K passes | Last-pass active workers |
|---|---|---:|---:|---:|---:|---:|---:|
| `kpack-c4` | Existing group-affine C4/W8/P4 | 16 | 512 | 256 | 64 | 3 | 32 |
| `kpack-c8` | Same template, C8/W8/P4 | 32 | 256 | 256 | 32 | 5 | 32 |
| `xplane` | Existing selected C4/W8 | 32 | 256 | 256 | — | — | — |
| `raw-reference` | Supplied FP32 reference, C4/WN8/WK1 | 32 | 256 | 256 | — | — | — |

C8 changes the existing kernel's column grouping, not the offline bytes,
decoder, FP32 group-affine formula or reducer algorithm. The work/reduction
assignment changes, so bit-identical C4/C8 floating-point output is **not**
required. Each is checked independently against the same official GGUF FP64
dot oracle, with zero-code/zero-A checks, output/workspace guards and exact
post-replay repeatability. All arms use FP16 A and FP32 accumulation/output.
The controls dequantize individual weights in FP16; K-pack uses group-affine
arithmetic. This remains an implementation comparison, not identical rounding.

Only C8 and a marker were newly compiled. C4, Xplane and raw-reference use
their original prebuilt DSOs. There is no AIU arm, new Split-K kernel, CPU
weight conversion in the timed region, JIT, or box compilation. More balanced
K passes and fewer CTAs do not by themselves prove a speedup: they also
change concurrency, load grouping and per-thread work.

The prior rotating times were 27.817 us for C4, 22.475 us for Xplane and
22.373 us for reference. Those numbers are context only, not reused samples.
The new pass threshold is 1.05 times the faster of the two controls measured
in the same new cohort (historical target approximately 23.491 us).

## Run on an otherwise idle PPU

```bash
(
  git switch develop &&
  GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin develop &&
  git lfs pull --include="prebuilt/ppu0010/q4-simt-ab-v1/*.so,prebuilt/ppu0010/q4-h800-port-v1/*.so,prebuilt/ppu0010/q4-cold-geometry-v1/*.so" --exclude="" &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
  bash tools/run_q4_cold_geometry_ppu_box.sh
)
```

Only the one fixture is created; `FIXTURES=/path/to/existing/fixtures` can
reuse it. Default ACU collects all four actual kernels with forced-cold
replay. ACU durations are not substituted for rotating event timings.
The identity-probe child exits before the experiment; the parent owns no GPU
context. `ACU=0` disables counters without changing timing conditions.

A failing child does not stop the other arms. Successful per-round receipts
and ACU reports are retained; `RESUME_RUN=/workspace/q4-cold-geometry-ppu.XXXXXX`
reuses only hash-verified results with the exact same source, images, runtime,
device and fixture. A missing or dirty result is not a performance pass.

Upload the printed `q4-cold-geometry-ppu.XXXXXX.results.tgz`. `status=PASS`
means the measurements are complete; only `parity_verdict=WITHIN_5_PERCENT`
means C8 meets the performance threshold. Production remains unchanged until
device results justify a selection change.
