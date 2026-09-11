# PPU replay of the three frozen H800 Q4 implementations

This is a native PPU experiment, not a production selection change. It ports
the three implementations confirmed in
[the H800 report](Q4_H800_OPTIMIZATION_20260911.md), retaining canonical
K-pack weights, FP16 A, and FP32 dot/reduction/output. The two `affine*`
implementations apply scale/min in FP32 after the group dot; Xplane and the
raw reference still reconstruct per-weight FP16 values. Thus this compares
implementations, not the layout alone.

| N x K | Implementation | (Columns, Warps, S) | Grid / threads |
|---|---|---|---|
| 512 x 2048 | cooperative metadata, vector A | (1,16,1) | 64 / 512 |
| 1024 x 5120 | eight-column group affine, vector A4 | (2,10,1) | 64 / 320 |
| 4096 x 2048 | four-column group affine | (4,8,1) | 256 / 256 |
| 4096 x 4096 | four-column group affine | (4,8,1) | 256 / 256 |
| 5120 x 8192 | four-column group affine | (4,8,1) | 320 / 256 |
| 8192 x 5120 | four-column group affine | (4,8,1) | 512 / 256 |

All are dense M=1, E=1, one kernel and S1, without an expanded scale workspace
or separate reducer. Each shape uses the same fixed K-pack recipe in warm
and rotating regimes. They are not claimed to be PPU-optimal before measuring.

## Scope and controls

- Three arms per shape: the frozen K-pack implementation, historical Xplane
  with FP32 accumulation, and the supplied raw-GGUF reader with FP32
  accumulation. The Xplane binary is unchanged from the earlier PPU package.
- On PPU, screen all 12 advertised Xplane recipes and all 60 raw-reference
  recipes. Freeze the best two per control, then confirm six alternating-order
  rounds, 15 event samples per round. K-pack has one candidate per shape.
- Ref's third recipe field is **intra-CTA K warps**, not inter-CTA Split-K.
- Every complete-call event sample traverses an entire cache ring. Rotating
  uses at least 2.25 times confirmed L2 capacity. Initial launches and graph
  upload are excluded, as is offline packing/host setup.
- Before timing, check all outputs against independent official-GGUF FP64
  dots, reject zeroed code planes, check zero A, and verify output/workspace
  guards. Repeat the oracle after timing and require deterministic FP32
  output. The error denominator is `sum(abs(A * W))`.
- A `WITHIN_5_PERCENT` cell must be less than 5% slower than **both** controls.
  `status=PASS` alone means the measurement completed, not performance parity.
- Native image/marker and numerical checks remain mandatory even if the
  installed SDK differs; SDK differences are recorded, not silently ignored.

The [2026-09-11 PPU retest](Q4_PPU_H800_PORT_RETEST_20260911.md) has now
completed: 1,236 numerical/timing records pass, but only 3/12 cells meet the
5% performance gate against both controls. Production remains unchanged.

Local compilation took 19.5 seconds with five parallel translation units.
The three candidate DSOs total about 206 KiB; the expanded raw-reference DSO
is about 684 KiB. Native device disassembly, exact entry symbols and runtime
linkage were checked. The immutable build manifest retains compile-only
status; device evidence is held separately in the retest receipt above.
Separate DSOs prevent helper interposition between signed per-weight-half
and unsigned group-affine implementations. No NVIDIA runtime is linked.

## Run on box

From the `quactlize` develop checkout, use a subshell so failures never exit
the interactive Docker shell:

```bash
(
  test "$(git branch --show-current)" = develop &&
  GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin develop &&
  git lfs pull --include="prebuilt/ppu0010/q4-simt-ab-v1/*.so,prebuilt/ppu0010/q4-h800-port-v1/*.so" --exclude="" &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
  bash tools/run_q4_h800_port_ppu_box.sh
)
```

No box compilation or JIT. Only run on an otherwise idle selected device.
The explicit 64 MiB L2 value is the supplied PPU capacity override; it is
recorded as an override when the SDK cannot report L2.

By default ACU collects nine forced-cold profiles: all three arms at
`512x2048`, `1024x5120`, and `5120x8192`, covering each implementation.
ACU times are not used as warm/rotating benchmark times. Use `ACU=0` to skip
profiling; the complete event-timed comparison still runs.

Successful child cells are bound to logs, payloads, runtime, device and
fixture identity. Failure does not discard other arms/shapes. To reuse only
the successful cells, set `RESUME_RUN` to the **exact existing run directory**
and rerun the script with the same inputs. For a fully fresh timing cohort,
omit `RESUME_RUN` (for example, after unrelated GPU activity).

The script prints `results=...results.tgz` and `summary=.../summary.tsv`.
The archive contains timings, raw logs, failed-case evidence and ACU reports.
Send that archive back; it is not necessary to copy fixtures or libraries.

## Rebuild locally, if needed

```bash
python dev/gemv_ppu/build_h800_port.py \
  --sdk /root/ppu-sdk/2.1.1 \
  --output /root/autodl-tmp/q4-h800-port-new-build \
  --jobs 5
```

Use a new output directory. Generation keeps each measured kernel body intact;
changes are native runtime spelling, exact shape/recipe admission, and a
small C launch wrapper. `tests/test_q4_h800_port_ppu.py` checks that boundary
and rejects mislabeled, incomplete, nonfinite and producer-only receipts.
