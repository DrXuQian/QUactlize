# Q4 dense versus grouped decode reproduction

Status: prebuilt-only runner; new PPU measurements are pending. No kernel,
offline format, production selector or native library is changed by this test.
ScaleFirst is intentionally excluded from this comparison.

## Equal logical work

The fixture selects eight distinct experts from E256 at N512/K2048. Each
active expert consumes the same FP16 activation row. Their Q4 code and packed
metadata planes are concatenated along **N within each plane** to construct
a canonical dense N4096/K2048 weight. Concatenating flattened expert buffers
would be wrong; local tests compare the result against the independent dense
reference packer and reject that incorrect concatenation.

Both sides therefore use exactly 4,718,592 active weight bytes (4.5 MiB),
the same logical weights, the same activation and the same 4,096 outputs.
Preparation and H2D are outside timing. This is synthetic model-shaped data,
not a tensor dump from a live llama.cpp request. All output values and every
FP32 Split-K plane are checked against official GGUF arithmetic. Reduction
is also checked against the downloaded partials in increasing split order.

The producer schedules differ: dense fixed Split-K owns contiguous K-tile
ranges, while grouped owns every S-th K tile. Each route therefore has its
own independently computed partial oracle. They coincide when each split
owns one K tile (S8 here), but not for S2/S4. The initial comparison script
incorrectly reused the grouped partial oracle for dense; that test-only
error is corrected without changing a kernel or relaxing numerical checks.

## Five bounded arms

| Arm | Geometry / provider | Split | Timed work |
| --- | --- | ---: | --- |
| dense-historical-s8 | TM8/TN128/TK256, WM8/WN32, stages2, AP1/DN64 | 8 | dense producer + reducer |
| dense-matched-s2 | TM8/TN64/TK256, WM8/WN16, stages2, AP0/DN64 | 2 | dense producer + reducer |
| dense-matched-s4 | same as preceding row | 4 | dense producer + reducer |
| grouped-compact-s2 | TM8/TN64/TK256, WM8/WN16, stages2, AP0/DN64 | 2 | GPU metadata + directory + grouped producer + reducer |
| grouped-compact-s4 | same as preceding row | 4 | GPU metadata + directory + grouped producer + reducer |

The historical dense configuration measured 11.12 us in the reviewed
overnight archive. This runner reuses its published native module; it does
not claim byte identity with the original overnight executable or guarantee
the old time on a different fixture/run.

"Matched" means tile/provider geometry, **not identical collective types**:
dense and grouped retain their actual metadata publication and epilogue
implementations. Grouped includes its two preparation kernels. Timing
also reflects contiguous versus interleaved K scheduling. Thus
differences are not automatically attributed to the expert lookup or router.
ACU provides the per-kernel breakdown.

Four alternating-order rounds each collect 11 event samples; each sample
replays 16 complete calls and is divided by 16. Correctness/initialization
and warmup are not timed. Results are warm fixed-weight effective bandwidth,
not measured DRAM utilization. MBU uses 4.5 MiB and the existing 2766 GB/s
nominal denominator, without crediting extra partial/reduction traffic.
Workspace guards preserve the 128-byte alignment required by dense's
existing M1 fast reducer; the guard must not silently select the fallback.

## Box

From the development checkout with one idle PPU:

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_kpack_dense_grouped_ab_box.sh
```

This compiles nothing. The three module payloads already exist in
`prebuilt/ppu0010/kpack-native-v1` and `kpack-gpu-compact-v1`. Only those
explicitly selected files are verified/loaded; missing modules do not
silently fall back or trigger JIT.

After timing, the default runner collects three native ACU reports:
`dense-historical-s8.acurep`, `grouped-compact-s2.acurep`, and
`grouped-compact-s4.acurep`. Correctness precedes capture and is checked again
after replay. Profiled durations are never mixed into the unprofiled table.
`--skip-acu` requests just the five timing arms. ACU targets the individual
nodes of one warmed graph, not the entire model.

Each arm prints progress. Ordinary per-arm exceptions preserve other timing
results and mark the run failed; a device/process crash is not masked as an
admitted result. Every run uses a fresh directory and preserves the Docker
shell. Return the printed `q4-dense-grouped.*.results.tgz`, which contains
`summary.tsv`, `timing.json`, module/source/device receipts, logs and reports.
