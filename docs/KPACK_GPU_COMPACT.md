# GPU compact grouped execution

Status: the bounded PPU gate passed 204/204 cells. Sixteen new modules compile
with SDK 2.1.1. Production policy and the deployed native bundle are unchanged;
this package is not yet a model-performance claim.

Reviewed archive: `kpack-compact.yAQMNP.results.tgz`, SHA-256
`621cf9ed0f7f956768be0e7938f408517383df9b4f0691960a8d2ebe7318996b`.
The source/kernel manifest and raw per-job JSON logs agree. Eleven jobs took
99.14 seconds; each cell has 3x11 timing samples, 16 calls per graph. There are
200 GEMM cells (145 with FP32 partials), 3,472 GEMM correctness checks, and four
unchanged SIMT controls. Maximum conditioned output/partial errors are
0.000153743 / 0.000236041 against the 0.005 bound. Reducer identity and guards
are exact checks; general GEMM equality with official GGUF is tolerance-based.

Same-parent TM16 model anchors, E256/top8/one token:

| Case | Old device S1 | GPU compact choice | Delta |
| --- | ---: | ---: | ---: |
| Q4 N512/K2048 | 19.3375 us | S2, 17.9400 us | -7.23% |
| Q5 N2048/K512 | 26.8000 us | S1, 14.7000 us | -45.15% |

Q4 compact S1 is 20.0600 us, so compact alone is not a universal improvement.
Persistent passes but does not beat ordinary compact on these two anchors.
Q5 TM8 reaches 14.6500 us, only 0.34% below TM16. The remaining same-split
host-compact differences are 3.635/3.550 us; these do not isolate metadata
cost from the changed producer schedule. No automatic production promotion.

For the requested single-token decode profiling, use **TM8/WM8**, not TM16.
The TM16 numbers above remain historical controls. The already-admitted TM8
module measures Q4 S2 at 18.1300 us and Q5 S1 at 14.6500 us. TM8 Q4 S4 measures
18.0475 us, within 0.5% of S2; the focused capture includes both S2 and S4.
They use the same module, with a runtime split selection. Q4 S4 doubles the
producer grid from 128 to 256 CTAs (128 threads each) and doubles the partial
workspace; a stable speedup is not established by this near tie. This is a
bounded profiling choice, not a global selector update.
Each active expert has M=1, so TM8 and TM16 both need one M tile per expert:
changing TM alone does not increase the 128/256 CTA counts in these calls.

The `kpack-decode.XZM60u` run passed 260/260 cells in 178.8 seconds. The same
Q4 TM16/TN64/TK256 parent measured 19.410 us on device-only S1 versus 16.035 us
on host-compact S1 and 14.2425 us on host-compact S2. Q5 measured 26.7925 us,
11.240 us and 19.365 us respectively. Host preparation was excluded; these
numbers do not promise deployable latency or isolate the metadata kernel alone.

## Implementation contract

- Keep grouped v2 ABI, device offsets, concatenated activations and canonical
  weight bytes. No per-run allocation, D2H, host wait or CPU routing.
- Reuse the existing device M-tile directory. Ordinary GPU execution launches
  a bounded one-task-per-CTA grid; persistent execution walks the same tasks.
- Extend the shared task identity with a K slice. Reads still select the real
  expert; FP32 outputs select `expert + slice * experts`. Reduction stays
  ordered and runs after every producer on the same stream.
- Preserve the existing host-prepared ordinary path and large-expert fallback.
  Persistent and compact directory builders currently support at most 1024
  experts. This is an explicit implementation bound, not a new offline format.

## Counts and performance guard

Before: ordinary device-only runs one metadata kernel, a rectangular producer
grid and, for S>1, one reducer. Persistent S1 additionally builds a directory.
The pair SIMT reader is unchanged by this work.

After: compact execution adds the existing directory build, removes expert-
padded producer work, and keeps the metadata and reducer launches. The data
mainloop, copy/converter types, intra-task barriers and arithmetic order are
unchanged for S1. At S>1 each task walks K tiles `s,s+S,...`; total logical MMA
work is unchanged, but prologues/epilogues increase and one reducer is required.
Persistent work transitions keep the existing shared-lifetime CTA barrier.

Tests must cover missing/duplicate/slice-rotated task negatives, safe row-count
bounds, changed device offsets in graph replay, every FP32 partial, reducer
identity and output/workspace guards. Compare both new schedules against the
previous immutable module in the same run. No timing is admitted from a wrong
result. GPU compact is not assumed faster until its directory cost is included.

## Run the prebuilt experiment

The package has 16 new modules, four immutable ordinary baseline modules and
the unchanged five-format SIMT library. Eleven isolated jobs check 204 cells:
five-format FQ controls; Q4/Q5 TM16 and TM8 model-shaped comparisons; Q4/Q5
resident-SF controls; and the previous best scalar/pair SIMT controls. Each
cell checks output/partial guards and every FP32 K slice before timing. Device
offset profiles change during graph replay, including the actual M-tile count.
Unused directory capacity must remain poisoned. SF controls exclude preparation
and are not billed as first-use SF timing.

Run on one idle PPU from the development checkout; no box compilation:

```bash
git pull --ff-only
git lfs pull --include='prebuilt/ppu0010/kpack-gpu-compact-v1/**' --exclude=''
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_kpack_gpu_compact_box.sh
```

Return the printed `/workspace/kpack-compact.*.results.tgz`. A failed job does
not discard other jobs. On the same source/SDK/device and parameters, set
`RESUME_RUN` to that run directory to reuse validated jobs and retry failures.
The expected final marker is `GPU_COMPACT_DONE status=PASS cells=204/204`.
Numerical admission is separate from reviewing the timing deltas.

## Single-operator profiling: ACU by default

Use ACU for a standalone operator's instruction, cache, shared-bank and warp
metrics. Use Asys for model/stream timelines and host-launch bubbles. The
following entry writes five native reports without compiling any DSO:

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_kpack_gpu_compact_acu_box.sh
```

Open the reported directory's `q4-up-tm8-compact-s2.acurep`,
`q4-up-tm8-compact-s4.acurep` and `q5-down-tm8-compact-s1.acurep`; matching
`*-tm8-baseline-s1.acurep` files contain the old device-only TM8 path.
All five captures require TM8/WM8; a missing TM8
module is an error, not a fallback to TM16. `acu-index.tsv` records TileM and
actual filenames. The old TM16 captures remain valid historical evidence and
are not overwritten. ACU starts only
after numerical checks and warmup, profiles one graph's individual nodes, and
does not kill the target before its post-profile output/partial check. The Q4
compact report contains metadata, directory, GEMM producer and reducer; Q5
compact has no reducer. These are standalone .so calls, not llama.cpp traces.

To add only Q4 S4 while preserving previously collected reports:

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_kpack_gpu_compact_acu_box.sh --case q4-up --arm compact --split 4
```

The runner creates a fresh results directory and reports `reports=1/1` for
this filter. Omit `--split 4` to collect the paired Q4 compact S2/S4 reports.

SDK 2.1.1 ACU cache control `all` clears L1/L2/LLC. Even its `none` mode clears
L1/L2. Therefore ACU replay timings are not the warm resident benchmark above;
use these reports to diagnose, not to replace the unprofiled performance board.
Return the printed `kpack-compact-acu.*.results.tgz`, including native reports.

## Remaining GEMV question

The previous pair candidate improved Q4 25.0325 to 18.600 us and Q5 best
37.350 to 22.755 us. These are not DRAM saturation measurements. C16/W8 static
inspection reports zero stack bytes, but still 1950/2339 instructions for
Q4/Q5, with substantial integer address/bit operations and FP16/FP32 conversion.
Static counts include branches and are not dynamic instruction counts. Repeated
warm weights, metadata decoding and activation broadcasts need source/ISA and
device-counter separation before assigning a bandwidth root cause.
