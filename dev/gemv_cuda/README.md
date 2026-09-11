# RTX 5090 SIMT and grouped post-operation experiments

Development-only CUDA diagnostics. This directory must not enter a PPU
production package. It compiles the production SIMT GEMV and reduction bodies
against real CUDA headers/runtime; it does not emulate PPU MMA or AIU opcodes.
The compatibility headers rename APIs, not arithmetic or synchronization.

The [2026-09-11 NCU follow-up](#ncu-guided-reader-follow-up-2026-09-11)
below supersedes the old "no counter evidence" limitation for the RTX 5070.
PPU and whole-model admission are still separate.

The subsequent [matched Q4 Xplane comparison](../../docs/Q4_SIMT_XPLANE_KPACK_5090_AB.md)
finds that the historical optimized Xplane SIMT reader remains substantially
faster than N2. Equal-shape, >L2 rotations give 4.880 vs 9.441 us at K2048
and 7.819 vs 14.807 us at K4096 (M1/N4096). These are independently tuned
complete calls, not a layout-only or PPU verdict.

The [matched NCU follow-up](../../docs/Q4_SIMT_XPLANE_KPACK_NCU.md) on 5070
finds 80.14% vs 43.98% DRAM throughput for rotating M1/N4096/K4096. N2's
76.38% SM throughput is ALU-heavy activity, not FMA utilization; it executes
5.78x the warp instructions. Recipes remain fixed to the 5090 winners, with
no production admission or claim of 5070-optimal tuning.

The [Q4 native-word follow-up](../../docs/Q4_KPACK_NATIVE_READER.md) keeps
the same K-pack bytes, N2 ownership and FP32 dot order. On 5070 it reduces
rotating M1/N4096/K4096 from 33.318 to 19.528 us and producer instructions
from 11.66M to 6.13M; request-side excessive sectors fall from 68% to below
1%. This is `--reader cuda-q4-n2`, not a production or PPU promotion.

## Reproduce

Requires an sm_120 NVIDIA GPU (tested on RTX 5090/5070), CUDA 12.8,
PyTorch with CUDA support, NumPy and `gguf` for the full Python gates.
The standalone profiler below only needs CUDA on its execution machine.
Use fresh output directories and do not run simultaneous GPU benchmarks.

```bash
python3 dev/gemv_cuda/prepare_fixtures.py --output /workspace/gemv-cuda-fixtures
python3 dev/gemv_cuda/build.py --cuda /usr/local/cuda-12.8 \
  --output /workspace/gemv-cuda-build --jobs 8
OPENBLAS_NUM_THREADS=1 python3 dev/gemv_cuda/bench.py \
  --build /workspace/gemv-cuda-build --fixtures /workspace/gemv-cuda-fixtures \
  --output /workspace/gemv-cuda-results.json
```

The manifest binds compiler, library and C++ source hashes. Fixtures use
official GGUF dequantization and an independent FP64 dot product. GEMV runs
256 configurations over five formats and three larger workloads. Checks
include zero-code negatives, nontrivial expert IDs, per-slot inputs, workspace
guards and fixed-order reduction of downloaded FP32 partials. Graph timings
include the GEMV reducer and exclude fixture transfers and host checking.

The direct-store test executes the actual grouped epilogue with a PPU
C-coordinate projection. Real hgcc separately proves that its C layout and
destination strides equal the production runtime types. This validates store
ownership, not PPU mainloop execution. Its 216 cells cover empty experts, later
M tiles, N tails and a wrong-split-pointer negative.

The reducer test compares the generic implementation, fixed-S vector bodies
and actual compact dispatcher. It checks cancellation-sensitive addition
order, +16/+128-byte workspace offsets, tails, strided output and weaker
output alignment. There are 120 records, including explicit rejection of
unsupported manual vector arms; the dispatcher must use its generic fallback
there.

## Measured development evidence, 2026-09-09

GPU: RTX 5090, 170 SMs; PyTorch 2.8.0+cu128. Final production-reader build took
9.42 seconds. All 256 GEMV cells and 216 direct-store cells passed; reducer
checks passed, including fallback cases. Eleven CUDA-event samples per cell;
GEMV samples contain 32 graph calls. These are warm, cache-resident timings,
not a DRAM bandwidth measurement or a PPU performance verdict.

For contiguous FP32 partials with M=8, N=512 and workspace base+16 bytes:

| Split | Generic reducer (µs) | Compact dispatcher (µs) |
| --- | ---: | ---: |
| 2 | 1.316 | 0.874 |
| 4 | 1.681 | 0.874 |
| 8 | 2.381 | 0.906 |

Final SIMT best measured cells (no gather/scatter):

| Workload | Reader / columns / warps / split | µs |
| --- | --- | ---: |
| Q4 dense 1×4096×2048 | pair / 16 / 4 / 8 | 10.648 |
| Q4 indexed 8×512×2048, E=256 | pair / 16 / 4 / 8 | 11.088 |
| Q5 indexed 8×2048×512, E=256 | pair / 16 / 8 / 1 | 12.931 |

The two Q4 fixtures have equal weight counts, not identical concatenated
weight bytes. Use the separate PPU dense/grouped A/B for that causal test.

An optional `build.py --reader cuda-half2` uses real CUDA `__hfma2` for paired
dequantization. In an earlier paired experiment, the Q4 indexed cell changed
11.150→10.715 µs and Q5 indexed 12.935→12.684 µs. Both full correctness suites
passed. This replaces CUDA's scalar fallback: the PPU reader already uses
native f16x2 instructions, so it is not an additional PPU optimization.
The SIMT performance investigation remains open.

## Matched llama.cpp comparison

The non-Q8-A control is the historical Q4_K/Q5_K **DMMV** dot body from
`467576b6cc7d2b9220f55bc635aa51469cf26fb5`, before its removal in `c3ea58aca`.
It is not a claim that current llama.cpp still selects DMMV. A thin device
wrapper rebases expert and input pointers from the same GPU IDs; the dot
body is not rewritten. It uses the current checkout's support headers.
The second control extracts current MMVQ and `quantize_q8_1` from checkout
`a0f0576b7741ede314d74ea08e84fb1122e78223`, including its small-K decision.
No attention, fusion, gather/scatter or CPU expert routing enters either arm.

Each comparison starts from the same raw GGUF weights and F32 activation
bytes. The K-pack arm reads their canonical offline reorder. Fixtures use
eight noncontiguous expert IDs, broadcast A for Q4 and per-slot A for Q5.
The comparison scans 24 K-pack pair recipes, then measures six alternating
arm-order rounds with eleven samples of 32 graph calls each. MMVQ includes
its online Q8_1 quantization; K-pack includes its Split-K reducer. All results
must pass the same official-GGUF FP64 dot oracle at conditioned error <0.005.
The A values are FP16-representable; this is not a wide-range F32 accuracy test.

Unchanged production reader on the same RTX 5090 (warm graph µs):

Here "production reader" names the existing source body, not an admitted
SIMT default: the current model recipe gate has selected no GEMV recipes.

| Workload | K-pack pair | DMMV, no Q8-A | MMVQ + Q8_1 |
| --- | ---: | ---: | ---: |
| Q4 indexed 8×512×2048 | 10.866 | 5.998 | 4.023 |
| Q5 indexed 8×2048×512 | 12.732 | 11.058 | 6.811 |
| Q4 dense 1×4096×2048 | 10.384 | 6.185 | 4.136 |

Thus the difference is not explained solely by the reference quantizing A.
This CUDA production-reader build uses the scalar fallback for PPU f16x2
assembly; do not translate these ratios into PPU performance predictions.

### Development optimization, not a production replacement

`--reader cuda-affine --schedule cuda-grid` keeps the offline planes but
accumulates `code*A` and `sum(A)` in FP32, applying the group's FP32 metadata
products afterwards. Four independent accumulators and aligned float4 A
loads shorten the serial instruction chain; weaker alignment and F16 inputs
use scalar loads. This changes intermediate rounding, not stored weight bits.
The optional grid experiment maps `(N tile, split, row)` directly to CUDA
grid dimensions and uses a row-aware reducer. Rows above 65,535 are explicitly
declined in this development-only schedule; production retains its flat grid.

Final v4 paired comparison, separate from the original-reader run above:

| Workload | Experimental K-pack | DMMV, no Q8-A | MMVQ + Q8_1 |
| --- | ---: | ---: | ---: |
| Q4 indexed 8×512×2048 | 8.960 | 6.025 | 4.108 |
| Q5 indexed 8×2048×512 | 11.167 | 11.078 | 6.779 |
| Q4 dense 1×4096×2048 | 8.127 | 6.016 | 4.129 |

Q5 is within 1% of DMMV in this run; Q4 is still 49%/35% slower, and all
three remain slower than current MMVQ. The target is **not met**. Removing
grid division/reducer indexing alone helped Q4 about 4%, not enough to
explain the gap. Single-call CUDA profiling of v3 showed Q4 producer/reducer
about 8.03/1.09 µs, Q5 producer 11.30 µs; these are diagnostic capture times,
not warm graph timings or hardware-counter evidence. The producer remains
the larger component. No DRAM bandwidth/stall root cause is asserted.

The final v4 regression passes 256 cells across five formats (192 new pair
and 64 unchanged scalar), plus 216 direct-store ownership cells. Maximum
new-pair conditioned error is 4.56e-8. Nine extra input controls cover F32
base+4, F16 base and F16 base+2; output/workspace guards, nontrivial expert
IDs, wrong-expert negatives, zero-code negatives and ordered FP32 reduction
are checked. This does not establish PPU numerical or performance admission.

Reproduction (fresh directories, no concurrent GPU benchmarks):

```bash
python3 dev/gemv_cuda/build_llama_reference.py --llama /path/to/llama.cpp \
  --output /workspace/llama-gemv-reference
python3 dev/gemv_cuda/compare_llama.py --fixtures /workspace/gemv-cuda-fixtures \
  --kpack-build /workspace/gemv-cuda-build \
  --reference-build /workspace/llama-gemv-reference \
  --output /workspace/gemv-llama-production.json
python3 dev/gemv_cuda/build.py --output /workspace/gemv-cuda-affine --jobs 8 \
  --reader cuda-affine --schedule cuda-grid
python3 dev/gemv_cuda/compare_llama.py --fixtures /workspace/gemv-cuda-fixtures \
  --kpack-build /workspace/gemv-cuda-affine \
  --reference-build /workspace/llama-gemv-reference \
  --output /workspace/gemv-llama-affine.json
```

Regenerate fixtures with this revision to include the original `raw` bytes.
The reference builder requires the historical DMMV revision in local git.
Its manifest records exact extracted fragments, current headers, compiler
and library hashes; it does not build or claim to benchmark an entire model.
`--profile-kernels` adds separate diagnostic events, not performance samples.

Evidence SHA256 (raw JSON under `/root/autodl-tmp/kpack-llama-comparison*`):

- Original-reader comparison:
  `580417acb946ae0db1f6921e7cd83752fb78872b15147c7c6415f93f5af350de`.
- Final v4 comparison:
  `96c6d4248cf05480ec5046fe4341e0f8928c85cef23d023ace8f6016f0da9f8f`.
- Five-format v4 gate (`kpack-gemv-affine-v4.json`):
  `595321503b204779e3c30815073857599d3a623d653e1669dda8a6fc0482a59c`.
- Experimental CUDA library:
  `51b7a0e2fc7ae24f0cbb8ded77c56fc10e2327bca01bf130a6b14ffe6957d013`.
- llama.cpp reference CUDA library:
  `4d5424719640640d177a94c0f59c58110527c5e0c8b919eb7a19b1da286e7cf6`.

Nothing in this experiment is packaged into the PPU native bundle. The next
SIMT step is producer/metadata-load optimization against these same two
controls; any change to production arithmetic first needs PPU validation.

Evidence SHA256:

- Final `result-v12.json`:
  `4932c49fe8353f2693a044d443dd80098d5a5153448c91ea6964c92ddc0e8ea9`.
- Its CUDA library:
  `a966a90207a1c2cc1d300df8bb93ab35ac92f6a537905719163f7445e9c8e137`.
- Paired production-reader `result-v11-base.json`:
  `ca00e24136956e8a57560cbcc330439dde0bf77fa0ac6e5d3992a444bb42c55a`.
- Paired native-half2 `result-v11-half2.json`:
  `74e7eab47a2d002d5d1a23e8f68abc9fe582598ed9b611f2c3f304129e1df6ce`.

Raw results remain in the development artifact directory
`/root/autodl-tmp/kpack-postops-cuda-evidence-v11/`. NVIDIA hardware-counter
profiling was denied with `ERR_NVGPUCTRPERM`; no counter-based bandwidth or
stall diagnosis is claimed. Driver permissions were not changed.

## Fixture stream-order check

`stream_poison.cu` is a separate six-cell CUDA diagnostic, not part of GEMV
timing. It delays the default stream and then compares an unordered device
memset with same-stream memset and an explicit default-stream drain, for
both eager and graph consumers. On the same RTX 5090 both unordered cells
returned four `0xa5a5a5a5` words (`EXPECTED_RED`); all four ordered cells
passed. This demonstrates a test-harness race mechanism, not PPU admission.

```bash
/usr/local/cuda-12.8/bin/nvcc -std=c++17 -arch=sm_120 -O2 \
  dev/gemv_cuda/stream_poison.cu -o /workspace/kpack-stream-poison
/workspace/kpack-stream-poison
```

## NCU-guided reader follow-up, 2026-09-11

The new `--reader cuda-n2` is a development candidate, not a replacement of
the production SIMT entry or a model-policy promotion. It preserves the
canonical K-pack bytes and metadata, converts A to FP16, computes the pair
reader's fused FP16 affine weights, and accumulates in FP32. Unlike the
older `cuda-affine` experiment, it does not move the affine transform after
the group dot. Its FP32 addition order changes, and its one-rounding pair
affine is not the scalar reader's two-rounding affine.

Each thread now owns two adjacent N columns. A contiguous b32 load serves
both b16 code words, a half2 affine serves both weights, and the same A value
feeds both dots. Aligned float4 A loads and four accumulator chains reduce
instruction overhead. F16/unaligned A and b16-but-not-b32 weight pointers
retain in-bounds scalar loads. No offline conversion, new scale workspace,
gather/scatter, or activation quantization is added.

`columns` in this candidate counts thread positions along N, **not** final
output values: 16 positions deliver 32 columns. Its manifest records this
factor; the benchmark reports the actual grid. Do not export these recipes
to a production reader with different ownership. Public scalar Q2-Q6 entry
bodies are unchanged. Q8's already-paired entry uses the experimental reader
only inside this development build.

### Complete-call measurements

Four alternating-order rounds, 15 samples per round, 32 graph calls per
sample; five first graph replays are excluded. CUDA events include producer
and Split-K reducer, with F32 input/output and no gather/scatter. Weights are
warm/cache-resident; this is not DRAM MBU. No clocks or driver permissions
were changed. The baseline is the existing **pair experimental entry**, not
a claim that the model previously selected SIMT. Its scalar Q4 control was
separately reproduced at about 14.4 us on 5090.

| Workload | 5090 pair → N2 (us) | Change | 5070 pair → N2 (us) | Change |
| --- | ---: | ---: | ---: | ---: |
| Q4 indexed 8×512×2048, E=256 | 10.790 → 7.613 | -29.45% | 29.689 → 18.365 | -38.14% |
| Q5 indexed 8×2048×512, E=256 | 12.708 → 9.132 | -28.14% | 37.195 → 21.563 | -42.03% |
| Q4 dense 1×4096×2048 | 10.258 → 7.309 | -28.76% | not measured | — |

The 5070 uses fixed 5090 recipes, not a per-device optimum. A separate
24-recipe screen and six alternating rounds on 5090 selected N2
16/4/S8 for Q4 and 16/4/S1 for Q5. The earlier pair control uses 16/4/S8
for Q4 and 16/8/S1 for Q5. Do not pool the six-round and four-round datasets.

In the four-round matched 5090 run, historical DMMV takes 5.773/10.828/5.691
us and MMVQ plus Q8_1 takes 4.040/6.663/4.039 us, in table order. N2 now
beats DMMV on Q5, but **Q4 remains slower than DMMV and both remain slower
than MMVQ**. Those controls use different intermediate arithmetic, as
described above. The no-slower-than-native target remains open.

### What NCU established

NCU 2025.1.1 counters work on RTX5070 WSL. Baseline Q4 scalar shows about
79% SM throughput, 31% of warp issue latency waiting for a math pipeline,
and less than 1% DRAM activity in this warm-cache capture. It is not evidence
of a saturated DRAM channel. Reducing instruction work is productive here.

| Producer | Executed warp instructions, pair → N2 | Registers/thread | Achieved occupancy |
| --- | ---: | ---: | ---: |
| Q4 indexed | 10,762,240 → 6,096,896 | 47 → 64 | 74.09% → 56.21% |
| Q5 indexed | 13,473,792 → 7,211,008 | 54 → 86 | 61.03% → 35.10% |
| Q8 dense 1×8192×2048 | 18,935,808 → 14,413,824 | 43 → 52 | 77.59% → 67.62% |

The kernels become faster despite lower occupancy. Register count and
occupancy alone are not a latency verdict. Q4/Q8 keep the same separate
reducer body; its measured instruction count is unchanged. The intermediate
`cuda-vector --schedule cuda-grid` ablation reduces Q4 producer instructions
to 9,091,072; N2 removes substantially more repeated A/decode work.
NCU replay durations are diagnostics, not the unprofiled table's timings.

### Q8: scan S8 before deciding

Q8 remains W8A16, using original resident FP16 d, with no Q8_1 A conversion.
The old shipping-domain check covered only S1/S4. This run additionally
scanned the existing pair domain: columns 16/32, warps 2/4/8, S1/2/4/8.
All 360 cells per arm pass across 15 dense/grouped/indexed contexts.
Fixed-recipe four-round confirmations on 5090 are:

| Q8 workload | Pair → N2 (us) | Decision |
| --- | ---: | --- |
| Dense M1, N8192, K2048, 16/4/S8 | 17.843 → 13.962 | -21.75%, candidate benefit |
| Dense M1, N1024, K5120, 16/8/S8 | 8.459 → 8.459 | effectively tied |
| Indexed 8 rows, E4, N256, K512, 16/8/S1 | 4.106 → 4.342 | +5.75%, retain old reader |

This is not a Q8-vs-native-llama speed verdict or a reason for one global
reader default. The production selector and its admitted recipe domain are
unchanged.

### Correctness and reproduction

Final N2 passes 256 existing K-quant cells, plus 360 new input controls over
Q2/Q3/Q4/Q5/Q6: 1/2/3/4 tokens, top8 with broadcast/per-slot inputs, dense
M1/M2/M4, ragged grouped rows including empty experts, F32/F16, nontrivial
row strides, A offsets +4/+2 and weight-plane offsets +2. Output/workspace
guards, three graph replays, ordered downloaded-partial reduction, and
wrong-expert negatives are checked. Maximum conditioned error is 1.85e-4
against the 0.005 bound. The existing direct-store projection's 216 cells
also pass; they are not PPU MMA execution.

The first N2 Q5 attempt incorrectly duplicated the high word for adjacent
columns. The actual map shares columns separated by 8, not 1. The host
`n2_layout_check.cpp` enumerates seven actual offline maps (114,688 codes),
proves aligned adjacent word addresses for even/odd N, and reproduces 8,192
errors with the duplicated-word negative. The corrected GPU reader passes
the independent oracle without changing packing or relaxing its threshold.

```bash
python3 dev/gemv_cuda/build.py --cuda /usr/local/cuda-12.8 \
  --output /work/gemv-base --jobs 8
python3 dev/gemv_cuda/build.py --cuda /usr/local/cuda-12.8 \
  --output /work/gemv-n2 --jobs 8 --reader cuda-n2
python3 dev/gemv_cuda/prepare_fixtures.py --input-controls \
  --output /work/gemv-input-fixtures
OPENBLAS_NUM_THREADS=1 python3 dev/gemv_cuda/check_inputs.py \
  --library /work/gemv-n2/libkpack_gemv_cuda.so \
  --fixtures /work/gemv-input-fixtures --output /work/gemv-inputs.json
```

The CUDA-only `standalone.cu` and `run_standalone.py` require neither PyTorch
nor NumPy on the profiling machine. Export the existing independent fixtures
on the development host, copy their `.bin` files, the runner and CUDA DSO:

```bash
python3 dev/gemv_cuda/export_standalone.py --fixtures /work/gemv-fixtures \
  --output /work/gemv-binary-fixtures
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 -I. -Iquactlize/include \
  dev/gemv_cuda/standalone.cu -ldl -o /work/gemv-standalone
python3 dev/gemv_cuda/run_standalone.py --runner /work/gemv-standalone \
  --fixtures /work/gemv-binary-fixtures \
  --baseline /work/gemv-base/libkpack_gemv_cuda.so \
  --candidate /work/gemv-n2/libkpack_gemv_cuda.so --output /work/gemv-ab
ncu --kernel-name regex:kpack_gemv --launch-skip 10 --launch-count 2 \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats --section SourceCounters \
  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --cache-control none --clock-control none --export /work/q4-n2 \
  /work/gemv-standalone /work/gemv-binary-fixtures/q12-n512-k2048-e256-c1.bin \
  /work/gemv-n2/libkpack_gemv_cuda.so pair 16 4 8 --profile
```

For S1 use `--launch-skip 5 --launch-count 1`. NCU outputs must use fresh
names. The profiling mode checks a positive output, then launches five warm
calls; it does not time wrong-expert/zero-code negative arms. Normal timing
also requires those negatives to be detected.

Q8's standalone test can scan all 24 pair recipes or run one fixed case:

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 -I. -Iquactlize/include \
  dev/gemv_cuda/q8_check.cu /work/gemv-n2/q8.o -o /work/q8-check
/work/q8-check --pair-sweep
/work/q8-check --pair-case 8192 2048 1 0 1 16 4 8
```

[Summary and counter identities](../../docs/measurements/gemv_ncu_20260911.json),
[raw samples, CSV counters and source-bound manifests](../../docs/measurements/gemv_ncu_20260911.logs.tgz).
Full NCU reports remain at `E:/kpack-gemv-ncu.7YMu6i` on the 5070 host and
`/root/autodl-tmp/kpack-gemv-ncu-evidence-20260911` locally; their hashes are
in the summary. No NVIDIA DSO, recipe or compatibility code was added to the
PPU bundle. Next: portable PPU implementation, bounded numerical/ACU gate,
and shape-specific full-call admission before changing model selection.
