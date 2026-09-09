# RTX 5090 SIMT and grouped post-operation experiments

Development-only CUDA diagnostics. This directory must not enter a PPU
production package. It compiles the production SIMT GEMV and reduction bodies
against real CUDA headers/runtime; it does not emulate PPU MMA or AIU opcodes.
The compatibility headers rename APIs, not arithmetic or synchronization.

## Reproduce

Requires an RTX 5090, CUDA 12.8, PyTorch with CUDA support, NumPy and `gguf`.
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
