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
