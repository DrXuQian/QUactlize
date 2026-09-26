# H800 reference comparison

This is an experimental prepared-kernel comparison, not a vLLM Python
benchmark or a production dependency. The goal is to beat **both** references
at each declared shape. Beating the original Quactlize baseline is not enough.
H800 evidence does not admit native PPU1.7 execution.

The device implementations are unmodified vLLM Machete and Marlin, pinned to
`11ba93f3646d4c5476c3b3fd56835589701f0fb1`. Their CUTLASS dependency is v4.4.2,
`da5e086dab31d63815acafdac9a9c5893b1c69e2`. Do not mix those headers with
the Quactlize PPU CUTLASS3.6.0 include graph. Only the host Tensor wrappers are
replaced by a small diagnostic C boundary. The strict host-check shim is for
Marlin's scalar type descriptor, not a disabled-check build.

## Compared work

- Dense W4A16: M1/8/128, N512/K2048 and N4096/K4096, FP16 and BF16 A/C,
  FP32 accumulation and FP32 Marlin cross-CTA reduction.
- Nine unique Machete default-heuristic schedules and all three upstream
  small/large Marlin launch geometries. Reference leaders are measured,
  never selected to make a candidate look faster.
- Machete supports group64/128, not arbitrary GGUF Q4 group32. The common
  fixture uses logical group64; canonical K-pack repeats each affine pair
  into two group32 entries. This is a common subset, **not** a claim that
  Machete consumes general GGUF group32 or canonical K-pack bytes.
- All16 codes, nonconstant scale/min, exact dyadic typed weights. A complete
  CPU FP64 dot of those typed weights checks every output, including guards
  and nonfinite values. Machete's zero is additive after scaling; Marlin's
  floating zero is a code-domain value multiplied by scale inside the kernel.
- Allocation, offline packing and plan creation are excluded. Required
  per-call initialization and reduction are included. Sequential plans share
  one scratch allocation, as a resident caller can; weights rotate across a
  full ring exceeding2.25x the queried L2 capacity. The helper does not create
  an independent cold scratch allocation for every weight copy.
- Graph capture and upload are excluded. Each configuration has>=250ms of
  device warmup,11 event samples, matching before/after SM/memory clocks and
  a live GPU process guard. Other GPU workloads invalidate the run; they are
  never terminated by this test. Clocks and driver settings are not modified.

The separate `cached-decode` numerical gate uses genuine different group32
metadata, N257 tails, dense M1..8, empty/ragged experts and all three storage
types. Common-fixture reference parity alone is not the production numeric
gate. GEMV and grouped GEMV performance admission are separate.

## Build

Use clean pinned source checkouts and an external fresh output directory:

```bash
python3 tests/ppu17/reference/build.py \
  --vllm /path/to/pinned-vllm \
  --cutlass /path/to/pinned-reference-cutlass \
  --compiler /usr/local/cuda-12.8/bin/nvcc \
  --output /workspace/q17-reference-build
```

No vLLM or Torch development package is linked. The CUDA driver stub is used
only at link time; the actual installed driver must provide tensor-map APIs
at execution. NumPy and Torch are required by the Python fixture/runner.

Run CPU-only fixture checks, optionally comparing Marlin packing byte for
byte with the pinned upstream Python functions:

```bash
python3 tests/ppu17/reference/test_fixture.py \
  --vllm /path/to/pinned-vllm --references /workspace/q17-reference-build
```

## Execute on an idle physical H800

`BUNDLE` is a separately built, validated PPU1.7 CUDA-source bundle. The native
PPU compiler is not used for this reference comparison.

```bash
python3 tests/ppu17/reference/cohort.py \
  --references /workspace/q17-reference-build \
  --bundle "$BUNDLE" \
  --backend-source quactlize/backends/ppu17 \
  --expected-uuid "$GPU_UUID" \
  --output /workspace/q17-reference-screen
```

This screens12 points and records every configuration. `compare.py` runs
one point; `--configs arm:index ...` is only for a declared bounded challenge.
`reconfirm.py --previous SCREEN --rounds 3` retains the two measured leading
configurations of each reference and challenges decode with the previous
Quactlize leader and the two16-warp candidates. It accepts the same bundle,
backend-source and device arguments, and requires a fresh output directory.

Do not use this test to claim group32 reference support, full-model latency,
MoE speed, native PPU1.7 readiness or an improvement on unmeasured shapes.
