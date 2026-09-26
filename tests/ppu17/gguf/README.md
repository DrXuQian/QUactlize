# GGUF Q4_K cooperative SM90 path

This is a private PPU1.7 experiment on PPU CUTLASS3.6. Existing PPU1.0/1.5,
canonical K-pack bytes, runtime selectors and the shipping q17 ABI are unchanged.
It extends the pinned Machete collective to **GGUF group32 affine metadata**,
not GPTQ group128 and not a re-quantization of GGUF weights.

Current scope: dense Q4_K, resident ScaleFirst FP16/BF16 A/S/Z/C, FP32
accumulation and Stream-K fixup. N must be divisible by128; K by256. Three
cooperative tiles `(weight-N, activation-M, K)` are128x16/64/128x64, Cluster1,
four pipeline stages. BF16 conversion is direct. Explicit pair multiply and
add retain two compute-type rounding points, including GGUF's converter bias.

## Data path and ownership

Original Q4_K codes are repacked losslessly into128N x64K microblocks. CuTe's
RS WGMMA ownership determines the thread/value permutation. A thread's32
nibbles occupy one contiguous aligned16-byte shared load. This private layout
is **not** identified as canonical `Q4_KPACK4_MAPPING_ID`; the benchmark uses
the separate `GGUF_Q4_K_G32_SF_PRIVATE_MMA_V1` contract. Scale/zero are separate
compute-type `[K/32,N]` arrays, not expanded floating weight matrices.

Each64K stage transfers two independent metadata rows. The physical shared
layout counts actual TMA bytes; a separate logical view broadcasts each row
only over its32 K values. Metadata loads occur once per two16K MMA atoms.
The prologue lookahead keeps the current stage: upstream's next-stage argument
was harmless only while metadata loaded exclusively at the first K atom.
`metadata.cu` proves thread/value ownership, group/stage mapping and bytes.

The adapted collective is attributed in `gguf_prepacked_mainloop.cuh`; its
Apache2 license is retained under `third_party/LICENSE.machete`. Pinned upstream
files are never edited. Explicit stages avoid the upstream auto-size helper's
one-metadata-row assumption. No barrier or retirement wait is removed.

## Build and replay

Use the same clean dependencies as the GPTQ comparison: PPU CUTLASS
`023e82d03e80b4d5982f664925e15499033df314`, vLLM
`11ba93f3646d4c5476c3b3fd56835589701f0fb1`. The latter supplies the pinned
packing/converter helpers; its CUTLASS4.4 reference tree is **not** included.

```bash
python3 tests/ppu17/gguf/build.py \
  --compiler /usr/local/cuda-12.8/bin/nvcc \
  --cutlass /path/to/ppu-cutlass360 --vllm /path/to/pinned-vllm \
  --output /workspace/gguf-q4-sf-build

python3 tests/ppu17/gguf/check_codegen.py \
  --bundle /workspace/gguf-q4-sf-build \
  --cuobjdump /usr/local/cuda-12.8/bin/cuobjdump \
  --output /workspace/gguf-q4-codegen.json

# Run each dtype separately; the controller refuses other GPU jobs.
python3 tests/ppu17/gguf/run.py \
  --bundle /workspace/gguf-q4-sf-build \
  --backend-source quactlize/backends/ppu17 \
  --expected-uuid GPU-REPLACE_WITH_PHYSICAL_UUID \
  --dtype 1 --output /workspace/gguf-q4-bf16-numerical
```

The runner requires PyTorch/NumPy only for fixtures and CUDA events. Its CPU
oracle decodes raw144-byte GGUF blocks independently, rounds each dequant step
in the requested type and computes the dot in FP64. It exercises one-hot
exact decoded weights, general inputs, tails, stage wrap, poisoned outputs,
guards, changed-input graph replay and a stale-group metadata negative.
The fixture is synthetic raw GGUF, **not a real model accuracy result**.

`--timing --m 128 --n 4096 --k 4096` times all three configs under a rotating
weight ring larger than2.25x the queried L2. Metadata, codes and every
Stream-K workspace reset/fixup are included. Allocation, descriptors,
packing, SF expansion and first graph upload are excluded. Eleven device
samples follow steady-state warmup; clocks and foreign processes are checked.
No measured SF GEMM time is labeled as FQ or SF end-to-end time.

`cohort.py` runs M1/8/128 x N512/K2048 and N4096/K4096 in both dtypes,
alternating arm order. `--baseline DIR` must identify the immutable initial
q17 bring-up library. It is an honest implementation control, **not** proof of
an advantage over optimized Machete/Marlin, nor a layout-only ablation. A clean
screen is not repeated confirmation or production winner admission.

For a bounded memory test, wrap `run.py --single --m 65 --n 256 --k 1024`
with `compute-sanitizer --target-processes all --tool memcheck
--error-exitcode 97 --check-bulk-copy yes --check-warpgroup-mma yes`.
The parent still guards the actual child GPU PID. Repeat for both dtypes and
use fresh result directories. Sanitizer results are never timing evidence.

## Remaining work

1. FQ: consume original packed scale/min units inside the collective. Do not
   hide an SF expansion kernel behind an FQ label. Define metadata stage
   ownership before attempting header reuse over successive K tiles.
2. Canonical-reader control on this same cooperative pipeline, preserving all
   other axes, before deciding that a shipping offline format should change.
3. Bounded stage/tile/scheduler confirmation, retaining prior winners; then
   Q2/Q3/Q5/Q6 two-plane/group16 paths and compact grouped scheduling.
4. Separate expansion cost, actual model integration, native PPU1.7 compiler/
   simulator/hardware admission. H800 assembly and timing prove none of these.
