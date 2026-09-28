# GGUF Q4_K cooperative SM90 path

This is a private PPU1.7 experiment on PPU CUTLASS4.3. Existing PPU1.0/1.5,
canonical K-pack bytes, runtime selectors and the shipping q17 ABI are unchanged.
It extends the pinned Machete collective to **GGUF group32 affine metadata**,
not GPTQ group128 and not a re-quantization of GGUF weights.

Current scope: dense Q4_K, resident ScaleFirst or original packed-unit FQ,
FP16/BF16 compute and output, FP32 accumulation and Stream-K fixup.
N must be divisible by128; K by256. Three
cooperative tiles `(weight-N, activation-M, K)` are128x16/64/128x64, Cluster1,
four/eight pipeline stages. Configs0/1/2 are stage4;3/4/5 are the same tiles
with stage8. BF16 conversion is direct. Explicit pair multiply and
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
`0eb6a3b218977892e2b6cd30d59880c440e54fd6`, vLLM
`11ba93f3646d4c5476c3b3fd56835589701f0fb1`. The latter supplies the pinned
packing/converter helpers; its CUTLASS4.4 reference tree is **not** included.
The PPU dependency is verified against the supplied4.3 archive's full content
hash, also when extracted without `.git`. Old3.6 results are not4.3 admission.

```bash
python3 tests/ppu17/gguf/build.py \
  --compiler /usr/local/cuda-12.8/bin/nvcc \
  --cutlass /path/to/ppu-cutlass430 --vllm /path/to/pinned-vllm \
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

Add `--metadata fq --configs 0 1 2 3 4 5` to test original packed-unit FQ.
The combined diagnostic library exports a distinct `q17_gguf_prepare_fq`
entrypoint and `GGUF_Q4_K_G32_PACKED_UNITS_PRIVATE_MMA_V1` contract. Its input
is `[K/256,N,16B]` canonical metadata, not raw GGUF's interleaved scale bytes
and not SF arrays. `packed_abi.cu` checks the reader against the existing
canonical writer. No SF expansion kernel is launched on this route.

FQ uses one contiguous TMA metadata copy per64K stage. Its128 columns occupy
256 uint64 elements (2048B); flattening into512 uint32 elements exceeds the
TMA box dimension limit. Storage bytes are unchanged. The stage's group
phase is derived in registers from the existing scheduler's absolute K start,
then advances once per64K stage. No mutable shared phase is published. The
build generates one optional coordinate hook at the pinned outer kernel's
MMA call; its original/generated hashes are recorded and the dependency tree
is unchanged. A missing hook rejects FQ instantiation rather than assuming K=0.
Whole units are deliberately reloaded for each64K stage in this first FQ
implementation. Resident bytes and issued per-stage bytes are different.

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

For bounded memory checks, use `memory.py --application BUILD/smoke
--bundle BUILD --sanitizer /path/to/compute-sanitizer --backend-source
quactlize/backends/ppu17 --expected-uuid UUID --output FRESH_DIR`.
This tests direct C++ calls in both dtypes with memcheck/racecheck/synccheck,
verifies instrumentation using a scalar OOB negative, and stops if a foreign
job arrives. Do not treat a sanitizer banner from an outer Python wrapper as
proof that its child was instrumented. An undersized **TMA output** allocation
was not reported by this H800 tool: a scalar negative proves only scalar
instrumentation. Output guards/ownership tests remain mandatory. Sanitizer
results are never timing evidence.

`stages.py` performs a one-axis stage4/8 comparison and requires identical raw
outputs for each same-tile pair. Both it and `memory.py` accept
`--resume-from PREVIOUS/summary.json` with a fresh output directory. Valid
cells are retained with identity/log checks; interrupted cells are not reused.

Use `memory.py --application BUILD/fq-smoke` for the separate packed-unit
memory gate. `fq_compare.py` compares the same stage4/8 tiles between SF and
FQ for the12 bounded points; pass the same bundle/backend/UUID arguments as
`stages.py`. It requires matched raw output fingerprints, checks clocks,
and preserves complete arms on resume. SF expansion remains excluded, so
this is **resident SF vs complete FQ GEMM**, not an SF/FQ end-to-end decision.

`memory.py --application BUILD/split-start --configs 0 3` additionally forces
the real scheduler to Split-K3 at K1280. It asserts scheduler parameters that
produce K-tile starts0/7/14, then checks exact one-hot outputs and guards in
both dtypes/stage depths. This tests non-superblock-aligned starts without
inventing a second scheduling policy or changing the measured default.

## Remaining work

1. Confirm packed-unit FQ performance and inspect field/header decoding cost;
   reuse across successive K tiles needs a separately proved lifetime.
2. Canonical-reader control on this same cooperative pipeline, preserving all
   other axes, before deciding that a shipping offline format should change.
3. Bounded stage/tile/scheduler confirmation, retaining prior winners; then
   Q2/Q3/Q5/Q6 two-plane/group16 paths and compact grouped scheduling.
4. Separate expansion cost, actual model integration, native PPU1.7 compiler/
   simulator/hardware admission. H800 assembly and timing prove none of these.
