# Quactlize handoff snapshot, 2026-09-17

This is curated technical memory, not a replacement for fresh PPU measurements.
The originating source is `d494291` on `develop`. Resolve full commits and
current pins from the checkout before a new run. Do not run against a moving
branch tip or describe later changes as this baseline.

## Current authority

- Runtime pin: `tools/kpack_q4_model_artifact.json`.
- Artifact commit: `58a50648375403d6a6e2685ead4c50559dbd9a9b` on
  `artifacts/kpack-model-runtime-v1`.
- Package: `prebuilt/ppu0010/kpack-model-runtime-v1`.
- Execution SHA256:
  `47d75e168b71ada4cb42edf29d0a2700af0de0868d81d8a220880a9e007fa30e`.
- Manifest SHA256:
  `f67c67e138793f74d39fe7965064874fb06f342932c2a29adc10772b869bdd34`.
- Caller: `dev/quactlize-v0.3.0`,
  `bd7ad99f156a4656997b1a3370dc3f2bfca5872f` in the owner's llama.cpp fork.
- Returned evidence: `kpack-q4-model.Y8Wky8.results.tgz`, SHA256
  `e7ae4e1e8408564d3059d0a872b1e7687c5b9cd94324e8ac48abbf583b121d49`.

The immutable build receipt still says device admission is pending. The later
device review is recorded in `docs/MODEL_DECODE_MBU_20260916.md`; do not edit
an old receipt to make it claim a later test.

## Priority GEMV cohort

N and K are per-expert dimensions. Indexed cases have E=256, top-k=8 and
one request token; count distinct active expert slices, not all 256 weights.
All five store input/output as F32. Accumulation is F32.

| Case | Qtype | N | K | Compute | Current measured implementation |
|---|---:|---:|---:|---|---|
| Dense large Q8 | 8 | 2048 | 4096 | F16 A | V5/C8/W4/P4/S8 plus ordered float2 reducer |
| Dense small Q8 A | 8 | 512 | 2048 | F16 A | Retain current per-shape policy; do not substitute the large recipe |
| Dense small Q8 B | 8 | 2048 | 512 | F16 A | Retain current per-shape policy |
| Merged gate/up MoE | 12 | 1024 | 2048 | BF16 A | V3/C4/W4/P4/S1, Changes=1, H32 metadata, channels=1 |
| Down MoE | 13 | 2048 | 512 | BF16 A | V3/C4/W2/P8/S1, Changes=3, H32/index/fold, channels=8 |

Read the selectors rather than reconstructing their meaning from these labels:
`policies/kpack_q8_vector_v1.{json,hpp}`, the Q4 decode policy, and
`quactlize/execution/{simt_kernel.cuh,simt_q8_vector.cuh,simt_codegen.py}`.
Q8 is canonical K-pack2, not a raw-GGUF or Xplane reader. Q2/Q3 use K-pack8
low planes; Q3/Q5/Q6 retain their separate high planes. The arrangement
registry, not a name, is the byte-layout contract.

The following are forced-cold ACU durations from the returned package,
not rotating-event baselines and not whole-model timings:

| Case | Producer us | Reducer us | Registers | Shared bytes |
|---|---:|---:|---:|---:|
| Dense Q8 2048x4096 | 9.708 | 1.690 | 60 | 512 |
| Dense Q8 512x2048 | 6.905 | none | 60 | 512 |
| Dense Q8 2048x512 | 6.621 | none | 60 | 512 |
| Q4 indexed | 18.076 | none | 64 | 256 |
| Q5 indexed | 14.317 | none | 114 | 256 |

Freeze a small/large classification in the new workload manifest before timing.
The initial overnight profile treats the two small Q8 cases as small and the
other three as large. This is an explicit task binning, not a hardware law.
Reference peak bandwidth is the user's 2700 GB/s assumption; report it as such.
Use measured active packed bytes from the artifact, including metadata.

## What the last model run proved

Qwen3.5-35B-A3B Q4_K_M, PP2048/TG128, request batch1, warmed ABBA:

| Metric | Native llama | K-pack |
|---|---:|---:|
| Prefill us/token | 140.723145 | 106.291748 |
| Decode ms/token | 7.681504 | 7.410473 |

Component gates passed: router alias360x4, production reader24 checks/18
replays, Q8 numeric12480, BF16 capability746, five-format metadata and mixed
chains. This perf-only run did not rerun full-model perplexity.

Asys found600 fused M1 prepares and40 remaining prefill top-k calls. The
prepare is7.733 us/call and SwiGLU4.087 us/call in this trace. They do not
load weights. Do not claim all remaining latency is GEMV, or modify these
helpers in a GEMV-only task. Extra caller work is a separate admission scope.

The original `model-acu` failure was a checker error: the new Q8 reducer name
and alternate M1/prefill symbols were rejected. `d494291` re-imported all nine
reports successfully without changing a production binary. Recheck saved
reports before asking for an unnecessary new device run.

## Lessons that transfer, not universal recipes

- Q4 C4-to-C8 on one cold shape reduced L1/L2 traffic by about49% with nearly
  unchanged DRAM bytes. Lower occupancy was faster. See the cold skill's C4/C8
  reference for the exact scope and the later stricter raw-reference criterion.
- Q5 H32/index/fold removed shared-bank conflicts and reduced per-CTA shared
  storage16640 to256 bytes in its admitted scope. Q4 preferred H32 alone.
- A fast reducer is part of a Split-K winner. Eight-byte-aligned float2 stores
  need the existing scalar fallback for a public four-byte alignment contract.
- NVIDIA H800/5070 wins did not transfer reliably to PPU. ACU and native ISA
  decide PPU admission. A high nominal occupancy is not an objective itself.
- The all-format reader supports Q2/Q3/Q4/Q5/Q6/Q8 and M1..8, but old generic
  SIMT is not the optimized Q4 incumbent. Keep both admitted Q4 and TC choices.
- BF16 is not F16 decode followed by a cast. Activation243383.484 is finite
  in F32/BF16 but overflows F16; clipping is not a valid numerical repair.

## Sources to read on demand

All paths below are in the Quactlize checkout:

- `docs/MODEL_DECODE_MBU_20260916.md`: latest returned evidence first;
  older sections retain earlier pending states and must not override it.
- `docs/measurements/model_decode_promotions_20260916.json`: promotion data.
- `docs/SIMT_ALL_FORMATS_20260915.md`: initial all-format reader and upload-order
  failure; numerical evidence on NVIDIA is not PPU performance admission.
- `docs/LLAMA_CPP_KPACK_HANDOFF.md`: caller/package integration.
- `dev/fold_derivation/TODO_KPACK_INCREMENTAL_OPTIMIZATIONS.md`: open work.

If a claimed archive or counter is unavailable, mark it historical/unverified
in the new task. Missing local raw evidence is not permission to invent it.
