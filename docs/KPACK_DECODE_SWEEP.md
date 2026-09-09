# Bounded grouped decode experiment

Status: local compilation/proofs complete; PPU correctness and performance
pending. This experiment does not change the production dispatcher, existing
native bundle, llama.cpp wiring, or offline bytes.

## What changed

The ordinary grouped kernel already traverses K tiles `s, s+S, s+2S, ...`.
The native module previously rejected every grouped `S != 1`. It now builds
an FP32 pointer-array epilogue for those partitions, output descriptors
`[split][expert]`, partial data `[split][all real rows][N]`, and the existing
ordered FP32-to-FP16 reducer. The mainloop and canonical metadata coordinates
are unchanged. Persistent grouped Split-K remains unsupported.

The grouped device-only metadata kernel updates shapes/rows once per expert
and publishes all slice pointers. It runs on the same stream as producer and
reducer. No allocation, host readback or synchronization is added to `run`.
S1 retains its FP16 epilogue and original workspace size. The host-prepared
compact arm is a diagnostic, not a replacement for dynamic GPU routing.

The SIMT candidate explicitly loads each b16 code word once per K8 cohort,
reuses it across the group's slots, converts pairs, and uses two FP32 dot
accumulators. It retains direct indexed activation/expert access and F32
output. It is **not** a tensor-core GEMM and needs no new offline shuffle.

The pair candidate uses FP16 **fused** affine dequantization. Its rounding
can differ from the scalar two-rounding oracle. The magic integer is removed
before scaling, avoiding cancellation by a large folded zero. Numerical
admission is against independent official GGUF arithmetic, not claimed bit
identity with the scalar path. The old scalar entry remains the default;
the pair entry must be requested explicitly.

## Scope and denominator

| Arm | Scope |
| --- | --- |
| Q4 grouped | N512/K2048, E256, top8, one token; six measured parent geometries including the deployed winner and TM8; S1/2/4/8 |
| Q5 grouped | N2048/K512, E256, top8, one token; five measured parent geometries; S1/2 (only two TK256 tiles exist) |
| Grouped controls | N256, same K; rows `[9,0,3,1]` and changing-router replay `[0,3,1,9]`; second TM8 tile and empty experts |
| SIMT model shapes | Both model shapes; scalar 8 recipes, pair 24 recipes: columns16/32, warps2/4/8, S1/2/4/8 |
| SIMT controls | All five formats, dense / grouped / indexed broadcast / indexed per-slot inputs; three recipes per control |

Total: **16 isolated jobs, 260 measured cells** if all pass. Parents are
looked up in the historical measured inventory, not recreated from a second
tactic authority. The existing winner is mandatory. No full Cartesian sweep
or model run is started.

Each cell checks independent conditioned dot error `< 0.005`, output/workspace
guards, and eager/graph execution. Grouped device-only handles also replay
changed offsets and activations without host metadata preparation. Every
FP32 partial is poisoned and checked against independent **logical K**
partitions; final FP16 output must agree bitwise with an ordered host reduction
of the downloaded partials. CPU negatives cover rotated code slots, missing
and swapped partials, reducer corruption, invalid geometry and overflow.
This is a bounded admission test, not an exhaustive all-shape numerical proof.

Timing defaults: three rounds of 11 event samples, 16 complete calls per
graph, divided by 16. Split-K time **includes the reducer**. Device-only GEMM
also includes metadata generation. Host compact excludes host preparation.
SIMT includes indexing and F32 input/output; GEMM uses pre-gathered FP16.
Cross-algorithm comparison is therefore still core-scoped, not a replacement
for a matched llama.cpp adapter/fusion benchmark.

All weights and IDs stay resident; there is no cache flush. Bandwidth is
necessary active-expert bytes divided by elapsed time, not measured DRAM
traffic. Asys/ACU can establish component time and actual traffic later.
The package manifest includes static SIMT ISA/register counts; those are not
dynamic instruction counts or evidence of a speedup.

## Box (no compilation)

From Quactlize `develop`, with one idle PPU:

```bash
git pull --ff-only
git lfs pull --include='prebuilt/ppu0010/kpack-decode-sweep-v1/**' --exclude=''
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 bash tools/run_kpack_decode_box.sh
```

Run with `bash`, not `source`: the script's failure cannot exit the caller's
Docker shell. It prints progress every 30 seconds, preserves each job's
log/result, and continues other jobs if one fails. The resume unit is one
parent (or one SIMT format), not an entire large bundle:

```bash
RESUME_RUN=/workspace/kpack-decode.REPLACE_WITH_PRINTED_ID \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 bash tools/run_kpack_decode_box.sh
```

Return the printed `*.results.tgz`. Successful admission prints
`KPACK_DECODE_DONE status=PASS cells=260/260`. A failed job cannot contribute
an admitted timing. Do not pool old 18.44/28.44-us samples with this run.

The small prebuilt package replaces nothing. Only after device results can
we select a candidate, revisit empty-CTA overhead, and measure adapters / the
reference MMVQ fusion on equal work. Grouped Split-K and SIMT remain pending
performance debt until those measurements exist.
