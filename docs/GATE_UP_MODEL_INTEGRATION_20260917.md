# Paired-N4 gate/up model integration

Status: returned PPU integration gates and actual fusion execution pass;
warmed model latency improves in the declared cohort. Model accuracy
admission remains separate; see the [returned review](GATE_UP_MODEL_REVIEW_20260917.md).
Enable explicitly with `MODEL_GATE_UP=1`
in `tools/run_kpack_q4_model_box.sh`. `MODEL_COMPUTE=bf16` applies BF16 only
to routed projections; dense/shared projections retain F16 compute and F32 I/O.

## Scope and selection

Both projections have N512/K2048. Selection is deliberately bounded to the
returned [16-point comparison](GATE_UP_PAIRED_N4_PERF_REVIEW_20260917.md).

| Consumer | Tokens | Config |
|---|---|---|
| Q8 shared, E1, F16 | 1..8 | SIMT C4/P8/W8, S1 |
| Q4 routed, E256/top8, BF16 | 1 | SIMT C4/P8/W8, S1 |
| Q4 routed | 2 | SIMT C4/P8/W4, S1 |
| Q4 routed | 3 | TC TM16, S2 |
| Q4 routed | 4 | SIMT C4/P8/W8, S1 |
| Q4 routed | 5..6 | TC TM16, S1 |
| Q4 routed | 7..8 | TC TM8, S1 |

Token4 uses the measured SIMT finalist within 1% of the fastest TC finalist,
avoiding an extra reduction launch. Other shapes, dtypes and prefill retain
their previous canonical path; there is no nearest-shape extrapolation.

## Weight and execution ownership

Canonical cache bytes are not reinterpreted. A separate GPU auxiliary copy
uses layout `0x47554e3400000001`, with G4/U4 physical N order. The first
pre-capture preparation copies canonical words and metadata together. It
synchronizes that preparation stream once; replay does not repack, allocate,
copy to CPU, or wait on a host event. Auxiliary weights live with the execution
context and are reused across graphs/token counts. This first integration
does not write them to the existing disk cache.

Canonical weights remain necessary for prefill and out-of-scope operations.
For 40 layers with these dimensions, the auxiliary planes require about
11.33 GiB, in addition to the original model. This is an integration cost,
not a byte-neutral replacement claim.

Shared fusion matches only a closed `MUL_MAT, MUL_MAT, SWIGLU` subgraph.
It uses the graph's existing allocator-overlap guard. Routed fusion reuses
the router/prepare/down/weighted-finish chain. It skips the old gate/up
producer and activation stage, writing activation directly into down's input:
F32 original slot order for SIMT down, or BF16 compact order for TC down.
The existing compact-row map supplies input ownership; no new gather/scatter
kernel is added. Invalid routing status propagates NaN, never stale data.

## Build and validation

The small execution library was rebuilt because the common SIMT templates
gained the fusion finish hook. Existing TC modules, packer and prefill images
are reused without recompilation. The dedicated fusion DSO contains the
new repacker, typed compact-row binding and confirmed configuration selector.
No llama binaries, compiler objects or build logs are published to Quactlize.
The caller is built on the box using `.aoneci/scripts/build.sh`.

Local release checks: 226 host regressions pass; the caller's execution,
loader and graph-walker translation units compile with the PPU SDK. The
fusion library took 114.8 seconds and the small execution rebuild 196.4
seconds with eight local jobs. These checks are not device numerical or
performance admission.

Box runner order:

1. Verify source/payload/runtime identities; 4 canonical-to-paired byte checks,
   144 mapped-output cells, 288 changed-input replays and 148 negatives.
2. Existing selected decode/BF16 checks, plus 16 paired routed-chain cases
   (tokens1..8, router absent/present), retaining the selected down kernel.
3. Original-corpus model likelihood comparison against the native reference.
4. Qwen3.5-35B-A3B Q4_K_M warmed ABBA prefill/decode benchmark.
5. Matched reference/native Asys captures, second request only. Require both
   shared and routed fusion device symbols, not merely selection log lines.

`results/trace/<model>/{reference,native}/proof.asysrep` contains the full
traces. `results/paired-model-proof.json` records observed fusion operations.
The usual `.results.tgz` contains compact summaries and logs; profiler reports
remain on the box. First-use packing/JIT costs are excluded from steady timing
but are not claimed to be zero. The returned review records the measured
model gains and the remaining accuracy/memory boundaries.
