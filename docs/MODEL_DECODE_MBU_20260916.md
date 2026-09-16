# Decode MBU: current model and bounded reader follow-up

## Returned W00MDq evidence and next Q8 round

The `kpack-model-mbu.W00MDq.results.tgz` archive is complete. Its SHA-256 is
`b7750decdd615e71649785722f21818ff50185233cc20d13f637c7e640f0f27d`.
All 76 numeric rows and the graph/guard/negative checks pass. The 13 ACU
reports were imported and compared with the returned raw CSVs. Medians below
are recomputed from six rounds of fifteen rotating-weight event samples;
the targets and shipping control in the following sections are unchanged.

| Point | Shipping us | Best new arm us | Delta | Decision |
|---|---:|---:|---:|---|
| Q8 2048 x 4096 | 12.405294 | 13.282941 | +7.075% | Retain shipping; index-only loses |
| Q8 512 x 2048 | 5.760441 | 5.862353 | +1.769% | Retain shipping |
| Q8 2048 x 512 | 4.911176 | 4.912647 | +0.030% | Retain shipping |
| Q4 merged MoE 1024 x 2048, eight experts | 19.292500 | 16.878125 | -12.515% | Keep H32-only candidate |
| Q5 MoE 2048 x 512, eight experts | 12.129259 | 11.237408 | -7.353% | Keep H32 + index/fold candidate |

Q5's observed shared bank conflicts fall from 16384 to zero and its shared
allocation from16640 to256 bytes. Q4's H32-only candidate is slightly faster
than the combined arm. These results do not authorize a global index change.
The two gains multiplied by40 layers estimate0.132 ms/token saved; this is
an arithmetic estimate, not a measured model improvement. Neither candidate
has been installed in production by this experiment.

The missing TC reports are now present. The exact Q6 output-head parent
`closure_q14_0_tm16_tn64_tk128_wn16_dn64`, S1, has419.399 MB DRAM reads,
1862.745 MB L1/L2 traffic and16337371 bank conflicts (13122160 loads,
2234880 stores,980331 stores-from-global-load). The48.02% occupancy is not
the small-Q8 grid problem. These aggregate counters do not identify which
shared instruction caused each conflict. Static inspection finds separate
opaque AIU/TSM B readers and scalar scale/zero consumers. Historical scale
XOR experiments in `dev/fold_derivation/TODO.md` did not reduce read conflicts;
do not transplant a logical swizzle or attribute every conflict to metadata.
An exact-parent instruction/read-path witness is still needed for a TC fix.

### Q8 topology experiment

Entry: `tools/run_q8_topology_box.sh`. It uses the same immutable execution
DSO and a373 KiB isolated PPU candidate, without JIT or a caller rebuild.
The three M1 dense shapes are512x2048,2048x512,2048x4096. Only canonical Q8
K-pack2 is tested; storage and accumulation areFP32, A register rounding is
FP16. No activation quantization toINT8, clipping or offline-layout change.

Thirteen(C,W,P) geometries times two existing A readers compile26 bodies.
The manifest publishes290 full-call cells(110/30/150) and134 excluded
geometry/split pairs with empty K workers. It keeps each shipping incumbent,
including the generic V1 reader on2048x4096. S1 and S2/4/8 complete calls
compete; split timing includes either the original scalar reducer or the
existing ordered float2 reducer. The latter is restricted toM1 and8-byte
output/partial alignment. It is not a new public ABI guarantee.

| B request geometry | Adjacent bytes/group of lanes | 64-byte footprint utilization |
|---|---:|---:|
| C4/P2 | 16 | 25% |
| C4/P4 | 32 | 50% |
| C8/P4 | 64 | 100% |
| C16/P2 | 64 | 100% |

These are source-request footprints, not DRAM transactions. C4/P2 supplies
more N tiles at the cost of narrower requests; C16/P2 reduces per-thread N
work while filling64 bytes. Every cell records A and scale addresses too,
actual base alignment,32/64/128-byte footprints, workers, passes and CTAs.
Native PPU ISA preserves32/64 `v.lop3.b32` and66/132 `v.fma.f32.rtte`
static sites forP2/P4 respectively; they are not dynamic instruction counts.
P2 uses32-bit B loads andP4 uses64-bit B loads. C16's cooperative A path
emits a64-bit load instead of C4's two128-bit loads per packet.

Local PPU compilation and8 host tests pass. OnRTX5070, all290 configurations
pass879 independent GGUF/clone checks,296 changed-input/zero-A graph controls,
six weak-metadata-alignment controls, and336 exact scalar/float2-reducer
comparisons. Maximum normalized dot error is2.576e-8. NVIDIA results are
numeric evidence only, not PPU performance or production admission.
Cold PPU screening retains the top two overall, the best S1, best split and
best C16 candidates. Finals use6x15 alternating samples. ACU captures the
shipping reader and best candidate, including the reducer when required.
The40%/60% MBU targets remain open.

### Router/prepare integration gap

The latest model native trace has585 ordinary BF16 `moe_chain_prepare_m1`
calls,15 `prepare_detail::once` calls and625 independent `topk_moe_cuda`
calls. These counts are consistent with585 unfused decode routers and40
prefill routers: only15 of600 decode chains use the full router/prepare
fast path. Gate/up weight fusion and shared projection preparation do not
imply router fusion. When the caller takes the non-router chain entry,
`plan.router.version==0`; ordinary prepare consumes existing IDs rather than
computing top-k a second time, but its separate launch still costs time.

The caller tries `match_moe_router`, checks the full span's memory ranges,
and calls `moe_run_router`; any decline falls back to native top-k followed
by `moe_run`. The present trace does not record which predicate rejects
each span. Record that exact rejection before changing alias/lifetime guards.
This is a separate, still-open model integration issue, not fixed by Q8 GEMV
tuning or by relabelling the ordinary prepare as fused.

`moe_chain_swiglu_compute<BF16>` also remains a separate launch:600 calls,
2557996 ns total,4.263 us/call and0.171 ms/token over40 layers. For the M1,
top8,hidden512 chain it reads32 KiB of F32 gate/up values and writes16 KiB
of F32 down input, plus small row/status metadata; it does not read weights.
The current body performs scalar loads/stores, BF16 rounding of producer
outputs, and`expf`. Low effective byte bandwidth on this small helper is not
proof of uncoalesced reads: isolate instruction/launch cost and vector width.
Preserve BF16 range and the required rounding. Fusing SwiGLU into a SIMT
producer additionally requires one CTA to own matching gate/up columns;
current contiguous gate/up halves generally belong to different CTAs.
The existing offline gate/up merge alone does not make this fusion happen.

## Scope and targets

Qwen3.5-35B-A3B Q4_K_M, one request, M=1 decode, PPU-ZW810.
Canonical weight bytes and production selection remain unchanged.
Dense compute is FP16; indexed MoE compute is BF16. Storage endpoints and
accumulation are FP32. No gather/scatter is included or required by these
selected M1 endpoints.

Effective weight MBU = distinct packed weight bytes, including metadata,
divided by full-call microseconds and 2,700,000 bytes/us. For indexed MoE,
count the eight used experts, not all 256 resident experts. For Split-K,
include the reducer. This model is not an ACU DRAM-counter percentage.

The requested targets are 40% for small weights and 60% for large weights.
For this five-SIMT-point follow-up, small means less than 2 MiB of distinct
weights: the two 1.0625 MiB Q8 points. This boundary is an experiment label,
not a new dispatch rule. Their 40% target is 1.032 us; the Q4/Q5 MoE 60%
targets are respectively 5.825/3.560 us. These are targets, not proven
attainable bounds. A separate XOR-checked Q8 load-only reference measures
an optimistic memory floor; it does not qualify as a GEMV result.

## Current trace

Authority: `kpack-q4-resume.5k41x3_y`, runtime manifest SHA-256
`40189c0938e87534f97b080ab0e6c5a32f880543716dfacea6563a06c3ba7a79`.
Execution DSO SHA-256:
`780d24dd4dc3620aa6b9eaafec22d3939426c706ca86a6cf0c5453ac13e1a9af`.
Uploaded native Asys SHA-256:
`47fc088c351e2eb0603a68524531674fb6253c25c507cf3227cf96c5bfe3872e`.
Reference Asys SHA-256:
`7f7c0a8e2d24f0569386f6a2f47666473be2924547466b2de948eb94e41072cd`.

The complete trace has 15 decode intervals after the first output-head
endpoint. Each interval covers 40 layers. Every symbol's aggregate count and
duration was checked against the returned JSON. First prefill/JIT work is
not a decode interval. The table uses decode producer averages, not the
prefill-plus-decode totals in the profiler summary.

| Work | N x K; active experts | Calls/token | Producer us | Reducer us | Full-call weight MBU |
|---|---|---:|---:|---:|---:|
| Q8 TC S8 | 8192 x 2048; 1 | 40 | 15.355 | 1.295 | 39.65% |
| Q8 TC S8 | 4096 x 2048; 1 | 30 | 10.457 | 1.241 | 28.22% |
| Q8 SIMT V1/C4/W8/P4 | 2048 x 4096; 1 | 40 | 13.428 | 0 | 24.58% |
| Q8 vector V5/C4/W8/P4 | 512 x 2048; 1 | 100 | 6.291 | 0 | 6.56% |
| Q8 vector V5/C8/W4/P4 | 2048 x 512; 1 | 40 | 5.910 | 0 | 6.98% |
| Q4 merged MoE V3/C4/W4/P4 | 1024 x 2048; 8 | 40 | 20.349 | 0 | 17.18% |
| Q5 down MoE V3/C4/W2/P8 | 2048 x 512; 8 | 40 | 14.339 | 0 | 14.90% |
| Q6 output head TC | 248320 x 2048; 1 | 1 | 407.904 | 0 | 37.88% |

Across these 331 calls/token: 2,518,261,760 effective weight bytes,
4.126 ms producers + 0.089 ms reducers, modeled full-call MBU 22.13%.
This is kernel-active time from a trace, not a replacement for unprofiled
TPOT. Current paired model timing is 7.636 ms reference versus 7.776 ms
native TPOT; the native prefill is faster. No claim of a decode speedup.

## Evidence and isolated candidates

The existing five SIMT ACU reports show near-minimum DRAM reads but much
larger L1/L2 traffic. Q8's two small launches have only 32 and 64 CTAs;
their achieved occupancy is approximately 5.5%. The Q5 MoE report has
16,384 shared bank conflicts and 16,640 shared bytes/CTA, although its
explicit partial array needs only 256 bytes.

Native ISA shows runtime-indexed metadata words stored to and read from
shared memory. Applying the existing Q4 H32 extraction to the identical
Q5 scale/min unit removes the extra 16 KiB in local PPU compilation:

| Same-geometry arm | Q4 vector registers / shared B | Q5 vector registers / shared B |
|---|---:|---:|
| 0: unchanged clone | 66 / 256 | 118 / 16640 |
| 1: H32 metadata extraction | 64 / 256 | 118 / 256 |
| 2: unsigned nonnegative indexing + static CTA fold | 60 / 256 | 114 / 16640 |
| 3: both | 60 / 256 | 114 / 256 |

The unchanged clones match the shipping image's register/shared resources.
This is static evidence of unnecessary metadata materialization, not proof
of bank-conflict removal, correctness or faster execution. The Q8 points
test arm 2 against arm 0 and the immutable shipping DSO; they have no H32
metadata to change. Their index-only register counts do not improve.

All candidates keep N/K ownership, vector load widths, offline bytes,
activation precision, FP32 affine/dot order and S1 geometry. The harness
records lane addresses and 32/64/128-byte footprints for A, both weight
planes and metadata using actual base alignment. H32 changes field
extraction, not the packed-code fast-dequant instruction family.

## Box handoff and interpretation

Entry: `tools/run_kpack_model_mbu_box.sh`. It uses the previous run's exact
runtime and JIT cache; no caller rebuild, new model run or broad bundle
rebuild. Missing cached TC modules may still invoke the existing JIT.
The five candidate DSOs are locally PPU-compiled LFS payloads in
`prebuilt/ppu0010/model-simt-followup-v1` (about 0.4 MiB combined).

1. Bind the three TC profiles through the same matched-table query used by
   the model, checking parent/build/policy/split/algorithm/grid.
2. For each of five SIMT points, require official-GGUF factorized-dot
   correctness, matched FP32 bits against shipping, changed-input graph
   replay, BF16 large-range inputs, output/workspace guards, and zero-A
   negatives. A failed numeric point has no timing admission.
3. Rotate weights over at least 2.25 times the operator-verified 64 MiB L2.
   Validate every physical ring copy. Exclude graph upload and first use;
   alternate arm order over six rounds of fifteen samples.
4. Capture shipping and best candidate in fresh ACU processes; verify exact
   symbols and geometry. Forced-cold profiler counters and rotating event
   timings remain separately labeled.

Independent points and phases continue after failures. Successful JSONs,
samples and reports remain valid evidence; `PASS` means the diagnostic
completed, not that 40%/60% targets were met or a production change admitted.
Avoid other inference work on the selected device. Device identity is
recorded; an otherwise idle machine is not automatically proven.

Next decisions depend on these results: admit only measured wins; retain
shipping when a candidate loses. If the small-Q8 memory floor already
exceeds 1.032 us, separate launch/coverage limits from decoder cost before
promising the 40% target. TC policy changes require the three missing ACU
profiles, not a transplant of the SIMT conclusions.
