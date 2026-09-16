# Decode MBU: current model and bounded reader follow-up

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
