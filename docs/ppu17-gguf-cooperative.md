# GGUF cooperative pipeline checkpoint

Current builds use [PPU CUTLASS4.3.0](ppu17-cutlass430.md). The following
2026-09-26 measurements belong to3.6 and are not relabeled as4.3 results.

2026-09-26; isolated branch `dev/ppu17-cutlass360`, parent `ee73b4e`.
This adapts Machete's register-dequant TMA/WGMMA pipeline to Q4_K's real
group32 scale/min semantics. No format merging or requantization. Existing
PPU1.0/1.5 code, canonical bytes, default dispatch and dependencies are intact.

## Implemented and checked

- Private lossless128N x64K code packing; CuTe/scalar inverse proof over8192
  coordinates and256 aligned16-byte thread vectors.
- Two physical scale/zero rows per64K stage, with group32 logical broadcasts.
  Host proof covers FP16/BF16, N128/256 and stages4/8. Wrong group/stage
  substitutions are rejected. Pipeline transactions count physical bytes.
- Three dense cooperative configs, explicit four-stage pipeline, native SM90
  Stream-K workspace/fixup included in the prepared call.
- CUDA assembly contains128-bit shared loads, TMA, direct BF16 pair multiply/
  add, and FP32-accumulating WGMMA. Six device bodies, five codegen negatives.
- H800:54/54 numerical cells for FP16 and54/54 for BF16; each has output
  guards, poison checks and three changed-input graph replays. Exact one-hot
  weights, distinct group32 metadata, M1/2/7/8/9/63/64/65/128, N128/256,
  K256/512. Stale-group data negatives are rejected for both dtypes.
- Largest relative L2 versus the typed raw-GGUF CPU-dot oracle:
  FP16 `2.494e-5`, BF16 `6.637e-5` in that numerical cohort.

The new descriptor remains diagnostic-only and is not published under a
canonical K-pack mapping ID. This milestone is **SF resident GEMM**, not FQ,
SF expansion performance, full-model accuracy or native PPU1.7 admission.

## Initial rotating-weight screen

H800 PCIe,114SM, queried L2=52,428,800 bytes; ring>2.25x L2; clocks1755/1593MHz
at both endpoints. Idle guard and sampled process monitoring admitted every
listed run. Each arm/config checks correctness before11 timed graph samples.
All scheduler clearing/fixup is included; first graph launch, allocation,
packing and SF expansion are excluded. Both arms use the same original GGUF
bytes and typed affine oracle.

| M/N/K | FP16 new SF (us) | BF16 new SF (us) |
|---|---:|---:|
| 1/512/2048 | 11.980 | 11.939 |
| 8/512/2048 | 11.938 | 12.002 |
| 128/512/2048 | 13.786 | 13.462 |
| 1/4096/4096 | 19.433 | 19.777 |
| 8/4096/4096 | 19.875 | 19.967 |
| 128/4096/4096 | 26.433 | 26.348 |

Selected among three configurations in each screen, **not confirmed winners**.
For example, M128/N4096/K4096's original bring-up control is443.411/449.882us
for FP16/BF16. This removes major bring-up overhead, but that old control is
not a mature library baseline. Packing, RS operand delivery and pipelining
changed together, so this is not evidence that shuffle alone caused the win.
GPTQ group128 Machete/Marlin timings are not equivalent GGUF baselines.

## Evidence and next gates

Artifacts: `/workspace/quactlize-ppu17-20260926/gguf-mainloop/`.

- `build-v1/manifest.json`, DSO SHA256
  `538068ed7286724cd5aa7cd6b00f6036e59e45099f05d18707b8129150794a2f`.
- `codegen-v1.json`, `numeric-f16-v2`, `numeric-bf16-v3`, `screen-v1`.
- The two earlier BF16 attempts were rejected before launch for foreign GPU
  processes; they were not numerical failures and were not reused as results.

Next: bounded stage-depth comparison with existing configs retained, memory/
race checks, and repeated finalist timing. Then add FQ packed metadata and a
canonical-reader control on this same pipeline before any shipping-format
decision. Other qtypes, grouped scheduling and native PPU1.7 remain open.
See [build/replay instructions](../tests/ppu17/gguf/README.md).

## Bounded stage-depth follow-up

`build-v2` retains configs0/1/2 and adds only stage8 counterparts3/4/5.
Both dtype numerical cohorts now pass108/108 cells (216 total in this build).
All36 same-tile stage4/8 comparisons in the12-point screen produce identical
output fingerprints, including changed-input replay.

At N4096/K4096 and M1/8, the128x16x64 stage8 tile measures17.40–17.58us
versus19.49–19.97us for the same stage4 tile, approximately10.6–12.4% lower
in this screen. At N512/K2048 the same change is only about2–4% and needs
repeated confirmation. At M128/N4096/K4096, BF16 tile128x128 is26.40->25.18us;
the analogous FP16 pair has different endpoint clocks and its delta remains
**UNRESOLVED**, not a win. No default configuration is changed.

Evidence: `stage-numeric-{f16,bf16}-v1`, `stage-screen-v1` valid first three
points retained in `stage-screen-v2`; the interrupted large-M1 timing was
discarded and rerun. `report-v1/benchmark.csv` records all180 screen cells,
not just the fastest row. The report counts324 numerical cells across the
initial and stage cohorts, deliberately retesting unchanged parents; these
are not324 independent workloads.

Sanitizer admission is separate. The direct executable's scalar OOB negative
is detected, but an undersized TMA output was not reported by the installed
tool. Do not treat that as proof of valid output sizing. Output guards and
host ownership checks are retained; full memory/race/synchronization coverage
is tracked independently in the task artifact directory. The stage4 direct
cohort is complete:18 positive cells (three configs x two dtypes x three
tools) plus the scalar negative, all admitted in `direct-memory-v7`. Foreign
job interruptions were resumed with each retained log's identity checked.
Stage8 subsequently completed19/19 in `stage8-memory-v7`, retaining only
identity-checked passing cells. The memory gate is closed; repeated finalist
timing and production admission are still separate. Earlier interrupted
attempts remain preserved. The runner now remembers proven descendants after
exit because NVML can briefly report them after `/proc` disappears; live PID
reuse and actual foreign GPU processes are still rejected.

The final build/replay script was exercised as `build-v3` (155.77s total),
including the direct smoke executable. Its12 device instruction bodies match
the measured `build-v2` exactly; timing receipts remain bound to `build-v2`,
not relabeled as measurements of a new DSO.

## FQ continuation boundary

The next provider must load original16-byte Q4 packed units (one unit per256
codes), then produce the exact same typed group32 affine values in registers.
It may not call an SF expansion and report only the following GEMM as FQ.
The code packing can stay the same; metadata delivery is a separate axis.

Start with explicit ownership/byte proofs before optimizing reuse. Fetching
one whole unit every64K tile is a simple correctness control but reads that
unit four times: per column per64K, codes32B + metadata16B, versus ideal
packed amortization32B +4B and current SF32B +8B. It is not automatically a
performance improvement. Reusing a256K metadata unit requires a lifetime
that covers four operand stages, arbitrary Stream-K slice starts/ends and
successive output tiles; it cannot borrow the ordinary operand stage release.
Do not delete waits or change barrier byte counts based on an unproved cache.

Retain the original FP16 header as source data. In BF16 compute, scale/min
must use that header's exact value with the original integer field, then round
to BF16. Converting the header to BF16 before multiplying, or reconstructing
FP16 weights before BF16, changes the contract. Converter-bias correction and
typed weight multiply/add keep their established rounding boundaries.

Before format adoption, compare canonical K-pack and private MMA packing on
the same group32 collective with identical tile/scheduler/metadata. Only this
control can separate layout cost from the large original bring-up overhead.

## Packed-unit FQ implementation

The new FQ input is the existing `[K/256,N,16B]` canonical metadata plane and
the same private lossless code packing as SF. It has a separate diagnostic C
entrypoint and contract; it never launches an SF prepass. The host decoder
matches the existing canonical writer and both typed affine readers in49152
checks. Original FP16 header values are preserved before typed scale/min
rounding. BF16 weight conversion uses native BF16 pair arithmetic, not FP16
weights followed by BF16 conversion. Accumulation and fixup remain FP32.

Two failed designs are retained:

- A512-element uint32 TMA box is invalid. The same2048 bytes are now described
  as256 uint64 elements; this is a descriptor change, not a format change.
- Publishing the absolute group phase in a mutable shared stage passed216
  numeric cells, but direct racecheck reported RAW hazards on that scalar.
  Those builds (`fq-build-v4/v5`) are **not** timing-admitted. A tiny single-
  phase probe passing racecheck did not exonerate the reused GEMM pipeline.

The admitted candidate eliminates that shared communication. The existing
Stream-K work tile already owns the absolute K start; an optional capability
hook passes it to the collective immediately before MMA. Each consumer keeps
its cursor in registers. The generated build-local overlay changes only this
call site and records the original/generated hashes. Pinned CUTLASS files,
the scheduler, waits, barrier counts and existing backends are unchanged.
Without the hook, an FQ instantiation fails at compile time rather than using
an assumed zero start. The precise dynamic cause of the old shared hazard is
not claimed to be a compiler defect or sanitizer false positive.

`fq-build-v6` DSO SHA256:
`83c499e608d950c0bb5b8d5ac7142ccac7a540796d109e9ab8d0c36d4ae3a4a7`.
H800 admission:108 numeric cells per dtype per metadata mode,432 total;
FQ/SF exact one-hot, typed raw-GGUF oracle, guards, metadata negatives and
changed-input graph replay. Direct FQ memcheck/racecheck/synccheck passes36
positive cells plus the scalar OOB negative (`fq6-memory-v1`). All24 generated
GEMM bodies pass codegen checks with zero local-load/store spill sites and six
rejected codegen negatives. This remains H800 CUDA evidence, not PPU1.7 native
compiler or device admission.

The focused `split-start` target forces Split-K3 at K1280 and asserts actual
scheduler parameters for K-tile starts0/7/14. It checks exact one-hot outputs
and guards for FP16/BF16 and stages4/8; its direct memory result is tracked
separately from the default-scheduler cohort.

The reproducible `fq-build-v7` includes this target; its24 GEMM instruction
bodies matchv6 exactly. `fq7-split-memory-v1` passes12 positive direct tool
cells plus the scalar negative. Timing authority remainsv6, not relabeled as
a measurement ofv7. The build took291.42s including host proofs and all three
direct smoke executables.

### First FQ/SF rotating-weight screen

`fq6-screen-v1` completed24 arms/48 configuration measurements and24 matched
same-tile pairs: every pair has identical raw output and matched endpoint
clocks. Timing uses the same idle/rotating-weight protocol as above. FP16
examples (microseconds; stage8, except noted):

| M/N/K | Resident SF | FQ | FQ delta |
|---|---:|---:|---:|
| 1/512/2048 | 11.59 | 13.93 | +20.3% |
| 1/4096/4096 | 17.45 | 24.41 | +39.9% |
| 8/4096/4096 | 17.28 | 24.33 | +40.8% |
| 128/4096/4096 | 25.46 | 31.63 | +24.2% |

Across both dtypes and all24 same-tile pairs, FQ is11.5–44.7% slower than
resident SF. This is a bounded screen, not repeated finalist confirmation.
The M128 points compare the128x128 tile only; do not mistake them for the
best result among all three tile shapes in earlier screens.

FQ reduces resident weight bytes by10% in this layout, but its first reader
issues more metadata bytes per64K stage and decodes headers/fields during
MMA. These are concrete optimization targets, **not** a measured attribution
of the entire timing gap. SF expansion remains excluded, so these numbers
alone cannot select the full FQ versus SF route. FQ is not promoted, and no
performance parity with optimized GGUF or GPTQ libraries is claimed.
