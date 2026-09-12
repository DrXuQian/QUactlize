# Q4 small latency: one more shape closed

N512/K2048 now meets the user's **raw-reference +5%** criterion in all six
confirmation rounds. N1024/K5120 improves but remains **+5.2845%**, not a
strict pass. Together with the preceding four shapes, this closes **five of
six Q4 dense M=1 cold-weight shapes**. Production dispatch, shipping DSOs,
canonical offline bytes and llama.cpp have not changed.

Source `9388a5a`; uploaded archive `q4-small-latency-ppu.J4XfKL.results.tgz`,
SHA256 `463ae2aa27ac76a089db75f9b3f5d23e9706885c3336a38fa7a20a6043ef2f21`.
The [review receipt](measurements/q4_small_latency_20260912/review.json)
contains source/runtime/image/device authority, all 80 screen medians, six
round medians, selected geometries, log/report hashes and imported ACU metrics.

## Rotating event results

Microseconds; medians of six alternating round medians, fifteen samples per
round. Baseline is the actual per-shape winner remeasured in this cohort.
Negative deltas mean faster. ACU instrumented durations are not used here.

| N x K | Baseline us | New selected us | Ref us | vs baseline | vs ref | Gate |
|---|---:|---:|---:|---:|---:|---|
| 512 x 2048 | 2.913164 | 2.524766 | 2.437812 | -13.33% | +3.57% | PASS |
| 1024 x 5120 | 5.652115 | 5.582308 | 5.302115 | -1.24% | +5.28% | OPEN |

The smaller winner is `meta-h0-l1-a1-w16`: per-weight FP16 reconstruction,
FP32 accumulation/output, 64 CTAs x512 threads, TileN8, one K pass and one
kernel. Its paired reference gaps are +1.95% to +4.39%. It beats this
cohort's FP32 Xplane by 26.38%.

The medium selection is `affine-h1-l1-a1-w10-c4-p2`: FP32 group-affine,
128 CTAs x320 threads, TileN8, two K passes and one kernel. Its six paired
reference gaps are +5.13% to +5.37%, so the open verdict is not driven by
one unusually slow round. The strict target is 5.567221 us, leaving
**0.015086 us (15.09 ns), about 0.27% of selected latency**, to remove.
This is an arithmetic target, not a predicted optimization gain.

Three medium candidates are essentially tied:

| Candidate | Confirmed us |
|---|---:|
| affine-h1-l1-a1-w10-c4-p2 | 5.582308 |
| affine-h1-l1-a0-w10-c4-p2 | 5.585000 |
| affine-h1-l0-a0-w20-c4-p4 | 5.585577 |

The full spread is only 0.059%. Do not infer a robust unique winner from
this ordering. The two P2 A-load variants have the same recorded native
instruction/load structure; a wider source vector type did not create a
distinct load-width improvement. Retain the P4 one-pass alternative too.

## Verification and timing scope

- Reparsed **158 timing cells** and ten profiled cells from 66 hashed logs;
  all expected cells, identities, sample medians and shortlist verdicts
  match their receipts. All ten ACU hashes match. No failures were hidden.
- 835 source hashes, the build manifest and native ISA receipt match the
  checkout/prebuilt package. Runtime libraries match the build. Both fixture
  hashes match the preceding experiment.
- All 80 candidate contexts pass independent original-GGUF dot, zero-code,
  zero-A, guards and post-replay determinism. FP32 fingerprints agree across
  variants at each of ten same-body/same-geometry groups. Eligible controls
  also match their immutable preceding binaries. Maximum conditioned error
  is `5.080224e-5`; no cross-family bit identity or FP16 accumulation is claimed.
- PPU-ZW810 PCI `0000:08:00.0`, ordinal0; the64MiB override is corroborated
  by the runtime L2 attribute. The ABI property's `sm=1` is not the physical
  72-CU count. No concurrency warnings occur, but logs alone do not prove
  complete device exclusivity.
- Rotating weights exceed2.25xL2 and timed graphs traverse complete rings.
  A is resident. First-use/graph upload is excluded. No box compilation/JIT.
- Campaign wall time was **244.80 seconds (4.08 minutes)**, excluding fixture
  preparation, within the handoff's4–8minute estimate.

## What changed in the small winner

All ten ACU reports were imported locally, 653 metrics per report. Exact
kernel tags and grid/block sizes match the intended arms. These use
forced-cold replay, not the event timing regime. Bytes are decimal MB.

| N512/K2048 arm | Registers | Occupancy | DRAM MB | L1/L2 MB | Vector loads |
|---|---:|---:|---:|---:|---:|
| Old cooperative winner | 34 | 20.48% | 0.6011 | 4.5322 | 5,120 |
| New H0/L1/A1 winner | 32 | 21.37% | 0.5997 | 4.5551 | 6,144 |
| H1/L1/A1 runner | 32 | 21.49% | 0.5996 | 4.6185 | 6,144 |
| Raw reference | 84 | 21.82% | 0.5999 | 0.8622 | 2,560 |

The winner is faster despite **more vector loads**, almost unchanged DRAM
bytes and slightly higher L1/L2 traffic. It is not a bandwidth-byte reduction
or an offline format change. B ownership/coalescing stays unchanged.

New A1 loads each lane's four FP16 activation residues directly, removing
the old A register transpose. The native body has five fewer butterfly
shuffle instructions than new A0, and issues all six A/B/unit loads before
its first vector-load wait. There is still one CTA barrier, no A staging
and no shared bank conflict. Source and ISA plus the controlled screen
support a shorter load/decode dependency path; counters alone do not assign
every saved nanosecond to one instruction.

The five-sample screen at the same H0/W16 geometry shows the interaction:
A0/L0 2.8398 us, A0/L1 2.7706 us, A1/L0 2.7248 us, A1/L1 2.5038 us.
Only shortlisted configurations have six-round confirmation; these screen
values are not promoted into equally strong standalone causal measurements.
Shared A loses to the selected direct-A reader. H1's branchless bit-select
also loses here, so neither switch becomes a universal default.

## Remaining medium shape

The selected P2 reader has 128CTAs rather than the old P4 anchor's64;
threads/CTA drop from640 to320, with two K passes rather than one. Its
ACU L1/L2 traffic rises from11.844MB to22.206MB, vector loads from20,480
to35,840, and registers from58 to62. DRAM bytes stay about2.973MB and
achieved occupancy changes from24.99% to26.47%. Reference reads3.707MB
through L1/L2 at43.06% occupancy. Lower dependency/issue ratio accompanied
higher pipe-busy/issue ratio; these are not wall-time percentages.

The aligned P2 B request is4bytes/lane, four adjacent lanes forming16bytes:
25% unique utilization of a64-byte footprint, versus50% for old P4. This
is a source request model, not measured DRAM efficiency. The tiny event
gain therefore does not prove the more fragmented reader is the only
useful direction. The tied H1/P4 reader was not among the two selected
ACU captures; do not assign it the unchanged P4 anchor's counters.

Next scope is **only N1024/K5120**: keep both P2/two-pass and P4/one-pass
anchors, inspect load issue and emitted CTA-reduction unrolling, and test
a bounded nearby warp-count change only with explicit coverage/footprint
and numerical checks. Avoid spending another full campaign on identical
A0/A1 code generation. Do not rerun the five closed shapes or mark5.28%
as a strict5% pass. Dense M2–7, indexed/grouped GEMV and llama.cpp remain
separate device-admission tasks.

## Combined closure record

Each row uses its own contemporaneous reference cohort; this is a ledger,
not one new six-shape run.

| N x K | vs ref | State |
|---|---:|---|
| 512 x 2048 | +3.57% | CLOSED, this cohort |
| 1024 x 5120 | +5.28% | OPEN, this cohort |
| 4096 x 2048 | +1.00% | CLOSED, preceding followup |
| 4096 x 4096 | -0.42% | CLOSED, preceding followup |
| 5120 x 8192 | +1.58% | CLOSED, preceding reader-reuse cohort |
| 8192 x 5120 | -7.46% | CLOSED, preceding reader-reuse cohort |
