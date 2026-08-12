# Box run preregistration: dense Marlin WK4 and finite GEMV sweep

This document was written before either queued box run produced a result.  Its
purpose is to make the interpretation a property of the source revision rather
than a story chosen after seeing timings.  The two frozen entry points are:

```text
tools/run_dense_marlin_wk4_box.sh
tools/run_gemv_sweep_box.sh
```

The source tree used by either runner must be clean.  A result from a different
root SHA, a dirty submodule, a different binary hash, or an unrecorded device /
driver is a different experiment and is not merged with this one.

## Common identity and durable evidence

Every published result must carry all of the following in one committed result
directory:

- root Git SHA, recursive submodule status, and actlize SHA;
- binary SHA-256 of the linked executable actually launched;
- device model, PCI identity, driver version, and SDK/compiler identity;
- the complete argv, protocol/sample count, command exit status, and the
  unedited stdout/stderr log;
- the raw machine-readable samples used by the summary.

The binary itself need not be committed.  Its hash does.  Temporary paths under
`/tmp` are staging, not evidence.  After identity and coverage are audited, copy
the files without rewriting them beneath:

```text
dev/box_runs/<root-sha>/dense-marlin-wk4/
dev/box_runs/<root-sha>/gemv-sweep/<binary-sha256>-samples20/
```

For dense Marlin this means the build log, the top-level runner log, every
`bpc*.log` or `bpc*.not-run`, and `illegal-bpc.log`.  For GEMV it means
`manifest.json`, `raw.jsonl`, `progress.jsonl`, `result.json`, `run.log`, the
per-job `logs/` directory, and the dry-run audit/summary.  A screen-only PASS or
winner is not a result.

## A1. Classic-aligned dense Marlin

### Fixed comparison

The only primary shape is `M=1, N=4096, K=4096, L=1, gs=128`.  The queued
target is the isolated ordinary-int4 `1M x 2N x 4K`, 256-thread, four-stage
consumer using the shipping xplane artifact.  Its two fixed anchors are:

- standalone classic on the same PPU and shape: **17.8 us / 17.5% of
  nameplate**;
- historical collective `4N x 1K`: **21.14 us / 14.5% of nameplate**.

Let `T` be the queued WK4/B1 median and let the historical gap be
`D = 21.14 - 17.8 = 3.34 us`.  The predeclared recovered-gap fraction is
`R = (21.14 - T) / D`.

| observed WK4/B1 median | registered interpretation |
|---|---|
| `R >= 0.75` (`T <= 18.635 us`) | **CONVERGED-OR-BETTER.** The aligned topology/consumer recovers at least three quarters of the historical gap.  This supports using the aligned arm as the decode baseline; it does not claim that the retained A/B movement, reduction, or epilogue is source-identical to classic. |
| `0 < R < 0.75` (`18.635 < T < 21.14 us`) | **PARTIAL.** The alignment is useful but insufficient.  Report `R`, then use the B ladder to separate extra runnable CTAs from scheduler/fixup cost.  Do not rename a partial recovery “classic-equivalent.” |
| `R <= 0` (`T >= 21.14 us`) | **NO-RECOVERY / WORSE.** Do not invent a new mechanism.  First close the two remaining code-generation questions in `MARLIN_STANDALONE_ALIGNMENT.md`: classic's `__launch_bounds__(256,2)` versus `MinBlocksPerMultiprocessor=1`, then the standalone-versus-repository toolchain/codegen boundary. |

The 20-sample `[min,max]` band is printed beside the median.  If it crosses a
classification boundary, retain the median category above but append
`BOUNDARY-UNRESOLVED`; do not present the category as a resolved ordering.
Correctness, shipping-artifact roundtrip, exact fixture, and all eight stable
lock fingerprints are prerequisites.  A timing from a failed prerequisite is
discarded, even if it is fast.

### `(WarpK, blocks_per_cu)` matrix

There are two different meanings of “default”: the shipping default topology
is WK1/B1; inside the isolated aligned target, omitting the scheduler override
is WK4/B1.  They must not be conflated.

| topology | B=1 | B=2 | B=4 | B=6 |
|---|---|---|---|---|
| WK1, historical `4N x 1K` | **shipping/default control** | scheduler-only diagnostic | scheduler-only diagnostic | scheduler-only diagnostic |
| WK2 | compile-time negative; no timing | compile-time negative; no timing | compile-time negative; no timing | compile-time negative; no timing |
| WK4, aligned `2N x 4K` | **primary queued result; aligned-target default** | occupancy/scheduler diagnostic | occupancy/scheduler diagnostic | occupancy/scheduler diagnostic |

`run_dense_marlin_wk4_box.sh` produces the WK4 row only.  B=1 deliberately
omits `--marlin-blocks-per-cu`; B=2/4/6 are explicit diagnostics and any value
above `Gemm::maximum_active_blocks()` is `NOT RUN`, not a slow point.  WK2 is a
deliberate unsupported control, not a missing measurement.

The permanent WK1 controls must still say that the new axis leaves the shipping
builder type and xplane bytes unchanged (`0/8192` byte-map differences).  If a
WK1 device control is included, its output must be raw-bit-identical to the
historical arm.  Any WK1 type/byte/output drift invalidates the whole WK4 batch:
the experiment would then mix a changed baseline with a changed candidate.

For every supported WK4 B point, record median/mean/min/max/spread,
`G/I/active/idle`, handoffs and max peers, achieved warp/CU, registers/thread,
block limits, and the named stalls/counters requested by the box recipe.  The B
ladder is judged as follows:

- decreasing time with gently increasing handoffs means B1's one-CTA/CU guard
  is too conservative for this FP32 fixup protocol;
- superlinear time growth with peer/handoff growth reproduces the classic
  ordered-chain cliff and supports the guard;
- a flat ladder means the shape is limited elsewhere; it is not evidence for
  either scheduler claim.

## A2. Finite GEMV sweep

Only the ten-group `partial_space=false` binary may establish a winner.  The
`i4-native` run is a compile/device smoke test and can report only
`LOWEST_IN_PARTIAL_SPACE`.

### Winner policy, fixed before timing

For each `(shape, format)` group, publish the leader, runner-up, both raw
sample bands, the median gap, and that shape's timer record:
`quantum.status`, `quantum.us`, `quantum.reason`, and
`quantum.minimum_claimable_us`.  The default minimum claimable quantum is
**0.01 us per shape**; it is not replaced after looking at the gap.

- **Winner changes:** accept the new leader only when the full manifest is
  complete and the analyser says `RESOLVED`: a runner-up exists, the raw bands
  do not overlap, the timer quantum is known, and the gap is greater than one
  quantum.  Otherwise report the observed leader but label it `UNRESOLVED`; it
  is not a shipping-routing decision.
- **Winner does not change:** always report the runner-up and its absolute and
  relative gap.  If the bands overlap, the quantum is unknown, or the gap is at
  most one admitted quantum, the prior “winner” was never established at this
  resolution; label it `UNRESOLVED`, not “confirmed.”
- A launch refusal, timer failure, or output-witness mismatch is an exclusion,
  not a very slow sample.  Any missing expected outcome makes the full result
  incomplete and unable to publish a winner.

### Pruning accounting

The underlying tactic Cartesian product has 27,360 rows.  Its static census is
fixed at 10,260 legal and **17,100 pruned**, with the complete first-failure
histogram:

| static reason | rows |
|---|---:|
| `STEP_TOO_SMALL_FOR_SPARSEST_PLANE` | 10,944 |
| `CTA_N_NOT_WHOLE_CHUNKS` | 6,156 |
| **total** | **17,100** |

The expanded full-shape manifest must independently close at `jobs=86`,
`total=165600`, `legal=63180`, `pruned=102420`, with
`STEP_TOO_SMALL_FOR_SPARSEST_PLANE=64512` and
`CTA_N_NOT_WHOLE_CHUNKS=37908`.  The i4/native smoke manifest must close at
`jobs=18`, `total=18288`, `legal=11430`, `pruned=6858`, all 6,858 under
`CTA_N_NOT_WHOLE_CHUNKS`.  Any extra reason, missing reason, or histogram whose
sum differs from `pruned` is a failed census, not a performance result.  These
reason groups must accompany the result so an apparently strong winner cannot
hide an over-aggressive pruning rule.

## A3. What may be concluded

These runs answer only the registered comparisons above, on the recorded
binary and device.  Dense WK4 does not establish a MoE result.  The GEMV sweep
does not turn a controlled-unshipped int1 row or a partial-space minimum into a
shipping winner.  No result is interpreted until provenance, raw coverage,
correctness/exclusion status, and the preregistered resolution/pruning checks
all close.
