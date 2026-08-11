# Dense Marlin sweep rejection census

This audit answers a deliberately narrower question than a performance sweep:
did A2 recover every already-legal dense tactic that PRE_A2's 2/4-warp
implementation whitelist prevented from reaching the Marlin sweep?

The row-level result is
[`MARLIN_SWEEP_REJECTION_CENSUS.tsv`](MARLIN_SWEEP_REJECTION_CENSUS.tsv).
It retains every PRE_A2-rejected row's source file and one-based source index,
all seven tactic fields, derived `(M-warps,N-warps)`, CTA warps/threads, both
the historical and current states, and the current capability guard.  It is
regenerated and checked by
`ci/check_dense_marlin_rejection_census.py`; a changed table or shifted guard
therefore cannot leave a stale green report.

## Universe and result

The denominator is the **committed dense table after the ordinary tactic-space
exclusions**, not the raw Cartesian product.  `PRE_A2={2,4}` is deliberately a
named historical baseline, not the current implementation.  The current CMake
path derives CTA threads from `(TM/WM)*(TN/WN)*32` and accepts the inclusive
warp-aligned `[32,1024]` capability.  Consequently every delta below is
relative to the same legal set; it must not be reconstructed by subtracting
independently counted exclusions from a raw space.

<!-- BEGIN GENERATED MARLIN REJECTION CENSUS -->
Historical baseline: `PRE_A2={2,4}` CTA warps (threads `64|128`).
Current capability: warp-aligned CTA threads in `[32,1024]`.

| format | committed legal rows | PRE_A2 admitted | PRE_A2 rejected | current admitted | current rejected | A2 recovered by cohort |
|---|---:|---:|---:|---:|---:|---|
| i4 | 1772 | 746 | 1026 | 1772 | 0 | w1=206, w8=384, w16=283, w32=153 |
| i2 | 2140 | 942 | 1198 | 2140 | 0 | w1=274, w8=450, w16=318, w32=156 |
| i1 | 878 | 414 | 464 | 878 | 0 | w1=130, w8=178, w16=112, w32=44 |
| **total** | **4790** | **2102** | **2688** | **4790** | **0** | **2688 recovered** |

All committed rows have integral `TM/WM` and `TN/WN`.  A2 removes exactly
the former current-implementation rejection:

| transition id | category | rows | meaning |
|---|---|---:|---|
| `A2_COHORT_CAPABILITY_RECOVERED` | `CURRENT_IMPLEMENTATION_REMOVED` | 2688 | Every row rejected only by PRE_A2's 2/4-warp whitelist is admitted by the current structural capability. |
| current rejection | — | 0 | Current capability admits 4790/4790 committed-legal rows. |
| hardware/ISA limitation | — | 0 | A2 adds no claim of device speed or correctness; those remain box gates. |
<!-- END GENERATED MARLIN REJECTION CENSUS -->

The checker requires the current implementation to express a capability rather
than another numeric cohort list:

1. CMake derives threads as CTA warps times 32 and uses one inclusive
   `[32,1024]` range;
2. `PersistentTileSchedulerPPUMarlin::fixup_thread_count_capable` spells the
   same warp-aligned range using `cutlass::NumThreadsPerWarp`;
3. the scheduler's explicit and derived arms both consume that helper while
   retaining `FixupThreadCount == DerivedThreadCount`;
4. the named kernel and generated wrapper consume the helper and independently
   assert `FixupThreadCount == MaxThreadsPerBlock`.

Deleting an anchor, reintroducing a list, changing either endpoint, or making
CMake and C++ disagree fails before a TSV is generated.  The TSV records the
current line of the CMake capability guard.

## Constructive compile evidence

Run:

```sh
bash dev/fold_derivation/run_l131_marlin_rejected_cohorts.sh
```

`l131_marlin_rejected_cohorts.cu` retains one real committed i4 row from each
A2-recovered cohort:

| CTA warps | threads | representative `(TM,TN,TK,WM,WN,st,bc)` |
|---:|---:|---|
| 1 | 32 | `(8,16,64,8,16,2,0)` |
| 8 | 256 | `(8,128,64,8,16,2,0)` |
| 16 | 512 | `(8,256,64,8,16,2,0)` |
| 32 | 1024 | `(32,256,64,16,16,2,0)` |

The ordinary DP compile remains a positive control.  A second positive compile
enables `DENSE_MARLIN_SWEEP=1` on the same rows, proving the released wrapper
path reaches the Marlin type without changing tactic, format, group size,
artifact TileK, B-chunk or collective guards.  A separate negative deliberately
mismatches the explicit fixup cohort and must fail the exact-binding assertion;
capability admission therefore cannot weaken the CTA/barrier/workspace seam.

The same TU instantiates the underlying `NamedBarrierManager<Cohort>` and
`BlockStripedReduce<Cohort,...>` types for 32/256/512/1024.  L124 separately
covers every table-derived accumulator layout and its residue predicate.  This
is constructive local evidence for the released capability; it does **not**
establish device correctness or speed.

## What A2 establishes—and what it does not

A2 establishes a closed source-level transition:

- PRE_A2 admitted 2102/4790 rows;
- the current capability admits 4790/4790;
- each cohort delta equals its independently counted table population:
  `w1=610`, `w8=1012`, `w16=713`, `w32=353`, total `2688`.

That closure proves there is no second hidden enumerator filtering a released
cohort.  Device numerical correctness, lock behavior, occupancy and performance
remain box questions, and no number in this document claims otherwise.  The
full performance sweep remains intentionally blocked on the B2 device result.
