# Dense Marlin sweep rejection census

This audit answers a deliberately narrower question than a performance sweep:
which already-legal dense tactics never reach the Marlin sweep, and why?

The row-level result is
[`MARLIN_SWEEP_REJECTION_CENSUS.tsv`](MARLIN_SWEEP_REJECTION_CENSUS.tsv).
It contains every rejected row's source file and one-based source index, all
seven tactic fields, derived `(M-warps,N-warps)`, CTA warps/threads, the
supported cohort set parsed from CMake, the first rejecting guard and a
reason/category.  It is regenerated and checked by
`ci/check_dense_marlin_rejection_census.py`; a changed table or shifted guard
therefore cannot leave a stale green report.

## Universe and result

The denominator is the **committed dense table after the ordinary tactic-space
exclusions**, not the raw Cartesian product.  The Marlin test in
`quactlize/csrc/CMakeLists.txt.in` is a second-stage filter over that list.  The
checker parses its actual strict OR-of-`_DENSE_MARLIN_CTA_WARPS EQUAL N`
condition; it does not carry a second hard-coded allow-list.
Consequently these numbers are relative to the legal set; they must not be
combined by subtracting independently counted exclusions from a raw space.

<!-- BEGIN GENERATED MARLIN REJECTION CENSUS -->
Parsed supported CTA warp cohorts: `2|4` (threads: `64|128`).

| format | committed legal source rows | Marlin rows (parsed supported set) | rejected | rejected by CTA warps |
|---|---:|---:|---:|---|
| i4 | 1772 | 746 | 1026 | w1=206, w8=384, w16=283, w32=153 |
| i2 | 2140 | 942 | 1198 | w1=274, w8=450, w16=318, w32=156 |
| i1 | 878 | 414 | 464 | w1=130, w8=178, w16=112, w32=44 |
| **total** | **4790** | **2102** | **2688** | |

All committed rows have integral `TM/WM` and `TN/WN`.  There is exactly one
rejection reason:

| reason id | category | rows | meaning |
|---|---|---:|---|
| `MARLIN_FIXUP_COHORT_NOT_IN_SUPPORTED_SET` | `CURRENT_IMPLEMENTATION` | 2688 | The named Marlin cooperative is currently implemented and gated for the parsed set: exact 64/128-thread (2/4-warp) CTA cohorts. |
| hardware/ISA limitation | — | 0 | No rejected row reached a hardware/ISA diagnostic. |
| accidental independent enumeration rule | — | 0 | The CMake rule mirrors four fail-closed implementation assertions; it is not an otherwise unexplained second filter. |
<!-- END GENERATED MARLIN REJECTION CENSUS -->

The parsed CMake warp set is converted to threads and required to equal all
four implementation enforcement sets below.  A changed guard structure, a
non-OR condition, a duplicate value, or any CMake/kernel/wrapper/fixup mismatch
fails closed before a TSV is generated:

1. `third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin.hpp`: scheduler type admits derived/64/128 cohorts;
2. `quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_marlin.hpp`: named kernel requires 64/128 threads;
3. `benchmarks/lowbit_dense_unit.inc`: generated Marlin wrapper rejects every other row;
4. `ppu_tile_scheduler_marlin.hpp` fixup: the exact derived cohort is again required before `NamedBarrierManager` and `BlockStripedReduce` are used.

The TSV records the current line of the first CMake guard.  The checker also
resolves all four enforcement anchors, so deleting or renaming one makes the
audit fail instead of silently changing the reason.

## Constructive compile evidence

Run:

```sh
bash dev/fold_derivation/run_l131_marlin_rejected_cohorts.sh
```

`l131_marlin_rejected_cohorts.cu` takes one real committed i4 row from each
rejected cohort:

| CTA warps | threads | representative `(TM,TN,TK,WM,WN,st,bc)` |
|---:|---:|---|
| 1 | 32 | `(8,16,64,8,16,2,0)` |
| 8 | 256 | `(8,128,64,8,16,2,0)` |
| 16 | 512 | `(8,256,64,8,16,2,0)` |
| 32 | 1024 | `(32,256,64,16,16,2,0)` |

The ordinary DP compile is the positive control: all four rows compile in one
TU with zero accepted noise and zero new diagnostics.  Each negative changes
only `DENSE_MARLIN_SWEEP=1`, which is the source-level equivalent of admitting
that row beyond the CMake cohort filter.  It does **not** alter the tactic,
format, group size, artifact TileK, B-chunk or collective guards.  Each row then
fails on the authored Marlin diagnostics, with normalized counts
`scheduler=1, fixup=2, kernel=4, wrapper=4` (the four kernel/wrapper instances
are the four compiled group-size arms).

The same TU also instantiates the underlying `NamedBarrierManager<Cohort>` and
`BlockStripedReduce<Cohort,...>` types for 32/256/512/1024.  They have no
compile-time 64/128 restriction.  This, plus the ordinary DP compile of the
exact rows, establishes that the *observed rejection* is ours, not an ISA
error.  It does **not** establish that a generalized Marlin cooperative is
device-correct or fast.

## What it would take to admit the rows

This is a current-implementation backlog, not permission to delete the CMake
filter.  Admission requires, in this order:

1. generalize the scheduler/fixup cohort contract to 32/256/512/1024;
2. prove per-cohort accumulator coverage, residue poison isolation, lock reset
   and peer ordering locally;
3. compile and run numerical/lock stress on ppu001, including repeated launches;
4. only then widen the named-kernel and generated-wrapper assertions and make
   the CMake filter consume that shared supported-cohort predicate.

Until those gates exist, widening only the enumerator would merely replace a
visible rejection with an unvalidated cooperative.
