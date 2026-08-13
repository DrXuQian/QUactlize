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

<!-- BOX_RUN_POLICY_V1_BEGIN -->
{
  "schema": "quactlize-box-run-policy-v1",
  "prose_sha256": "97dee361f4a336b4ef392eddab1e64865021bde528eab513adf9d9c49dab416c",
  "dense": {
    "classic_anchor_us": "17.8",
    "historical_anchor_us": "21.14",
    "converged_recovered_fraction": "0.75",
    "sample_count": 20,
    "problem": {
      "m": 1,
      "n": 4096,
      "k": 4096,
      "l": 1,
      "group_size": 128
    },
    "decomposition": {
      "output_tiles": 32,
      "k_tiles": 32
    },
    "invocation": {
      "flags": [
        "--marlin",
        "--streamk_exact_fixture"
      ],
      "options": {
        "mode": 1,
        "alpha": "1",
        "beta": "0"
      }
    },
    "primary_cell": {
      "warp_k": 4,
      "blocks_per_cu": 1
    },
    "required_prerequisites": [
      "correctness",
      "shipping_artifact_roundtrip",
      "exact_fixture",
      "lock_fingerprints_8_stable"
    ],
    "wk1_admission": {
      "byte_map_total": 8192,
      "byte_map_diff": 0
    },
    "cells": [
      {
        "warp_k": 1,
        "blocks_per_cu": 1,
        "role": "shipping_default_control"
      },
      {
        "warp_k": 1,
        "blocks_per_cu": 2,
        "role": "scheduler_diagnostic"
      },
      {
        "warp_k": 1,
        "blocks_per_cu": 4,
        "role": "scheduler_diagnostic"
      },
      {
        "warp_k": 1,
        "blocks_per_cu": 6,
        "role": "scheduler_diagnostic"
      },
      {
        "warp_k": 2,
        "blocks_per_cu": 1,
        "role": "compile_negative"
      },
      {
        "warp_k": 2,
        "blocks_per_cu": 2,
        "role": "compile_negative"
      },
      {
        "warp_k": 2,
        "blocks_per_cu": 4,
        "role": "compile_negative"
      },
      {
        "warp_k": 2,
        "blocks_per_cu": 6,
        "role": "compile_negative"
      },
      {
        "warp_k": 4,
        "blocks_per_cu": 1,
        "role": "primary"
      },
      {
        "warp_k": 4,
        "blocks_per_cu": 2,
        "role": "occupancy_scheduler_diagnostic"
      },
      {
        "warp_k": 4,
        "blocks_per_cu": 4,
        "role": "occupancy_scheduler_diagnostic"
      },
      {
        "warp_k": 4,
        "blocks_per_cu": 6,
        "role": "occupancy_scheduler_diagnostic"
      }
    ]
  },
  "gemv": {
    "minimum_claimable_us": "0.01",
    "timer_normalization_us": "0.001",
    "sample_count": 20,
    "publication_partial_space": false,
    "resolution_rule": {
      "require_runner_up": true,
      "require_disjoint_bands": true,
      "max_unresolved_quanta": "1"
    },
    "incumbent_rules": {
      "geometry_cta_m": {
        "S068": 1,
        "S069": 1,
        "S070": 1,
        "S071": 1,
        "S072": 2,
        "S073": 2,
        "S074": 2,
        "S075": 2,
        "S076": 3,
        "S077": 3,
        "S078": 3,
        "S079": 3,
        "H-G8-2048": 1,
        "D-EXT-O": 1,
        "D-EXT-K1024": 1,
        "D-EXT-Q": 1,
        "D-4096": 1
      },
      "format_axes": {
        "int4": {
          "layout": "native",
          "step_k": 16,
          "threads": 128,
          "cta_n": 8,
          "chunk": 2
        },
        "int2": {
          "layout": "native",
          "step_k": 16,
          "threads": 128,
          "cta_n": 8,
          "chunk": 2
        },
        "q3": {
          "layout": "native",
          "step_k": 32,
          "threads": 64,
          "cta_n": 8,
          "chunk": 2
        },
        "q6": {
          "layout": "native",
          "step_k": 16,
          "threads": 128,
          "cta_n": 8,
          "chunk": 2
        }
      }
    },
    "base_census": {
      "total": 27360,
      "legal": 10260,
      "pruned": 17100,
      "prune_reasons": {
        "STEP_TOO_SMALL_FOR_SPARSEST_PLANE": 10944,
        "CTA_N_NOT_WHOLE_CHUNKS": 6156
      }
    },
    "full_manifest": {
      "jobs": 86,
      "total": 165600,
      "legal": 63180,
      "pruned": 102420,
      "prune_reasons": {
        "STEP_TOO_SMALL_FOR_SPARSEST_PLANE": 64512,
        "CTA_N_NOT_WHOLE_CHUNKS": 37908
      }
    },
    "smoke_manifest": {
      "jobs": 18,
      "total": 18288,
      "legal": 11430,
      "pruned": 6858,
      "prune_reasons": {
        "CTA_N_NOT_WHOLE_CHUNKS": 6858
      }
    }
  }
}
<!-- BOX_RUN_POLICY_V1_END -->

<!-- BOX_RUN_POLICY_MIRROR_V1_BEGIN -->
policy_sha256=445b5c9eaf60fca421312408c98b73411d80e34b838055fb1f7939f6fc723eef
dense anchors_us=17.8/21.14 recovered_fraction=0.75 derived_gap_us=3.34 derived_boundary_us=18.6350 samples=20
dense problem={"group_size":128,"k":4096,"l":1,"m":1,"n":4096} decomposition={"k_tiles":32,"output_tiles":32} invocation={"flags":["--marlin","--streamk_exact_fixture"],"options":{"alpha":"1","beta":"0","mode":1}}
dense primary=WK4/B1 cells=WK1/B1:shipping_default_control,WK1/B2:scheduler_diagnostic,WK1/B4:scheduler_diagnostic,WK1/B6:scheduler_diagnostic,WK2/B1:compile_negative,WK2/B2:compile_negative,WK2/B4:compile_negative,WK2/B6:compile_negative,WK4/B1:primary,WK4/B2:occupancy_scheduler_diagnostic,WK4/B4:occupancy_scheduler_diagnostic,WK4/B6:occupancy_scheduler_diagnostic
dense prerequisites=correctness,shipping_artifact_roundtrip,exact_fixture,lock_fingerprints_8_stable wk1_byte_map=0/8192
gemv minimum_claimable_us=0.01 timer_normalization_us=0.001 samples=20  publication_partial_space=false resolution_rule={"max_unresolved_quanta":"1","require_disjoint_bands":true,"require_runner_up":true} incumbent_rules={"format_axes":{"int2":{"chunk":2,"cta_n":8,"layout":"native","step_k":16,"threads":128},"int4":{"chunk":2,"cta_n":8,"layout":"native","step_k":16,"threads":128},"q3":{"chunk":2,"cta_n":8,"layout":"native","step_k":32,"threads":64},"q6":{"chunk":2,"cta_n":8,"layout":"native","step_k":16,"threads":128}},"geometry_cta_m":{"D-4096":1,"D-EXT-K1024":1,"D-EXT-O":1,"D-EXT-Q":1,"H-G8-2048":1,"S068":1,"S069":1,"S070":1,"S071":1,"S072":2,"S073":2,"S074":2,"S075":2,"S076":3,"S077":3,"S078":3,"S079":3}}
gemv base_census total=27360 legal=10260 pruned=17100 reasons=CTA_N_NOT_WHOLE_CHUNKS:6156,STEP_TOO_SMALL_FOR_SPARSEST_PLANE:10944
gemv full_manifest jobs=86 total=165600 legal=63180 pruned=102420 reasons=CTA_N_NOT_WHOLE_CHUNKS:37908,STEP_TOO_SMALL_FOR_SPARSEST_PLANE:64512
gemv smoke_manifest jobs=18 total=18288 legal=11430 pruned=6858 reasons=CTA_N_NOT_WHOLE_CHUNKS:6858
<!-- BOX_RUN_POLICY_MIRROR_V1_END -->

The JSON block above is the sole machine authority.  The mirror is generated
from it and checked byte-for-byte; the block also pins the SHA-256 of all text
outside the block.  A missing/duplicate block, an edited mirror, or prose drift
therefore makes adjudication `VOID` rather than selecting a fallback threshold.

## Common identity and durable evidence

Every published result must carry all of the following in one committed result
directory:

- root Git SHA, recursive submodule status, and actlize SHA;
- binary SHA-256 of the linked executable actually launched;
- device model, PCI identity, driver version, and SDK/compiler identity;
- the complete argv, protocol/sample count, command exit status, and the
  unedited stdout/stderr log;
- for GEMV, the raw machine-readable samples used by the summary; for dense,
  the unedited harness log containing its 20-independent-event aggregate.

The binary itself need not be committed.  Its hash does.  Temporary paths under
`/tmp` are staging, not evidence.  After identity and coverage are audited, copy
the files without rewriting them beneath:

```text
dev/box_runs/<root-sha>/dense-marlin-wk4/
dev/box_runs/<root-sha>/gemv-sweep/<binary-sha256>-samples20/
```

For dense Marlin this means the build log, the top-level runner log, every
`bpc*.log` or `bpc*.not-run`, `illegal-bpc.log`, `wk1-admission.log`,
`commands.jsonl`, and `submodule-status.txt`.  For GEMV it means
`manifest.json`, `raw.jsonl`, `progress.jsonl`, `result.json`, `run.log`, the
per-job `logs/` directory, the dry-run audit/summary, `base-census.json`,
`base-census-authority.log`, `build.log`, `runner.log`, `commands.jsonl`, and
`submodule-status.txt`.  A screen-only PASS or winner is not a result.

`tools/adjudicate_box_runs.py` accepts those result directories, not a hand
written verdict JSON.  Each directory must also contain `provenance.json` with
`root_sha`, `submodule_status`, `actlize_sha`, `binary_sha256`, `device_model`,
`pci_identity`, `driver_version`, `sdk_compiler_identity`, the complete `argv`,
`runner_exit_status`, and `protocol_sample_count`.  The `commands` array is an
exact copy of `commands.jsonl`, not a summary reconstructed after the run.
Dense additionally carries `wk1-admission.log`.  Its L143 portion is the exact
expected output of a locally executable host oracle, committed beneath the
result SHA and retrieved with `git show <root_sha>:dev/fold_derivation/`
`l143_wk4_production_delivery.expected.txt`; it is explicitly **not** a fresh
box execution.  The local tier executes that oracle and byte-compares its output
with the committed file, while the box contributes the separately recorded
static target check.  Both frozen runners now synthesize
these files themselves and fail closed on a dirty root/submodule, missing
identity, missing command evidence, or a non-20-sample policy.  Missing or
inconsistent fields produce a named `VOID`.  Publication mode reads this policy
with `git show <root_sha>:<path>`, so a newer worktree cannot reinterpret an
older binary.  It also byte-compares both `tools/adjudicate_box_runs.py` and
`benchmarks/sweep_gemv_perf.py` with those files at the result SHA; a changed
reader or raw-event analyser must adjudicate from a clean checkout of that SHA,
not silently reinterpret the old bundle.  `--policy` and the normalized observation adapter exist only
behind explicit `--fixture-mode` for the synthetic CI controls.

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

The dense harness currently publishes `n/median/mean/min/max` but not its 20
individual event words.  The adjudicator can validate the independent-pair
count and consume that aggregate; it cannot independently recompute the dense
median.  This is a declared evidence limit, not permission to call the summary
raw samples.  The 20-sample `[min,max]` band is printed beside the median.  If it crosses a
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
deliberate compile-time negative in L142, not a missing measurement.  The box
runner does not rerun L142, so the adjudicator labels these cells
`PREREGISTERED_COMPILE_NEGATIVE_NOT_RERUN` rather than pretending they are fresh
box evidence.

The permanent WK1 controls must still say that the new axis leaves the shipping
builder type and xplane bytes unchanged (`0/8192` byte-map differences).  If a
WK1 device control is included, its output must be raw-bit-identical to the
historical arm.  Any WK1 type/byte/output drift invalidates the whole WK4 batch:
the experiment would then mix a changed baseline with a changed candidate.

For every supported WK4 B point, the mechanical adjudication records
median/min/max and `G/I/active/idle`, handoffs and max peers.  The optional ACU
pass separately records achieved warp/CU, registers/thread, block limits, and
named stalls/counters; those counters are unregistered observations and do not
alter the primary timing verdict.  The B ladder is judged as follows:

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
  complete and the sealed JSON rule says `RESOLVED`: a runner-up exists, the
  raw bands do not overlap, the timer quantum is known, and the gap is greater
  than one quantum.  The analyser supplies decoded samples, ranking facts, and
  the inferred quantum; its convenience verdict/reasons are not a second
  decision authority.  Otherwise report the observed leader but label it
  `UNRESOLVED`; it is not a shipping-routing decision.
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
