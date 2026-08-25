# PPU Split-K shared partial epilogue

Use this case when a producer's completed fp32 accumulator is exact but its
Split-K workspace contains intermittent, stripe-aligned corruptions.  The
authoritative repository record is
`dev/fold_derivation/Q4K_FQ_SPLITK_PARTIAL_EPILOGUE_ROOT_CAUSE.md`.

Current status: production repair closed; internal root incomplete.  The
shared prefix factorial localizes a trigger in the custom S>1 kernel but does
not explain ordinary S1 correctness, because shipping S1 is a different
compiled kernel and output epilogue.

## Frozen incident

```text
shape       M/N/K=1/1024/5120, group_size=32
format      Q4_K, ArtifactTileK=64
tactic      TM/TN/TK=8/64/256, WM/WN=8/16, stages=2
providers   standard-aiu and packed-row
splits      S=2 and S=4
```

Canaries remained intact, while bad values appeared as complete 32-output
fp32 stripes in changing split planes.  Independent per-plane host reduction
proved that the corruption was already present in producer partials.

## Decisive prefix localization (not a cross-context root)

Keep the real mainloop and append one prefix at a time before the proven direct
accumulator-to-workspace store:

- opaque accumulator liveness, an extra register clone and CTA-only sync were
  clean;
- one exact-once flat constant store to disjoint shared storage was the first
  arm to fail;
- a flat live-accumulator store and vectorized R2S variants also failed;
- the historical shared R2S/barrier/S2R output path reproduced strongly;
- the production direct store remained clean.

The device verdict was `SHARED_STORE_BACKEND_OR_FOOTPRINT_CAUSAL` within the
frozen custom S>1 context.  Exact codegen kept 16 MMA instructions and added
one vector TSM store at the first failing arm.  No stack allocation was
reported.  This localizes the first corrupting operation to the extra
shared-store lowering or its compiler/scoreboard footprint; it does not prove
that this operation is sufficient in ordinary S1, nor justify blaming the
CuTe layout or reducer.

## Do not diagnose this as a missing barrier

Ordinary S=1 epilogues are clean controls.  In the failing Split-K producer,
the mainloop already ended with `cp_async_wait<0>() + __syncthreads()`, and the
legacy epilogue synchronized on both sides of its shared readback.  Historical
user, reserved epilogue, and full CTA barrier variants all failed; an extra
pre-R2S CTA barrier and physically disjoint shared storage also failed.

The first failing prefix was an exact-once constant write to disjoint shared
memory, bracketed by CTA barriers and never read, before the correct direct
store.  It cannot be repaired or explained by adding a logical producer/
consumer barrier.  The honest remaining boundary is a Split-K-kernel-specific
codegen/scoreboard interaction with the additional vector TSM store.  Do not
claim that the split count alone is causal: S=1 delegates to a different
shipping kernel and output epilogue.

A host observation made only after device completion still saw the wrong FP32
values in individual producer planes.  This rejects a producer/reducer stream
publication gap as the explanation for this incident; cross-kernel ordering
cannot repair a value already corrupted inside the producer.

If distinguishing "the custom Split-K kernel context" from "runtime S>1" is
material, use the custom fixed Split-K kernel itself at S=1 as the control.
Do not compare against the shipping S=1 launcher and call the split integer
causal; those arms are different instantiated kernels.

Use `tools/run_fq_q4k_custom_split_count_box.sh` for that control.  It freezes
the generated AP0/AP1 symbols and runs the same custom kernel at runtime
S=1/2/4 in both direct-store and legacy-shared binaries.  Interpret only its
registered verdicts:

- `RUNTIME_SPLIT_DECOMPOSITION_NECESSARY`: custom legacy S1 is clean and
  custom legacy S2/S4 fail;
- `CUSTOM_KERNEL_CONTEXT_SUFFICIENT`: custom legacy S1/S2/S4 fail;
- `PROVIDER_DEPENDENT_CUSTOM_S1`: S1 differs across AP0/AP1 and the simple
  cross-provider explanation remains incomplete.

## Repair and controls

Publish each completed accumulator directly to its split-major fp32 workspace
through the production `TiledMma::partition_C` ownership.  This removes the
unnecessary R2S/barrier/S2R roundtrip while preserving accumulator order,
workspace bytes and deterministic reduction.  Keep the historical shared
path only as an exact negative.

Required closure:

- direct-store ownership oracle is exact-once, with hole/duplicate and rotated
  fragment negatives;
- legacy shared output reproduces corruption at high repeat count;
- direct production path is raw-bit exact for both A providers and S=2/S=4;
- same-run producer timing does not regress.

Do not retain cadence changes, alternate barriers or metadata-owner changes as
repairs: they changed incidence rates but did not close the row.

Audit adjacent paths rather than blanket-rewriting them.  In particular,
grouped MoE Split-K still runs an ordinary converting pointer-array epilogue
per slice, and generic vendor `GemmUniversalParallel` can still instantiate
`EpilogueParallel`.  Their current numerical tests are useful controls but do
not substitute for a high-repeat per-plane raw-bit/canary closure.
