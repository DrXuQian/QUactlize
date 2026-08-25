# PPU Split-K shared partial epilogue

Use this case when a producer's completed fp32 accumulator is exact but its
Split-K workspace contains intermittent, stripe-aligned corruptions.  The
authoritative repository record is
`dev/fold_derivation/Q4K_FQ_SPLITK_PARTIAL_EPILOGUE_ROOT_CAUSE.md`.

Current status: the rare direct-store failure remains open, but its value is
now classified exactly as a stale previous A tile.  The shared prefix
factorial localized one earlier trigger.  A later 8192-repeat control found
the direct-store packed-row S4 cell wrong once at repeat 2142 (`raw_bad=32`).
Separating the packed m8 stages' complete physical x4 footprints at
`ca01dc6` did not close it: the same AP1/S4 row failed again at repeat 714 with
the wrong FP32 producer plane (`1.0 -> 6.0`).  Do not cite the stage-pitch
change as the root or as device-closed correctness evidence.

## Refuted complete-root hypothesis: compressed physical stage footprint

`PPU0010_TSM_LD_SWZL_M8` publishes two registers (v0/v1) to the logical m8
CuTe fragment, but executes an `m8n8.x4` shared load and physically reads four
registers.  `Copy_Traits` describes the values consumed by MMA, but that
logical x2 source layout was incorrectly reused as a storage-lifetime
authority.  The packed-row provider placed consecutive pipeline stages only
`4 * 64 = 256` fp16 elements apart.

The exact failing TM8/TK256/Stages2 footprint oracle found:

```text
historical_stage_pitch=256  cross-stage physical read/write addresses=432
repaired_stage_pitch=1216   cross-stage physical read/write addresses=0
```

The 432 addresses are reads into hidden/discarded x4 rows that can overlap the
other stage's cp.async row-0 writes.  This is a real physical-footprint
fragility that the logical x2 CuTe view does not encode, so keeping the wider
stage pitch is defensible structural hardening.  It is not the complete root:
the frozen incident reproduced after the overlap count became zero.

The repair keeps the 64-half cube pitch but gives stages separate physical
footprints:

```text
stage_pitch = cube_pitch * (cubes_per_stage - 1) + physical_cube_span
            = 64 * 3 + 16 * 64 = 1216 fp16 elements
```

For TK256/S2, packed A grows from about 2.9 KiB to 4.75 KiB, still far below
the ordinary 16 KiB A allocation.  Both the writer and the m8 copy atom carry
this stage pitch.  L186 proves the physical crossing was removed; it does not
prove the intermittent numerical failure was removed.

## Exact stale-A incident fingerprint

`l224_fq_packed_m8_prepare_consume_layout.cu` composes the exact failing
TM8/TK256/WM8/WN16 CuTe A fragment and copy views.  It proves:

```text
MMA K atoms = 16, A copy blocks = 16, B deliveries = 4
A copy block i maps exactly to MMA atom i
prepare(next B delivery) vs consume(current B delivery): zero logical offsets overlap
```

For the frozen fixture, AP1/S4 plane 2 at output column 32 contains
superblocks 10..14.  Their expected contributions are:

```text
+3, -14, +1, -4, +15  -> 1.0
```

Replacing exactly superblock 13's A tile with the preceding A tile changes
only `-4 -> +1`, producing the observed `6.0`.  None of the other four
single-tile stale substitutions produces 6.  B repeats its code pattern every
256 K elements, while the fixture's active A offset moves by 37, so this is a
specific previous-A-tile fingerprint rather than a generic arithmetic match.
The remaining boundary is therefore A delivery/publication or a physical
register/backend lifetime not represented by the exact logical CuTe views.

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
that this operation is sufficient in ordinary S1 and is superseded as the
explanation for the rarer direct-store packed-row failure.

## Reopened direct-store evidence

The first custom split-count run produced:

```text
direct standard-aiu S2/S4: clean at 8192 repeats
direct packed-row  S2:    clean at 8192 repeats
direct packed-row  S4:    raw_bad=32 at repeat 2142
custom S1, both providers: REAL_CAN_IMPLEMENT (not launched)
```

The S1 rejection was a diagnostic ABI error: CUTLASS canonicalizes the batch
stride to zero when `L=1`, while the physical Split-K plane ABI requires
`stride_L=M*N` even for one plane.  Construct that stride explicitly before
interpreting any custom-S1 result.

On a direct-store failure, preserve the original producer+reducer cadence,
then snapshot the already completed FP32 workspace and rerun the reducer:

- wrong FP32 plane bytes prove producer/mainloop/direct-store corruption;
- exact FP32 planes plus a clean reducer replay prove a same-stream
  publication gap;
- exact planes plus a still-wrong replay keep the reducer open.

This post-failure observation must not insert a synchronization or host copy
before the original failure, because doing so can suppress a publication bug.

## Do not diagnose this as Split-K synchronization

Ordinary S=1 epilogues are clean controls.  In the failing Split-K producer,
the mainloop already ended with `cp_async_wait<0>() + __syncthreads()`, and the
legacy epilogue synchronized on both sides of its shared readback.  Historical
user, reserved epilogue, and full CTA barrier variants all failed; an extra
pre-R2S CTA barrier and physically disjoint shared storage also failed.

The first failing prefix was an exact-once constant write to disjoint shared
memory, bracketed by CTA barriers and never read, before the correct direct
store.  It cannot be repaired or explained by adding a logical producer/
consumer barrier.  For that earlier shared-output trigger, the narrow boundary
is a Split-K-kernel-specific codegen/scoreboard interaction with the additional
vector TSM store.  It is not the explanation for the later direct-store
packed-row S4 failure.  Do not claim that the split count alone is causal: S=1
delegates to a different shipping kernel and output epilogue.

A host observation made only after device completion still saw the wrong FP32
values in individual producer planes.  This rejects a producer/reducer stream
publication gap as the explanation for this incident; cross-kernel ordering
cannot repair a value already corrupted inside the producer.

Each producer CTA writes a distinct split plane.  No producer-to-producer
publication is required, and ca01dc6 removed the proposed cross-stage physical
alias without closing correctness.  Do not add a global fence, counter, or
inter-CTA barrier.  Isolate the intra-CTA A-stage source instead.

If distinguishing "the custom Split-K kernel context" from "runtime S>1" is
material, use the custom fixed Split-K kernel itself at S=1 as the control.
Do not compare against the shipping S=1 launcher and call the split integer
causal; those arms are different instantiated kernels.

Use `tools/run_fq_q4k_custom_split_count_box.sh` for the current closure.  It
freezes the generated AP0/AP1 symbols and runs the same custom S=1/2/4 kernel.

The completed eight-arm factorial at `3e83a45` reported:

```text
baseline/prepare-after/explicit-stage/direct-x4/separate-group/sync-store:
    0/8 clean S>1 cells
compiler-fence: 4/8 clean
A-before-B:     6/8 clean
```

Every dirty cell localized to a wrong producer FP32 partial; none of the seven
counterfactuals closed both providers and S=2/S=4.  The two partially cleaner
arms are code-layout/timing sensitivity, not repairs or causal proof.

The current first reduced closure builds two arms: baseline and an exact
inline-asm shared-memory contract.  The candidate keeps the physical x4
instruction and adds `memory` clobbers to the fp16 AIU/packed cp.async
producer, commit/wait and m8 ldmatrix consumer.  Numeric rc=1 is retained;
both arms and independent attempts complete before a fail-closed verdict is
emitted.  A logical-x2 scalar load that removes the physical x4 swizzle opcode
is source-proved by L186 (512 exact logical addresses plus an offset negative),
but is deliberately reserved for a separate next run if the contract arm
remains dirty.

Missing memory side-effect declarations are a real inline-asm contract defect,
but do not by themselves prove the stale-stage explanation.  A compiler may
hoist or CSE pure address arithmetic only while preserving the runtime stage
value.  Call the contract causal only if the x4-preserving memory-contract arm
closes while baseline reproduces.  If it remains dirty, test the reserved
opcode-removing arm as the next single-variable closure.

## Repair and controls

Publish each completed accumulator directly to its split-major fp32 workspace
through the production `TiledMma::partition_C` ownership.  This removes the
unnecessary R2S/barrier/S2R roundtrip while preserving accumulator order,
workspace bytes and deterministic reduction.  Keep the historical shared
path only as an exact negative.

Independently, packed m8 A should keep pipeline stages separated by the
complete x4 hardware footprint even though its CuTe value layout exposes only
x2.  This closes a genuine physical-overlap fragility but must not be presented
as the numerical repair after ca01dc6 reproduced.

Required closure (minimum 8192 correctness repeats for this frozen row):

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
