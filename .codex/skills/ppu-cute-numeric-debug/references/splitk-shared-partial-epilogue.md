# PPU Split-K shared partial epilogue

Use this case when a producer's completed fp32 accumulator is exact but its
Split-K workspace contains intermittent, stripe-aligned corruptions.  The
authoritative repository record is
`dev/fold_derivation/Q4K_FQ_SPLITK_PARTIAL_EPILOGUE_ROOT_CAUSE.md`.

Current status: the rare direct-store failure remains open and is localized to
one wrong 32-output band in a completed producer FP32 partial.  That places the
defect somewhere from operand delivery through accumulator state to the direct
store; it does not separately observe or certify the accumulator.  The shared
prefix factorial localized one earlier, independent trigger.  A later
8192-repeat control found the direct-store packed-row S4
cell wrong once at repeat 2142 (`raw_bad=32`).  Separating the packed m8
stages' complete physical x4 footprints at `ca01dc6` did not close it.  One
captured value pair (`1.0 -> 6.0`) admits a stale previous A tile, but the full
failure family does not: another captured delta (`-6`) cannot be produced by
any complete A-tile substitution in the exact fixture.  Do not cite either
the stale-A match or the stage-pitch change as the root.

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

## One stale-A-compatible signature is not the family root

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
single-tile stale substitutions produces 6, so that one value pair is
compatible with stale A.

Do not generalize it.  L224 now enumerates all K superblocks, all eight fixture
column residues and every complete source-A/destination-B tile pairing.  The
observed `+5` and `-5` deltas occur for adjacent previous-A substitutions and
`-4` occurs for some non-adjacent complete-A substitutions, but `-6` occurs
for none.  Therefore stale complete-A delivery cannot classify the entire
incident family.  The honest remaining boundary is operand delivery,
MMA/accumulator state, or a backend lifetime not represented by the exact
logical CuTe views.

The same oracle binds the 32-output footprint to `partition_C`: for M=1 each
of the four warps owns 16 consecutive live N values, so an aligned 32-value
band is two adjacent output warps.  This footprint alone does not identify an
A/B copy issuer or prove a wait/barrier defect.

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

## Separate Split-K publication from intra-CTA operand publication

Ordinary shipping S=1 epilogues are clean controls, but they are a different
kernel.  In the failing custom producer,
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
inter-CTA barrier.  The remaining synchronization question is narrower and
intra-CTA: whether PPU async-shared completion needs an explicit proxy fence
before ordinary shared consumers, in addition to the established
`cp_async_wait + __syncthreads` protocol.

If distinguishing "the custom Split-K kernel context" from "runtime S>1" is
material, use the custom fixed Split-K kernel itself at S=1 as the control.
Do not compare against the shipping S=1 launcher and call the split integer
causal; those arms are different instantiated kernels.

The first custom-S1 runner did route S1 to the custom type, but its verdict
parser silently discarded that row.  A claim that the defect is S>1-only was
therefore not established.  The current runner requires an explicit S1 census
for both providers and treats a candidate as closed only when S1 and every
S2/S4 cell are clean.

Do not use `PPU_PACKED_SPLIT_GROUPS` to adjudicate A/B wait ownership.  On
PPU0010, `cute::copy_aiu` has warp 0 issue both ordinary A and B; the warp0/A,
warp1/B split is the non-HGGC branch.  Packed A is issued by its explicit
logical copy threads, while wrapped extra-input partitions also give CTA
threads metadata-copy work.  `PPU_PACKED_SPLIT_GROUPS` only redistributes
packed metadata decode groups between duplicate metadata owners.  It may
change cadence, but it does not change the A/B issuer/wait relation.

The Stages2 driver also issues one redundant final-tile prefetch into an
already retired, physically disjoint stage while draining.  Removing it is a
useful cadence/traffic ablation, but it changes the number of async operations
without identifying a violated address or ownership relation.  Do not accept a
clean no-final-prefetch arm as a root-cause repair.  Reserve it for a second
localization step only if the exact proxy-fence and logical-x2 consumer arms
both remain dirty.

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

The first reduced closure at `d6e5589` was initially described as an exact
inline-asm shared-memory contract.  Source re-audit proved that description
false.  The macro covered the ordinary fp16-A AIU producer, packed-A zfill
producer, commit/wait, and the common swizzle reader, but did **not** cover the
Q4 B bulk-AIU producer or the non-zfill packed-metadata producer.  Both still
wrote shared memory through input-only inline asm with no `memory` clobber.
Therefore its 0/8 result cannot refute the compiler-memory-contract
hypothesis; it tested only a partial chain.

The current `asm-memory-contract` arm closes the complete frozen Q4 chain:
ordinary fp16 A, Q4 B, packed A, packed metadata, commit/wait, and the common
swizzle reader.  Keep this as a separate arm from the logical-x2 consumer and
async-proxy-fence arms.  Only the complete arm may adjudicate the compiler
contract.

A logical-x2 scalar load that removes the physical x4 swizzle opcode is
source-proved by L186 (512 exact logical addresses plus an offset negative).
Another remaining arm adds only `fence_view_async_shared()` after each
`cp_async_wait<Stages-2>()` and before packed decode/CTA publication.  The
default closure runs one uncontaminated baseline plus three orthogonal arms:
complete asm-memory contract, logical-x2 consumer, and async-shared proxy
fence.  None of the candidates are combined.

Missing memory side-effect declarations are a real inline-asm contract defect,
but do not by themselves prove the stale-stage explanation.  A compiler may
hoist or CSE pure address arithmetic only while preserving the runtime stage
value.  Do not cite the earlier partial contract arm as a device refutation.
The current closure separately tests the complete contract, physical m8 x4
consumer, and async-shared proxy visibility.  One uniquely clean arm selects
the next repair; multiple clean arms are cadence-sensitive and remain
unadjudicated; all dirty arms retire all three seams.

The `plane2: 1.0 -> 6.0` value pair remains valuable evidence for a specific
stale previous A tile, but physical output index 32 is not part of that
mechanism identity.  At `d6e5589`, baseline AP1/S4 attempt 2 reproduced the
same output and producer-plane bits at aligned index 576.  The first runner
incorrectly called this nonreproduction because it required index 32.  Freeze
the symbol, fixture, provider, S4 cell, one-hot plane and value pair; require
the output and partial indices to agree and remain 32-aligned, but print the
specific index separately.  Baseline admission is no longer tied to that exact
cell or value pair: it requires producer-partial corruption in all four
AP0/AP1 x S2/S4 cells across the independent attempts.  The historical value
pair remains a diagnostic field only.

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
- the same custom producer path is raw-bit exact for both A providers at
  S=1/2/4;
- same-run producer timing does not regress.

Do not retain cadence changes, alternate barriers or metadata-owner changes as
repairs: they changed incidence rates but did not close the row.

Audit adjacent paths rather than blanket-rewriting them.  In particular,
grouped MoE Split-K still runs an ordinary converting pointer-array epilogue
per slice, and generic vendor `GemmUniversalParallel` can still instantiate
`EpilogueParallel`.  Their current numerical tests are useful controls but do
not substitute for a high-repeat per-plane raw-bit/canary closure.
