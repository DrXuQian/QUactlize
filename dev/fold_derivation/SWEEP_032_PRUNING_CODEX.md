# Sweep 032: Codex's independent pruning rules

This is the independent answer requested by INBOX 032.  It was written before INBOX 033 existed and deliberately
does not try to anticipate the other rule set.

INBOX 032b arrived while this answer was being committed and was applied only after the independent record landed.
It carries a user scope decision, not the other rule set: the stage axis is exactly `{2,4}` and stages above 4 are
out.  The rules below reflect that scope.

The winner must be named per `(operator, stored schema, M)`.  A format is not a tactic for another format, and a
dense result is not a grouped result.  Rules below marked **EXACT** cannot remove the winner under their stated
scope.  **MEASURED** means the repository has direct device evidence but not a proof over this whole space.
**BELIEVED** is a pruning hypothesis; each such rule includes the condition that falsifies it.

## Two corrections before pruning

### Same bytes do not make two TileK kernels duplicates -- **EXACT**

The 1023-cell "pinned" count in the first `size_sweep.cpp` is not a free reduction of kernel instantiations.  It
confuses two questions:

1. Do two candidates require two stored weight buffers?
2. Do two candidates instantiate the same kernel?

For an unfolded arrangement, `PlacedArrangement.layout_is_tile_free()` explicitly permits every compatible
`TK <= 256`.  Those kernels still have different A-smem (`TM*TK*2`), K-loop counts and AIU run widths.  The
repository has measured TileK move time materially: grouped i4 prefill improved 12.4% from TK64 to TK32, while
grouped decode improved 21% in the other direction from TK32 to TK256.  Equal bytes can eliminate a repack; they
cannot eliminate either timing.

For a folded arrangement, the resident descriptor pins TK.  For an unfolded arrangement, retain every compatible
TK tactic.  For either case, compare the actual byte descriptor, not just `(F_lo,F_hi)`: folded `xplane` also
depends on warp-columns-per-tile, and `xplane_hi` records deliveries and folded high rows.  This is lossless because
it removes only consumers that cannot read the one resident buffer; it does not select among consumers that can.

### The stage axis is `{2,4}` -- **USER SCOPE DECISION**

032b sets the axis to stages 2 and 4 and removes every stage above 4.  Stage 3 has historical measured winners, so
its omission must be described as scope, not as measured dominance.  Under the user-defined axis and without the
invalid TileK de-duplication, the unpruned count is 2331 instantiations per operator.

## Lossless rules

### L1. Apply legality and exact resident-descriptor compatibility first -- **EXACT**

Keep the existing ordered kernel/topology/producer exclusions.  After choosing the tensor's resident arrangement,
also reject a tactic whose placement descriptor differs from those bytes.  It cannot be the winner because it
cannot consume the tensor being timed.  If the experiment intentionally compares two descriptors, pack and label
them as two arrangement candidates; do not disguise that comparison as two tactics over one buffer.

F remains derived from `(bits,TK)`.  Over-folds remain out because no online consumer selects them.  This rule must
use the byte-equivalence record, not fold equality alone.

### L2. Scope the 025 crossover experiment to one stored schema -- **EXACT for 025**

025 asks whether one tensor's optimum moves with M.  Choose one schema and run both operators' complete candidate
rules for that schema.  Compiling the other five schemas cannot change that tensor's winner because they are not
legal interpretations of its bytes.  The conclusion must be reported as format-local.  If the desired deliverable
changes to a six-schema tactic table, this rule no longer applies and all six schemas are separate strata; no one
format may prune another because format-dependent stage winners and dense/grouped reversals are already measured.

### L3. Never prune M -- **EXACT**

Run every compiled candidate at M = 1, 8, 64, 512 and 2048.  M is a runtime loop, so retaining all five values adds
no kernel instantiation.  Dropping one saves only timing seconds and can hide the crossover the experiment exists
to find.

## Axes that must remain protected

### P1. Do not prune TileK among tactics compatible with the resident bytes -- **MEASURED**

TileK changes three independent mechanisms: A-smem, K-loop depth and AIU contiguous-run width.  Its preference has
already reversed between grouped prefill and decode.  The only exact TileK exclusion is resident-descriptor
incompatibility from L1.  In particular, F=1 does not authorize selecting one representative TK.

### P2. Do not prune TileM -- **MEASURED**

TileM stays a full axis in the primary search.  This directly preserves the repository's recorded lever:
`A-smem = TM*TK*2` per stage.  The grouped prefill winner was the interior TM64 point, not an endpoint, while
decode moved to TM16.  TileM also trades masked rows and CTA count against that footprint, so neither "smallest"
nor "largest" is a dominance rule.

The later decode result that TK moved more than all tile knobs does not invalidate this rule.  It applies to one
decode band, and even there TM16 bought 7.8%; it does not prove that TM is inert at M=64..2048.  Therefore no hard
TileM pruning is part of this proposal.

### P3. Do not globally prune TileN or WarpN -- **MEASURED**

The recorded preference changes with operator and regime: WN32 beat WN64 on dense measurements, WN64 beat WN32
by 13% in grouped Q6 prefill, and grouped decode eventually won at WN16/TN32.  TileN changes grid size, reuse,
shared footprint and tail behavior.  There is no measured monotone direction that is common to both operators and
all M, so these axes may be sampled adaptively but cannot be hard-pinned globally.

## Practical, falsifiable pruning rules

The following rules make the build runnable.  They are not mathematical dominance proofs.  A result produced with
them is a measured winner of the guarded search; it becomes a full-space claim only after every guard below stays
negative.

### H1. Use the largest legal WarpM for each TileM as the primary row -- **MEASURED + BELIEVED**

Primary rule: `WM = min(TM,64)`.  Keep WM16 when TM16 forces it.  Smaller voluntary WM values are guards, not part
of the initial Cartesian product.

Why it is believed not to lose the winner: converter work per mma is exactly `128/WM`.  WM16 is the measured
throughput ceiling (`cvt/mma=8`) and stayed flat across a fourfold occupancy range; WM32 crossed that ceiling, and
WM64 cost essentially nothing at the measured 272-register point.  All recorded grouped prefill winners use the
largest legal WM, and the decode winner's WM16 is forced by TM16.  The direct TM32/WM16 decode probe lost.

Failure mode: a smaller WM can create enough grid warps or relieve a register cliff to outweigh conversion work.
Guard it by compiling the next-smaller WM at the resource-light and resource-heavy N shapes for every
`(operator,schema,TK)` stratum.  Because those kernels run all M, this costs no M cross-product.  If either guard is
inside the winner's confidence set at any M, expand all legal WM values for that stratum before selecting a winner.

### H2. Use `TileN/WarpN = 2` as the primary N geometry -- **MEASURED + BELIEVED**

Ratio two is the common geometry of the recorded grouped prefill winners and the final grouped decode winner, at
different absolute TN/WN values.  It balances two warp-columns of reuse against CTA count; the stored low-plane
placement also treats this ratio as the relevant geometry when folding is active.

Why it is only believed: neither the resource equations nor byte equivalence prove that ratios one or four cannot
win, and P3 records operator/regime reversals in the absolute WN.  Therefore keep the nearest legal ratio-one and
ratio-four rows as guards for every `(operator,schema,TK)` stratum, at the TileM values that minimize and maximize
A-smem.  If a guard enters the confidence set at any M, expand all legal TN/WN pairs for that stratum.  This rule
must not be used to redefine the artifact silently: a folded guard with a different descriptor is a separately
labelled arrangement candidate.

### H3. Fully cross both allowed stages; make no shared-memory cut -- **EXACT for the user-defined axis**

Stages 2 and 4 stay on every primary geometry.  There is no deep-stage sampling rule because stages above 4 are
outside the requested space, and there is no `smem <= 128KB` or minimum-blocks rule.  Such a rule under `{2,4}`
would mostly remove large TileM rows and would therefore violate P2 while pretending to prune pipeline depth.

Keeping s2 explicitly addresses the occupancy caveat.  The d128 FA result is a measured cross-kernel warning:
FA's 64KB block is already limited to four blocks/25% occupancy, and adding shared storage dropped it to three
blocks and made it slower even after halving a memory stall.  It does not prove s4 loses on GEMM, but it rules out
keeping only the deeper allowed stage.  Both s2 and s4 are therefore crossed everywhere; neither is inferred from
the other.

### H4. Make pruning adaptive, not a one-shot truncated grid -- **BELIEVED process rule**

Build the primary rows from H1/H2 with all protected TileM, compatible TileK and both stages, plus every guard.  Run
all M.  Expand the affected stratum whenever a guard is competitive; do not merely annotate the missing cells
after choosing a winner.

This cannot give a mathematical guarantee against an arbitrary high-order interaction.  Its argument is that each
excluded mechanism has an explicit falsifier in the same binary, and expansion happens before the result is called
a winner.  The claim should therefore state which guards were negative.  If the user requires an unconditional
winner over the user-defined Cartesian product, no performance pruning rule is honest: all 2331 kernels per operator
must be built.

### H5. Use confidence-set elimination, not one timing -- **MEASURED**

The same configuration has shown 13% cross-run spread, and cold single-row launches differ from warmed sweep rows.
Time candidates interleaved in one process and repeat blocks.  A guard is negative only when its uncertainty band
is disjoint from the current best; a point inside the band survives and triggers expansion.  This rule cannot lose
a statistically plausible winner under the chosen confidence procedure.  It also prevents old, cross-run numbers
from becoming silent pruning predicates.

## Bottom line

The exact cuts are legality, actual resident-byte compatibility, the one-schema scope of question 025, and the
user-set stage domain.  They do not include TileK de-duplication.  The performance cuts are maximum WarpM and
ratio-two N geometry; both are explicitly measured beliefs with expansion guards.  TileM remains untouched, and
both allowed stages remain fully represented.  A deeper-only sweep is rejected because the FA occupancy cliff is
direct evidence that extra per-block shared storage can remove the warps needed to hide latency.
