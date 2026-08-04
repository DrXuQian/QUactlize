# Sweep 032: Codex's independent pruning rules

This is the independent answer requested by INBOX 032.  It was written before INBOX 033 existed and deliberately
does not try to anticipate the other rule set.

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

### Stage 3 is part of the starting space -- **MEASURED**

The first sizing program listed `{2,4,6,8,12}`, but the harness exposes stage 3 and the recorded product sweep has
stage-3 winners.  Stages 2, 3 and 4 have each won for some format/shape; omitting 3 is already performance pruning,
not sizing arithmetic.  The corrected unpruned count is 5791 instantiations per operator, including stage 3 and
without the invalid TileK de-duplication.

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

### H3. Fully cross stages 2, 3 and 4; probe 6, 8 and 12 with sentinels -- **MEASURED + BELIEVED**

Stages 2/3/4 stay on every primary geometry because each has a measured win and one fixed stage gave a 1.54x spread
on a single shape.  Deep stages do not get a full shape cross-product initially.  For each
`(operator,schema,TK)` stratum, compile stages 6/8/12 at:

- the legal geometry with the smallest per-stage shared footprint, where depth has the best chance to fit without
  reducing resident blocks; and
- the stage-4 primary winner's geometry, where the comparison changes only depth.

Why deep-only is specifically rejected: whenever shared memory binds,
`resident_blocks * stages` is approximately invariant.  The grouped decode example is 76 in-flight loads at s4
versus 72 at s12, so depth concentrates roughly the same loads into fewer CTAs and fewer warps.  Deep stages also
lost in the measured decode ladder.

The d128 FA result is a measured cross-kernel warning, not proof about GEMM: FA already sits at four blocks/25%
occupancy, and adding shared storage dropped it to three blocks and made it slower even after halving a memory
stall.  It rules out the argument that deeper is automatically safer; it does not rule out deep stages on a tiny,
register- or grid-limited GEMM tile.  That is why the minimum-footprint sentinel exists.

If any deep sentinel enters the winner's confidence set at any M, expand that stage over the corresponding
resource bucket first, then over the whole stratum if the gain survives.  The belief being tested is that the
geometry most able to retain occupancy is also the geometry most able to expose a deep-pipeline win.  A
shape-specific stage interaction could violate it, so a negative sentinel is evidence-backed pruning, not a proof.

### H4. Make pruning adaptive, not a one-shot truncated grid -- **BELIEVED process rule**

Build the primary rows from H1/H2 with all protected TileM, compatible TileK and stages 2/3/4, plus every guard and
deep sentinel.  Run all M.  Expand the affected stratum whenever a guard is competitive; do not merely annotate the
missing cells after choosing a winner.

This cannot give a mathematical guarantee against an arbitrary high-order interaction.  Its argument is that each
excluded mechanism has an explicit falsifier in the same binary, and expansion happens before the result is called
a winner.  The claim should therefore state which guards were negative.  If the user requires an unconditional
winner over the original Cartesian product, no performance pruning rule is honest: all 5791 kernels per operator
must be built.

### H5. Use confidence-set elimination, not one timing -- **MEASURED**

The same configuration has shown 13% cross-run spread, and cold single-row launches differ from warmed sweep rows.
Time candidates interleaved in one process and repeat blocks.  A guard is negative only when its uncertainty band
is disjoint from the current best; a point inside the band survives and triggers expansion.  This rule cannot lose
a statistically plausible winner under the chosen confidence procedure.  It also prevents old, cross-run numbers
from becoming silent pruning predicates.

## Bottom line

The exact cuts are legality, actual resident-byte compatibility, and the one-schema scope of question 025.  They do
not include TileK de-duplication.  The primary performance cuts are maximum WarpM, ratio-two N geometry and sentinel-
only deep stages; all three are explicitly measured beliefs with expansion guards.  TileM remains untouched, and
stages 2/3/4 remain fully represented.  A deep-only sweep is rejected because both this kernel's in-flight-load
arithmetic and the FA occupancy cliff show how it can preserve only the configurations least able to fill the
machine.
