# Native PPU lowering and execution structure

## A compilable PTX path can still be an expensive compatibility path

Some PTX operations unsupported natively on the selected PPU can compile
through compatibility lowering. That can expand one source operation into
many native instructions, register exchanges, shared-memory round trips or
synchronization. Other operations simply fail compilation. Treat compiler
acceptance, numerical correctness and efficient native execution as three
different facts.

For the exact shipping specialization, inspect actual native code and
resources before attributing a performance gap to hardware or the algorithm:

- Bind compiler/SDK, target architecture, compatibility flags, source and
  binary. Instantiate the device body; a parsed header or unused template is
  not coverage of a target-specific branch.
- Check native operation families, operand forms, load/store widths, register
  transfers, barriers, spills and local-memory accesses. A source intrinsic
  or inline-PTX mnemonic is not proof that a single equivalent native opcode
  was emitted. Renaming/removing a PTX prefix is not an implementation proof.
- Normalize by useful work (matrix fragment, MMA, token, or output element),
  including any emulation/transpose/exchange sequence. Keep static body size
  separate from dynamically executed per-PC counts. Check the enclosing
  loops: equal static counts can hide very different repetition.
- Prefer a documented native SDK operation or a proven actlize atom when a
  compatibility expansion is costly. Prove its lane/register/shared-memory
  mapping and rounding first; then test the exact body and the full call.
  Compile-only access to a native opcode does not certify that mapping or
  predict a latency win.

Do not equate every large instruction count with compatibility emulation.
Generic signed layout arithmetic, scalar publication, repeated conversions
and a poor register layout also expand code. Attribute the extra sequence to
the actual source/IR/native lowering before naming its cause.

### Nonnegative coordinates must remain unsigned through reader helpers

Q6 PPU0010 follow-up: an unsigned outer index still incurred sign correction
after conversion to local `int k` or signed `word_index`/`word_slot` parameters.
Keeping validated nonnegative coordinates unsigned end to end removed native
shift/add/multiply corrections without changing canonical bytes or dequant
math. Preserve 64-bit address products, signed scales and invalid expert IDs.
Prove actual helper addresses at slot/group boundaries and maximum ABI sizes.
Local non-NOP BF16 sites fell 2107->1630 and vregs 108->96; global load widths,
FMA and code-conversion counts did not change. This is compile evidence only.
The Box compiler can allocate differently; fewer instructions or registers
are not themselves a speed verdict. Inspect control flow when static stores
increase: separate error/normal exits can still execute just one store.

If ACU reports show different CE frequencies, never attribute the converted
microsecond ratio entirely to source changes. Cycles are also affected by
core/memory clock ratios: collect repeated alternating pairs under matching
clock conditions and report raw elapsed cycles, not instruction totals, when
the task uses a cycle score. Record disagreements rather than replacing that
score with event time. Evidence/scope: `docs/measurements/q6_index_20261019.txt`.

The A3 device follow-up (b0b450f, Q6 indexed M8/N512/K2048/E256/topk8,
F32 storage/BF16 activation, V3/C4/W8/P8/S1, ZW810/1.7GHz) passes50 numerical
checks and RAW equality. Three alternating same-clock CE pairs reduce cycles
107742->87023,107806->88029,107968->88529 (median -18.345%). Actual dynamic
SHR4.678M->3.490M, SHRA1.638M->0 and MADL2.491M->0.098M support the address-
arithmetic mechanism, not attribution to SHR alone. FMA/conversions/global
loads,7.42MB DRAM and113MB L1/L2 traffic remain essentially fixed. Actual
registers124->104 still allow only4CTA/CU, so this is not doubled occupancy.
Remaining memory-dependency ratios rise despite faster execution; do not
interpret them as saved/cancelled wall time. Preserve remaining field-extract
shifts unless another measured change helps. Focus evidence is not permission
to promote all M/qtypes/shapes. Archive SHA256:
`a9ee9d00e411db0a7b7ab9773344b9e7f6474f4b49b3eb1d17f265dc089ec0ea`.

When testing fewer warps after the unsigned-coordinate improvement, keep the
A3 control binary body unchanged. Warp count changes FP32 reduction association;
use the independent oracle across warps, and RAW equality only between matched-
warp scheduling variants. A depth2 residue-load experiment preserves every
word and arithmetic operation while interleaving loads with FMAs: local codegen
keeps the load/FMA/conversion counts and registers fixed, but starts FMA after5
rather than9 vector128 load sites. This is native scheduling evidence, NOT
measured latency overlap. Do not use fewer SHRL sites or the earlier static FMA
position to predict a speedup. Next discriminating test: alternating same-clock
CE cycles plus actual waits/traffic under direct one-kernel cold ACU replay.
The follow-up (9446927, sameQ6/M8/N512/K2048/ZW810/BF16/1.7GHz) passes150
numeric checks and18 reimported ACU reports. W8/W4/W2 ordinary median cycles
87775/84389/78938: fewer warps save10.068% atW2, while achieved occupancy falls
46.56->40.48%. PU instructions22.553M->20.319M and TSM ld/st1/4 support less
per-warp setup/reduction work, not higher occupancy or fewer useful FMAs.
DRAM stays7.42MB; L1/L2 bytes rise4.72%. Warp-op SHRL drops only12288: do not
reuse the earlier address-signedness explanation for this different gain.
Pipeline W8/W4 has mixed signs; W2 adds only.945% (all3 pairs lower but small).
Actual Box emits the first FMA after8 rather than9 vector128 loads, NOT the
local5vs9. Compiler-specific schedule evidence matters. Memory-dependency
ratios rise while compute-dependency falls; this is not an exact wall-time
decomposition. Keep W2/pipeline as scoped experimental candidates, not global
policies; next test other existing workload controls before more pipelining.
Evidence: `docs/measurements/q6_schedule_20261019.txt`; production unchanged.

Generalizing this path (eff5d4c, ZW810/1.7GHz, returned20261020) reduced
Q2/Q3/Q4/Q5 indexed M8 focus cycles33.09/33.41/18.19/17.00%, with unchanged
useful FMAs. Q3 shared29696->1024B, TSM ld/st401408->8192 each and near-fixed
DRAM show removable on-chip metadata delivery, not lower HBM weight bytes.
Q8 focus improved15.90%, but12 event points regressed>1%, including dense
M8/N5120/K8192 +14.68%; Q6 versus the already-optimized W2-pipeline was
+0.62%/mixed. Retain old controls; don't generalize a mode's win globally or
count old Q6 gains again. Current experiment separates mode specialization,
2D geometry and index signedness, preserving the full dot/dequant body.
An unsigned K-loop changes native load forms; that compile observation alone
does not explain dense regressions. Native/codegen control bytes must remain
identical when constructing ablations. Evidence:
`docs/measurements/q8_dense_ablation_20261020.txt`.

If production clock changes (user's historical1.0GHz sweep vs current1.7GHz
ACU), do not rescale old times or assume the recipe ranking survives.
Reconfirm old champions/close runners-up/new candidates and distinct compute
routes as a bounded shortlist at the target core/memory clock regime. Preserve
old numeric evidence separately; cycles also depend on core/memory balance.
A matched1.7GHz old/new reader improvement remains valid for that experiment,
but does not establish parity with every historical full-sweep winner.

## Keep AIU SWZL and matrix-load layout contracts paired

For matrix tiles written by **AIU.swzl, use the matching ld.swzl consumer**.
Treat the producer, physical shared layout and consumer as one contract:
element width, cube geometry, swizzle, cube/stage byte pitch, coordinates,
transpose and lane/register ownership. Never change just one endpoint or
replace ld.swzl with a plain/NCOM load on the same bytes without proving a
separate complete delivery path. Matching producer/load/MMA traits should
not need a software unswizzle between load and MMA.

For eligible regular global input tiles in a PPU port, start with this native
pair. Retaining NVIDIA-style per-thread cp.async plus software swizzled
destination arithmetic is an explicit alternative, not evidence that the
native AIU pair is being used. Record why it is needed and measure the full
delivery cost. Values produced in registers are different: AIU global loads
do not replace their shared stores for free. Prove their producer/consumer
layout separately, or preserve them in registers when possible.

Do not confuse shared-layout placement, address operand conversion and
post-load fragment rearrangement. A SDK2.1.1 PPU0010 compile control kept
ld.swzl fixed: fixed cube base produced zero `v.mov.v2s`, while a warp-varying
base produced one move whose SGPR destination was the load address. Both
NCOM controls had zero such moves. This proves that move prepares an address,
not that SWZL universally needs it, that all `v.mov` are layout repair, or
that NCOM wins the complete kernel. Cube/descriptor selection still costs
instructions even when AIU supplies the right physical layout.

GDN example provenance: upstream aa04271 used NVIDIA cp.async/LDSM. Our PPU
port1af3d5c introduced the software AIU-swizzled shared layout/SWZL adapter;
the later WY path inherited it. It was our backend delivery choice, not an
upstream PPU optimization. Evidence and four-cell compile probe:
GDN-QSA-sm80 `docs/PPU_WY_FLA_REWRITE.md`,
`dev/ppu/l015_swzl_address_class_compile.cu` (compile attribution only).

The subsequent same-binary GDN device A/B validates the native pair's benefit,
not universal SWZL superiority: at S2048/Hk16/Hv32/D128/C64, g=-1.0,
PPU-ZW8101.700 GHz, state/output executed instructions fell45.0%/46.1%
and profiled times fell17.0%/27.5%. MMA counts, main traffic and active warps
were essentially unchanged. Many descriptor `v.mov.v2s` remain. Barrier
instruction counts stayed fixed but absolute sync-stall counters increased
while commit-dependency waiting decreased. After removing address work,
recheck arrival/wait/overlap behavior; fewer instructions do not imply an
equal latency reduction or permission to remove a required barrier. Treat
single-issuer arrival imbalance as a hypothesis until localized, and keep
stall-per-issue ratios separate from wall-time shares. The new dominant gap
was the unchanged prepare stage, not the stage just optimized. Evidence:
GDN-QSA-sm80 `docs/PPU_WY_AIU_ACU_20260924.md` (capture tool banners and scope
recorded there; local compile SDK identity is not the box compiler identity).

## Reference implementations should guide structure, not only tile sizes

When Triton is faster, compare its **selected/generated** kernel with ours:
CTA/warp ownership, resident accumulators, tile layouts, global-to-shared and
shared-to-register paths, intermediate materialization, pipelining and
synchronization. Source-level `tl.dot` is not that implementation description.
Inspect the compiled primitive sequence as well as the tensor formula.

Equal MMA counts but roughly twice the delivery/address/control instructions
justify inspecting or replacing that execution layer. They do not prove that
the recurrence is mathematically wrong. Changing shared layout, fragment
ownership and load atom together may be necessary; replacing only the load
opcode is unsafe. Keep the admitted old path as a counterfactual while a
separate implementation is proved. Fusion is not intrinsically faster: a
large fused CTA may force unlike stages to share an unfavorable resource
budget or materialize intermediates inefficiently.

The next GDN split-prepare experiment (same S2048 scope, clean c2bdb5c,
PPU-ZW8101.700 GHz, g=-1, ACU all-kernel sums) is a concrete counterexample
to minimizing launches or total bytes in isolation. Prefix/solve/WU used
independent resource budgets; vector publication removed the measured W/U
KVD-store amplification512->32 MiB. But total prepare L2 stores increased
32.25->40.25 MiB and DRAM reads24.27->40.91 MiB. Complete prepare still fell
136.34->101.57 us; whole-call kernel sum304.99->269.31 us. Keep hierarchy
units separate and include all split children/intermediates. The combined
gain does not isolate publication from resource separation. State became
the largest remaining excess, despite equal MMA, traffic and active warps
versus FLA; its lower barrier count coexisted with more waiting. Next checks
are generated operand/prefetch lifetimes and W/U conditioning roundtrips,
not a blanket fewer-barriers or higher-occupancy rule. FLA still wins223.58
vs269.31 us; no default promotion or weak-gate speed is established. Evidence:
GDN-QSA-sm80 `docs/PPU_WY_SPLIT_ACU_20260924.md`.

For an async operand pipeline, verify the native CFG, not linear PC order
or a copy/wait opcode count. SDK2.1.1 rotated the GDN state loop so UPDATE
appeared at lower PCs than the copies which precede it dynamically; it also
hoisted the K/U wait after12 of16 source-level PROJECT MMAs. Next-W retained
16 UPDATE MMAs before its next-loop wait. A same-opcode-count wait relocation
must fail the schedule gate. Prove old-reader retirement before issuing an
async overwrite AND completion before consuming it: a CTA barrier alone is
not an async-copy wait. Source guards and native branch structure are separate
evidence; do not assert infeasible exit edges without proving predicates.
Earlier issue/native overlap windows remain compile facts, not measured hidden
latency; scoreboard/pipe-flush waits can still matter. Evidence and negative
controls: GDN-QSA-sm80 `docs/PPU_WY_STATE_PIPELINE.md` (local proof; measured
follow-up below).

That candidate's device follow-up (clean8eaaa56, same S2048/GVA1:2/C64,
g=-1/1.700GHz, one full ACU capture per arm) validates a scheduling gain:
state122.196->105.335us, whole-call kernel sum271.483->252.191us. State MMA,
AIU counts, main traffic and occupancy limits stay fixed; instructions fall
only0.91%, while sync/commit-stall averages fall48.55%/55.53%. TSM dependency
increases10%; these counters are not disjoint wall-time shares. New prefetch
ordering and one fewer barrier were combined, not independently timed.
Unchanged phases contribute another2.431us of observed variation, which must
not be credited to the state source change. Actual box resources are232regs
and1956static instructions, versus local230/1880: bind each to its compiler.
FLA still wins224.775us; preparation is now the largest combined excess.
Evidence: `docs/PPU_WY_STATE_PIPELINE_ACU_20260924.md` in GDN-QSA-sm80.

PPU global metadata lookahead can expose the other queue's wait. On the later
GDN HV control (source6a102fd, sameS2048/g=-1/1.700GHz, siteACU), native gate/
beta loads genuinely precede8UPDATE MMAs and their `vldcnt` memory samples
fall1007->38, yet AIU `commit_group(0)` samples rise196->625 and INPUTS_READY
sync samples3735->4094. Old metadata work sat between current AIU issue and
completion; moving it changes that overlap window too. Added generic future-
address/validity work raises native dynamic instructions9.38%; state89.882->
91.944us and all-kernel187.008->189.575us show no gain. Registers122->120 do
not lift the fixed grid ceiling (activewarps14.16->14.15); BC/math/staging stay
fixed. Check both vector-load and AIU completion queues after such a move;
do not equate fewer global-wait samples with saved wall time or infer an
exact transfer between sample buckets. This is one implementation/capture,
not a universal rejection of prefetch. Evidence/negative controls:
GDN `docs/PPU_GDN_RESIDUAL_METADATA_ACU_20260925.md`; defaults unchanged.

Using CUTLASS/actlize atoms in hand-written orchestration is not equivalent
to using a mature tuned collective. A Triton reference also emits native
MMA. Compare delivered work and materialization, not implementation language:
this W/U pair has equal262144 MMAs but1.716x native instructions on our side,
including avoidable-looking shared conditioning roundtrips. That is a target
for an independent numerical/codegen experiment, not a proved free removal
or an automatic promise of1.5x speed merely by switching to C++.

Pre-AIU GDN example (local SDK2.1.1, PPU0010, B1/S2048/Hk16/Hv32/D128/C64;
device capture20260922):
WY and FLA state/output had equal BF16 MMA counts, but WY executed about
2.17x/2.35x as many native instructions. WY used SWZL loads with scalar
descriptors; FLA used NCOM lane-address loads. The pinned actlize header
disabled plain PPU0010 LDSM because its old PTX grammar was unproved, while
the installed public `hggc_mma.h` exposed `awmma::ldmatrix`. A local SDK probe
compiled its normal/16x16-transpose forms to one native NCOM load each,
without vector-to-scalar transfers or stack. That was compile-only reachability,
not a lane/register numerical gate. This identifies an adapter/codegen seam
to test, not evidence that SWZL is always slow or
that swapping it alone closes the entire gap. Detailed evidence:
GDN-QSA-sm80 `docs/PPU_WY_SHARED_ACU_20260922.md`, commit4267b28.

## An API win and slower GPU kernels can coexist

A full-public-API event span can include GPU idle gaps while the host
allocates/dispatches the next operation. Summed ACU kernel durations omit
those gaps and come from a different instrumented/replayed protocol. A C++
multi-launch entry and Python/Triton launch path can have different dispatch
costs even when their arithmetic scopes match.

Report full-call and kernel-only results separately. Do not subtract timings
from different runs/protocols and label the difference measured Python,
allocation or launch overhead. Establish that attribution with one complete
call timeline or an explicitly paired prepared-workspace/graph experiment;
verify that the runtime supports the latter. Do not let host overhead hide
a slower kernel or remove required work from only one timed arm.

## Reassociation changes precision and workspace lifetimes

Removing a materialized product is not merely a delivery variant. For GDN,
PBV/PBDK-style materialization versus a gated residual has different BF16
rounding boundaries; prove the real-algebra identity and admit the new
association against an independent recurrence, while retaining RAW-BIT checks
for old delivery-only variants. Give the algorithm a distinct API/receipt
identity; a close oracle error cannot certify which variant actually ran.

Reaudit global scratch ownership when removing a stage: the old GDN solve
stored its inverse in future H-snapshot scratch, safe only because W/U consumed
it before state started. In a direct P@residual recurrence, independent V-slice
CTAs may overwrite another CTA's still-live inverse. A per-CTA barrier cannot
repair that inter-CTA race; use separate storage or prove a different lifetime.
Retaining a padded inverse pitch can erase the allocation-size saving even
when W/U traffic disappears. Fewer total GEMMs may also lengthen the serial
recurrence. Evidence: GDN-QSA-sm80 `docs/PPU_GDN_RESIDUAL.md`, SDK2.1.1 local
algebra/layout/native checks; the initial handoff had device results NOT_RUN.

The follow-up at source1a843eb (S2048/Hk16/Hv32/D128/C64, g=-1,
PPU-ZW8101.700GHz, site ACU, SDK2.1.1) passed30 residual device cases and
reduced full kernel time255.226->230.584us, still behind FLA224.309us. It
deleted47.644us of W/U but added22.365us to serial state. Compare new state
with old/reference **W/U+state**, not state alone; no precision or auto-routing
promotion follows. The1.5x goal remains unmet. Both state grids128x128 yield
about7.1warps/CU, matching the grid ceiling despite capacity for16; common
underfill can be an optimization target without explaining the between-library
gap. Smaller independent V slices duplicate K/P delivery and need a measured
mapping/resource tradeoff, not simply more launched CTAs.

Before blaming operand delivery, compare the actual precision algorithms:
this solve used98304 TF32 MMAs versus FLA32768 because it explicitly retained
three high/residual products. BF16 KKT counts were equal. A single-TF32
replacement is a separate numerical experiment, not a layout-only change.
Source `CUTE_UNROLL` also did not eliminate all diagonal register-array loops:
native `column[k]` delivery used196608 `ivreg` reads plus loop/pipe-flush
machinery. Inspect that sequence before calling ordinary moves unswizzling;
most observed solve moves loaded literal operands. BF16 conversion totals
closed on explicit H/residual/Vnew/scaledV boundaries, so optimize their
materialization without silently deleting required rounding. Bank-conflict
totals and wait instruction counts are not latency attribution: the faster
FLA state had more counted bank conflicts, and its solve more barrier opcodes.
Evidence: `docs/PPU_GDN_RESIDUAL_ACU_20260924.md`; weak-gate timing and isolated
register/pipeline changes remain unmeasured.

For this SDK2.1.1 case, explicit row/term template indices did remove the
indirect delivery without changing precision: diagonal sites612->405,
ivreg sites12->0 and pipe-flush sites42->0; no diagonal backedge remains.
Coefficient loads combine into8scalar+8two-word+24four-word operations,
still120FP32words and120ordered RTTE FMAs. Whole solve1678->1472sites,
84regs/zero stack, same three-product TF32 scheme. This shows a rolled
register loop need not be fixed by a larger straight-line footprint, and
fewer load opcodes need not mean missing data. Verify actual native indexing,
word coverage, FMA order/contraction and resources; the retained old lowering
under the new symbol must fail the gate. That initial handoff was compile-only.
Evidence: GDN `docs/PPU_GDN_SOLVE_STATIC_INDEX.md`; measured follow-up below.

The PPU1.0 follow-up (actlizeLA77fee23, same S2048 scope, g=-1/-0.1,
HGCC2.2.0-dev box, 1.700GHz, site ACU) confirms a same-precision gain:
solve52.50/52.48->44.88/44.95us, complete4/4/7-kernel sums
185.71/187.10->180.07/181.19us, about1.247xFLA, not1.5x. Dynamic diagonal
indirect reads196608->0 and pipe_flush waits466944->0 close the generated-code
mechanism; whole-solve opcodes-22.69%, same491520ordered diagonal FMAs,
same BF16/three-productTF32 MMA,84regs/zero stack, shared capacity and BC.
Higher sync/memory per-issue ratios coexist with lower latency; do not turn
those changing-denominator ratios into wall-time cancellation. Three unchanged
stages have identical opcode dictionaries but add1.62-1.99us variation, which
is not credited to the edit. One capture per arm/gate supports confirmation,
not automatic routing. Bind actual box compiler/dependency history, not the
local SDK label: only4/37 box bodies match the local build. Evidence:
actlizeLA `docs/PPU10_STATIC_SOLVE_ACU_20260928.md`.

Full chunks can still pay PPU runtime-predication costs even when no lane
ever takes a tail path. On the subsequent full-C64 GDN state experiment
(actlizeLA41ade14, sameS2048/g=-1/-0.1, HGCC2.2.0-dev,1.700GHz), making
the admitted extent constant removed262144`v.csel`,32768`s.min`,139264
execution-mask operations and131072`v.mov.v2s` executions. State opcodes
fell13.648->12.154M; complete calls174.845->168.924us /178.407->169.362us.
BC4.440M, matrix traffic/MMA, shared46080B and roughly14.1activewarps/CU
stayed fixed. Registers124->120 did not lift occupancy; wait instructions
increased. This is a generated control/address simplification, not a BC or
bandwidth-volume win. Prove full-chunk eligibility and retain generic tails,
then verify the actual native body and dynamic counts: localSDK2.1.1 body
sizes differed from box2.2.0-dev. Keep unchanged-stage timing variation and
per-issue stall ratios separate from saved wall time. Both are single captures
and remain short of1.5xFLA. Evidence and36case x8 RAW-BIT admission:
actlizeLA `docs/PPU10_FULL_CHUNK_ACU_20260928.md`.

Before adding explicit shared-to-register double buffering, inspect the old
native schedule too. SDK2.1.1 already prefetched several GDN residual operand
tiles: a two-slot source rewrite left UPDATE lookahead unchanged and altered
KH/PR scheduling without proving longer useful overlap. Registers rose242->244
with zero stack. Keep such a candidate distinct from next-chunk global
prefetch and from V-slice geometry; source load-before-MMA order is not a
performance verdict. An independent V32->V16 compile reduced registers242->98
and shared45568->35328B, doubling target CTAs128->256 but also duplicating K/P
delivery. Halved static MMA sites per CTA do not mean halved dynamic work.
Both require same-math RAW-BIT device admission and full-call profiling.
Evidence: GDN-QSA-sm80 `docs/PPU_GDN_RESIDUAL_DELIVERY.md` (local compile-only).
Bank-conflict elimination is a subsequent isolated experiment: bind counters
to specific shared PCs/bank maps, preserve the paired writer/reader contract,
and require a latency gain rather than optimizing the aggregate counter alone.

The V16 device follow-up (GDN5a3ebc9, same S2048/g=-1/PPU-ZW8101.700GHz,
site ACU) confirms the tradeoff: active warps7.08->14.12, but all-kernel time
226.03->224.41us, effectively tied with FLA225.67us in that capture. KVD-to-TSM
transaction bytes112->224MiB; read BC+36.36%, write BC-1.62%, transposed matrix
loads+50% with unchanged useful MMA. Do not name BC alone as the lost speedup:
splitting V duplicates K/P and changes request mix/granularity. Normalize
conflicts by useful work AND request class; sampled waits are not wall time.
More CTA-local warps with the original tile can preserve global/shared reuse,
but still duplicate shared operands and needs an independent proof/experiment.
PPU result fragments place adjacent columns in different lanes: packing
BF16 stores requires a proved exchange or different paired intermediate
layout, not a per-thread reinterpret-cast. Inspect native lowering before
accepting that exchange. Evidence: GDN `docs/PPU_GDN_V16_ACU_20260924.md`;
V32/8warp ownership is host-proved only, not implemented or timed.

When BC elimination is explicitly required to add no work,
restrict that experiment to compile-time shared layout plus supported native matrix loads,
including AIU.swzl/ld.swzl. No runtime unswizzle, lane exchange, extra copy,
barrier or rearrangement kernel may be disguised as a free layout fix.
Constexpr CuTe coordinates can still lower to dynamic address instructions;
verify actual native lowering. Register-produced intermediates need a separate
writer/reader proof. If the primitive cannot express the required permutation,
report the boundary instead of relaxing the constraint. Do not elevate a
design suggestion into a hard gate: this GDN user later clarified that added
instructions are acceptable when the complete same-correctness ACU call is
faster. Keep instruction/BC counts as explanations, not performance verdicts.
Compare BC fixes within the same warp geometry; more warps are separate.

For register-produced, non-published B intermediates, direct placement in
consumer orientation can replace a transposed matrix load without adding a
transpose operation. GDN residual/scaled-V are eligible; H snapshots also
have a vector global-publication consumer and cannot be changed by the same
argument. Prove BOTH consumers for externally published intermediates.
SDK2.1.1 local compilation of a wide32x64 B-oriented cube added111 static sites
and2registers; native16x16 microcubes plus a precomputed lane base retained
242regs/zero spill and reduced recurring body sites899/905->897/903, despite
whole-body sites2092->2093. Static totals alone miss setup versus recurring
cost. These are compile facts, NOT measured performance/BC elimination.
Evidence: GDN-QSA-sm80 `docs/PPU_GDN_RESIDUAL_B_LAYOUT.md`; native C/B traits,
actual actlize reader simulation, unchanged global AIU planes. The bank model
must state bank width and request phases; do not equate modeled conflict-free
addresses with a measured whole-kernel ACU bank-conflict counter of zero.

The B-layout device follow-up (GDN3c7da09, same S2048/g=-1/PPU-ZW8101.700GHz,
site ACU) reduced BC6.930432M->5.357568M (-22.70%) at unchanged useful MMA,
shared-load/store counts and112MiB KVD-to-TSM bytes. The intended131072
transposed loads became non-transposed; no stale/unused-branch explanation.
Yet state130.484->130.334us and all-kernel230.678->232.220us show no gain;
unmodified output accounts for most total-time variation. Eligible warps
stay0.25 in the same WE metric; native dynamic instructions fall only0.213%.
Memory-dependency ratios fell, but do not turn them into wall-time savings.
A bank-layout improvement can be real without shortening the critical path.
Do not infer all residual BC is harmless, or force promotion from a cleaner
layout. Retain the counterfactual and change the next experimental axis.
Extra wait opcodes do not prove exact cancellation of saved conflicts.
Evidence: GDN `docs/PPU_GDN_RESIDUAL_B_LAYOUT_ACU_20260924.md`; no routing
promotion,8-warp/original-tile follow-up remains unimplemented.

That follow-up's local SDK2.1.1 eight-warp implementation makes the two traffic
directions explicit: V32/grid128 keeps28KiB global-to-shared input perCTA/chunk,
but shared matrix reads224->288 (+28.57%) because H/residual/scaled-V register
reuse is split across warps. Native40->20 MMA sites times4->8warps preserve
useful work; global staging remaining112MiB is NOT all shared traffic unchanged.
Registers242->162/thread,0stack,shared45568B are compile facts, not a speedup.
Do not force a128-register cutoff without checking the live grid: on the
captured131072-register/CU target,162x256 permits roughly3 register-limited
CTAs while grid128/72CU supplies only1.78 on average. A fourth slot at128 does
not lift the grid ceiling; larger grids or another bottleneck may change this.
Never trade into spill merely to hit that number. Validate the real owner map
AND vector publishers when doubling threads; the latter cannot retain the old
iteration stride. Evidence: GDN `docs/PPU_GDN_RESIDUAL_WARPS8.md`; device
RAW-BIT/traffic/occupancy/latency remain separate, initially NOT_RUN.

The eight-warp device follow-up (GDN dfe76c3, same S2048/g=-1/
PPU-ZW8101.700GHz, one full site-ACU capture per arm) passes30 cases x8
RAW-BIT and lowers state127.745->104.810us, all-kernel228.641->207.002us
versus FLA226.969us (1.096x, NOT1.5x). Actual162regs/zero stack and3register-
limited CTA slots still yield14.12warps/CU, near the grid ceiling14.22;
eligible warps/WE rise0.25->0.36. Global staging remains112MiB, but shared
matrix reads+28.57%,BC+15.13% and native dynamic instructions+9.72%.
Memory-dependency/sync per-issue ratios also rise while latency falls. More
issue supply can outweigh extra operand work; lower per-warp stall or fewer
instructions is not necessary for a faster complete call. This does not
isolate occupancy from shorter fragment work/codegen or make more warps a
universal policy. Keep BC and instruction counts as diagnostics, not gates.
The extra1024BF16 conversions are a zero-constant setup PC, not new recurring
rounding. Unchanged stages contribute+1.296us variation; do not attribute it
to changed state source. Evidence: GDN
`docs/PPU_GDN_RESIDUAL_WARPS8_ACU_20260924.md`. Next layout test must use the
eight-warp control and re-prove paired maps; old four-warp BC reduction is
not admission of their combination. Default routing remains unchanged.

Static BC attribution has a further boundary: native fragment/byte maps are
not the native memory service phases. GDN eight-warps already uses
`ldmatrix.swzl.trans`; its extra262144 trans loads are H/R/scaled-V reuse
split across warps, not extra K-transpose. Removing trans on unchanged bytes
fails3840/4096 H and1920/2048 R/scaled tags. R/scaled can instead be stored
directly in B orientation; H has a second canonical vector-publication
consumer, and K is used in both orientations. Prove all consumers.
The empirical account4*trans+3*scalar-BF16-reads fits six captured arms,
but a32x4B word-bank model gives identical old/B-layout scalar-store signatures
while measured writeBC differs1048576. Exact aggregate fits are not independent
per-PC bank proof; label inferred coefficients, broadcasts/subword rules and
unknown TSM grouping explicitly. The later paired-Vnew layout is a
counterexample to the opcode-only fit:
state read BC rises393216 with unchanged transpose/scalar-load counts,
while the paired output's read BC falls to zero. Its16B shared-publication
reader has different addresses despite equal instruction/byte counts;
AIU/SWZL matrix-consumer admission does not certify that publisher's bank
behavior. Evidence: GDN `docs/PPU_GDN_RESIDUAL_WARPS8_HV_LAYOUT_ACU_20260925.md`.
Also inspect a transfer atom's `Trans` flag:
the pinned b16 AIU true/false wrappers share the same bulk opcode and differ
in coordinate argument ordering; the flag alone does not prove a physically
transposed destination. Evidence: GDN
`docs/PPU_GDN_RESIDUAL_BC_STATIC_20260924.md`, host-only l026; no kernel edits.

Combining a proved layout with a new warp geometry still requires a new
producer proof: reuse the physical microcubes/reader, derive the producer
base from current owner coordinates rather than copying the old warp-index
shortcut. GDN's eight-warp R/scaled-V combination checks both planes, all
tails/slices/warps and fixed writer/reader denominators; the stale four-warp
base is a constructive wrong-address/OOB negative. SDK2.1.1 local compilation
kept math/copies/barriers and all28 old native bodies unchanged, replaced8
transposed load sites, and naturally allocated124 rather than162 registers
without a cap/spill. That compile fact does not imply more achieved occupancy
at the same underfilled grid or a latency win. The new capture must compare
the SAME eight-warp geometry, with scalar residual retained only as the
numeric anchor; otherwise layout and occupancy gains are confounded.
Evidence: GDN `docs/PPU_GDN_RESIDUAL_WARPS8_B_LAYOUT.md` (initial compile
handoff); measured follow-up below.

The combined8warp B-layout follow-up (GDN0273ad3, same S2048/g=-1/
PPU-ZW8101.700GHz, one full site-ACU capture/arm) passes numerics and reduces
BC7.979->5.882M, but all-kernel205.359->205.854us shows no speed gain.
Report occupancy as three layers: resource capacity, grid/work supply, and
issue readiness. Registers162->124 increase register slots3->4, but shared
limit5,warp limit8 and grid128/72CU remain fixed; active warps14.12->14.11
and eligible warps/WE0.36->0.34 do not improve. Do not optimize a register
threshold alone. Native pairs show MMA inputs immediately reused as subsequent
load destinations; per-PC compute samples rise there while aggregate TFU-WAR
ratio0.00->0.39 increases. This supports an operand-lifetime/prefetch hypothesis,
not proof that fewer registers caused all the loss. Distinguish newly changed
scheduling hazards from a previously hidden bottleneck. Per-issue memory
ratio1.74->1.71 is not measured wall time saved or an exact compute-stall
cancellation. Keep the low-BC branch as a counterfactual and test independent
operand lifetimes before conflating them with further layout changes.
H/K conflicts remain legitimate targets, with H's canonical publisher and
K's two orientations proved/accounted together; no blanket claim that BC is
irrelevant or that eliminating it guarantees speed. Evidence: GDN
`docs/PPU_GDN_RESIDUAL_WARPS8_B_LAYOUT_ACU_20260924.md`; defaults unchanged.

In the next same-layout8warp compile screen, two/four source operand slots
and an empty-assembly liveness constraint did not improve KH/PR native
scheduling. Keep only the effective UPDATE edit: its B operands gain2/3
intervening MMAs and targeted immediate input reuse falls3->0 at124regs,
zero stack, unchanged math/traffic/barriers. This is a partial compile result,
not global WAR elimination or a speedup. Track physical register-word writes,
not only array names; classify both A/B load orientations and fixed per-phase
MMA denominators. An old-native-body replacement with unchanged useful work
must fail specifically for unchanged lookahead. Do not retain inert asm or
deeper arrays as an optimization. Evidence: GDN
`docs/PPU_GDN_RESIDUAL_WARPS8_OPERANDS.md`; SDK2.1.1, initially device pending.

The UPDATE-only device follow-up (GDN0755b93, same S2048/g=-1/1.700GHz,
one site-ACU capture/arm) confirms the changed native lead/reuse but no gain:
state104.528->104.361us, all-kernel204.835->206.176us, FLA224.985us.
Unchanged phases contribute+1.507us variation; do not call it source regression.
124regs/zero stack, BC5.882M, traffic and useful math stay fixed. Targeted
matrix-load compute samples444->379 coexist with whole-kernel3565->3620,
TFU-WAR ratio.39->.40 and eligible warps.35->.34. Samples are not cycles;
removing immediate input reuse is not proof of complete hardware retirement
or kernel-wide WAR removal. Bind every phase's actual native load PCs and
fixed denominator; do not extrapolate a local scheduling improvement into
latency savings. Reject this as an effective speed optimization without
declaring the untouched KH/PR/H/K paths exhausted. Evidence: GDN
`docs/PPU_GDN_RESIDUAL_WARPS8_OPERANDS_ACU_20260925.md`; no route promotion.

A register-produced tile with a global downstream consumer cannot inherit an
internal-only B-layout fix unchanged. GDN H needs both KH and global snapshot
publication; transposing only shared H breaks contiguous canonical stores.
One bounded alternative changes the PRIVATE snapshot to[V,K] and pairs the
output AIU reader, preserving public initial/final state and rounding. Prove
the entire C-writer/native-B-reader/16B-publisher/output-reader chain with
distinct tags and stale-publisher/old-output negatives. Equal allocation and
logical bytes still do not imply equal copies:128x64->64x128 raises output
H AIU cubes1->2 because of the native128B cube width. Price that extra copy
instruction and measure both changed kernels, not state alone. This is a
cube-geometry cost, not a universal transpose tax: a paired64x64 Vnew swap
still occupies one AIU cube. SDK2.1.1 native compilation removed its eight
output transpose-load sites without extra AIU instructions or staging bytes.
Check the actual cube tiling and linked native body before assuming either
extra traffic or a free orientation change; this does not establish speed.
Evidence: GDN `docs/PPU_GDN_RESIDUAL_WARPS8_HV_LAYOUT.md`.
SDK2.1.1 local
codegen kept vector publication, zero spill and30 old bodies intact while
reducing state/output registers124/94->122/78; performance remains pending.
For a source gate that normalizes candidate names back to a control, first
require the actual candidate binding: otherwise an already-old downstream
call can pass the normalization despite the new private format. Evidence:
GDN `docs/PPU_GDN_RESIDUAL_WARPS8_H_LAYOUT.md`; no automatic route promotion.

The paired-H device follow-up (GDN14c7268,same S2048/g=-1/1.700GHz,
one full site-ACU capture/arm) improves state104.952->95.412us and
output45.708->44.084us; all-kernel205.732->194.466us,FLA224.350us
(1.154x,not1.5x). BC falls22.28%/38.10%,but this is not BC-only evidence:
state KVD/L2 store transactions fall98->66MiB while logical and L2-to-LLC
stores remain50MiB. A16B vector store does not prove cross-lane coalescing.
The production publisher's64B-at256B H rows become contiguous256B rows;
128B footprint accounting matches the32MiB transaction difference, but
label that service-granularity inference rather than per-PC measurement.
Do not translate it into saved compulsory DRAM bytes. Both changed kernels
and the extra2048 output AIU instructions are included; staging112/80MiB
and useful MMA/rounding are fixed. Actual warps stay14.1/37 despite fewer
registers. State TFU-WAR decreases while output TFU-WAR/sync increase;
the latter does not cancel the full-call gain. Keep memory/layout/scheduling
effects coupled until isolated. Analysis filters must select exact state
AND output symbols when both share the candidate namespace; a former
state-only substring can silently mix their PC samples. Evidence: GDN
`docs/PPU_GDN_RESIDUAL_WARPS8_H_LAYOUT_ACU_20260925.md`; routing unchanged.

## Quantized Hopper control: a scheduler is not the complete implementation

Quactlize's existing standalone Marlin collective has a different packing,
K-warp/CTA topology, cp.async cadence, register dequant and reduction path
from a generic cooperative WGMMA collective using Marlin stripe scheduling.
Reusing its pure integer scheduler does not port that complete implementation.
The PPU001 m8n16 register maps must not be renamed into Hopper m16n8 maps.
Prefer a proved architecture-native provider over an incomplete bring-up
collective when restoring a representative baseline, and explicitly identify
upstream reuse instead of claiming a new algorithm.

Evidence: `docs/ppu17-gptq-baseline.md`, H800/CUDA12.8, INT4 symmetric GPTQ,
group128, FP16 A/weights/C and FP32 accumulation/reduction, rotating weights
>2.25x queried L2. The scheduler-only/MMA-prepacked route still lagged native
Marlin by20–24% for M1/8,N4096,K4096. Integrating the complete pinned Marlin
body removed that large gap in screening; all six instruction bodies match
the reference and18 same-config comparisons are RAW-equal. This is neither
native PPU1.7 admission nor a GGUF group16/32 performance result.

Audit prepared-call resets separately: removing an unnecessary reset is safe
only if the last peer already resets every lock, first-use initialization is
retained, and scratch reuse is serial on the same stream. Do not apply that
contract to Stream-K without proof. Stage4/8 variants are likewise measured
alternatives, not universal defaults: fewer stages help short stripes in
this small-M experiment but regress large-M calls. Preserve the old winner
and transitively include same-math controls in confirmation. Clock-rejected
points remain invalid; retry only missing points and bind an assembled report
to each original receipt rather than relabelling a failed cohort complete.

## Fine GGUF groups on a Hopper prepacked collective

Extending a group128/TileK64 Machete-style collective to GGUF group32 is not
just relaxing `can_implement`. Give each raw64K stage two physical affine
rows, count both in the TMA barrier bytes, and present a separate logical-K
broadcast view to the MMA partition. Reuse metadata only within the two16K
atoms belonging to that group. Explicit stage counts are needed until the
auto-sizing helper accounts for every metadata row.

Audit prologue lookahead after changing metadata cadence: the upstream call
passed the next pipe index, harmless while metadata loaded only at atom0.
Loading a second group inside the tile makes that index wrong. Prove both
thread/value ownership and current-stage selection with group/stage negatives,
then require exact one-hot decode and changed-input replays on device.
Use explicit rounded half2/BF16x2 multiply and add if the contract has two
rounding points; inspect actual SASS rather than assuming overloaded operators
will preserve both vectorization and rounding.

Evidence: `docs/ppu17-gguf-cooperative.md`, H800/CUDA12.8, Q4_K group32 SF,
FP16/BF16, FP32 accumulation.108 initial numerical cells passed. The large
win against a bring-up canonical-reader kernel changes pipeline and packing
together; it is not proof that canonical K-pack itself must be replaced.
It is neither FQ packed-metadata evidence nor native PPU1.7 admission.

### Packed-unit extension: coordinate ownership and shared sidebands

The subsequent Q4_K FQ control kept the same code packing, math and scheduler.
A shared per-stage group phase passed216 numeric cells but failed direct
racecheck with RAW hazards. Do not promote numerical success over a failing
lifetime gate, or call a small single-phase probe proof of a reused pipeline.
One initial scalar probe was optimized to constants; inspect native LDS/STS
before treating a synchronization probe as meaningful.

When the outer scheduler already owns an absolute K start, pass that value
through an optional collective capability and keep its cursor in registers.
This removes shared publication instead of adding speculative fences. The
isolated overlay must bind the exact seam and reject missing integration.
Check nonzero slice starts as well as K=0 and ordinary stage wrap. Avoid
inventing another scheduler solely to recover coordinates.

On H800/CUDA12.8, Q4_K FQ/SF x FP16/BF16 passed432 numerical cells after
removing the sideband; FQ passed36 direct memcheck/racecheck/synccheck cells
plus a scalar fault negative. Scope and receipts: `docs/ppu17-gguf-cooperative.md`,
`gguf-mainloop/fq6-memory-v1`, `fq6-screen-v1`. Exact dynamic cause of the old
hazard is not established as a compiler or tooling defect.

Packed resident size alone does not predict speed. Reading an entire16B
unit each64K repeats its metadata four times, whereas a unit covers256K.
The first FQ reader was11.5–44.7% slower than resident SF in24 matched
same-tile pairs despite10% lower resident bytes. Next distinguish repeated
traffic, field/header conversion and register pressure with bounded ablations;
measure SF expansion separately before end-to-end selection. This is CUDA
Hopper evidence, not a universal native PPU1.7 performance claim.

## Dense full-dequant: packed reuse and exact FP32 conversion

PPU0010/SDK2.1.1 helper R3 keeps each canonical32-bit low word's two N
columns together in shared memory. One consumer owns2N x8K and issues
two independently coalesced16B output stores. Q5/Q6 retain a compact16-bit
high sideband. Compared with R2 byte-expanded exchange, logical code
shared reads fall4->2B/weight for Q4 and4->3 for Q5/Q6; affine math is
unchanged. Do not confuse this with reduced compulsory HBM bytes.

For raw codes<=63, bitcast(0x4b000000|raw)-(8388608+bias) produces the
exact FP32 integer, including Q6 bias32. Keep multiply and minimum
subtraction separately rounded; folding the bias into a scaled constant
is a different arithmetic contract. This is not a cast through FP16.
Native N32K128 int->F32 sites fall34->2, but OR/subtract work increases;
retain ordinary conversion as an independent control. Fewer converts
alone do not prove lower cycles. Final output already uses packed F16/BF16
conversions, so scalar-looking source casts are not evidence of scalar ISA.

Local codegen: Q4 N32K128 F16 static instructions532->394 for reuse,
428 with fast FP32; Q6 N16K256 BF16717->575/576. Zero stack, all52 old
dequant instruction bodies unchanged.108 host raw-GGUF cells, all integer
codes, producer/consumer coverage and planted mapping faults passed. These
are compile/host proofs, NOT GPU correctness or speed admission. R3 Box
uses matched R2 controls, isolated event timing and single-helper ACU ABBA
cycles; full calls are confirmed separately, including reset and IO.

Audit global footprint per instruction: N32 low loads span64B per K
residue, N16 spans32B, not128B. Q5 high words also repeat across N-bit3
and K32; Q6 high repeats across K32 halves. Those existing limitations
are not fixed by shared reuse. `tests/test_prefill_dense.py` preserves
32/64/128B footprint and duplicate-read checks instead of assuming a
uint4 load guarantees full sectors. Current readers are consolidated in
`quactlize/dequant/dense_reader.hpp`. The unselected candidates and R1–R5
runners were retired; historical experiments remain at commit
`8a004d42a91a23664e69762509d34c3b76bbf8e8`, not in the installed runtime.

R3 device follow-up (`dc18965`, review-1792640379512214929.tgz,
SHA256 22fc34cb4ab784e597bf3d4ec8b94797e054cacd6f4cde542d3329614398f18e,
ZW81072CU/1.700GHz/SDK2.1.1) passes all42 arms and24 ACU captures.
Paired reuse wins small Q4/Q6 event medians by11%/6--7%, but large shapes
only0.2--1.7%. Exact FP32 bit construction does not win. Q4 large shared
read transactions fall40% while shared-write conflicts triple; Q6 large
reads fall20% while conflicts rise67%. These are overlapping effects,
not proof of speed proportional to static instructions or TSM bytes.

Q6 N5120K25600 paired BF16 profile10 reports968.75MiB LSU global-read
transaction bytes,393.617MiB L2 reads,102.561MiB HBM reads versus102.539MiB
packed weights; output is250MiB. The9.45x amplification is on-chip request
traffic, NOT9x compulsory HBM. Canonical36B metadata units still use byte
loads and repeat the header per group. A uint4 code load does not solve
metadata coalescing. R2 IO similarly halved expanded-side KVD transactions
without materially reducing large-shape latency; low eligible warps persist.

The bounded R4 hypothesis separates low32/high16 shared maps and producer
ordering, cooperative16B unit staging, wider N, and full-flat four-load IO.
Q6 unit staging copies the unused paired-superblock half and adds one CTA
barrier: both costs must remain in the component time. Host bank models
at16/32 lanes are not actual ACU bank-conflict admission. Local native code
removes cooperative metadata global byte loads, keeps zero stack, and
preserves108 old device bodies; this is compile evidence, not a speedup.
Use `round4.py` matched bank-only/metadata pairs and separate full-call
confirmation; no production route changes until device regression checks.

Q3/Q8 paired-plane follow-up (953c9ca, ZW810/72CU/1.7GHz, returned
review-1792722733794805356.tgz) passed48 F16/BF16 raw-bit cells. F16-only
rotating component medians improved Q3 by85.0--85.9%, Q8 by25.2--39.2%.
Q3 N17408K5120 KVD/LSU read transactions7.13GB->52.9MB and dynamic
instructions-91.9%, with nearly unchanged compulsory DRAM reads, identify
repeated metadata/delivery work rather than a7x HBM-volume reduction.
Q8 N8192K2048 instructions-33.6%, KVD store bytes64->32MiB and lower cycles
support more efficient store servicing; useful output bytes stay32MiB.
K128 wins Q8; Q3 K128/K256 differences are within round variation. Retain
both until full-call confirmation; BF16 performance was NOT measured in
that return. ACU forced-cold single launches and rotating event medians
remain separate protocols. This is not model/native-parity admission.

Q8 shared-exchange follow-up (2026-10-23, SDK2.1.1, local codegen/host only):
N64 paired Q8 K8 tasks read each shared code word twice; adjacent K8 phases
use the same physical word and different byte shifts. A naive K16 consumer
is NOT automatically a coalescing win: an aligned32-byte extended-vector
store compiled into two16-byte native stores, each with holes in its lane
footprint. Keep the K8 store map and compare packed-word lane exchange.
The swizzle seam matters: shared(pair,row+4), not shared(pair,row)+4;
pair bit2 swaps these vectors. A host ownership oracle catches the latter
at pair4 and an independent raw-GGUF oracle confirms the repaired mapping.
This does not justify changing canonical bytes or rounding. The candidate
lowers to native butterfly shuffle (not extra shared staging), retains zero
stack, and leaves all42 old device bodies byte-identical. New K128/K256
vector register counts40/54 versus control44/56 are compile observations,
not GPU speedups. Measure both precisions on Box and inspect shared conflicts,
cycles, issue pressure and actual output transactions before promotion.
Full-only helper measurements must not silently change a shared TC/SF adapter
lookup; their route-specific scopes now have separate host policy entries.
Evidence/workload and bounded ACU command: docs/PPU_PREFILL_DENSE_PRODUCTION.md.
