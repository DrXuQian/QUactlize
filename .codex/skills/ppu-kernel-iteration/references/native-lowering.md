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
