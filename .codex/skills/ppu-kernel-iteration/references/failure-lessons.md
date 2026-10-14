# PPU failure lessons

Use the category that matches the observation; these are not reasons to add
every diagnostic to every kernel.

| Observation | First discriminating check | Do not conclude |
|---|---|---|
| Outputs remain poison; sync returns success | Clear/check prior status, immediate launch status, then sync/deferred status; run an existing marker in the same carrier | Decode formula is wrong just because output is zero |
| Immediate status200 / InvalidKernelImage, including marker | Inspect SDK/compiler/runtime identity and native image admission; compare known runnable carrier | Header-defined kernel, aggregate ABI or CuTe must be broken |
| Failure appears after adding F32 endpoints | Inspect first nonfinite tensor and exact activation/dequant precision boundaries | F32-to-F16 cannot overflow, or clipping is harmless |
| Changing-input replay intermittently wrong | H2D must be ordered on the consumer stream and host memory kept alive | Passing once excludes a race |
| Error at second M8 tile, N-column stripes | Enumerate actual epilogue ownership; first CTA must not write next tile's row | Rebase the A pointer without evidence |
| Profiler checker rejects a new implementation | Import saved reports; match exact producer, geometry and required reducer for that call | All symbols observed in a whole request must occur in one M1 profile |
| A grouped graph has two `prepare` roles and no producer | Classify the callable or CUTLASS's outer kernel type, not words in the full demangled template/argument list | `GroupDirectoryMixedInputKernel` is a directory-builder launch because its name contains `directory` |
| Query says JIT source differs | Rebuild the small dispatcher against the candidate source and refresh the package | Disable hashes or point an old manifest at a replacement DSO |
| Benchmark is days rather than hours | Measure wall-time fixture/build/child overhead and prune a declared inventory | Kernel duration times the cell count predicts campaign duration |
| Port compiles but native instructions balloon | Inspect exact PTX/SDK lowering, operand delivery and resources; see [native lowering](native-lowering.md) | Compiler acceptance means efficient native support, or the hardware lacks the arithmetic capability |
| Full API wins but ACU kernels lose | Separate event spans from per-kernel replay timing; measure same-call launch gaps before attribution | The difference of unlike measurements is measured Python overhead, or the kernels already beat the reference |
| A mixed-provider winner is presented as one family's parity | Bind the compiled reader/collective/scheduler and filter that family before finalist selection | A Machete reader plus Marlin scheduling proves Machete/Stream-K performance |
| Correct PDL graph is slower than ordinary/direct launch | Retain ordinary, protocol-only and early-load arms; change one dependency edge or resource policy and inspect the graph timeline | All kernels should enable PDL, or a longer kernel interval is entirely time waiting at the dependency instruction |

For independent family parity, retain that reference's full relevant config
inventory in screening and compare finalists with identical arithmetic and
complete-call work. H800 GPTQ example: Machete's StreamKScheduler may itself
choose data-parallel, Split-K or Stream-K decomposition; that is not a switch
to Marlin. Keep required workspace clearing/reduction inside timing. A5%
parity result is not strict no-slower, model admission or native PPU evidence.

Stable before/after clocks per arm need not be equal BETWEEN paired arms.
If matched clocks are part of the gate, retain the mismatched attempt and
retry only that point, admitting the first clock-matched attempt rather than
the fastest. Never edit failed raw receipts or recompute a campaign from a
hand-picked fastest sample. Evidence and reproducible independent gate:
`docs/ppu17-gptq-baseline.md`, `tests/ppu17/reference/machete_parity.py`
(H800, GPTQ INT4/group128, FP16 A/C, FP32 accumulation/reduction,2026-09-26).

SDK environment changes once caused even standalone markers and dequant to
fail image admission. A later machine/environment passed the same production
header bodies. Never infer a code fix from a machine switch alone.

An allocated MoE graph reused dead router logits storage. Whole-chain alias
rejection blocked most fused prepares; the bounded M1 fix relies on a register
snapshot before any store. This does not authorize generic overlap or deleting
live-input guards in multi-token cases.

Use fresh child processes after runtime/launch errors. Keep failed logs and
mark their timings invalid. Preserve unrelated valid experiments. If the same
environment failure recurs after targeted checks, report it and continue safe
host analysis; do not reset devices, kill another workload or rewrite the SDK.

The 2026-10-14 ZW810 resource-reuse pilot reported 8/12 PASS, followed by a
captured inventory `prepare, prepare, postprocess`. The host classifier matched
`directory` anywhere before testing `gemm`, misclassifying the grouped producer.
Callable/outer-type regressions reproduce and correct this parser error; the
device retry remains necessary for extracted replay and isolated timer gates.
Retry failed cells in a new source-bound run, leaving old PASS receipts under
their original authority; do not disable source checks or relabel old timing.

Keep diagnostics small: the exact first bad cell, immediate runtime code,
source/binary identity, and saved detailed logs are more useful than thousands
of repeated warnings. A CPU/profiler receipt is not device admission.

On physical H800 / CUDA12.8 SM90a / driver595.71.05, a fixed Q4
GEMM -> SF -> TMA GEMM graph regressed even without early weight loads.
Making only SF ordinary restored the baseline; target shared/L1 carveout25
removed much, but not all, of the slowdown. The target device bodies were
unchanged in the resource-policy control. This isolates an edge/resource
interaction, not its hardware mechanism; counters or a phase-instrumented
kernel are still needed before claiming L1 misses or dependency-wait cycles.
Treat carveout as a controlled diagnostic, not a universal preset. Keep native
PPU1.7 status separate. Evidence: `docs/ppu17-pdl-prefetch.md` (2026-10-01).

## Bounded fallback-rule audits (2026-10-14, host-only)

When the task asks rules to recover old winners, hide the complete target N/K
family before generation, including other M, precision and format rows. Reveal
historical winners afterwards as separate controls; a control-only win must
not be reported as rule recall. Exact recipe recall is still not performance
proof. A new-device holdout and original-wiring A/B remain necessary.

Count compiled parents, runtime recipe arms and independent helper jobs
separately. An SF parent exposes both ordinary and persistent entries: a
persistent donor must not hide ordinary Split-K. Conversely, resource probing
must use the exact ordinary/persistent, split and endpoint/compute entry.
Deduplicate B/mode aliases using actual algorithm/S/grid and probed CUs, not
an assumed 72-CU budget. High-coverage one-axis expansion can still be too
large: the 2026-10-14 host replay expanded 308846 arms for only 93.23% exact
recall. It was rejected as the default, not called an overnight plan.

Use durable per-arm receipts and fresh processes after failures. Test resume
with real fault-injected child processes, including tuple/list JSON identity
roundtrips. Bind imported fixture/oracle sources as well as the top-level
runner. A declared helper/grid domain is not executed coverage; fail the
preflight when its driver is incomplete. Evidence and host regression seams:
`dev/strategy/`, `tests/test_strategy_{space,journal,resources,provenance}.py`.

Follow-up blind replay: remove duplicate seed slots only when their ENTIRE
generated closures agree (warp-only expansion can collapse donor warps;
one-axis warp+column expansion cannot). Equal-ranked seeds should not lose
because of an arbitrary symbol hash. Preserve narrow/wide output tiles for
decode and narrow/throughput row tiles for prefill. Unrestricted cross-M
transfer increased work and regressed recall; vacant decode strata can borrow
legal cross-M geometry without replacing nearby donors. These are development
rules, not universal hardware performance laws.

Q4 specialized SIMT is not the generic all-format image: N/K and reader
parameters instantiate a new body. Count these specializations and require
an explicit carrier; a selected-only public API cannot benchmark arbitrary
proposals. For the existing Q4 row readers, whole-group K pass counts can
derive non-power-of-two warp neighbors without embedding a target winner.
They still require numerical, native-code and PPU performance admission.

After freezing candidates, report same-point historical confirmed-pool regret
separately from exact-config recall. A missed route champion may be irrelevant
to the overall winner, while high aggregate recall can hide a large isolated
regression. Near-tie within 5% is not strict parity; missing measurements are
not losses. Never feed this post-generation scoreboard into the target pool.
Keep production unchanged and admit no new-device performance on host tests.
Seams: `dev/strategy/{prescreen,q4_space,replay}.py` and their tests. Source
seals must include shell orchestration; resource resume binds host, PCI,
driver/runtime versions, SDK and exact module/compute/schedule identity.
