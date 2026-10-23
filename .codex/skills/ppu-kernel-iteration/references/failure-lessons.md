# PPU failure lessons

An Asys capture-boundary omission need not invalidate later complete samples.
The 2026-10-23 ZW810/SDK2.1.1 dense Q4 M256/N10240/K5120 trace contained48
successful reset API calls but47GPU reset activities. Only call0 of12already
discarded primer calls lacked the activity; all36formal calls had stable,
complete inventories. The old parser rejected before applying primer exclusion.
Allow only this evidenced first-primer exception: correlate the input launch
with a successful reset immediately after ProfilerStart, validate every other
kernel, and retain an explicit audit. Never time the API or invent the absent
GPU duration. Missing formal/reset/other-helper records still fail. Offline
corrections retain raw hashes, original failures and old measurement authority;
they do not mutate PASS receipts or authorize cross-source GPU resume. The
reason the profiler omitted the first GPU activity remains unknown. Seams:
`dev/prefill_asys/trace.py`, `tools/review_prefill_cohort.py` and their tests.

Do not use ast.dump as a cross-Python-version measurement compatibility seal.
Python3.12 adds FunctionDef.type_params=[] to the dump of unmodified Python3.10
functions. The R26 prefill supplement rejected its unchanged child/jobs before
device probing for exactly this reason (reproduced on both interpreters).
Seal the reviewed function source spans, including decorators, instead; reject
missing/duplicate functions and retain all helper, image, launch, SDK and
physical-device checks. Test the real entry boundary on both interpreters,
including changed-body negatives. Never bypass the check or relabel prior
timings. Seams: prefill_supplement.py, test_strategy_prefill_supplement.py.

Resource and timing drivers must not independently acquire the same device
`flock` while one waits for the other. Passing the parent's locked fd to a
child does not authorize the child's second `open()` to acquire that lock.
The large-M wrapper did exactly this; a host-only real-subprocess regression
reproduces `BlockingIOError: [Errno 11]` before the resource probe can work.
Run the probe under its own lock first, then acquire the measurement lock;
retain the inherited fd only for measurement children that do not reopen it.
Keep failures' bounded log tails visible in the parent instead of reporting
only "resource phase incomplete". This fixes orchestration, not numerical or
performance admission. Reuse compiled module caches in a new source-bound
run; never delete lock files, disable exclusion, or bypass old source seals.
Seams: `tools/run_kpack_strategy_prefill.py`, `tests/test_strategy_prefill.py`
(2026-10-15; pre-fix regression RED, post-fix host suite 63 PASS; real PPU
resource/timing retry remains required).

Result transfer is not raw-run backup. For large rule campaigns, recursively
archiving JSON also copies expanded plans, per-family manifests and resource
queries even when binaries/attempts are excluded. Export all current candidate
accuracy summaries and costs (including losers/failures), deduplicated config/
kernel identities, referenced component samples, finalist confirmation and
bounded failure-log tails. Keep source/SDK/device identity once. Leave raw
receipts on Box for targeted follow-up; the compact file is not a full
admission certificate or a resume cache. Do not modify an active source tree
just to package results: use a standalone stdlib exporter outside that tree.
The archived five-point ZW810 screen shrank5280850 to168476bytes in the initial
host check, retaining416currentarm records and454components. This is a measured
sample reduction, not a promised full-campaign size. Seams:
`tools/pack_kpack_strategy_review.py`, `tests/test_strategy_review_pack.py`.

Large-M component harness additions (2026-10-14, host validation only): do
not extend a decode-only indexed binder by removing its M check. Use the
general typed grouped endpoint with resident GPU offsets, retaining its real
metadata/directory and reducer kernels. Time caller adapters separately and
state whether creation of routing maps is outside scope. For indexed full
dequant, reuse helper timings across M only for the exact same active ID set,
original E-axis strides, precision, binary/source and cache protocol; SF's
all-expert metadata expansion is a different domain. A family cold-ring bound
can use the minimum active set across every declared correctness replay, but
must assert that bound at runtime, not assume all experts are always active.

Python JIT providers may allocate temporary directory tensors during graph
capture. Extracting native kernel nodes without retaining the framework graph
allocator can leave stale device pointers. The large-M harness captures with
Torch, clones the graph before capture ends through a compiled SDK bridge,
and retains the original graph/pool owner until extracted graphs are gone.
CUDA permits operations on an active capture graph except destruction/node
removal ([runtime contract](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__STREAM.html));
this is an API/design basis, NOT proof of PPU runtime support or speed. Local
ownership/ABI tests and HGCC object compilation do not replace the Box gate.
Evidence seams: `dev/strategy/prefill_*`, `tests/test_strategy_prefill.py`.

Follow-up ZW810/SDK2.1.1 observation (2026-10-15): the dense BF16 cuBLAS
provider failed the combined capture-info/clone bridge with801 for Q3
M128/N1024/K5120; the preceding full-dequant jobs passed. That message alone
does NOT distinguish GetCaptureInfo failure from GraphClone failure. Log
each API and graph node type before blaming the provider or changing kernels.
cuBLAS can capture allocation/free nodes, which CUDA GraphClone disallows;
an explicit256-byte-aligned persistent workspace, bound AFTER SetStream,
is a candidate mitigation, not yet proven for this PPU implementation. Keep
the workspace through graph replay and handle destruction, with redzones.
Do not remove allocation/free or other non-kernel nodes to make extraction
pass. The one-point default/resident probe uses separate processes and the
same physical-device lock as the campaign; active sources/results stay fixed.
Follow-up evidence fromqiUnYdHg: resident workspace yielded60kernel nodes,
zero non-kernel nodes and successful clone, but runtimeGetKernelNodeParams
failed500 (named symbol not found). Do not mistake successful clone for
usable extracted kernels, or blame numerical correctness from this failure.
The user requested NO CUDA Graph for cuBLAS. That capture probe is retired.
The replacement Asys/HGTX path then reported0/135 measured ranges; that alone
does not prove missing GPU activity, and its range failure root is unverified.
The user explicitly chose direct CUDA events, without Asys. Current
`probe_kpack_cublas_ordinary.py` warms ordinary calls then times only the
provider API; input/output conversions and expansion remain separate. A
provider span includes any internal reducer AND host-submission/device gaps;
do not relabel it a single GPU kernel, subtract guessed launch latency or add
the internal reducer twice. Keep this method distinct from isolated graph
kernel costs; mixed-method sums are screening estimates, not model E2E.
Offline review may reuse the previous UNPROFILED events worker only after its
exit, authority, numerical controls and complete sample inventory validate.
Do not turn a failed profiler receipt green or invent unmeasured adapter times.
Existing TC/dequant/grouped timers remain unchanged; host tests are not PPU proof.

Saved ordinary-call events from rtPK66KP (Q3 M128/N1024/K5120, ZW810/72,
SDK2.1.1) had provider round medians59.32/65.76/59.68us and first-sample
spikes183.64/147.08/118.12us. The source placed Python/Torch stream/view/
pointer work AFTER the start event, synchronized each phase, and performed
D2H/CPU oracle work between rounds. Those are confirmed harness seams, not
proof that cuBLAS changes algorithms or clocks. Use prebound native ordinary
submissions with separate event pairs, one final completion sync, and oracle
work outside the whole timed sequence. Retain all samples and CPU submission
diagnostics without subtracting them. No Graph or profiler is required.
The uT4gbzkq follow-up on the SAME physical card/SDK/workload passed nine
numerical checks and135phase event samples: provider26.24us, round medians
26.40/26.24/26.24us (spread0.61%). Input12.80us and output4.28us were separate.
This resolves that single-point harness gate, not every large-M shape. Native
CPU submission24.656us is a diagnostic, NOT subtractable from26.24us. Do not
call59.68->26.24us a GPU-kernel optimization or retroactively change old scopes.
ArchiveSHA256:983e5184fc25208303b83e57e99b49e1ad09082ee97fc2a53247bace13480d8f.
Keep CUDA compatibility cuBLAS headers in a separate HOST translation unit
from native HGGCRT/device headers; combining their include roots caused
type/math declaration conflicts on this SDK. Bind the actual SDK ABI and
pass the already-loaded cuBLAS function/handle, not a second provider DSO.

Changing one provider's timing protocol need not invalidate unrelated PASS
components. The reviewed4ad5bb8->4e46a0f migration retains byte-identical TC,
dequant and isolated-adapter DSOs and old receipts under their OLD authority;
only the ordinary-cuBLAS helper is new. Re-run EVERY dense provider job, even
if its old graph attempt passed. Require identical workloads/incumbents/rules,
SDK, physical PCI/host, resources and non-provider timer/oracle sources. A
small-M policy/catalog hash changed while all430large-M points stayed equal;
do not disable whole-source validation to accommodate it. Bind an explicit,
reviewed changed-file/hash transition and retain negative tests for altered
fixtures/timers/images/devices. Match multi-GPU continuation by physical card,
not ordinal. Export old/new authority and method separately. Scope: component
screen only; no model admission. Seams: `dev/strategy/prefill_upgrade.py`,
`tests/test_strategy_prefill_retry.py`, `tests/test_strategy_prefill_multigpu.py`.

Two equal captured launch inventories do not prove tuning completion or
optimality. Use ordinary untimed warmup, synchronization, and event timing;
for this experiment retain the requested separate kernel-stage timers.
The provider default is not an external algo search. Explicit workspace size
belongs in the measurement identity because it can affect provider selection.
See [cuBLAS workspace/graphs](https://docs.nvidia.com/cuda/cublas/index.html),
[CUDA graph restrictions](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html),
and [NVIDIA explicit cuBLASLt tuning sample](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtSgemmSimpleAutoTuning).

Use the category that matches the observation; these are not reasons to add
every diagnostic to every kernel.

| Observation | First discriminating check | Do not conclude |
|---|---|---|
| Outputs remain poison; sync returns success | Clear/check prior status, immediate launch status, then sync/deferred status; run an existing marker in the same carrier | Decode formula is wrong just because output is zero |
| Immediate status200 / InvalidKernelImage, including marker | Inspect SDK/compiler/runtime identity and native image admission; compare known runnable carrier | Header-defined kernel, aggregate ABI or CuTe must be broken |
| Failure appears after adding F32 endpoints | Inspect first nonfinite tensor and exact activation/dequant precision boundaries | F32-to-F16 cannot overflow, or clipping is harmless |
| Changing-input replay intermittently wrong | H2D must be ordered on the consumer stream and host memory kept alive | Passing once excludes a race |
| Extracted replay differs after an asynchronous linked-backend launch | Wait for the producer stream BEFORE saving the reference output; tensor_get may wait only for its own copy stream | A later synchronized oracle repairs an already-stale host snapshot |
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

Native-component diagnostic pdvGnj2J (ZW810, 2026-10-19) exposed a missing
producer wait in its changed-input reference download. llama graph compute
uses a nonblocking stream; tensor_get used cudaStreamPerThread and waited
only for the D2H copy. A host two-stream negative reproduces this race, but
device closure at that point still required the fixed diagnostic.
Keep comparison phase/residual and partial independent proofs on failure.
Separate this from the six completed Q4/Q5 dense precision controls: the
original F16 path has about9% abs-dot-normalized error even when extraction
matches; BF16/F32 controls are below.031%. Do not relax the gate, blame
Quactlize, or substitute a faster/different native arithmetic baseline.
Evidence: docs/measurements/native_component_failures_20261019.txt.

Follow-up sHDpIkfa (95.33seconds):17/17diagnostic jobs completed; all34
linked/extracted full-output comparisons across two inputs have zero residual.
Q4/Q5 native Q8_1 residuals are below8.621e-8. Preserve the separate statuses:
this admits the sampled harness seam, not every historical failed point or
new performance. Retry only the reviewed failures in a new run with explicit
old/new authority. User-deferred native-provider numerical failures must stay
marked deferred, not become successful rows or forced-precision baselines.

On physical H800 / CUDA12.8 SM90a / driver595.71.05, a fixed Q4
GEMM -> SF -> TMA GEMM graph regressed even without early weight loads.
Making only SF ordinary restored the baseline; target shared/L1 carveout25
removed much, but not all, of the slowdown. The target device bodies were
unchanged in the resource-policy control. This isolates an edge/resource
interaction, not its hardware mechanism; counters or a phase-instrumented
kernel are still needed before claiming L1 misses or dependency-wait cycles.
Treat carveout as a controlled diagnostic, not a universal preset. Keep native
PPU1.7 status separate. Evidence: `docs/ppu17-pdl-prefetch.md` (2026-10-01).

In the ZW810 five-point cold-rule screen (IdVdoJoi, source9edbbf5), execution
wall was984.89s, with about715s in the small dense Q4 point's110 arms. A cold
SF prepass reads only64KiB of metadata, requiring2304 copies for2.25*64MiB L2;
using that ring for every GEMM also multiplied handle/capture preparation.
Keep an independent metadata-only ring for expansion and an active-packed
ring for producers. Each still needs its own verified cold-read footprint;
shrinking the SF ring using total packed bytes would falsely measure hot SF.
This observation identifies harness work, not a kernel speedup or a measured
speedup for the revised harness. Do not linearly project984s/406 to all shapes.
Also distinguish a noisy losing screen arm from an unstable winner: an
irrelevant slow outlier need not force resweeping every candidate. Retest
competitive finalists and the actual incumbent in alternating rounds, keeping
noisy candidates whose best rounds remain competitive. Raw archive SHA256:
fc836f44dcc15a5603b3455e30bb116b89ee225c0c91c6a62f6256f74c274c20.

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
roundtrips. Exercise the freshly generated plan after its LAST binder, not
only fixtures already decoded from JSON. In the 2026-10-15 prefill retry,
`bind_original()` reintroduced tuple recipe parameters after `make_plan()`
normalized them; direct comparison with saved lists rejected an unchanged
430-point inventory before GPU work. Normalize at the freeze return boundary,
and retain negative tests for actual workload/config/shard changes. Reproducer:
`tests/test_strategy_prefill_retry.py::test_fresh_full_prefill_plan_matches_its_json_for_retry`.
Bind imported fixture/oracle sources as well as the top-level
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

Updating the production table can change the donor catalog even when the
proposal code is unchanged. On 2026-10-16, leave-NK-out replay of the updated
small-M catalog recalled 877/930 selected host-key recipes and missed 53
(22 TC, 11 generic SIMT, 20 Q4). These configurations already had measurements;
this was a rule-regeneration gap, NOT 53 missing tests or a measured runtime
regression. Q4 indexed M5/N512/K2048 W4 was confirmed in the frozen campaign
but absent from the regenerated pool. Treat proposal catalog + rule revision
as one versioned input. Audit after table promotions, keep measured evidence
immutable, and distinguish exact-table availability, blind generation recall,
competitive-pool regret and model A/B. Repair generic coverage before claiming
the legacy selector is unnecessary; never silently inject the target champion.
Evidence: `policies/kpack_component_decode_v1.json`, `dev/strategy/{prescreen,
neighbors,q4_space}.py`, and the task's `decode-rule-recall-20261016.json`.

## Confirmation roles are provenance, not permanent eligibility (2026-10-19)

In the 430-point ZW810/SDK2.1.1 prefill confirmation, a helper variant labelled
as an old control was sometimes also independently generated by the blind
R27 rules. Freezing the screen's helper winner hid valid faster measured
compositions. Replay eligibility without the target N/K family, then choose
the fastest candidate that meets the full confirmation gate, not the fastest
median first followed by rejection of the whole point. Keep original roles,
verdicts, raw samples and the revised decision as separate records. Never
invent an unmeasured pairing or drop a reducer/adapter from a measured job.

The review retained 412 PASS/16 RULE_REGRESSION/2 UNSTABLE raw verdicts while
qualifying all 430 selections; two noisy controls had strictly separated
observed ranges from stable candidates. This is a disclosed review criterion,
not permission to ignore overlapping noise or claim statistical confidence.
Transfer route, precision, exact TC recipe and helper together into the
caller: a second fixed-route selector can silently discard the measured win.
Test the materialized ticket as well as the pure lookup. A bounded caller
pilot must cover selected provider routes too; the former 22-format pilot
contained no FULL path despite that path winning 186 prefill points. None of
these component results establishes native llama parity or model TPOT.
Evidence and limits: `docs/measurements/kpack_prefill_confirmation_20261019.txt`,
`dev/strategy/prefill_review.py`, `tests/test_strategy_prefill_review.py`.

When reusing a CMake target's flags for a narrow test adapter, preserve its
compile working directory as well as argv. The native comparison runner
copied `--options-file CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp` but launched
from the top-level build; the file belongs under `build/ggml/src/ggml-cuda`.
Local SDK2.1.1 compilation reproduces "Could not open options file" at the
wrong cwd and succeeds at the target cwd. Keep the native response file and
flags intact, seal its contents in the receipt, and expose compiler stderr
in worker/coordinator errors. Object compilation is not full linking or GPU
execution. Seams: `dev/native_components/build.py`,
`tests/test_native_components_build.py` (2026-10-19).

Apply that diagnostic path to host carriers as well as device adapters. A
truncated g++ argv/CalledProcessError does not distinguish a source error,
missing dependency or unresolved symbol. Preserve and inspect the existing
carrier-build.log before another GPU campaign; host syntax success is not
proof of linking against the Box backend and its transitive dependencies.

The 2026-10-20 new-clock small-M refresh completed1098/1114points; all16
incomplete points stopped at a strict1700MHz check with1537..1687MHz actual
CU clocks. The user changed this experiment's contract: request1700MHz, then
treat actual frequency as diagnostic, not an automatic failure/retry gate.
SDK2.1.1 ppu-smi v1.28 uses --lock-ppu-clocks=1700,1700, not --lock-gpu-clocks.
Target only the resolved assigned PCI devices; preserve power/memory settings.
A successful request does not rule out hardware throttling, and frequency
alone does not prove a TDP or temperature cause. Keep unscaled event times and
numerical/launch/identity checks. Preserve strict-policy raw receipts; retry
only incomplete points under a separately sealed advisory policy. This is
user-authorized cohort policy, not a universal exemption from clock controls.

An experiment can preserve the production geometry yet execute the wrong
implementation. The 2026-10-20 MoE M8 gap audit found the explicit SIMT carrier
calling measured_decode/generic launch_v2 but missing the newer common-reader
dispatch above them. Raw captured register_reuse symbols proved the mismatch;
an "implementation=shipping" label did not. Mirror dispatch ordering and
guards, compare native bodies, and assert the captured producer entry. When
this changes a comparison's control body, remeasure every required router
profile for that key; do not merge the old control's absolute times or call
them measurements of the new body. Keep unrelated archived verdicts intact.
See docs/measurements/decode_profile_gap_handoff_20261020.txt. This corrects
the experiment, not a production selector or a model-performance claim.

Test symbol guards with raw captured names, not only hand-written readable
names. GNU c++filt can return a valid deeply nested CuTe symbol unchanged at
its recursion limit; seven such TC symbols caused all49 profile-gap points
to fail a post-timing exact-name guard. --no-recurse-limit reproduces their
actual device_kernel identity on the host. For outer-entry classification,
recognize only the exact mangled namespace/name/template boundary instead
of recursively expanding types or substring matching nested helpers. Keep
unknown/mismatching entries rejected. Save raw measurements as UNADMITTED
when postchecks fail, separately from PASS receipts; otherwise a host-only
validator bug discards the samples and requires another GPU run.
