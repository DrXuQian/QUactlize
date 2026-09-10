# K-pack execution follow-up

Updated: 2026-09-10. The O0ki3q native micro gates and adapter tests pass;
model prefill improves, but decode remains about 31% slower and the short
trace lacks selected native dense compute. This is not optimized-routing
closure. Canonical offline planes are unchanged.

Latest JIT retry `5gSImm`: **28/28 native contexts pass in both processes**,
including the original Q4 SF-grouped and Q6 SF-dense failures. The unchanged
23 device modules were reused in all 56 cache resolutions. Metadata and
eager/graph checks pass; max normalized error is `2.45e-4`. Native operator
closure is complete for this gate, not for the full llama model or additional
shapes. See [evidence, timing and remaining boundaries](KPACK_JIT_GATE_REVIEW.md).

## Complete delivery backlog

This is the delivery-level list; the local/box tables below break down only
part of it. Earlier gate results are evidence for their recorded scope, not
proof that the complete production chain is finished.

Current decisions: automatic decode stays on FQ. SIMT GEMV is provisionally
accepted for this milestone by user decision and further optimization is
parked; this does not rewrite individual benchmark results or promote GEMV
over FQ. ScaleFirst must expand metadata on the GPU for every SF call, then
consume it. Workspace may be reused, but expanded values must not be cached
across calls or amortized across requests in the selector.

| ID | Workstream | Actual status / remaining work | PPU box boundary |
| --- | --- | --- | --- |
| T01 | Per-call ScaleFirst execution and selection | Native PPU gate passes both processes, including metadata poison and eager/replay checks; per-call cost is included. V2 pairwise decisions agree 14/14. No production routing change in this review | Validate the updated llama adapter/full-model path; do not promote SF for every prefill shape |
| T02 | Production single-parent JIT | `kpack-jit-v2` passes 28/28 twice using the original 23 compiled parents. No rebuild on 56 resolutions. Cold compilation remains expensive; prewarm is recommended | Same-parent prebuilt/JIT comparison and final model deployment scope remain |
| T03 | Model-load preparation and JIT misses | Implemented bounded prewarm: unsplit GGUF header inventory or explicit requests, actual C++ heuristic, deduplicated parallel compilation; llama enables JIT before graph preparation. Unknown families still explicitly miss; sharded/TP discovery remains explicit-request mode | Cold startup, disk-cache hit, process restart, new-M/router and graph replay |
| T04 | Production module-cache lifecycle | Host contracts pass; PPU two-process reuse passes with exact stable parent keys. Cache helper median remains about 0.4 s per resolution in this gate, so startup cost is not zero | Full-model startup/TTFT, moved-image/device scope and resolution-tail cost remain |
| T05 | Small-library delivery and legacy dependency removal | Small JIT dispatcher/execution candidate built (about 1.1 MiB); intake/conversion/admission extraction remains. The model still needs the old six libraries for intake and explicit FQ fallback; do not remove them before replacement coverage | Load and execute with legacy libraries deliberately absent; compare cold/hot startup and model results |
| T06 | Complete heuristic/recipe contract | Partial: measured/heuristic C++ selection is wired, not globally optimal coverage. Keep one owner for parent, AP, delivery, Split-K, scheduler/grid and legal optimization axes; cover misses without compiled-default guesses or restoring a Cartesian sweep | Bounded challenge around coverage gaps and historical winners, not universal 5% claims |
| T07 | Q8_0 production integration | Partial: controlled ScaleFirst/I8 collective and sweep exist. Reuse them; add the production GGUF/device producer, arrangement, ABI, selector and dense llama admission. Do not feed the historical Xplane fixture through the K-pack API | Producer bytes, decode/prefill numerics and matched Q8 native-reference timings |
| T08 | Q6 output-head native coverage | TODO: N248320/K2048 still takes labelled legacy FQ. Inventory/build a bounded candidate set and provide a measured selected-native route | Same-shape correctness/performance and actual dense compute trace |
| T09 | End-to-end adapters and scheduling | Partial: GPU routing/compact directory, gather/scatter and prepared handles exist; newest FQ decode route is locally compiled. Verify selected TM8/S4 Q4 and TM8/S1 Q5 in model execution; diagnose launch/adapter gaps | New Asys timeline and unprofiled request-batch-1 latency |
| T10 | Final accuracy/performance release gate | Pending for the final module/route set. Reuse unchanged evidence, but do not relabel historical model gates as validation of new paths | Paired GPU-reference numerical/PPL/GSM8K, actual kernel execution and full-model latency |
| T11 | Optimization-axis coverage | TODO: reconcile recorded B-chunk, AP/packed-A, delivery N, fused scale/zero, stage and scheduler results with generated/native tactics. Distinguish untested combinations from rejected or numerically invalid ones | Only missing or changed promising combinations need bounded PPU checks |
| T12 | Reader/provider extensions | Deferred: AIU+UniversalCopy, cp.async+reader and DeepGEMM-compatible placement need an explicit offline-layout/fragment contract and adapter boundary. Existing provider experiments are not blanket correctness/performance admission | Exact reader/layout numeric and performance gates before promotion |
| T13 | Product main cleanup and documentation | Pending: minimal public entry points, reproducible config/JIT scripts, README/ABI/handoff, unused flag/debug removal; enforce the recorded main-admission skill, PPU-only code and required production formats | Final release regression after selective cleanup; no Xplane/NVIDIA diagnostic path admitted implicitly |
| T14 | SIMT GEMV further optimization | PARKED: user accepts the current comparable level for this milestone. Preserve code and evidence; not a blocker and not selected by automatic decode | No new gate solely for this parked item; reopen only for a new regression/target |
| T15 | Cross-call weight prefetch | Two bounded experiments implemented: target benefit and concurrent-prefetch interference. Reuse Q4 TM8/S4 and Q5 TM8/S1; no production changes or hard CU partition. Initial box attempt stopped on captured-event timing; repaired runner uses explicit timestamp nodes and an early five-arm timer gate, with unchanged binaries | Repaired PPU timer admission, then unprofiled A/B, real timeline concurrency and cache-pressure sensitivity; ACU kernel replay must not erase prefetch state. [Protocol](KPACK_PREFETCH_EXPERIMENT.md) |

The locally implemented path is heuristic -> one complete tactic ->
prebuilt/disk-cache hit or explicitly enabled single-parent compilation ->
prepared execution. See [JIT package, commands, timing and boundaries](KPACK_JIT.md).
PPU admission is limited to the recorded native gate. Keep online multi-candidate tuning opt-in
and outside the default model path; the old six-library dependency is not
removed by shrinking the dispatcher.

Implement/compile T01-T08 and their host tests locally where bounded; actual
PPU admission follows the last column. T02-T04 can be developed alongside
T07/T08 and do not require another full sweep. T05 depends on those coverage
and miss-path contracts. T13 follows stable runtime/ABI selection. Do not
restart T14 as an active performance project.

## Tracked delivery

Grouped direct FP32 partial publication and the compact fixed-S reducer are
implemented and packaged separately. Nine PPU parents compile; actual PPU
coverage now passes nine jobs / 384 cells across two preserved runs. Q4 compact
S4 measures 14.59 µs versus 17.58 µs for its same-parent baseline; Q5 compact
still favors S1. S1, mainloop, offline bytes and workspace are unchanged. The
native selector now adopts Q4 compact TM8/S4 and Q5 compact TM8/S1 only on
their exact M8/E256/max_rows1 anchors; all other choices remain unchanged.
Model admission with this new selection remains open; see
[the bounded A/B gate](KPACK_GROUPED_POSTOPS.md#reviewed-ppu-closure).

| Item | State | Completion condition |
| --- | --- | --- |
| 3. Native selected-module binding | Micro gates pass; Q6 output-head policy coverage remains open | Both hooks are wired, but N248320/K2048 dense head misses the native policy and retains labelled legacy K-pack FQ. No Python/JIT/online timing in inference |
| 4. Device-only grouped metadata | Correctness and changing-router replay pass; performance open | No per-token D2H; isolate the cost of rectangular device-only scheduling against the same-parent compact diagnostic |
| 5. ScaleFirst prefill | Native per-call PPU gate passes; updated full-model adapter gate remains | Cross-call expanded-value reuse removed. V2 timing includes every expansion. No inference wait on cache D2H; do not generalize the measured FQ/SF crossover to untested shapes |
| Decode GEMV | Provisionally accepted for this milestone; optimization parked | User accepts the current comparable level. Preserve the matched evidence and its scope; automatic decode remains FQ |
| Uniform FQ decode | Automatic dense/grouped single-token routing changed in llama.cpp; deployment requires adapter rebuild | Ignore GEMV policy in `auto`; preserve measured FQ parent/Split-K/compact selection and prefill FQ/SF policy. Forced GEMV/SF stay diagnostic. No Quactlize kernel DSO rebuild; this does not by itself close model performance debt |
| Model decode regression | Open performance debt | Isolate the measured per-token gap with matched work; retain the old GEMM incumbent and do not attribute the whole gap to one missing algorithm |
| Grouped Split-K / SIMT pair reader | 260/260 PPU cells pass; performance reviewed | Q4 compact S2 improves on compact S1; Q5 compact S1 remains best. Pair reader improves both SIMT anchors. Host compact excludes CPU preparation and is not a production replacement |
| GPU compact / persistent Split-K | PPU 204/204 pass; newer postops choices integrated below | [Reviewed gate and ACU capture](KPACK_GPU_COMPACT.md) include directory cost, mutable GPU routing and FP32 partials. Persistent is not the anchor winner. External ABI unchanged |
| SIMT NVIDIA diagnosis | Comparison evidence retained; no further optimization in current milestone | [Matched evidence](../dev/gemv_cuda/README.md#matched-llamacpp-comparison) remains unchanged. T14 is parked by user decision, not a new claim of all-shape parity or PPU promotion |
| Equal-weight dense/grouped reproduction | [Five-arm prebuilt runner](KPACK_DENSE_GROUPED_AB.md) ready for PPU measurements | Q4 N4096/K2048 dense versus eight N512/K2048 experts, identical logical weights/A; historical winner plus matched tile controls. SF excluded |
| Split-K reducer optimization | Bounded PPU gate: 9/9 jobs, 384 cells; measured Q4/S4 and Q5/S1 native choices integrated | Only two modules added; old 214 preserved; no device compilation. ACU reducer 5.77→2.16 µs is separate from warm timing. Full-adapter/model measurement remains; [review](KPACK_GROUPED_POSTOPS.md#reviewed-ppu-closure) |
| Q8_0 production integration | Existing controlled resident ScaleFirst/I8 kernel and sweep; production integration missing | Reuse the existing int8 collective and converter. Add the GGUF/device producer, production arrangement/ABI, selector and llama.cpp admission with independent tests. The historical A32/F1 Xplane fixture is not a production K-pack artifact. There is no shipping Q8 FQ reader; keep llama.cpp Q8_0 routing until the replacement passes PPU gates |
| PPU GEMV / FQ / SF comparison | k5Tp2m: 5 cases, 240 SIMT recipes, 25 confirmed arms and 25 ACU reports pass | [Reviewed results](KPACK_GEMV_FQ_SF.md#reviewed-ppu-results-k5tp2m-2026-09-09). Per-call SF must charge the ~55 us all-256-expert expansion on these small MoE inputs; resident-only timing is not its full-call cost. No new measurement or binary change |
| SF grouped decode selection | Open scheduling debt | Q5-down SF remains a rectangular 8,192-CTA `DEVICE_BOUNDS` choice versus 256 compact FQ CTAs. Measure compact SF before treating this as the format's best performance; Q4-up resident SF/FQ are within 3% |

## Local work and required PPU gates (2026-09-09)

"Local" means no PPU box is required for the listed deliverable, not that
compilation proves numerical correctness or performance. RTX 5090 experiments
are optional additional device evidence and cannot admit PPU kernels. Keep
hour-scale builds on the box as requested; small selective SDK builds remain
local. No fresh full Cartesian sweep is required for the tasks below.

Q8 status was rechecked in source: the `Q8` specialization in
[`test_scalefirst_bench.cu`](../benchmarks/test_scalefirst_bench.cu) calls
`FinegrainedScaleOnly` with `int8_t`, group size 32, and no zero plane.
[`controlled_scalefirst_row`](../tools/prefill_sweep.py) explicitly limits it
to controlled resident GEMM; the checkpoint split/reorder producer is not
connected. The resident code is `q+128`, not the raw signed GGUF byte.
[`fully_quantized_internal_matrix.py`](../tools/fully_quantized_internal_matrix.py)
records Q8 FQ as unavailable. This is an integration task building on existing
SF kernel work, not a claim that Quactlize has no Q8 implementation at all.

### Can be completed without a PPU box

| ID | Deliverable | Status / completion boundary |
| --- | --- | --- |
| L1 | Audit reusable Q8 ScaleFirst implementation and production gaps | Source audit done above; no Q8 production admission claimed |
| L2 | Q8 production format and wiring | TODO: reuse the int8 collective; define the separate weight/scale arrangement, GGUF GPU producer and inverse/reference, C ABI, selector and llama.cpp dense admission. Host round trips, unsupported-type rejection, CuTe layout proofs, SDK compilation and ELF/ABI checks are local; B2/B3 admit execution |
| L3 | Q6 output-head native coverage | TODO: inventory candidates for N248320/K2048, reuse historical measured configurations where applicable, prepare a bounded comparison and compile missing candidates. Do not replace the labelled legacy FQ miss with an unmeasured supposed winner; B2/B3 remain required |
| L4 | Adapter and evidence tooling | TODO: strengthen route/parent/Split-K receipts and graph/lifetime tests; inspect uploaded traces for CPU waits, allocations and routing/cast/scatter cost. Trace capture is B1; fixes can be implemented/compiled locally, with device replay still required |
| L5 | SIMT optimization backlog | PARKED by user decision; retain current implementation and evidence, no additional optimization required for this milestone |
| L6 | Packaging and maintainability | TODO: inventory required exports/modules and unused flags, keep the single integration handoff current, clarify reader/provider boundaries and prepare cleanup. AIU+UniversalCopy/provider extensions remain separate experiments; product main admission is not authorized by static cleanup alone |
| L7 | Per-call SF and production JIT | Local implementation, SDK compilation and host tests completed for T01-T04's bounded scope; new candidate package and gate ready. PPU numerical/replay/performance admission remains separate |

### Must execute on a PPU box

| ID | Test | What it establishes |
| --- | --- | --- |
| B1 | Rebuilt llama adapter tests plus one new model Asys capture | Automatic dense/grouped decode really uses FQ; Q4-up selects compact TM8/S4 and Q5-down compact TM8/S1 on the measured anchors. Inspect launch gaps, auxiliary kernels and explicit Q6 fallback. Existing Q8_0 remains native llama |
| B2 | Correctness and graph replay for changed/new device paths | Q8 GPU producer bytes, dense decode/prefill output, tails, scales and stream lifetimes against independent oracles; include grouped only if exposed. Also test any added Q6 parent and adapter changes on PPU |
| B3 | Matched operator/model performance | Measure selected FQ and new Q8/Q6 paths against the appropriate same-PPU baseline. Include adapters/reducers in full-call timing; use ACU for operator counters and separate unprofiled samples for latency. SIMT further optimization is parked |
| B4 | ScaleFirst per-call expansion timing | Measure GPU expansion+GEMM for every SF call and graph replay, including scratch footprint and correct stream ordering. Reusing allocation is allowed, reusing expanded values across calls is not; compute must not wait on background cache D2H |
| B5 | Final model accuracy and release evidence | Paired GPU-reference numerical/PPL/GSM8K checks with actual route evidence after final selection changes; then review the release bundle/main admission |

Immediate order: B1 can run while L2/L3/L4 are prepared locally. Q8 and Q6
new paths remain unselected until their B2/B3 results are reviewed. Already
passed unchanged kernel gates need not be rebuilt/repeated just because the
consumer's automatic route changed.

Profiling default: standalone operators use ACU native reports; model/stream
timelines and launch bubbles use Asys. The compact ACU entry selects only the
two measured anchors and their old-module controls, not the entire gate.

The Q4 N512/K2048/E256/top8 single-token case was not omitted: overnight
screen/neighbor/confirmation evidence includes TM8, and the earlier runtime selected
the confirmed TM16/TN64/TK256 winner. Its historical 18.44 us is itself only
about 9.25% effective weight bandwidth; replaying that number is not closure.
See [the coverage audit and no-recompile single-op probe](KPACK_GROUPED_DECODE_REVIEW.md).

## Native model gate

Entry: `bash tools/run_kpack_native_box.sh` (see `--help`). This consumes
`prebuilt/ppu0010/kpack-native-v1`: 216 selected parent modules plus the native
host dispatcher and execution DSO. It does not replace the old six libraries,
which still provide intake/admission and explicitly logged K-pack FQ misses.
The box rebuilds llama.cpp for the changed context layout, not Quactlize.

1. Twenty dense and eight grouped contexts use the actual C++ selector and
   prepared module, independent GGUF arithmetic, output guards and changed
   router graph replay. ScaleFirst prepass is checked and timed separately.
   The updated exporter compares per-call expansion plus GEMM with a 2%
   margin. Existing resident-only tables are rejected; rerun this gate before
   choosing SF under the new execution contract.
2. Fourteen model-shaped GEMV contexts (Q4/Q5 experts, both broadcast/per-slot
   A, and Q6 N248320/K2048 dense output) compare all eight recipes in 3x11
   samples. The comparison uses selected FQ when covered, explicitly tagged
   legacy FQ otherwise. GEMV includes indexed access and reduction; FQ here
   excludes adapter casts/gather/scatter. This conservative gate is not a
   proof of global cross-algorithm optimality.
3. Production adapter tests use device tags to verify pointer/stride/routing,
   cached preparation, eager/capture/replay and mutable IDs. The `auto` case
   offers a GEMV recipe but requires FQ for single-token dense/grouped and
   retains the SF-preferring policy for prefill. Arithmetic is
   intentionally stubbed in this seam test, not in the separate numeric gate.
4. Real model ABBA: reference/native/native/reference, one business request
   at a time, 128/512 input tokens, 128 generated tokens, chunk=128. First-use
   samples are separate; steady timings include the whole model adapters.
5. A separate short Asight run must observe native dense and grouped compute.
   Its timings are never performance samples. It proves that short request,
   not every parent in all ABBA requests.

Reports retain parent/build ID, split, grid, policy class, explicit fallback,
prepass time and raw per-request timers. `fully_selected=false` is not admitted
as complete optimized wiring. A nearest-family grouped proposal uses the
caller-known token bound, not an unobserved exact router; it remains labelled
`DEVICE_BOUNDS`. FQ/SF route measurements cover exact gate contexts, not every
router distribution. Missing prefill entries retain selected FQ. The model
timings test whether those measured core choices win with actual adapters.
First-use allocations may have runtime synchronization cost; the explicit
ready-event protocol does not prove allocator calls nonblocking. That cost
belongs in first-use wall timing, not the prepass kernel timer.

### Native SF metadata oracle correction

The partial box log from `/workspace/kpack-native-model.q40qV3` reports all
four Q2_K contexts and both Q3_K FQ contexts passing; the Q3_K SF contexts
stop at the zero-plane byte comparison, before GEMM. That gate incorrectly
used the historical timing fixture's resident zero plane. For scale-only
Q3/Q6, that fixture starts zero at `-0`, whereas production `unit_group`
starts it at `+0` and applies the canonical correction in float before
rounding to FP16. Do not change the production kernel to match that fixture.

With the exact deterministic N1024/K5120/E1 fixture, the production C++ host
decoder differs from the historical zero plane only at 5,152 signed zeros
for Q3_K (first index 170, historical `0x8000`, canonical `0x0000`) and 1,180
for Q6_K (first index 29). Nonzero values and scale bits agree. This proves
an oracle defect locally; the partial box log did not include device bits,
so it cannot by itself prove that every device difference is signed zero.

The native gate now uses the independent packed-unit decoder already checked
against production C++ in the GEMV gate. Comparison remains raw-bit exact,
including zero signs. First-launch output is poisoned, and
`KPACK_NATIVE_METADATA` records unit SHA, both plane denominators, mismatch
counts, nonfinite/signed-zero counts and the first expert/group/N coordinate
and bits. A failed metadata check cannot produce admitted timing or policy.
Historical fixtures/calibration hashes, offline bytes, selected recipes and
all prebuilt DSOs remain unchanged. Local coverage includes all five formats,
multi-expert placement, the exact Q3/Q6 geometry and eight negative plants.
Pull and rerun the native box entry; no Quactlize binary rebuild or LFS update
is required for this correction. PPU closure and model performance remain
pending, as does the original llama.cpp adapter rebuild in that entry.

### Loader test environment isolation and adapter resume

The box subsequently reports 17/31 stub-loader cases failing. The test driver
overrode only `LD_LIBRARY_PATH`; the model runner's absolute
`QUACTLIZE_PPU_BUNDLE` and `QUACTLIZE_PPU_PACK_LIBRARY` overrides won, so the
stub fault switches were ignored by real libraries. `rc=256` is a `system()`
wait status for exit code 1, not a device error code. This failure does not
establish a kernel numerical failure.

The private llama branch clears production library/SDK overrides only for
stub child cases, leaving `--real`, `--bench`, and inference unchanged.
Buffer/sidecar CTests explicitly select their stub libraries. A separate
hostile-environment loader test prevents recurrence; both 31-case loader
runs plus buffer and sidecar pass locally with real library paths exported.

Set `RESUME_RUN` to the failed model run directory when calling the native box
entry. It validates all 28 native and 14 GEMV contexts, the saved numerical
and timing evidence, current native/execution/legacy binary identities, SDK,
recorded device identity, and unchanged measurement source. It copies only
the prior microbenchmark results into a fresh run, records their origin and
hashes, then regenerates the policies and rebuilds/tests the llama adapter.
Model timings and traces are always fresh. The old run is not overwritten;
partial/changed inputs are rejected rather than silently reused. No Quactlize
DSO rebuild is needed. Local resume/export coverage: 27 tests pass.

### Model tensor override scope

After the reference performance arm, the native server aborted in
`qz_buffer_set_tensor`: `blk.3.attn_output.weight` is Q8_0 but had received a
K-pack buffer. This explains the client's `Connection reset by peer`; it is
not evidence of a GEMM numerical failure or a valid native timing sample.

The model runner used `(ffn_.*_exps|output\.weight)`. The loader applies
`std::regex_search`, so the second alternative also matches the suffix of
`attn_output.weight`. A non-CPU explicit buffer override directly chooses
the requested buffer instead of calling automatic `weight_buft_supported`.
The K-pack setter's unsupported-format rejection was correct and remains.

Private llama commit `5837b4d86` anchors both alternatives to complete tensor
names: `^(blk\.[0-9]+\.ffn_[a-z0-9_]+_exps\.weight|output\.weight)$`.
The ABBA runner and separate trace share this rule. Regression tests check
161 intended names and 206 excluded names with both Python and the loader's
C++ regex semantics; the old expression demonstrably admits the failing
Q8_0 tensor name. Five local tests pass, including immediate startup abort
and connection-reset reporting with the arm, phase, process status and log
path. The runner records the exact override in its protocol receipt.

No production kernel, library or heuristic changed. All Quactlize DSOs and
complete micro gates remain reusable through the validated resume entry.
New model ABBA timings and the separate device trace are still required;
the failed native arm cannot establish performance or accuracy admission.

### Dense coverage correction

The old +29% GSM8K command overrode only `ffn_.*_exps`. Its 120 K-pack weights
were grouped; it did not measure dense K-pack integration. The old grouped
null config selected `16x128:16x16:s2`, Split-K=1. Historical dense Q4 selection
was different and already contained S1/S4 choices; it must not be described as
the same fixed grouped default.

The new command also selects Q6 `output.weight`. The model's 311 Q8_0 tensors
remain on ordinary GPU kernels: Q8_0 support is a separate task, not silently
covered by Q2_K..Q6_K. A model-wide speedup cannot be inferred from K-quant
microbenchmarks or from the grouped route alone.

## Reviewed GSM8K pilot

Archive `llama-kpack-gsm8k.EEZKMu.results.tgz`, SHA-256
`cb8b7906bd9af587e1c7d5b3dd98a50d6d76e16eced7823c37fd2619e923a624`,
records llama.cpp `a39917a66acd742ae02b2034549fded8ce9506bb`.
All 128 paired raw responses, route receipts and prompt/decode timers were
replayed against the summaries. Both processes exited zero. Each arm scored
122/128 (95.3125%), with no paired correctness flips. Each has one truncated
and unparseable answer, but those are different questions between arms.
Only 43 complete outputs are identical. This is scoped task-accuracy evidence,
not bit equality, full GSM8K, or a new device trace.

| Aggregate | Ordinary GPU reference | K-pack |
| --- | ---: | ---: |
| Input tokens | 14,012 | 14,012 |
| Generated tokens | 42,433 | 42,822 |
| Prompt time, s | 22.492235 | 8.467084 |
| Decode time, s | 343.420132 | 448.025652 |
| Token-weighted decode, ms/token | 8.093232 | 10.462511 |
| Request wall time, s | 366.056342 | 456.649109 |

Decode latency per token is 29.27% higher. The 43 identical-output pairs
still have a median paired increase of 29.17%. All cache accesses hit, no
GPU repack occurs, and both arms reuse graphs. The ordinary reference is
llama.cpp's GPU path, not the historical standalone Xplane board. Runs are
reference-first, not interleaved ABBA; no clock/co-tenant receipt or kernel
trace was collected. These timings establish an observed regression, not
its isolated kernel cause.

The existing sweep explicitly excluded canonical K-pack BC/GEMV
(`NO_CANONICAL_KPACK_BC_READER`). The current llama.cpp K-pack branch calls
FQ grouped GEMM even for one token, while its ordinary quantized branch can
use MMVQ. Add GEMV as a separate algorithm candidate, not as another GEMM
tile or an unmeasured default. Keep prior measurements and coverage labels.

## ScaleFirst preparation cost and lifetime

For E experts and logical weights N x K, let W = E*N*K, G be the scale
group size, and U the packed metadata bytes per 256-weight superblock.
The current ScaleFirst contract materializes **both** FP16 scale and zero
planes, including the canonical affine correction for Q3/Q6:

    packed metadata bytes      = W * U / 256
    expanded scale+zero bytes  = W * 4 / G
    ideal prepass traffic      = W * (U/256 + 4/G)

This is a minimum logical read/write volume, not measured DRAM transactions
or time. Actual duplicate loads, decode arithmetic, launch overhead and
allocation can increase cost. Weight-code planes are not expanded or copied.

| Format | G | U | Expanded metadata / packed code bytes |
| --- | ---: | ---: | ---: |
| Q2_K | 16 | 20 | 100% |
| Q3_K | 16 | 14 | 66.67% |
| Q4_K | 32 | 16 | 25% |
| Q5_K | 32 | 16 | 20% |
| Q6_K | 16 | 18 | 33.33% |

The reviewed model's cache manifest contains 80 Q4_K tensors at
N512/K2048/E256 and 40 Q5_K tensors at N2048/K512/E256. Every tensor has
16 MiB of packed metadata and requires 32 MiB of expanded metadata.
Across 120 tensors: 1.875 GiB read, 3.75 GiB written, at least 5.625 GiB
logical traffic. Keeping packed units for GEMV/FQ and the FP16 planes for
SF adds 3.75 GiB of resident device memory; it is not byte-neutral.
At a *hypothetical effective* 500 GB/s, traffic alone is about 12.08 ms for
all tensors. This is a scale estimate, not a PPU timing prediction or bound.

Long prefill shares one call's metadata preparation over that call's tokens,
not across requests. A
short MoE chunk may touch few experts while a full prepass expands all 256;
using active-expert bytes to price that full prepass is incorrect. Current
metadata preparation has no active-expert selection. A future partial cache
would also need initialization state and stream-safe publication; it is not
part of the first implementation.

The reviewed model has 256 experts and top-8 routing. At the measured
128-token submission size there are 1,024 routed rows, **four rows per
expert on average**, not M=128 for every expert. A longer prompt submitted
as 128-token chunks does not change that per-call mean. SF/FQ selection must
consider token chunking and expert work, not only the total sequence length.
The device-only gate therefore uses the caller-available max-rows bound 128,
not the tighter maximum visible only to its host oracle.

Implementation and measurement requirements (corrected per-call contract):

1. Expand on the GPU for each SF call, using temporary scale/zero workspace.
   Put expansion before its GEMM in stream/capture order so every graph replay
   expands again. Reuse workspace allocation if safe, not expanded values
   across calls. Automatic FQ decode reads packed units without SF expansion.
2. Keep preparation and weight backcopy independent. Compute waits only for
   the device data it consumes, never for D2H/disk publication.
3. Budget per-stream scratch plus concurrent uses; do not keep expanded
   planes for all 120 weights. The 3.75 GiB total above describes the previous
   cached implementation, not the required temporary-workspace residency.
4. Time prepass, GEMM and their complete per-call sequence against FQ on
   identical routing/endpoints. Include allocation in startup wall timing
   separately; no CPU reference or disk-cache work belongs in kernel timing.
5. For serial expansion and GEMM, compare `T_expand + T_gemm` plus matching
   adapters against the FQ full call. For R calls, charge every expansion:
   `sum(T_expand_i + T_gemm_i)`, not one expansion plus R GEMMs. Existing
   T01 now removes `scale_ready` caching and rejects resident-only policy data;
   the new per-call device gate still must be executed.

## Locally compiled box package

`prebuilt/ppu0010/kpack-execution-v1` contains the 560,232-byte execution DSO
and ten separate grouped parents, 2,176,200 bytes of DSOs total. The execution
DSO SHA-256 is `53165c9ec37a7f73057b31289434c90a2a4da0065f37c93ca78208cecd4ba1de`.
Builds use the local PPU SDK 2.1.1; no PPU execution is claimed. All manifest
flags remain `device_validated=false`, `heuristic_admitted=false`.
Local validation: 75 reader/metadata/fixture/selected-policy tests pass;
the historical fixture and its calibration hashes remain unchanged. The
execution ELF contains all 20 format-specific GEMV entry kernels, five
reducers and five prepasses. All ten grouped DSOs expose the old identity
and new query/prepare exports. No old six-library bundle is replaced.

The ordinary v2 grouped route uses a conservative per-expert 3-D grid bound,
whereas v1 can use an exact host-derived flat grid. Persistent v2 rebuilds
the compact device directory. Correctness of the same parent does not make
these scheduling costs equal: the gate records v1/v2 resident timings, and
v2 ordinary does not inherit old flat-grid performance admission.

Run `bash tools/run_kpack_execution_box.sh` after pulling the LFS package,
with the existing `PPU_SDK` and `QUACTLIZE_PPU_BUNDLE`. This performs **no
compilation** and does not run or modify llama.cpp. It preserves the Docker
shell and archives even incomplete results. Both independent gates run:

- GEMV: 32 weight geometries / 81 work points / eight recipes, three rounds
  of 11 samples. Five formats, real dense families and the Q4/Q5 E256 model
  anchors; signed activations, all-output GGUF oracle, output guards and
  zero-low negatives. Compare with the current legacy GEMM incumbent; this
  is not proof against every optimal GEMM from the historical search.
- Grouped: ten parents / fourteen parent-shape cases, three routing profiles
  each. Same-parent v1 raw equality, independent GGUF tolerance, output guards,
  and captured replay with changed GPU bounds. SF reports first prepass,
  resident GEMM and repeated prepass+GEMM. The latter is not cold model
  allocation/JIT/TTFT and must not be labeled as such.

Model anchors use the real N/K/E/top-k/chunk sizes with reproducible synthetic
weights/routing, not IDs captured from GSM8K. Before promoting a GEMV rule,
also compare it with the already-measured optimal GEMM candidate for the same
work (including supported adapters). Beating the current legacy default alone
does not establish that broader algorithm choice.

Return the printed `kpack-execution.*.results.tgz`. Results do not automatically
change the heuristic. C++ selected-module binding and final llama.cpp route/
metadata ownership are still open; the current model route is unchanged.
