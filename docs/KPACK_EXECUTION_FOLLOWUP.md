# K-pack execution follow-up

Updated: 2026-09-09. The O0ki3q native micro gates and adapter tests pass;
model prefill improves, but decode remains about 31% slower and the short
trace lacks selected native dense compute. This is not optimized-routing
closure. Canonical offline planes are unchanged.

## Tracked delivery

| Item | State | Completion condition |
| --- | --- | --- |
| 3. Native selected-module binding | Micro gates pass; Q6 output-head policy coverage remains open | Both hooks are wired, but N248320/K2048 dense head misses the native policy and retains labelled legacy K-pack FQ. No Python/JIT/online timing in inference |
| 4. Device-only grouped metadata | Correctness and changing-router replay pass; performance open | No per-token D2H; isolate the cost of rectangular device-only scheduling against the same-parent compact diagnostic |
| 5. ScaleFirst prefill | Native metadata/GEMM gates pass; model prefill improves | Immutable-weight prepass is reused. Full-model first-use and resident timing remain separate; no inference wait on cache D2H |
| Decode GEMV | 14 contexts x 8 recipes pass; zero recipes admitted | Current FQ comparison excludes casts/gather/scatter while GEMV includes indexed I/O. Revisit equal-work timing; this is not proof GEMV cannot win |
| Model decode regression | Open performance debt | Isolate the measured per-token gap with matched work; retain the old GEMM incumbent and do not attribute the whole gap to one missing algorithm |
| Grouped Split-K / SIMT pair reader | 260/260 PPU cells pass; performance reviewed | Q4 compact S2 improves on compact S1; Q5 compact S1 remains best. Pair reader improves both SIMT anchors. Host compact excludes CPU preparation and is not a production replacement |
| GPU compact / persistent Split-K | PPU 204/204 pass; Q4 compact S2 -7.23%, Q5 compact S1 -45.15% | [Reviewed gate and ACU capture](KPACK_GPU_COMPACT.md) include directory cost, mutable GPU routing and FP32 partials. Persistent is not the anchor winner. External ABI unchanged; deployed native bundle not switched |
| SIMT NVIDIA diagnosis | Real CUDA/half probe passes on RTX 5090; kernel port in progress | Development-only bridge, independent GGUF oracle, then memory/instruction counters. No NVIDIA result substitutes for PPU admission |

Profiling default: standalone operators use ACU native reports; model/stream
timelines and launch bubbles use Asys. The compact ACU entry selects only the
two measured anchors and their old-module controls, not the entire gate.

The Q4 N512/K2048/E256/top8 single-token case was not omitted: overnight
screen/neighbor/confirmation evidence includes TM8, and the runtime selects
the confirmed TM16/TN64/TK256 winner. Its historical 18.44 us is itself only
about 9.25% effective weight bandwidth; replaying that number is not closure.
See [the coverage audit and no-recompile single-op probe](KPACK_GROUPED_DECODE_REVIEW.md).

## Native model gate

Entry: `bash tools/run_kpack_native_box.sh` (see `--help`). This consumes
`prebuilt/ppu0010/kpack-native-v1`: 214 selected parent modules plus the native
host dispatcher and execution DSO. It does not replace the old six libraries,
which still provide intake/admission and explicitly logged K-pack FQ misses.
The box rebuilds llama.cpp for the changed context layout, not Quactlize.

1. Twenty dense and eight grouped contexts use the actual C++ selector and
   prepared module, independent GGUF arithmetic, output guards and changed
   router graph replay. ScaleFirst prepass is checked and timed separately.
   These paired measurements produce the FQ/SF route table before the model
   starts. SF must improve resident core time by more than 2%; otherwise FQ
   is retained. Reports include first prepass time and reuse break-even.
2. Fourteen model-shaped GEMV contexts (Q4/Q5 experts, both broadcast/per-slot
   A, and Q6 N248320/K2048 dense output) compare all eight recipes in 3x11
   samples. The comparison uses selected FQ when covered, explicitly tagged
   legacy FQ otherwise. GEMV includes indexed access and reduction; FQ here
   excludes adapter casts/gather/scatter. This conservative gate is not a
   proof of global cross-algorithm optimality.
3. Production adapter tests use device tags to verify pointer/stride/routing,
   cached preparation, eager/capture/replay and mutable IDs. Arithmetic is
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

Long prefill amortizes metadata preparation over many tokens/requests. A
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

Implementation and measurement requirements:

1. Own metadata with the immutable weight artifact. Prepare once, outside
   graph capture, then record readiness and reuse across prefill requests.
   Decode GEMV continues to read packed units without requiring SF allocation.
2. Keep preparation and weight backcopy independent. Compute waits only for
   the device data it consumes, never for D2H/disk publication.
3. Record memory capacity before enabling SF. Do not eagerly allocate every
   expert plane without accounting for KV/workspaces and the extra 3.75 GiB.
4. Time prepass, resident SF GEMM, first-use prepass+SF, and FQ on identical
   expert routing. Include lazy allocation in model first-use wall timing,
   not in the kernel-only prepass timer. Cold-D2H publication is separate.
5. First-use SF wins only if `prepass + SF < FQ`. Over R reused calls, compare
   `prepass + sum(SF)` with `sum(FQ)`. Do not train the first-use selector on
   resident-only timing or charge preparation again on every request.

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
