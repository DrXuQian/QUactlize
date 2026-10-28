# MoE sweep preparation

Host planning, bounded device experiments and the explicitly scoped exact
admissions below are separate milestones. None claims that the dense PR now
executes MoE or that all models are covered. Keep the caller PR unchanged.

## Current priority and backup list (2026-10-27)

Active: optimize the indexed SIMT decode reader for tokens 1–8, first with a
bounded same-expert two-row reuse experiment. Keep canonical K-pack bytes,
direct BF16 A rounding, original FP32 B group-affine math/accumulation/output
and existing Split-K partial order.
The existing common/selected reader is the control; no production admission
or table update follows from host tests or compilation alone.

Backup, retained but not on this experiment's critical path:

- After SIMT optimization closes, compare SIMT **and TC** in the final config
  sweep for every M, including M2–8. A SIMT-only improvement is not proof of
  the fastest implementation. Include both ordinary/persistent grouped TC
  and its real prepare/reducer/finish costs. Retain generated-rule recall of
  historical winners; do not inject a missing champion into the candidate set.
- Merged/paired gate-up and fused SwiGLU configuration scanning, including
  physical 2N shapes and each output/rounding contract below.
- Broader fused TC/SF/full-provider implementation and large-M MoE scanning.
- New MoE table/bucket admission and caller integration after fresh evidence.

The fusion inventory and recall tools remain available. Deferral does not
reinterpret an old dense result as a fused result or remove fusion coverage
from the eventual scan.

### Bounded decode experiment

The opt-in `moe-token-pair` cohort reuses the existing Asys-only runner; it
does not add a production selector or change an installed kernel. Its runtime
row-pair mapping is shared across N/K, not an exact-shape implementation.

- Qwen35 separate Q4 gate/up N512/K2048, and Q5 down N2048/K512.
- Per family: M1 spread, M4 repeat2, M8 spread and M8 cluster. These are named
  synthetic router controls, NOT a claim to have captured real model IDs.
- Two geometries, S1/S4, current selected body plus matched single-row/pair
  bodies: 8 points, 64 arms, 13 vector modules at the recorded baseline.
- M1 keeps the old body. For M>1, only matching experts share B/code/metadata.
  A, outputs and split partials retain original routed row and stride semantics.
- Pairing scans at most 64 IDs in each warp, without a new helper launch. It
  increases live accumulators/registers; zero-collision and high-collision
  results must both be reported. No universal speedup is assumed.

Local plan-only invocation (no device or performance claim):

```bash
python3 tools/run_smallm_regressions_asys.py \
    --cohort moe-token-pair --plan-only --output /tmp/moe-pair-plan
```

After these sources are committed and delivered to Box, the existing frozen
runner accepts `COHORT=moe-token-pair`. It uses `IDLE_CONFIRMED=1`, `GPUS`,
verified `L2_BYTES`, fresh Asys samples, separate reducer timing, the commit
and transitive source/binary identities. An old checkout does NOT contain the
new cohort; do not execute its archived HEAD and call it this experiment.
Only `review-lite-*.tar.xz` needs uploading initially; raw traces stay on Box.
GPU correctness, measured benefit, all-M/model coverage, native comparison
and production admission remain separate pending gates.

### Q5 single-row M1–8 completion

The `moe-single-row-q5` cohort completes **SIMT validation first**, before the
TC configuration sweep. It runs the public shipping bodies, not the private
pair-only launcher. This preserves the fixed M1 C4/W2/P8/S1 body and catches
an accidental benchmark of a similar-looking but different implementation.

- Q5 N2048/K512, E256/top8/channels8, F32 endpoints/BF16 A/FP32 affine math.
- 36 points: M1 spread; each M2–8 with spread, cluster, repeat2, repeat4 and
  synthetic hot-expert `real`. `real` is not a captured model trace.
- Current production choice plus the pre-admission control, both remeasured;
  three common geometries C4/W2/P8, C4/W8/P8 and C8/W4/P4, each S1/2/4/8.
- At this revision: 442 arms, five vector modules shared across all M/router/
  split cases. No pair candidate, no new TC search, no old timings reused.
- Fresh Asys component means with full source/commit/binary provenance;
  producer and real reducer are isolated, measured once per component and
  included in each arm's complete-call sum. Existing guards/oracle remain.

Local validation without a GPU:

```bash
python3 tools/run_smallm_regressions_asys.py \
    --cohort moe-single-row-q5 --plan-only --output /tmp/moe-single-row-plan
```

After this source is committed and delivered, use the existing Box wrapper
with `COHORT=moe-single-row-q5`, idle `GPUS`, and no `RESUME_RUN` from the older
pair experiment. This cohort is not all-model coverage or a TC/SIMT selection
admission. Qwen122B TP2 and DeepSeek TP4 still require their actual local expert
shapes, and Q8/BF16-storage reader optimization remains open.

### Three-model local-shape completion

`COHORT=moe-single-row-models` replaces the Q5-only handoff when all requested
models are required: Qwen3.5-35B-A3B TP1, Qwen3.5-122B-A10B TP2, and
DeepSeek-V4-Flash-0731 TP4. It reads all GGUF shard **headers**, then executes
the caller's source-bound TP split function on CPU. It neither launches llama
nor loads model weights. No llama binary or multi-device communication is
needed for these per-rank operator timings.

- Gate/up and down use their actual local N/K, quantization, expert count,
  selected-expert count and channel semantics. No all-formats Cartesian
  product or naive N/TP division; unknown/unsupported local shapes stay visible.
- Every weight family gets M1 spread and M2–8 across five synthetic routing
  profiles (36 points/family). Identical families across layers/models/ranks
  share measurements, retaining every consumer in the plan.
- Existing common SIMT geometries plus S1/2/4/8; the actual current auto
  incumbent remains even if it is TC or specialized Q4. Table misses use the
  old BF16 caller's v2/Q4 chain as control, checked against real public APIs
  by host tests. New TC and fused gate-up/SwiGLU candidates remain deferred.
- No historical latency reuse. Independent oracle, guards, rotating-weight
  Asys component means, commit/source/device/binary identities, and separate
  prepare/main/reducer/finish costs remain mandatory. These are shape-based
  synthetic operators, not model inference or captured routing distributions.

After committing/delivering these sources, run the existing Box wrapper with
`COHORT=moe-single-row-models`, `BASELINE=quactlize`, idle `GPUS`,
`LLAMA_CI_DIR` (the caller **source**, not its build directory), and verified
`L2_BYTES`. `MODEL_ROOT` defaults to
`/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo`.
`QWEN35_MODEL`, `QWEN122_MODEL`, `DEEPSEEK_MODEL` override exact first-shard
paths. `PLAN_ONLY=1` prints per-model local shape/format/point coverage without
GPU work. A timing run prints the same inventory before compilation. GPU
workers distribute whole comparison points; TP4 does not require four GPUs.

Use a new run, not a resume of the frozen Q5-only/pair experiment. The new
run's `RESUME_RUN` preserves its header inventory and original card bindings.
Only upload `results/review-lite-*.tar.xz`; headers/TP receipts/coverage,
numeric results, separate component timings and errors are included, while
raw Asys reports and binaries stay on Box. No new production routing is
admitted by generating this plan or by passing host tests.

### First Box result and Q4-only retry (2026-10-28)

Source `98f5cd4b94922653f928555d96c54a9abeb9cba1`, ZW810/72 CUs,
requested CE clock 1700 MHz, GPU 4. Uploaded
`review-lite-1793141000861683287.tar.xz` SHA256:
`709144bef858b1cef681f0e559285a43f5edadfeb638c7c2639f9ad2048d6a82`.
All 87 payload hashes, run/source/harness seals and the four passing Q5
receipts' saved raw durations, component sums, producer identities and numeric
proofs were revalidated locally. Raw Asys reports and device binaries remain
on Box, not in this small archive.

The run is correctly marked **INCOMPLETE: 4 PASS / 4 FAIL**. The four Q4
controls executed `q4_decode::kernel_bf16`; the host checker only recognized
`q4_decode::kernel`. This is a post-capture name-check failure, not a reported
numeric error. Because the old child failed before saving `capture.json`,
those four points cannot be admitted by reparsing logs alone. Do not edit
their status, source identity or old frozen checkout.

Q5 down N2048/K512, E256/top8/channels8, BF16 compute with F32 storage:

| Tokens / synthetic routing | Production (us) | Best matched single-row (us) | Best pair (us) | Winner vs production |
|---|---:|---:|---:|---:|
| 1 / spread | 9.320 | 9.155 | not launched | -1.8% |
| 4 / repeat2 | 23.601 | 23.703 | 21.359 | -9.5% |
| 8 / spread | 56.704 | 46.695 | 50.731 | -17.7%, single-row |
| 8 / cluster | 51.816 | 42.256 | 33.892 | -34.6% |

These are fresh Asys arithmetic means of separately profiled producer and
reducer calls, not model latency or native-comparison admission. The numerical
proofs passed (largest recorded relative L1 error < 1.7e-7).
Do not attribute all gains to pairing: M1 has no pair; M8 spread improves
from a different single-row geometry and pairing loses 8.6% to its matched
C4/W2/P8/S1 single-row control. With M4 repeat2, C8/W4/P4/S1 pairing saves
27.1% against that same geometry; with M8 cluster, C4/W2/P8/S1 saves 19.8%.
Split-4 pairing lost to its matched controls in all measured multi-token Q5
points. No universal pair selection or production change is justified.

The fixed checker distinguishes FP16/BF16 entries by the declared workload
and submitted ABI; it still rejects the wrong reader and unknown lookalikes.
It now saves an unadmitted capture after correctness, guard and cleanup checks
but before host name validation. A failed name check still exits nonzero;
only the timed worker can publish a PASS.
The harness fix passed 224 host/regression tests (135 optional/native tests
skipped); no GPU rerun was performed locally and no device kernel changed.

After delivering the fix, use `COHORT=moe-token-pair-q4` in a **new** run
(`unset RUN RESUME_RUN`), on the same otherwise-idle GPU 4. This is precisely
the original four Q4 workloads/32 arms/7 vector modules, not a reduced config
search. The source commit and hashes are recorded anew. Keep the four old Q5
PASS receipts under their original source identity; no Q5 rerun or relabeling
is needed for this harness-only fix. A new partial run does not itself certify
the complete eight-point experiment or native/model parity.

Local checkpoint: 170 host/regression tests passed; 15 PPU1.0 SDK2.1.1
Q2–Q6/geometry instantiations compiled with zero stack. The embedded common
reader/reducer code retained 75 identical machine-code bodies across the
initial and simplified carriers. These are not GPU numerical/performance
results. C4/P8 pair registers rise by 18–26 over the one-row body; C8/P4 by
12–14. Pairing is deliberately not enabled in production.

### Q5 single-row exact integration

`policies/kpack_moe_decode_exact_v1.json` binds the reviewed Q5 run to the
existing flat decode catalog. Regenerate/verify it with
`tools/fit_moe_decode.py --review <first-review-lite.tar.xz> [--check]`, then
`tools/generate_decode_selection.py [--check]`. This is build-time admission,
not another runtime heuristic, JIT tuner or router readback.

Only Q5 indexed N2048/K512/E256/top8/channels8, F32 input/output with BF16
activation compute, changes at these exact token counts:

| Tokens | Selected public single-row recipe | Measured gain vs old production |
|---|---|---:|
| 1 | V3/C8/W4/P4/S1 | 1.76%, spread |
| 8 | V3/C4/W2/P8/S1 | 17.65%, spread; 18.45%, cluster |

Recipes are selected from the intersection of non-regressing public
single-row candidates across the available fresh profiles, minimizing worst
mean ratio. The experiment-only `moe_pair_rows` flag is never aliased to a
shipping config. M4 remains unchanged. All other exact choices, old bucket
donors, Q4 paths and BF16-storage/grouped endpoints remain unchanged.
The 34.59% clustered two-row gain is retained as a candidate result, not
claimed as the benefit of this conservative single-row integration.

No arithmetic, device kernel, public ABI or llama caller change is needed.
Host tests exercise the compiled C++ selector and the public C ABI, including
unseen shapes/M and storage negatives. Local PPU compilation confirms both
selected kernel machine-code bodies are identical to the measured control
bodies and have zero stack; this is not a new GPU/model run.

Current network coverage is NOT complete: Qwen35 has two unfused expert
families at the four synthetic routing/token points per family; no new model
run. Qwen3-32B and Qwen3.8-27B have no MoE operators. Qwen122B TP2 and
DeepSeek TP4 still require current expert-weight inventories and fresh MoE
device/model validation. Historical dense sweeps do not supply that evidence.

At aligned bases, the unchanged source footprint model gives low-plane
32/64/128-byte unique-sector use of 100/100/50 percent for both geometries.
Q5 high is 100/100/50 for C4/P8 but 50/50/25 for C8/P4; Q4/Q5 A improves
from 50/50/50 to 100/100/100 when moving C4/P8 to C8/P4. This is why both
geometries remain controls. Pair reuse halves B/code/metadata requests for a
matched pair without improving these per-instruction footprints; A requests
and useful FLOPs are unchanged. It is not a measured HBM-traffic reduction.

## Rule-first, then reveal historical references

```bash
python3 tools/plan_moe_strategy.py --output /tmp/moe-rule-audit.json
```

The output must be a new file. Exit 0 means the declared historical recipe
recall passed, not performance or complete target-model coverage. A miss exits
1 and remains visible; it is never fixed by appending the missing champion.

The catalog is projected to configuration/geometry records with old times,
sample distributions, admission labels and timing priorities removed. Generate
candidates while withholding the entire target N/K family across M, qtype,
precision and router. Afterwards audit effective production choices,
individual router-profile winners and historical prefill winners. Fused
gate/up recipes have their own recall audit, not a tile-only alias.

Rule refinements retain indexed low-warp Q4 readers, one-axis SIMT column and
reader peers, distinct prefill M/N tile sizes, and portable SF delivery peers.
Every candidate still needs compiled resource and independent numeric gates.
These are sparse neighbors of blind seeds/observed template families, not a
Cartesian product of axes or per-shape champion exceptions. The ordinary
dense prescreen keeps its previous defaults.

`--seed-width 1` is a host budget experiment. It must pass the same recall
gate; a smaller pool is not useful if it drops known winners. Candidate and
module counts are reported before timing. Historical audit points are NOT
automatically scheduled on Box.

## Actual model and fusion denominator

Optionally supply `--models models.json --llama-source /path/to/llama.cpp`.
`models.json` is a list of `{ "name": "...", "path": "/path/first-shard.gguf",
"tp": 1 }`. TP2/TP4 use the unchanged caller split function in a CPU carrier,
not a model load or inferred `N/TP, K/TP`. The caller source hashes and complete
GGUF shard headers are bound to the inventory; unknown extents stay blocked.
Expert-axis partitioning is not admitted by this N/K-TP inventory.

Token anchors are 1–8, 16/32/64/128/256, 512/1024/2048/4096/5120. Only actual
quantized expert weights become model points; Q2–Q8 are not multiplied into
every model. Gate/up input channels=1; down channels=topk. Token T is not the
per-expert row count or the total routed row count T*topk. Real routing capture
remains separate; old fixtures called `real` were synthetic distributions.

The scan must include these execution forms, in addition to down:

| Form | Physical weight N | Output N | Extra contract |
|---|---:|---:|---|
| Separate gate/up | N each | N each | Two projections plus SwiGLU |
| Merged gate/up | 2N | 2N | Canonical per-expert gate-then-up; separate SwiGLU |
| Gate/up + SwiGLU | 2N | N | G4/U4 paired layout; F32 original-row output |
| Gate/up + SwiGLU | 2N | N | G4/U4 paired layout; BF16 compact-row output |

For Qwen3.5-35B-A3B headers, the two Q4 projections have N512/K2048;
their fused physical weight is N1024/K2048. Down is Q5 N2048/K512.
Gate/up pairing requires matching format, K/N, expert extent and local rank
partition. Native merged GGUF weights are not doubled again. Incompatible
pairs retain an explicit reason and their unfused controls.

Fusion keys include layout, projection rounding, output type/order, consumer
shape and epilogue. Even identical dense N/K does not supply its measurement.
Candidate templates hold out BOTH logical and physical target N/K. Generic
paired recipes are generated from the API/template family, not looked up in
the historical selector. New shape-specialized reader instantiations and
paired-register TC epilogues require compilation and numerical validation.
The current fused indexed TC processes one routed row at a time; expert-grouped
multi-token reuse and broader SF/full-provider composition remain explicit
implementation gates. `execution_ready`/`sweep_ready` stay false at this stage.

## Fresh measurement only

All scan times must be new **Asys GPU activity** samples. Every result must bind
the full Quactlize commit, actual source digest, actlize revision, compiler/SDK,
runtime, physical device, binary digest, recipe/body, router fixture and Asys
trace/range. The host plan reports `<commit>-src-<digest>`; this is a result tag,
not a Git tag or a substitute for a Box binary receipt. Dirty source is
fingerprinted rather than silently called the committed binary.

Old timings never enter the new table. Resume is only for valid results from
the same fresh campaign and implementation/measurement identity. Prepare,
metadata/full expansion, main, reducer and finish are profiled separately;
sum actual calls once. A fused reducer/SwiGLU is one measured component, not
an estimate plus the same standalone helper again. Native uses identical raw
weights/A/IDs and the same output boundary; record its actual arithmetic.
JIT, provider tuning and first-use warmup are outside measured samples.

Sequence: close host inventory/rule coverage → implement and validate the
BF16 indexed/fused candidates → fresh bounded Asys/native scan → measured
table/bucket admission → caller integration and model validation. No stale
timing transfer, automatic admission, or runtime configuration tuner.
