# Dense prefill model cohorts

Bucket construction is on hold pending user feedback. The inventory and review
export raw per-shape evidence and rankings only; `buckets` stays empty with
`bucket_status=PENDING_USER_FEEDBACK`. No production selection table changes.
DeepSeek TP4 is reserved for later validation, not bucket training. Its local
dimensions can be derived from the pinned llama TP code and GGUF metadata;
the source-derived inventory path is still being prepared.

The Qwen3-32B PP2048 screen reproduced the old-helper whole-model Asys subtotal
within -0.41% (457.337 versus 459.225 ms for 445 calls). The native subtotal
differed by -2.19%; the small Q4 KV control differed by -10.61%. This supports
continuing the screen, not exact model/cache equivalence or shipping admission.
The same-F16-operand/FP32-accumulator winners total 428.180 ms versus native
466.652 ms in the new screen. These are GPU-activity sums, not model latency.

## Inventory, without loading model weights

```bash
python3 tools/plan_prefill_model_cohort.py \
  --names qwen3-32b qwen35-35b-a3b qwen38-27b \
  --m 128 256 512 1024 2048 --output /workspace/prefill-cohort.json
```

Default model root is `/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo`.
Use `--model-root` or a `--models` JSON list of `{name,path,tp,tp_logs}`.
Qwen3.8's actual Q3/Q4/Q5/Q6/Q8 mix comes from GGUF, not its filename.
Quantized dense matrices outside the K-pack shape contract remain explicitly
native-only. MoE matrices, embeddings and the output head are listed separately;
this runner does not claim their coverage. Dense layer matrices are benchmarked
at the requested M, not treated as measured model call counts.

For DeepSeek-V4-Flash TP4, include `deepseek4-flash-tp4` in `--names`, specify
`--deepseek-model /path/to/first-00001-of-00005.gguf` and one or more
`--tp4-log /path/to/actual-llama.log`. All GGUF shards are checked. Each dense
weight requires records for local device ordinals 0..3 from `[quactlize-dense]`.
N-split, K-split and replicated extents are validated separately; an absent,
ambiguous or more complicated partition is an explicit blocker, never guessed.
This checks local extent coverage, not TP communication or numerical admission.

## Run the existing Asys harness

```bash
COHORT=/workspace/prefill-cohort.json GPU=1 IDLE_CONFIRMED=1 \
PPU_SDK=/usr/local/PPU_SDK \
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp \
LLAMA_CI_BUILD_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
bash tools/run_prefill_model_asys_box.sh
```

`PLAN_ONLY=1` performs host planning only. `SHARD=0 SHARDS=7` through
`SHARD=6 SHARDS=7` assign complete weight families to separate GPUs. Keep each
comparison on one physical card. `TC_CACHE` may point at a shared compiler
cache; its locks prevent duplicate builds. `RESUME_RUN` resumes the frozen
run on the same GPU. Do not edit a running source or replace its libraries.

Multi-GPU coordinator (only after confirming all listed GPUs are idle):

```bash
python3 tools/run_prefill_cohort_multi.py --cohort /workspace/prefill-cohort.json \
  --gpus 1,2,3,4,5,6,7 --confirm-idle --output /workspace/prefill-all-models
```

Use the same command plus `--resume` after interruption. It retains the original
source revision and GPU assignment. Families are greedily balanced by candidate
count and matrix work, not assigned by a hash modulo device count. This weight is
an orchestration estimate, never a measured latency or performance-selection cost.
The final `review-*.json.xz` is the compact upload; incomplete points keep their
valid samples and diagnostics but cannot replace a complete table row.

Candidates are bounded neighbor proposals plus separate current/exact controls,
not a Cartesian sweep. A per-shape limit (`MAX_CASES`, default 96) refuses an
oversized plan rather than silently dropping controls. Probe measured occupancy
on the actual device. This cohort retains proposed schedules instead of the
earlier exhaustive 1..32 grid expansion. A bounded screen is not a global-optimum
certificate: regressions or unstable finalists need a targeted wider search.

Full-F16 includes Q8_0. Its new packed-word/shared-exchange dequant keeps the
canonical K-pack2 layout, original FP16 scales, FP32 multiplication and a single
RNE conversion to F16 (BF16 is separately supported). The typed workspace query
is separate from the legacy provider; grouped/full-BF16 admission is unchanged.
No selector is automatically switched to this new implementation.

The runner keeps independent raw-GGUF checks, changed-input/zero/omitted-producer
negatives and redzones before timing. It warms ordinary calls, excludes the first
captured ring, and records every helper, producer and reducer in Asys. No cuBLAS
CUDA Graph, CPU-timing substitution or frequency-normalized latency is used.
1700 MHz is requested; actual throttling remains diagnostic, not automatic FAIL.

## Incremental component review and per-shape rankings

```bash
python3 tools/review_prefill_cohort.py --run /workspace/qwen32-prefill-asys.RUN \
  --output /workspace/prefill-review.json.xz
```

Repeat `--run` to merge disjoint shards. Upload this compact review; `.asysrep`,
SQLite, weights, binaries and compiler caches stay on Box. The review retains
source/SDK/device/binary authority, numerical evidence, all candidate recipes,
per-kernel signatures/geometry/raw samples and complete-call sums. Medians and
component sums are checked when exporting. `exact`, `components` and
`candidates` are separate sections in the same checksum-protected file. The
component registry carries input/dequant/main/reducer/output roles; no upload
of weights or compiler intermediates is needed. The previous seven-shape,
2295-measurement archive also passes this reader after extraction, without
rewriting its receipts or numerical hashes (about 1.8 MB compact output).

To optimize a single point, set `COHORT` and `POINT_ID=<inventory point id>` in
the same Box command with a new run directory. Then merge its complete measured
point with `--previous /workspace/prefill-review.json.xz --run NEW_RUN --output
NEW_REVIEW.json.xz`. Unchanged points remain intact; replaced evidence is kept
in history. New runs for the same point must be chosen explicitly, not minimized
across cards or attempts.

`--invalidate-component <component id>` marks affected measured calls stale.
A helper-only result does not overwrite an old complete-call measurement.
Reprofile the affected points before updating their exact winners. Missing or
stale points remain visible in coverage and cannot silently shrink a bucket's
validation set. Source/SDK/device changes stay attached to their measurements;
an SDK or hardware domain change requires a separate table.

Per-shape rankings use complete GPU-call cost at equal precision. The internal
bucket experiment remains testable but is not invoked by the inventory, sweep
or export commands. Bucket rules and their validation will be decided after
the user reviews the measurements. TP/model consumers remain attached to
deduplicated local workloads; no collective cost is inferred.
