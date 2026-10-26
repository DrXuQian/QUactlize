# Dense prefill whole-call selection

## Current exact admission: 2026-10-26

The M-span results replace all 225 previous exact rows and add 180 rows:
**45 dense families x M128/256/512/1024/2048/3072/4096/5120/8192 = 405**.
This includes the recorded TP-local dense families, not MoE or collectives.
259 select Full-F16 and 146 SF; none select FQ. Input/output are F32,
operands F16, accumulation FP32. The public provider honors each recipe's
own dequant, I/O, reducer, schedule and stage order, without a helper override.

Ranking is the arithmetic mean of the complete Asys GPU component sum across
both forward/reverse captures, including all samples and error reset.
It is not the fastest sample, the old median ranking, or model wall time.
Exactly one accepted difference against the same-run recorded control is
Q4/M3072/N25600/K5120: Full-F16 **2423.915981us** vs old Full-BF16
**2410.623111us**, **+0.551429%**, explicitly accepted by the user. The
exception is bound to that point/candidate/control; no blanket tolerance.

The 312 original complete points and 93 retried complete points retain their
own authorities. Sources are reviews `dcb0fb9c...53344` and `8d9b87e7...b4fe`;
full digests, archive hashes and approval are in the catalog and
`policies/dense_prefill_m_span_admission.json`. All candidate samples were
audited before export. Raw reports/binaries remain on Box. Uploaded harness
and DSO hashes are preserved; a full native-build commit receipt was not
exported, so `source_commit` remains null. The retry harness matches local
`8a41376f`, which is not by itself proof of the native binary's source commit.

These runs explicitly omitted native comparison. The old 225 rows, including
their native losses and median-based helper admissions, remain in
`superseded_points`; they are not runtime selections or a fresh native A/B.
No model speedup or native no-regression claim is made by this table update.

Exact matches now extend to M8192. The existing 147 predicted buckets and
their historical training remain unchanged and bounded to M<=2048; M3072
must not accidentally inherit the M2048 log bin. Decode, BF16/MoE and other
devices are unchanged. This is not admission of arbitrary untested shapes.

Offline reproduction (the base is the frozen pre-update catalog):

```bash
python3 tools/fit_dense_prefill.py --base BASE_CATALOG.json \
  --mean-plan FROZEN_M_SPAN_PLAN.json \
  --mean-admission policies/dense_prefill_m_span_admission.json \
  --review review-1792993181069001430.json.xz \
  --review review-1793008645924230352.json.xz
python3 tools/fit_dense_prefill.py --check
```

Rebuild the complete public library/package to deploy the new generated table;
editing JSON alone does not update an installed `.so`. No llama ABI change is
needed. Host validation replays every exact/bucket decision and the real public
prepare/run orchestration with device doubles, not simulated GPU performance.

## Historical initial admission

2026-10-23: the complete `quactlize_ppu_dense_prepare_v1` provider now uses
the reviewed dense **F32 endpoints, F16 operands, FP32 accumulation** winners
on PPU-ZW810 (72 CUs, warp 32). Selection includes input/output helpers,
per-call scale/full dequantization, the main recipe and any reducer. It is
not a main-kernel-only ranking, nor an online tuner.

The policy is isolated from decode, MoE and explicit BF16 compute. It does
not change GGUF/K-pack bytes, clipping, caller structures or kernel bodies.
Full uses the existing typed F16 provider, including Q8; the old Full-BF16
path is retained outside this policy. Route 3 means Full provider and
`plan_info.compute_type` identifies the actual arithmetic.

## Priority and scope

1. Exact `(q,M,N,K)` row: policy **20**, all 165 measured points keep their
   individual lowest complete-call time. 116 Full-F16, 48 SF, 1 FQ.
2. Bounded same-format bucket: policy **21**, explicitly predicted. M/N/K
   logarithmic bins plus aspect ratio; N/K subdivisions where a common
   measured config would lose more than 5% on known members. N>K and K>N
   remain distinct even inside the balanced region. This is **not** a
   new-shape speed guarantee.
3. Outside these bounds, use the prior legal selector. No distant donor or
   arbitrary FQ replaces an exact measured winner.

Domain: contiguous dense, M128..2048, Q8_0/Q3_K/Q4_K/Q5_K/Q6_K. Q2_K has no
new observations in this cohort and is not relabelled measured. Unknown
M/N/K inside a leaf are predictions, even if its sole donor is measured.
BF16, noncontiguous storage and other devices do not receive policy 20/21.
This cohort comes from single-card Qwen3-32B, Qwen3.5-35B-A3B and Qwen3.8-27B.
Qwen3.5-122B-A10B TP2 and DeepSeek TP4 are NOT dedicated measured/validated
sets here. A coincident local-shard shape may hit an exact row, but that is
not coverage of the TP model, its MoE operators or communication. Their
post-split workload inventories and new-model holdout performance remain open.

The exact recipe binds its own dequant and I/O IDs. A later helper-only
lookup cannot overwrite them. TC grid uses the selected scheduler policy
and runtime occupancy/CU count, not a frozen launch size. A missing winning
module/provider is an error, not permission to choose a slower recipe.

## Evidence and maintenance

`policies/dense_prefill_v1.json` keeps per-point complete-call costs, execution
identities, winner receipts, source hashes and bucket membership. The
compact C++ table is generated by `tools/fit_dense_prefill.py`. No raw reports
or binaries are needed at runtime; the build seals both catalog and table.

Source reviews:

- `2f97b8da478468e26e8fb1ff8421cc95fd33111f763ca5cf77229af59c522f51`
- `1cd438d924f087d72cfd589c3419c97ff7d1857f665de62d84251c25f0cc27da`

47 complete two-direction confirmations replace whole earlier points; 118
unaffected points retain their evidence. Never combine a new helper time
with an old main time to claim a new complete-call win. Confirmation uses
the median of two capture medians. Native comparison retains the 12 Q8
losses (all below 5%); it does not claim zero regression or new model TPOT.

```bash
python tools/fit_dense_prefill.py --check
python tools/fit_dense_prefill.py --check --audit-buckets
bash run_tests.sh --suite host
```

For a later single-point optimization, export a complete source-bound review
and run `fit_dense_prefill.py --review ORIGINAL --review NEWER ...` in order.
Generation validates samples/components/numerics, replaces complete points,
reselects their minima and refits affected buckets deterministically. It
rejects incomplete reviews or changed SDK/device/clock timing domains. A
new catalog still needs explicit review; regeneration is not automatic
shipping admission. Host CI replays exact and bucket decisions through real
C++ selection and public prepare/run with device doubles. Actual PPU model
A/B after rebuilding remains required; host tests cannot prove it.

Leave-one-N/K-family-out audit (all M held out together): 33 comparable
predictions, worst regret 5.73%; 17 predictions lack a measured held-out
candidate, 115 requests have no donor in the bounded bin. These are not
passes or made-up costs. DeepSeek/new shapes still require their own check.

## TP2 / TP4 supplementary measurement

`tools/run_tp_prefill_supplement.py` covers the **local dense prefill** matrices
of Qwen3.5-122B-A10B-Q4_K_M TP2 and DeepSeek-V4-Flash-0731-UD-Q4_K_XL TP4.
It does not claim MoE-GEMM, decode, TP correctness, communication or model TPOT
coverage. Those exclusions remain in the inventory, not counted as passes.

The default `--inventory-mode headers` reads only the GGUF shard headers. A
small CPU-only helper executes the unchanged TP split function extracted from
`LLAMA_CI_DIR/src/llama-model.cpp`, with the caller's actual `llama-hparams.cpp`
and headers. It allocates no weights, initializes no GPU runtime and never
loads or runs a model. Planning needs the caller source and `g++`, not a llama
binary or SDK. No llama rebuild is required for this harness change.

The helper accounts for equal TP splits, replicated weights, N/K cuts, repeated
Qwen SSM segments, quantization-block alignment and per-layer rounding rotation.
It does not divide every N and K by the TP world size. Caller-source hashes,
model metadata and every rank's segment extents are retained in `tp-headers/`.
These are source-bound CPU inferences, not observed model calls. Missing or
unsupported rules stop planning; no model-run fallback happens automatically.
MTP is excluded (the caller default). DeepSeek `attn_output_a.weight` is also
explicitly excluded: its GGUF 2D shape becomes a grouped 3D projection in llama,
not an ordinary dense GEMM. MoE weights remain a separate cohort.

Optional `--qwen-log` / `--deepseek-log` provide a cross-check against complete
observations. `--inventory-mode native` retains the previous diagnostic path:
short PP64/TG1 native model runs with equal tensor splits, Quactlize disabled
and `GGML_META_DEBUG=1`, or those supplied logs. Only that explicit native mode
loads models and requires at least four assigned GPUs when probing both models.
No `--no-warmup` option is required or passed; inventory is not timed performance.

```bash
cd /sim/eec/shared/junfu.qx/quactlize
git pull --ff-only origin develop
PPU_SDK=/usr/local/PPU_SDK \
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp \
LLAMA_CI_BUILD_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
L2_BYTES=67108864 JOBS=8 \
python3 -u tools/run_tp_prefill_supplement.py \
  --qwen-model /sim/eec/shared/AI_workspace/llm-models/Qwen3.5-122B-A10B-GGUF-Q4_K_M/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
  --deepseek-model /sim/eec/shared/AI_workspace/llm-models/DeepSeek-V4-Flash-0731-UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf \
  --inventory-mode headers \
  --gpus 1,2,3,4,5,6,7 --confirm-idle --run
```

Only use `--confirm-idle` when the assigned GPUs are idle. Standalone local-shard
measurements run independently across the assigned cards; they do not allocate
a complete model or perform all-reduce. They still use the native llama build
for the reference operator, not for shape discovery.
The model resolver checks both `/sim/eec/shared/AI_workspace/llm-models` and
its `bench_model_zoo` subdirectory. It requires exactly one matching first
shard. Override ambiguous or differently named files with `--qwen-model` and
`--deepseek-model`; it will not choose an arbitrary GGUF quantization.

Default M: 128/256/512/1024/2048. Candidate generation keeps the current exact
or bounded-bucket choice, legal shipping recipe proposals and nearby per-route
leaders, plus the new Q8/Q3 Full-F16 helpers. A maximum of 32 candidates per
point is checked before TC candidate compilation; controls are never silently dropped to
meet a quota. Q2 geometry transfers are explicitly unmeasured proposals.
Every point, even an existing table hit, is remeasured against native F16
operands / FP32 accumulation. Old absolute times are not reused on TP shapes.
Targets outside the new table are labelled `OUTSIDE_NEW_TABLE`; this is not
an assertion that their older production fallback has been measured here.

Native and candidate passes alternate in forward/reverse candidate order.
Asys GPU activities retain separate input/dequant/main/reducer/output samples
and a sum over each complete call. No CUDA Graph for cuBLAS, no JIT/warmup in
the admitted samples, no total derived from unrelated component medians.
1700 MHz is requested; actual frequency remains advisory. This is a bounded
candidate comparison, not proof of a global optimum.

`TP_PREFILL_PLAN` prints actual deduplicated points, families, cases, modules
and coverage before timing. The component runner estimates remaining time
from observed work; compilation is separate from admitted timing.
Omit `--run` to stop after CPU-only planning (no idle confirmation is needed in
header mode). Resume using the same arguments with
`--resume --output /workspace/kpack-tp-prefill.XXXXXXXX` (the printed retained
directory); source, model paths and GPU assignment must stay unchanged.

Upload only `timing/review-*.json.xz`. Original Asys reports stay under
`timing/gpu-*/qwen32-prefill-asys.*/results/batches/*/attempt-*/trace.asysrep`.
No production policy is updated by the run. Review its complete points before
adding exact rows or refitting buckets.
