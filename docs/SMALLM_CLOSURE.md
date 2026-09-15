# Small-M selection closure

This is an **offline box measurement**, not an inference-time tuner. It does
not modify the shipping policy, llama.cpp checkout, or existing bundles.

The frozen scope is TP=1, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0, dense and indexed
MoE, tokens 1–8, F32 input/output, with F16 and BF16 compute measured separately.
Actual GGUF headers from the two Q4_K_M target models are joined with historical
measurements. Separate and possible fused gate/up shapes and output heads are
included. Unknown quantized matrix roles fail inventory admission; unsupported
formats and non-matmul tensors remain listed in the inventory.

## Run

Use idle PPU-ZW810 devices; do not overlap inference or profiler jobs:

```bash
(
  set -e
  cd /sim/eec/shared/junfu.qx/quactlize
  git pull --ff-only origin develop
  DEVICES="0 1 2 3 4 5 6 7" JOBS=192 L2_BYTES=67108864 \
    PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
    MODEL_ROOT=/sim/eec/shared/AI_workspace/llm-models \
    bash tools/run_smallm_closure_box.sh
)
```

`L2_BYTES` is the previously verified 64 MiB capacity, not a guessed default.
A positive SDK capacity that disagrees is an error. `MODEL_PLAN` can bind exact
GGUF filenames when directories are ambiguous. `COMPUTE_TYPES="f16"` narrows
the declared scope; it **cannot** certify BF16 coverage. `PLAN_ONLY=1` prints
the frozen denominator without compiling/running. `BUILD_ONLY=1` compiles only.

The wrapper fetches only the current execution DSO through Git LFS. It compiles
deduplicated typed TC modules on the box, with up to 192 concurrent jobs subject
to available RAM. It does not rebuild the old monolithic sweep bundle or build
llama.cpp. CPU underutilization is not a reason to terminate a build.

There is no promised overnight duration before observing this cohort. Build
and runtime progress report their own elapsed/remaining estimates. Those are
observed averages, not deadlines. The historical-only plan has about 2,900
contexts and 75,000 shortlisted screen cells, versus 750,000 with all compiled
SIMT configurations; model-header additions are printed before compilation.

## Candidate selection and timing

- Preserve historical winners and current exact/bucket choices without a
  shortlist cap. F16-only AP1/Q4 bodies are explicitly excluded from the BF16
  candidate domain, not relabeled as BF16 results.
- Extract two nearby measured SIMT geometries plus two warp-count neighbors;
  compare S1/S2/S4/S8. Include two historical TC geometries, two bounded seeds,
  and all exact historical TC incumbents. No full Cartesian configuration scan.
- Use one raw-GGUF fixture and one resident weight ring per weight family,
  shared across candidates/M/compute/profile cases. Only active expert bytes
  count toward the ring's minimum 2.25×L2 footprint.
- Time the complete public F32 endpoint. TC indexed calls include GPU route
  preparation, metadata/directory, actual Split-K reduction and indexed finish.
  SIMT includes its real reducer. No CPU routing or theoretical reducer timing.
- Screen with 3 event samples; confirm each implementation's top two and all
  incumbents in alternating-order 4×11 rounds. Competitive screen-only cells
  are automatically confirmed. Unstable winners get bounded extra rounds.
- Cross-confirm routed-profile winners. A production rule cannot choose using
  a benchmark's hidden routing histogram; `policy-review.json` therefore
  reports a single minimax choice and its worst measured profile regret.
- Setup, fixture creation, allocation, compilation/JIT and first graph upload
  are excluded. Correctness checks cover every output, changing A/IDs, graph
  replay, zero-A negatives, finite values and 128-byte-aligned scratch guards.

Successful per-candidate receipts and compiled modules are resumable. Keep the
same source, SDK, devices, model headers and options, then set
`RESUME_RUN=/workspace/smallm-closure.XXXXXX` and rerun the command. Failed
candidates are retried; successful receipts are not discarded. Device/runtime
faults can end one family process; other families continue. Changed identities
require a new run rather than silently mixing cohorts.

## Results and remaining limits

Upload the printed `smallm-closure.XXXXXX.results.tgz`. It contains no model
weights or compiled payloads. Important files:

| File | Meaning |
| --- | --- |
| `plan.json` | Model inventory, sources, candidates, incumbents, pruning and exact denominator |
| `summary.tsv`, `winners.json` | Best measured complete-call configuration per context |
| `coverage.json` | Missing/failed/unstable cells; separate measurement and policy admission |
| `policy-review.json` | Cross-profile minimax choices, regrets and missing comparisons |
| `tpot-estimate.json` | Per-model M1 projection sums weighted by actual layer counts; separate gate/up pairing scenarios, not measured TPOT |
| `evidence-audit.json` | Historical testing issues and approximations, including both sides of the old coverage difference |
| `cells/*/winner-access-pattern.json` | SIMT source addresses/widths and 32/64/128-byte footprints, not ACU counters |

`MEASURED_POOL_CLOSED` means no missing or failed comparison in the declared
pool. It is **not** a global-optimality proof. `policy_5pct_admission=PASS` also
requires one common confirmed choice within 5% across the tested profiles.
Otherwise the result must explain the missing evidence or router-sensitive
tradeoff, rather than advertising a fast per-profile oracle as a production
selector.

These measurements can close the standalone selection-data gaps. They cannot
certify whole-model speed or numerical safety: the generated policy still needs
production replay and the existing BF16/model numerical/performance gate after
integration. In particular:

- Historical TC geometry recompiled through the typed endpoint is not an
  immutable legacy DSO. The old fast legacy output-head observation remains a
  model-regression control, not a fabricated paired timing.
- Common-exact dyadic weights isolate reader/compute correctness. They do not
  prove that real SwiGLU activations fit FP16 or certify non-dyadic BF16 rounding.
- MBU is a model of unique packed-B bytes divided by event time and 2700 GB/s.
  It is not measured DRAM traffic, occupancy, or hardware utilization.
- Fused MoE chains, topk reuse and surrounding model operations are not certified
  by timing an isolated indexed projection. Inspect the full model trace.
- Large-M SF/full-dequant/provider component sums remain labeled estimates,
  not measured E2E. Prefill, TP>1 and unseen shapes are outside this decode scope.

No additional data collection should be needed **for a successfully closed
declared pool**. Numerical failures, unresolved instability, unacceptable model
regressions or a new workload legitimately require further work and retesting.
