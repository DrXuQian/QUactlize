# Exact model GEMV integration

The user authorized promotion of the seven candidates in the
[measured review](MODEL_GEMV_REVIEW_20260917.md). This is an exact M1
integration, not a new sweep or a global SIMT preference. Q6 output remains
TC; Q8 SSM remains S8, including its ordered float2 reducer. All endpoints
remain F32 with FP32 accumulation. Routed MoE computes in BF16; dense and
shared-expert operations compute in F16. No clipping or format change.

| Operation | Exact dimensions | Integrated reader |
|---|---|---|
| Q4 paired routed gate/up | logical N512, K2048, E256, top8, channels1 | H32, fixed shape, C4/P8/W8/S1 |
| Q5 routed down | N2048, K512, E256, top8, channels8 | H32, unsigned indexing, fixed shape, C4/P8/W2/S1 |
| Q8 paired shared gate/up | logical N512, K2048, E1 | hoist, TileN16, C4/P4/W8/S1 |
| Q8 shared down | N2048, K512, E1 | hoist, C4/P4/W2/S1 |
| Q8 SSM output | N2048, K4096, E1 | fixed shape, C8/P4/W4/S8 |
| Q8 QKV | N8192, K2048, E1 | hoist, fixed shape, C8/P4/W4/S1 |
| Q8 attention gate | N4096, K2048, E1 | C4/P4/W8/S1 |

Paired physical N is twice logical N. Paired G4/U4 placement and sequential
inter-warp reduction are unchanged. TileN16 preserves complete G4/U4 pairs.
The two TC-to-SIMT replacements require the exact old parent and geometry,
F16 compute, F32 storage, dense M1/E1/top1/channels1. A changed parent, dtype,
M or shape retains the old route. Nearest-shape buckets are not retrained.
M2/M8 checks are fallback numerical controls, not performance admissions.

## Local validation and publication

Execution and fusion compile with PPU SDK 2.1.1. The final execution build
took 184.401 seconds. The package refresh reuses all seven TC modules and
the packer, prefill, router probe and typed TC images. The dispatcher is
rebuilt against the new receipts. No llama binaries enter the package.
The caller change only recognizes new template names in trace evidence;
there is no caller kernel, ABI or graph change.

Local regressions: 280 kernel/dispatch tests and 119 model-runner tests pass,
one optional historical archive test is skipped; 37 caller tests pass.
All seven exact emitted functions
match the admitted experiment's opcode counts and resource declarations,
with zero stack allocation. This is not an instruction-schedule identity
or a device-performance claim.

The immutable artifact and caller SHA are recorded in
`tools/kpack_q4_model_artifact.json`. Source branch is
`dev/gemv-model-tuning`; caller branch is
`dev/quactlize-gate-up-v0.3.0` on the owner's fork.
Runtime artifact commit is `7c198d6313364e96723f335c113e6275b1e1e070`.
There are 64 LFS payload paths but only three new unique images (44,366,872
bytes). The execution image also supplies two typed gate aliases; 57 other
DSOs are byte-identical to the prior package.

Reproduce local builds with `tools/build_kpack_execution.py` and
`tools/build_kpack_gate_up.py`, refresh with
`tools/refresh_kpack_model_execution.py`, then attach with
`tools/attach_kpack_gate_up.py`. Use new output directories for each build.
`tools/inspect_model_gemv_integration.py` compares only the seven exact
emitted functions against the immutable experiment.

During integration, native inspection caught two dynamic Split-K wrappers
where the experiment had constant S1. Dedicated guarded S1 entries preserve
that compiler specialization without duplicating the reader body.
The exact-function inspection excludes adjacent device math helpers: the
older broad section scan's LOP3 count can include those helpers. Source
fast dequant may lower to native AND/OR plus half2 instructions; a source
LOP3 is not proof of a native LOP3 in the measured kernel.

## Box boundary

The integrated package is locally validated but **PPU integration and model
performance are pending**. Run `tools/run_kpack_q4_model_box.sh` with the
pinned caller, `MODEL_GATE_UP=1`, `MODEL_COMPUTE=bf16`,
`MODEL_PHASES=perf`, `MODEL_NAMES=qwen35-35b-q4km` and
`L2_BYTES=67108864` (the existing ZW810 receipt).

Before the model, `tools/run_model_gemv_integration.py` compares the frozen
incumbent, admitted experiment and production-selected implementation in
fresh point processes. It requires independent GGUF numerical checks,
M1/M2/M8 controls, exact admitted-versus-integrated M1 bits, changed inputs
and IDs, graph replay, BF16 range and negative controls. Timing traverses
an active-weight ring at least 2.25 times L2, six alternating rounds of 15
samples, including reducers/activation. Integrated median must beat the
original and remain within 5% of the admitted implementation. Failed
points retain their logs while the other six continue; the model gate
does not proceed after a correctness or performance admission failure.

The model uses warmed PP2048/TG128 ABBA and a second-request Asys capture;
upload, first JIT/use and initial graph replay are excluded. The result
archive contains component gates, selections and model summaries. Full
Asys reports remain under `results/trace/*/{reference,native}/proof.asysrep`.
Keep the selected GPU free of other inference requests throughout timing.
Component savings are not a measured TPOT improvement, and the 40%/60%
modeled MBU objectives remain open.

## Returned run y7SNPu: integration passes, model load OOM

Archive `kpack-q4-model.y7SNPu.results.tgz`, SHA256
`f4fc4a9b1192ebca8614994c52d6d8aa0b6e478600299e64b7d2f23be49992aa`,
contains source `2d5d7c827fe88ecd6ed1a084f58d405a403c8cb3` and the pinned
runtime/caller above. All seven reader points pass with zero M1 bit differences
between integrated and admitted implementations. The three-arm controls
contain 144 numerical rows across M1/M2/M8. The rotating comparison satisfies
the script's thresholds at every point (integrated versus admitted medians
range from -0.431% to +0.065%). GPU exclusivity was not demonstrated; these
observations must not be substituted for a clean whole-model benchmark.

Paired integration, router, Q8, BF16 capability (746), selected Q4 and mixed
chain checks also pass. However, every model benchmark process exits during
model loading. Device 0 / PCI 0000:08:00.0 reports only 9204 MiB free.
Reference allocation requests 20470.32 MiB and K-pack requests 20381.60 MiB;
both fail with out-of-memory. The reference Asys server also fails to load.
There are **zero model timing samples and no completed inference trace**.
The final `paired-model-proof` stage is a consequence, not a fusion failure.
The archive does not identify the other memory owner; do not kill workloads
or reset a GPU based on this receipt.

No kernel, library, caller rebuild or artifact replacement is needed.
After making a device idle, resume only the model phases:

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    git switch dev/gemv-model-tuning
    GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin dev/gemv-model-tuning
    PREVIOUS_RUN=/workspace/kpack-q4-model.y7SNPu \
    LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp \
    PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
    PERFORMANCE_ONLY=1 REPAIR_PREFILL=0 MODEL_ACU=0 CUDA_VISIBLE_DEVICES=0 \
    bash tools/resume_kpack_q4_model_box.sh
)
```

The continuation checks the same pinned runtime, every caller binary hash,
the saved ABBA/warmup protocol, and all eight component receipts. It preserves
the failed logs and writes new results. It runs warmed ABBA, second-request
Asys and paired-kernel proof, without numerical-model evaluation or component
reruns. `MODEL_ACU=1` explicitly adds profiling. Local continuation regressions:
132 tests pass, including altered-image, wrong-protocol and failed-gate negatives.
