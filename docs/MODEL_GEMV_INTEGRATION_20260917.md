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

Local regressions: 280 Quactlize tests pass, one optional historical archive
test is skipped; 37 caller tests pass. All seven exact emitted functions
match the admitted experiment's opcode counts and resource declarations,
with zero stack allocation. This is not an instruction-schedule identity
or a device-performance claim.

The immutable artifact and caller SHA are recorded in
`tools/kpack_q4_model_artifact.json`. Source branch is
`dev/gemv-model-tuning`; caller branch is
`dev/quactlize-gate-up-v0.3.0` on the owner's fork.

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
