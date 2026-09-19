# Resume

Latest returned integration,2026-09-19: `kpack-tp2.8VYueY.results.tgz`
matches source e624e44, caller1f7da3b and the published selected runtime.
Read `docs/selected-result-20260919.md` before the older pending statements.
All17 selected-entry numerical points and both cold/hot TP2 gates pass.
Current matched ABBA: native12.700852 versus selected11.239113 ms/token
decode (-11.509%); prefill285.995361 versus228.169678 us/token (-20.219%).
The previous K-pack model result was12.482082 ms/token. All measured selected
plans have zero legacy fallbacks. Token-batch-1 mean KLD has increased from
0.001708 to0.002444; finite short likelihood checks are not full accuracy
admission. Prefill likelihood metrics match the preceding result.

Only model trace failed: Asys shared session creation timed out before model
load; no per-kernel times/MBU are available from this upload. Existing
trace-only private namespace retry can reuse this exact run without a sweep,
build or numerical rerun. Shared service paths differ from the selected SDK;
the device-service PID is stale. Do not stop another profiler or delete locks.
Namespace permission and the private capture still need box confirmation.

Trace-only handoff is ready: `tools/run_kpack_model_trace.py --previous <this
run> --model qwen35-122b-q4km --profiler-scope private`. Caller/SDK/result roots
now default to the saved receipts; six-library discovery reuses the exact
manifest finder. Runtime, caller binaries, model cache and JIT cache remain
unchanged. 285 host regression checks pass; the live seven-module package
still verifies. No compiler or GPU task was run for this orchestration change.

Latest,2026-09-19: `dev/dispatch-results` now integrates the returned confirmed
minima, not merely the earlier behavior-preserving refactor. Read
`docs/selected-20260919.md` first; later entries below are chronological history.
Final selection is locally audited (1852 catalog rows,937 actual public SIMT
queries,15 ordinary winners), generated once, and backed by the published
execution inventory. New M1 timings do not change M2..8 rankings. Q6 head keeps
TC; Q4 TP2 paired remains unselected pending comparable chain cost.

Caller: `dev/quactlize-selected-decode`,1f7da3bd94b1fc6b9520e156456c89221fddd07f,
own fork only. Runtime: `artifacts/kpack-model-selected-v1`,
d8852d419a550623c4fd96c723f749c128798774. All63 ELF payload paths use LFS;
no caller binary is included. Artifact manifest:
5bda5fabf565c54494a358479a3ce9923bcde8d204338aede0742517491a4aab.
The seven production TC modules and BF16 gate identities are unchanged.
The SIMT reducer was isolated from the TC JIT contract so cached TC images
remain reusable without source relabeling or a large recompilation.

Next is device integration only, not a sweep:
`python3 tools/run_selected_decode_box.py --previous-run <previous TP2 run>`.
It recovers caller/NCP build directories and the exact prior compatibility
overlay, checks the pinned caller, and performs17 selected-entry numerical
checks, two-device/cache chains, model numerics, warm ABBA and Asys.
New-device performance is not yet measured. Ten promoted implementations are
instruction/resource-identical to the measured images; Q5 down has reviewed
address-lowering differences (no floating/memory opcode changes), so its new
image latency remains explicitly pending. No fresh PPU run was performed here.

2026-09-19: baseline TP2 model/ABBA results reviewed; no candidate admitted.
Read `docs/plan.md` for the 10 local decode shapes and eight restriction classes.

Current worktree is `dev/kpack-tp2`; only profiler orchestration and this audit
change here. Production runtime/caller binaries remain those in
`kpack-tp2.om9JQj`. Asys SDK path normalization and a trace-only TP2 resume entry
are host-tested. Private profiler namespace fallback is device-environment
pending; it does not stop host profiler services. No new decode performance
claim has been made.

Local implementation is complete on `dev/tp2-fastpaths`, based on 4326381,
with caller branch `dev/quactlize-tp2-fastpaths`, based on 1bac078de. The original
TP2 branches and runtime pin remain unchanged for trace recovery.

Structural capability and measured default selection are separate. Read
`README.md` for build/replay entrypoints. No new default selector was admitted.

Local evidence (2026-09-19):

- Quactlize host suite: 434 passed, 15 subtests passed (34.67 seconds).
- Caller native/trace suite: 41 passed; graph matching covers189 graphs.
- PPU candidate build: 17 points, 117 SIMT variants and four TC controls,
  95.25 seconds with eight compile jobs. ISA inspection and package verification
  pass; no device was used.
- Prepare gate binary, Q4/Q8 production fusion TUs, BF16 Q8/fixed Q5 smoke
  instantiations and the changed caller TU compile. Dispatcher syntax check passes.
- Package: `/tmp/tp2-fastpaths-package-20260919`, manifest SHA256
  `4addcd47773099db4d636f1c0e0d51b8ac1982624edf109733e17ccca7546460`.
- Prepare: `/tmp/tp2-prepare-build-20260919-r2/bench`, SHA256
  `6383bc26e6b4b6e2e2622ead04154202f994aa680edfd30002d2b59eb5fa8af2`.
- Caller object: `/tmp/tp2-fastpaths-caller-final-20260919.o`, SHA256
  `4df96eb6f82a361ec180f60c22e167c2cdc7462213d0ba764945608de3362ed4`.
- Additional compile receipt: `/tmp/tp2-fastpaths-extra-xu0b2n4n/receipt.json`.

The initial experiment wrapper used the non-Q4 arrangement constructor for
an unpaired Q4 point. The new host query check caught it before a device run;
the final package uses the exact Q4 registry descriptor. Discard the earlier
`tp2-fastpaths-build-20260919` and `-r2` candidates as handoff authorities.
No user files or old artifacts were deleted.

Next requires PPU: run independent numeric controls, cold full-call comparison,
prepare shape/alias gates and ACU. Then promote only measured scopes, compose
the runtime with matching caller, and repeat model ABBA/Asys. The original TP2
branches can still replay the unchanged baseline trace independently.

Device-result update,2026-09-19: the component run has now returned in
`/root/tp2-fastpaths.IwJJxB.results.tgz` (SHA256
`5944aa7547e4dd5c9b87dae0f906102d7df5d4e39f42d7007c25731408fee075`).
17 points/134 total arms/2576 numerical records and34 ACU exports pass local
receipt review. All4590 confirmation samples recompute correctly. Prepare
shape64/router-edge56/alias360 gates pass; K3072 M1 improves37.17%, M8 improves
87.38%. Read `docs/results-20260919.md` and `benchmark.csv` for the complete
table, retained incumbents, ACU evidence and limitations. No production
selection was changed. The next action is selective M1/prepare integration
and matched TP2 model replay, not another unrestricted sweep. Keep Q6 TC and
the old Q4/Q5/Q8-SSM winners; Q4 projection and paired-chain gains differ.

Fallback follow-up,2026-09-19, branch `dev/decode-fallback` in
`/tmp/quactlize-tp2-fastpaths-20260919`: the user explicitly requires common
optimizations to improve fallback, not just exact rows. Implemented donor
upgrade inheritance in the actual small-M dispatcher, dynamic Q8 hoist within
its recipe/precision family, Q5 generic unsigned/fold, shared row-vector
reduction and structurally selected all-SIMT prepare. Old measured exceptions,
precision boundaries, exclusions, scalar alignment fallback and mixed-TC
prepare remain. See `docs/fallback-20260919.md` for exact scope.

Local validation:111 tests and15 subtests pass in32.22s, including real C ABI
bucket selection without a GPU/JIT, exact-table retention and numerical host
mapping/guard controls. PPU real SIMT launchers compile/link in23.08s;52 kernel
ISA entries include all six formats, three aligned row reducers and scalar
fallbacks. Fast code decode, FP32 FMA and vector loads remain; the row reducers
emit paired loads/stores. Production `execution/moe.cu` F16/BF16 dispatchers
compile in11.65s. No PPU device was available and no new timing is claimed.

Compile evidence: `/tmp/decode-fallback-compile-6d9ninjd/local-receipt.json`
and its ISA/resource logs; prepare gate rebuilt under
`/tmp/decode-fallback-prepare-20260919`. These are local compile evidence,
not a published runtime or a source-matched box handoff. Keep the previous
artifact immutable; do not run its source verifier at this new source HEAD.
Next: bounded table-miss/old-shape integration against frozen per-point best,
then registry/paired-candidate work and a matching model package. The active
production runtime pin and llama caller are unchanged by this follow-up.

Source-matched handoff,2026-09-19: `tools/run_decode_fallback_box.sh` and
`fallback_prebuilt.json` now provide the separate bounded gate. Artifact commit
`331fc8d1690eba798073d1b6d79d96a1e15f7091` on
`artifacts/decode-fallback-v1` contains only the component package and prepare
gate, not a replacement model runtime or llama binaries. The GEMV manifest is
`2a40ab32742940ef0a9c614bb0f8b83ed909f71d73ba6c0c30fb1919c9cc5e7c`.
Eleven known points use frozen confirmed minima (independently checked against
the returned archive), two unseen Q8 points use explicit old controls, and
five BF16 reducer points are non-policy controls. The actual C ABI selector
is rechecked in every ordinary child; production launchers provide the body.
M1..8 numerics precede M1 rotating complete-call timing; both ACU arms bind
native producer/reducer identity and geometry. Prepare uses its real dispatcher.

Final local build: `/tmp/decode-fallback-gate-build-20260919-r4`,21.0 seconds,
six jobs. Candidate and frozen-incumbent native identities pass static checks.
119 host tests pass in34.82s. End-to-end no-device preflight passes at
`/tmp/decode-fallback.fHq22o`. No numerical/performance device claim is made.
Resume binds profiling requirements and retains every completed profiler arm
and later completed point. Static compilation is not performance admission.

Independent architecture review and staged TODO are in
`docs/architecture-20260919.md`. Priority is behavior-preserving final-decision
snapshots, an effective recipe catalog, and one selector shared by runtime and
build/prewarm. No architecture implementation was mixed into this gate.
The separately requested external CLI review could not start because its
authentication expired; do not attribute this report to that external model.
