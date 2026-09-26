# PPU1.7 integration worklist

Scope: complete independent PPU1.7 backend on PPU CUTLASS3.6.0, leaving the
PPU1.0/1.5 actlize implementation and production routing unchanged.

Source baseline: `fc354953671bdb2079af02507a94805f614f1496`.
Dependency baseline: `023e82d03e80b4d5982f664925e15499033df314`.

| Part | Acceptance criterion | Status |
|---|---|---|
| Isolation | Separate worktree, include graph, library and build identity | PASS; existing files unchanged |
| Format reader | Reuse canonical maps; six formats, direct BF16; CPU oracle | CPU and initial physical H800 PASS; PPU pending |
| SIMT | Dense/grouped, M1–8, typed I/O, masks | Implemented; 48 SM90a bodies assembled |
| Tensor core | Real FP32-accumulating SM90 GMMA bodies, dense/grouped FQ/SF | Implemented; 96 bodies assembled, including 24 TMA bodies |
| Scale expansion | Separate typed producer, bit-matches FQ metadata | CPU and 12 physical H800 bitwise checks PASS; PPU pending |
| Build/ABI | Independent C boundary, explicit config inventory, fail closed | Host load/C/Python PASS; only q17_* exported |
| Host/codegen gates | Legacy isolation, byte-map and numerical CPU tests, SM90a assembly | PASS; 5,533,890 checks, 156 bodies, 20 negative codegen/target plants |
| Native PPU1.7 | Correct SDK target and live emitted device bodies | SDK2.1.1 rejected by target guard; capable SDK needed |
| Device correctness | Single-launch CPU-oracle gates including tails/empty experts | H800 initial 300 PASS; extended 382/1,020 checked, remainder paused; PPU pending |
| Pipeline/performance | TMA overlap, measured configs, no inherited ZW810 claims | Two-stage TMA implemented; guarded physical timing runner built; no timing admitted |
| Production integration | Architecture-specific policy and caller admission after gates | Pending |

No machine is required for the source/host/codegen work. Hardware/simulator
availability must not block that work or be confused with device admission.

## Checkpoint — 2026-09-26 03:09 UTC

- Branch: `dev/ppu17-cutlass360`.
- Worktree: `/workspace/quactlize-ppu17-20260926/source`.
- Source-validation library:
  `/workspace/quactlize-ppu17-20260926/build/45f349d3df1d88fdd788/libquactlize_ppu17.so`.
- Library SHA256: `a3285663bff9d2527acf0e32585f5677bda8b41b8d568394857cb97129f12030`.
- Evidence: `/workspace/quactlize-ppu17-20260926/validation-final/validation.json`.
- Additional CPU-only fixture coverage: 14 cases (six formats with
  `[0,1,8,9,63,64,65,0]` expert rows plus dense M1–8), K1024 and N tails.
- Native target rejection:
  `/workspace/quactlize-ppu17-20260926/native-probe/ccb40bb794be29ad4da1/target.log`.

The library is a CUDA SM90a source-validation artifact, not a native PPU1.7
delivery. It must not be installed over the existing six libraries. No
PPU1.0/1.5 build, format owner, kernel, dispatch table, submodule pin or caller
was modified, and the user's dirty develop checkout was not touched.

## Physical Hopper checkpoint — 2026-09-26 03:41 UTC

- Device: H800 PCIe 80GB, UUID `GPU-1d5fdef3-4899-79d9-19e6-c9c815b2a59c`,
  driver 595.71.05. No device clocks, partitions, modes or global packages changed.
- Remote task root: `/workspace/ppu17-h800-20260926`.
- Kernel library is unchanged from the SHA256 above. The application SHA256
  is `555fde2939e6c7ee7c56f423b77e5dfd2ad5d031681ebb46e9cb4b32b9b34588`.
- `results/coverage-v1/summary.json`: 300/300 PASS, six qtypes, F16/BF16,
  six configs, FQ/SF, dense/grouped, separate bitwise scale expansion.
- `results/extended-v2/summary.json`: 382/1,020 checked, zero failures,
  including 136 checked rows recovered from `extended-v1`. This is an
  **incomplete campaign**, not an extended-suite PASS.
- Interrupted twice by other GPU processes: a Python job, then
  `/workspace/ppu17-hopper-control-20260926/hopper_fp8_control`. Neither
  foreign job was interrupted. Further GPU work is paused until the device
  can be used exclusively.
- Local evidence copy:
  `/workspace/quactlize-ppu17-20260926/h800-evidence/results`.
- **No performance measurements or sanitizer/profile results admitted.**
  The new physical benchmark and CPU-tested concurrency guard are separate
  from the numerical/simulation application. Timing requires an idle window.

Hopper numerics do not close native PPU compilation, PPU simulator correctness,
or native performance gates. New runner code changes no production routing.

## Remaining admission sequence

1. Run the same target probe and assemble the backend with a capable native
   PPU1.7 SDK. Do not force architecture macros around the rejection.
2. Use the single-launch application for each format/dtype/engine/route:
   dense M1–8 and M9/63/64/65, grouped empty and ragged experts, N tails,
   K512/1024 for TMA phase wrap, and separate scale expansion. The CPU oracle
   rounds weights/inputs in the requested type and accumulates in FP64.
3. Confirm runtime launch count and the actual target symbol in the simulator
   or device trace; collect hardware/model identity. The application's
   `timing=NOT_MEASURED` is intentional.
4. Measure admitted configs and SF expansion separately. Add only
   architecture-specific timing/policy data, keeping legacy routing untouched.
5. Admit the explicit new backend at the production caller boundary after
   numerical/performance gates, not by replacing actlize includes globally.
