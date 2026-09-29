# PPU1.7 integration worklist

Current dependency: **PPU CUTLASS4.3.0**; see [migration and current gates](ppu17-cutlass430.md).
The dated results below describe the retired3.6 build and are retained only
as historical evidence. They do not admit the new dependency on hardware.

Historical scope: independent PPU1.7 backend on PPU CUTLASS3.6.0, leaving the
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
| Host/codegen gates | Legacy isolation, byte-map and numerical CPU tests, SM90a assembly | PASS; 5,875,034 strengthened checks, 156 bodies, 20 negative codegen/target plants |
| Native PPU1.7 | Correct SDK target and live emitted device bodies | SDK2.1.1 rejected by target guard; capable SDK needed |
| Device correctness | Single-launch CPU-oracle gates including tails/empty experts | H800 extended 1,020/1,020 PASS for baseline and affine-reuse candidate; PPU pending |
| Memory and same-math replay | TMA/GMMA sanitizer checks; complete output-byte comparison | Both images: 25/25 sanitizer cases PASS; candidate/control 156/156 byte-exact pairs PASS |
| Pipeline/performance | TMA overlap, measured configs, no inherited ZW810 claims | 72 same-config Q4 comparisons complete; reverse confirmation interrupted by another job; affine cache not enabled globally |
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

## Physical Hopper checkpoint — 2026-09-26 05:15 UTC

This supersedes the interrupted numerical checkpoint above. The same H800
became idle; the completed `extended-v3` baseline has 1,020/1,020 PASS. The
metadata-reuse candidate also completed 1,020/1,020 in `affine-extended-v1`.
No legacy kernel or production routing was changed.

- `memory-v1` and `affine-memory-v1`: each has 17 memcheck, six racecheck and
  two synccheck cases, all PASS. Bulk-copy and warpgroup-MMA checking were on.
- `affine-replay-v1`: 156/156 baseline/candidate pairs have identical full
  output bytes, including guards and separate scale/zero expansion. Each
  arm also passes the independent typed CPU dot oracle.
- Baseline library SHA256 remains `a3285663bff9d2527acf0e32585f5677bda8b41b8d568394857cb97129f12030`.
- Experimental affine-reuse library SHA256:
  `39847acf9a8dc62f653196d822dcb6f44c41dee297398e10d89ff0c29cf79dc8`.
- The replay application SHA256 is
  `84cdea5282834245d45ac76a1a385f9c39a8cc68bacf9ae28fd863afd985c224`.
  The original numerical and benchmark binaries were not overwritten.
- Both dtype baseline timing campaigns have 36/36 PASS. These use rotating
  weights exceeding twice the queried 50 MiB L2, correct each weight copy,
  exclude three warmup graphs and retain 11 event samples. GPU processes
  were monitored; no foreign process was observed during admitted timing.
- `profile-v1` failed with `ERR_NVGPUCTRPERM`. No hardware counters were
  collected, and no host driver permission, clock or mode was changed.

Evidence is preserved under the remote root and its local copy documented
above. The experiment record is
`/workspace/quactlize-ppu17-20260926/tuning-hopper/docs/plan.md`.
These are Hopper source-validation results, not PPU1.7 native admission.

## Performance decision — 2026-09-26 05:25 UTC

See [the Hopper validation report](ppu17-hopper-validation.md) for the full
measurement boundary and representative numbers. The 72-point candidate
screen completed without observed foreign GPU work. All 36 FQ points became
faster, but 29/36 SF points became slower, by up to 11.46%. The generic affine
cache is therefore **not retained in the backend reader**. Its exact patch,
compiled library and raw evidence remain in the external experiment directory.

Reverse-order baseline confirmation completed ten clean points before another
GPU job appeared at 05:23:22 UTC. The overlapping eleventh point was marked
INVALID and our child stopped. FP16 reverse confirmation was not started.
This is not a completed confirmation or a new production routing policy.
The foreign job was not interrupted. Further performance work requires an
exclusive GPU window; next optimization should scope metadata reuse to FQ.

After restoring the reader, the strengthened baseline host/codegen gate also
passed: 5,875,034 CPU checks, 156 live assembled bodies and 20 negative plants.
Native PPU1.7 compilation, simulator correctness and performance remain open.
