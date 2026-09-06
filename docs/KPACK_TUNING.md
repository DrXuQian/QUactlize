# K-pack configuration tuning

The budgeted tuner searches the real workload inventory, not its Cartesian
product with every compiled configuration. It is independent of the older
exhaustive campaign and never reuses its timing samples.

## Selection and historical results

The design combines the strategies in `GEMM 配置选择方法综述与对比.md`:

- DeepGEMM-style tile/warp binding and multiple resource/traffic/tail rankings.
- A short list of complete tile proposals, followed by measurement and caching.
- Explicit historical geometry challenges, retained even beyond the soft budget.

DeepGEMM's W4A16 warp-K reduction, resource constants and weight layout are not
interchangeable with this library's K-pack kernels. We project recommendations
onto the existing **legal generated types**; numerical checks and real resources
remain owned by the existing compiled wrappers. Estimated occupancy is a ranking
hint, never a proof that a configuration cannot run.

`plan.json` reports each historical geometry's recall **before** forced inclusion.
Geometry recall is not measured performance recall. An optional `ANCHORS` JSON
array imports additional exact winners, each with `cell_key`, `route`, `symbol`.
An absent/inadmissible imported winner fails planning instead of disappearing.

The local historical replay against the uploaded FQ `summary.json` covers
**832/832 winner geometries**: 572 dense rows across Q2/Q3/Q5/Q6 and 260 grouped
rows across all five formats. Initially keeping only the grouped default lost
32 of these winners; the non-default grouped geometries are now mandatory
challenges too. This is CTA/warp/TK/stage coverage, not proof of the same runtime
provider/grid or a measured 5% bound. The source summary SHA-256 is
`0e6175907d1ff609f4c28aa4f7d92b440a50d1b122d70aff2fb6e34a80eb7f09`.

Replay a plan against the uploaded archive (or a plain `summary.json`):

```bash
python3 tools/replay_kpack_tuning_history.py \
  --plan /workspace/quactlize-kpack-budgeted-v1/plan.json \
  --summary fq-kquant-heuristic-handoff.tgz --require-all
```

The default soft budget is 32 **parent configurations** per route/workload.
Historical and axis challenges may exceed it. Actual persistent grids and
decode Split-K variants expand at runtime and are counted separately. Large
prefill FQ requests use S1; SF uses nonpersistent/persistent full-output boards.
At this revision, the full plan retains 1,381 workloads / 2,762 route-workload
pairs, selects 95,120 parent-workload pairs, and compiles a union of only 2,232
parents. These are different denominators, not kernel-launch counts.

## Execution and compilation

Only the selected parent **union** is compiled. Small shared modules expose a
private, source/SDK-bound benchmark registry. Drivers use the existing exact
fixtures and row functions; they do not implement a second kernel or oracle.

One process per device handles a sequence of requests with the same
qtype/route/N/K/expert geometry. It caches host-side placed weights and metadata.
A/router/golden are regenerated per request. That request's H2D allocations and
workspace are shared across every candidate module. Modules remain loaded until
the process exits. No module is unloaded while function pointers may be used.

Old executable bundles cannot become these modules without relinking/rebuilding.
The old full builder normally deleted its object cache. This entry therefore
builds the selected union, not all 70,483 parents. It retains objects and DSOs;
with one parent/module, later candidate additions do not invalidate existing
modules. A source/SDK change intentionally invalidates the affected cache epoch.

## Run on the PPU box

### Overnight staged search

```bash
python3 -u tools/run_kpack_overnight.py \
  --output /workspace/kpack-overnight-v1 \
  --sdk /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  --build-cache /workspace/kpack-minimal-95e0659/build \
  --jobs 192 --devices 0,1,2,3,4,5,6,7 --hours 10
```

Compilation is inside the ten-hour scheduling budget. The initial measured
calibration covers every qtype/route with small-M, large-work and large-weight
requests; it is not an extrapolation from Q4 alone. Before the full build,
`budget-admission.json` separates estimated compilation, screen and reserved
confirmation time, with a 2x safety multiplier. Predicted over-budget work is
rejected before starting the full campaign. An estimate is not a runtime proof.

To explicitly run without a total time limit, add `--no-budget-limit` and use a
fresh `--output` with the same `--build-cache`. This bypasses the cost estimator,
budget admission, and all campaign phase deadlines (`--hours` is not enforced).
The log reports `verdict=BYPASS_USER_REQUEST` and `remaining_minutes=UNLIMITED`.
Search bounds, numerical checks, device identity, per-request timeouts, failure
isolation, resumption, and the final 3x11 confirmation remain unchanged. This
mode may run longer than one night; it does not assert an overnight ETA.

The admitted sequence is: all-workload screen (default 32-parent soft budget,
three samples), measured-winner neighbors plus far challenges, wider boundary
audits, family propagation of audit gains, then three independent 11-sample
confirmation rounds. Confirmation retains the top four parents, up to four
near ties, and the initial incumbent. Rounds use different deterministic shuffle
seeds on the same assigned GPU. Matched runtime variants, not just parent
names, own the 33-sample denominator. Raw correctness repeats remain one.

Optional exploration may add at most 256 new compiled parents per stage;
compile-capped and deadline-skipped coverage is reported, not labelled slow.
Exploration cannot consume the estimated confirmation reserve. Deadline handling
terminates owned child work, allows up to 60 seconds of graceful shutdown, and
keeps completed artifacts. Missing rounds, missing variants and noisy winners
are explicit non-complete statuses. There is no exhaustive fallback or claim
that all shapes are globally within 5% merely because the timer expired.

Results are under `results/summary.tsv`, `results/summary.json`, and
`results/heuristic-input.json`; raw stage evidence is under `phases/`. The same
command resumes immutable phase inputs/results with a new time window. Source,
SDK, phase inputs and physical device identity must match. Change OUT for a new
measurement epoch. No production selector is updated automatically.

Physical-device uniqueness uses the existing host-only SDK probe and its
`hggcDeviceGetPCIBusId` result, including the PCI function. The driver's older
property-based PCI string is diagnostic only. `KPACK_TUNER_DEVICE_ID` prints
both values for each worker; an actual duplicate API identity still rejects
execution. The identity fix needs no kernel/module rebuild. Use a fresh OUT
with the previous build cache when changing orchestration identity. Cached
calibration builds retain their original compile-cost receipts, so new kernels
are not budgeted at zero compilation cost after a restart.

The prior Q4 minimal box probe reported a cold build of 45.023 seconds for nine
compile units and 8.325 seconds for eight route/workloads on one card. This
validates the module path, not the duration of the all-format overnight search.

### Minimal probe and initial screen

For a **minimal compile + runtime wall-time probe**, use the separate entry:

```bash
python3 -u tools/run_kpack_minimal_probe.py \
  --output /workspace/kpack-minimal-probe-v1 \
  --sdk /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK --jobs 192 --device 0
```

This only builds four Q4 modules/four drivers plus one layout object (nine
compile units), then runs eight route/workload pairs: FQ/SF dense M=1/2048,
N=1024 K=5120; FQ/SF grouped tokens=1/2048, N=512 K=2048, E=256. Each route uses
one `16x64x64_w16x16_s2_ap0_dn16` parent. It emits `KPACK_MINIMAL_BUILD`,
`KPACK_MINIMAL_RUN`, and `timing.json`. There is no full-campaign continuation.
The default watchdog is 30 minutes per build/run phase, with up to 60 seconds
for graceful termination. Use a fresh OUT for cold-build timing. Nine compile
units cannot saturate 192 cores; do not scale wall time solely by parent count.
This proves neither whole-night duration nor global performance coverage.

Use an idle device pool. Do not run the old sweep or another GPU benchmark at
the same time. All commands below run scripts as child processes; they do not
exit the interactive container shell.

```bash
OUT=/workspace/quactlize-kpack-budgeted-v1 \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
JOBS=192 DEVICES=0,1,2,3,4,5,6,7 \
bash tools/run_kpack_budgeted_tuner_box.sh
```

This first admits the module transport on four Q4 workloads/eight route-workload
pairs, then builds and screens all five formats and all real/control/historical
workloads. The gate modules are reused; gate timings are not copied into the
full screen. `MODE=pilot QTYPES=12` with a **different OUT** only exercises four
representative workloads/eight routes. `MODE=build` builds without using a GPU.
`PARENTS_PER_MODULE=1` is the default for incremental cache reuse and parallel
compilation. `JOBS` is a process concurrency limit, not a CPU-utilization promise.

For a later search with a different budget or candidate rule, use a **fresh OUT**
and set `BUILD_CACHE` to this run's `OUT/build`. With unchanged kernel source/SDK,
existing one-parent modules are reused and only new candidates compile. Each OUT
keeps its own `bundle.json` snapshot; a subsequent search cannot overwrite its
measurement authority. A kernel/SDK change requires a fresh build cache.

Repeat the same command/OUT to resume completed compilation and measurements.
Use `MODE=retry` on the same full-screen OUT to retry rejected candidates only.
Completed candidates remain valid inside the same source/SDK/device epoch;
their normalized cells are rederived from hashed raw logs before reuse.

Raw/numeric failure never becomes a timed result. A failed candidate terminates
that process to clear sticky runtime errors. Other candidates run in a fresh
process. Fixture/module/device failures leave the group incomplete rather than
declaring all its candidates slow. `failures.json` and raw logs retain evidence.

## Outputs and the 5% goal

- `plan.json`: workloads, candidate reasons, anchor recall and compile union.
- `bundle.json`: frozen module, driver, source and SDK hashes for this run.
- `screen/summary.tsv`: best measured full-output runtime variant per route/shape.
- `screen/heuristic-input.json`: exact-shape inputs for fitting the initial policy.
- `screen/results/`, `screen/logs/`: individual resumable results and raw evidence.

FQ Split-K measures **producer and real reducer inside the event span**. It is
not compared against S1 using producer-only time. Cross-route prepass costs and
amortization are not included: outputs are per-route kernel recommendations,
not an automatic ScaleFirst-versus-FQ inference route decision.

The initial two-sample screen provides a provisional policy input. It does not
prove a global 5% bound. Follow-up work is measured-winner neighborhood expansion,
wide-search holdouts, and selective multi-round confirmation. A 5% claim must
name its measured reference set and cover each shape, not only the mean.
No production selector, inference ABI, or llama.cpp behavior changes merely by
running this tuner.
