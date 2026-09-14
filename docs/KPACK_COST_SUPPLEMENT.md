# Remaining heuristic component measurements

This finite supplement collects missing cross-route costs. It does not run
an online tuner, change production dispatch, or repeat the config Cartesian
product. The actual C++ production selector supplies each FQ/SF candidate;
one applicable historical S1 winner per route is retained as a challenger.
Some challengers may coincide with the selected recipe and serve as controls.

| Phase | Workload points | Scope |
|---|---:|---|
| `dense-mid` | 30 | Q4/Q5, five already measured weight families, M128/512/1024 |
| `dense-rest` | 65 | Remaining historical Q4/Q5 dense geometries, M128/512/1024/2048/4096 |
| `formats` | 255 | Q2/Q3/Q6 dense and grouped real geometries, the same five token counts |
| `grouped-mid` | 36 | Q4/Q5, six grouped families, tokens128/512/1024 |
| `sparse` | 72 | Q4/Q5, six grouped families, noncontiguous active32/64 of E256, tokens128/512/1024 |
| `all` | 458 | Union of the five phases |

Grouped uses top8; total GEMM rows equal tokens times eight. No small-M
full-dequant candidate is added. The unmeasured range between decode and
M128 is not automatically admitted to full-dequant by this experiment.

Each point measures selected FQ, selected SF, applicable historical
challengers, and BF16 cuBLAS (dense) or the installed Python-JIT DeepGEMM
grouped entry. Dequant is separately deduplicated by weight and expert set,
not remeasured for every M. The 44 audited all-expert Q4/Q5 SF/full costs
are reused only after exact fixture, physical device, runtime and package
matching. Their receipts are in
[`kpack_cost_reuse_20260914.json`](measurements/kpack_cost_reuse_20260914.json).
The previous 88 selected large-M GEMM measurements remain valid.

The new active-expert full-dequant entry reads resident GPU IDs/count,
preserves original expert strides, and leaves unused experts untouched.
It reuses the existing vector/packed bodies; no proportional E256-to-active
timing estimate is used. Tests include an empty list, changing IDs inside a
captured graph, an invalid ID, poisoned unused output and official GGUF
weights. This is a measurement entry, not yet a production route. Creating
the unique active-ID list is outside its timed scope and remains an
integration cost to account for if not already available from routing.
SF expands only metadata, never full weights. It retains the existing
all-expert scale-plane contract in this experiment.

## Box command

Run on the same idle physical PPU as the previous measurements. For grouped
BF16, the installed `deep_gemm.jit_kernels.m_grouped_gemm` Python entry must
be available to the selected Python interpreter. No provider fallback is
silently substituted. No ACU/asys capture is requested by default.

```bash
cd /sim/eec/shared/junfu.qx/quactlize &&
git pull --ff-only &&
CUDA_VISIBLE_DEVICES=0 JOBS=192 \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
bash tools/run_kpack_cost_supplement_ppu_box.sh all
```

Use `dense-mid` instead of `all` for the first 30 points. The build closure
is shared by all phases: 134 deduplicated GEMM parent images. Existing exact
images/cache entries are reused; only missing parents and the small support
library/dispatcher compile. This is **not** a rebuild of the old giant
bundle. Compilation requests 192 concurrent slots; utilization is not
guaranteed at startup or the final few compilation units. Numerical device
admission is still performed on box, not inferred from a successful build.

The script prints `COST_BUILD_PROGRESS`, `COST_FIXTURE`, `COST_COMPONENT`
and `COST_PROGRESS`. Runtime ETA is explicitly a recent-weight estimate;
there is no proven whole-campaign wall-time yet. Do not derive a wall-time
promise by multiplying GEMM microseconds alone: fixtures, setup, compilation
and JIT are also wall time, although excluded from GPU timings.

Each successful component is checksummed and survives a later failure.
Rerun with the printed run directory to execute only missing/failed points:

```bash
CUDA_VISIBLE_DEVICES=0 JOBS=192 \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
RESUME_RUN=/workspace/kpack-cost-supplement.REPLACE_WITH_PRINTED_SUFFIX \
bash tools/run_kpack_cost_supplement_ppu_box.sh all
```

This also extends a completed `dense-mid` run to all phases. Reuse the same
checkout, SDK and build directory. Changed evidence identity is rejected;
failed attempts remain in `results/failures/`. The shell is preserved on
error. Send only the printed `*.results.tgz`, not modules or compile cache.

## Timing and admission

- First JIT, upload and graph replays are untimed. Each component has three
  rounds of five event samples over complete rotating weight rings.
- FQ/SF GEMM measurements include their actual internal directory/reducer;
  SF prepass and external activation/output adapters are excluded.
- BF16 provider timings exclude dequant, include the provider's own internal
  directory, and use weights made by validated device dequantization.
- Component sums are labelled `SUM_OF_ISOLATED_COMPONENTS_NOT_MEASURED_E2E`.
  They are route-cost evidence, not a measured end-to-end model speedup.
- Round-median spread over 5% is recorded as requiring confirmation, not
  hidden by averaging. Unrelated successful measurements are retained.
- One noisy small-M reducer point, `dense-m1-n4096-s4`, is rechecked. The
  other 137 stable points and the 48 large reducers are not repeated.

After result review, update the production heuristic from the cost evidence
and perform route/numerical/model regression. Do not add an isolated reducer
to a complete GEMM measurement which already includes it, nor substitute a
plain reducer for a fused reduce/scatter measurement. Sparse full-dequant
still requires the corresponding active-ID production integration before
shipping; this harness alone does not admit that path.
