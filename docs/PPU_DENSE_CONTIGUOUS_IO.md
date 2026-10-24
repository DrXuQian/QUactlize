# Dense conversion and compact reducer reuse

IO4 adapts the MIT-licensed contiguous body from llama.cpp's
`ggml/src/ggml-cuda/convert-ppu.cuh` at
`d30de5bbe45fabb23bcc7a6832d84f0046c16f23`. It does not depend on llama headers
or binaries. IO0..3, the measured helper tables, GEMM and dequant implementations
are unchanged. IO4 and the existing compact reducer are explicit measurement
candidates; neither is enabled by a production policy in this change.

## Contract and local evidence

Use typed restricted pointers, 32-bit eight-element chunk indices and one
chunk/thread, 256 threads/CTA. Host eligibility requires 16-byte aligned
endpoints, contiguous rows, count divisible by eight, and count/8 <= INT_MAX/2
so both `c` and `2*c+1` fit. Otherwise IO4 retains the old IO1 path. No added
shared memory, transpose, clipping, or BF16-through-FP16 conversion.

Output with a non-null producer-error flag uses a checked specialization;
nonzero status still writes canonical FP32 NaNs. A null flag uses the exact
unchecked body; input ignores the flag as before. Buffers remain nonoverlapping
and validated by the existing entry. Streams/addresses are per-call.

Local SDK2.1.1/PPU1.0 native-code comparison, not GPU timing:

| Body | Old IO1 static sites | IO4 static sites | Extracted native control |
|---|---:|---:|---:|
| F32 -> F16 | 92 | 52 | 52; instruction bytes identical |
| F16 -> F32, flag absent | 166 (generic flag branch) | 63 | 63; instruction bytes identical |
| F16 -> F32, flag present | 166 | 75 | Native has no corresponding flag contract |

These are complete static bodies including padding, not dynamically executed
instruction counts or predicted latency. All four unchecked F16/BF16 bodies
match their extracted native controls byte-for-byte; all 24 pre-existing
conversion bodies retain their instruction bytes. All use 32 registers/thread,
zero shared memory and zero local stack with this compiler. Box compiler/device
correctness and timing are separate gates.

The previous same-run Full-F16 example M2048/N6144/K3072 had input/output
casts 32.20025/21.610us versus native24.58025/18.410us. Its dequant was already
faster (31.79025 vs37.75025us), and main GEMM effectively equal. This motivates
cast reuse; it does not certify IO4's speed. SF's custom Split-K reducer is a
separate optimization. Full's provider GEMM and internal reducer are already
cuBLAS and should not be replaced by a copied undocumented reducer entry.

The in-library reuse candidate for SF is:
`PpuMixedInputSplitKParallelCompactReduction` already flattens contiguous
`[S,M,N]` to `[S,M*N]`, uses aligned vector access and compile-timeS2/4/8,
and preserves increasing-S FP32 addition followed by F16 rounding. Grouped
runtime uses it; `dense_splitk_parallel_ppu.cuh` instead instantiates
`PpuMixedInputSplitKParallelM1FastReduction` without `FlattenContiguousRows`,
so M>1 falls back to the generic reducer. Reusing the compact dispatcher is
an isolated experiment, not a new reducer algorithm or a proved speedup.
Do not switch it globally: CTA grain changes, contiguous/aligned/size checks
must still pass, and producer/output layouts must agree. A default-false
`PreparedMultiformatLauncher` template argument selects compact<2>. Generated
native F16 dense parents carry `dense_reducer=2` and an `_rc2` suffix, with their
own source/build identity. Existing parents, bound/dynamic call ABI and all
policies are unchanged. Typed decode and grouped modules reject this option.
The option changes only the reducer dispatcher: ordinary/Split-K GEMM, partial
layout, workspace size and increasing-S sum are unchanged. Native-code audits
compare every GPU body/resource against the retained pre-change module, rather
than inferring that from identical source templates.
Local validation for this combined candidate: 773 product-host tests passed;
all seven cohort TC modules and the actual reducer diagnostic carrier compiled
with PPU SDK2.1.1. Every GPU body/resource in those TC modules matches the
retained pre-change module. No local PPU runtime was available; this is not
a device numerical or performance pass.

## Bounded Box check

Use the latest completed Q8 TP refinement compact review, not the earlier
pre-refinement report. `--phase helpers` keeps the best measured SF and Full-F16
recipe for every point, adds IO4 to each, and remeasures native. The 22 points
whose SF control uses Split-K also get reducer-only and IO4+reducer comparisons.
For the current cohort this is 65 points, four or six candidate arms/point,
304 candidate arms, seven shared TC modules and two opposite-order confirmations
(738 timed arms including native). `--phase io` remains available for casts alone.
No GEMM geometry or DQ sweep and no
historical microseconds inserted into new comparisons. Full sums retain input,
expansion, GEMM, any reducer, output and reset; Asys also records each component.

Each worker first runs `tools/check_dense_io.py` against its actual built
library. The gate checks scalar/old/new raw equality, all16-bit input words,
rounding boundaries, overflow, infinities, NaNs, strides/tails, vector-unaligned
addresses, redzones/padding, source preservation and producer-error propagation.
NaN payloads must match the scalar device control; independent IEEE checks
compare every non-NaN bit and NaN classification. This gate is not timed.

Workers with Split-K candidates also run `tools/check_dense_reducer.py` against
the compiled existing-body diagnostic carrier. It deduplicates [M,N,S] across
K/families, checks actual cohort sizes plus S8, cancellation, F16 ties/overflow,
tails/strides/alignment, repeated execution, guards and invalid spans. Its
independent oracle adds partials in increasing-S FP32 order, rounds once to F16,
and requires raw-bit equality with both reducer variants. Asys must then show
`PpuMixedInputSplitKParallelM1FastReductionKernel<2,S>`, block32, grid=M*N/64 and
zero shared memory in each compact candidate; a generic fallback is a failed
experiment, not an accepted candidate timing. Complete-call numerical checks
remain mandatory as well. Standalone numerical gates are never timed.

After updating to the commit containing this candidate, on idle GPUs:

```bash
cd /sim/eec/shared/junfu.qx/quactlize
python3 tools/run_q8_tp_refine.py \
  --phase helpers \
  --review /workspace/kpack-q8-tp-refine.lj0co_ma/timing/review-1792806745232929096.json.xz \
  --llama-source /sim/eec/shared/junfu.qx/llama.cpp.ncp \
  --llama-build /sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
  --sdk /usr/local/PPU_SDK --gpus 1,2,3,4,5,7 \
  --jobs 8 --configs-per-batch 16 \
  --run --confirm-idle
```

The existing runner requests1700MHz; actual-frequency variation is advisory,
not a failure. Generated source/cache/raw traces stay on Box. Upload the printed
compact `review-*.tgz`; it also includes IO/reducer gate results/logs. Resume uses
the same arguments plus `--resume --output <printed run directory>`.

The coordinator takes explicit `--jobs` (1..64, default8) and
`--configs-per-batch` (1..32, default16), records both, and overrides ambient
`JOBS`/`CONFIGS_PER_BATCH` for its workers. A parent-shell build setting such as
`JOBS=192` must not leak into this bounded experiment. Worker logs also print
the effective values; invalid CLI input is rejected before launching any workers.
The first uploaded helper attempt (revision731c2db, all six workers rc2) failed
at argument validation, before compilation/numerics/timing; the old diagnostic
did not identify which inherited value was invalid. Keep that attempt intact
and start a fresh run with the fixed launcher, not `--resume` (no completed
device work exists there to reuse).

No production selection is promoted by this command. Compare complete calls
and both orderings before choosing IO4 or compact reduction for a measured scope.
Do not use `--profile-acu` for the IO/reducer phase: that flag belongs to the
independent dequant phase. The Asys reports and per-kernel durations are retained
for selecting a bounded follow-up profile if required.

Metadata erratum for earlier dequant reports: the old `NO_FLUSH` label meant
no software cold-ring loop in the profile range, but ACU was invoked with
`--cache-control all`. The new label makes both facts explicit. Never use those
profile replay durations as rotating-event or complete-call timings.
