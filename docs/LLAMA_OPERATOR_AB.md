# Native llama operator regression gate

`tools/run_kpack_llama_ab_box.sh` compares actual llama CUDA/PPU dispatch with
raw GGUF weights against explicit K-pack weights in the **same shared-library
build**. It does not copy a particular reference kernel, resweep configs, or
use the old Quactlize selector as the native baseline.

The fixed list has 1544 points: 1114 decode profiles (M=1..8) and 430 prefill
points (M=128,512,1024,2048,4096). There are 810 dense and 734 indexed points.
Formats are Q8_0 and Q2_K..Q6_K; the historical prefill registry has no Q8
large-M points. Router profiles are preserved even after a selector promotion
collapses its host keys. The list is not the Cartesian product of these axes.

## Prerequisites

- Use the **full PoC caller** with both MUL_MAT and MUL_MAT_ID enabled for the
  full cohort. The dense-only MR cannot cover the 734 indexed points. The
  pilot will reject it, not skip those points or reinterpret packed weights.
- `LLAMA_CI_DIR` is the exact source corresponding to `LLAMA_BUILD_DIR`.
  Build llama only through its existing `.aoneci/scripts/build.sh`. The test
  compiles a small host-only carrier, not llama or a new device kernel.
- Keep the candidate runtime environment from the working model test:
  `QUACTLIZE_KPACK_EXECUTION`, `QUACTLIZE_PPU_BUNDLE`, and
  `QUACTLIZE_PPU_PACK_LIBRARY` when the PoC requires them. The public dense MR
  instead loads its installed `libquactlize_ppu.so` and private runtime.
  No library is downloaded or silently replaced. A stale caller that rejects
  a new policy number must be updated before this gate can admit that policy.
- All assigned physical cards must be idle for the whole test. The script
  uses the same advisory locks as the component campaign; unrelated programs
  need to be stopped separately. CUDA Graph would not isolate other work.
- L2 is queried using the installed SDK attribute. If it reports zero, supply
  independently verified `L2_BYTES`; 67108864 is the user's verified ZW810
  value, not a guessed default for other devices.

```bash
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp \
LLAMA_BUILD_DIR=/path/to/the/matching/full-poc-build \
PPU_SDK=/usr/local/PPU_SDK L2_BYTES=67108864 \
DEVICES=1,2,3,4,5,6,7 \
bash tools/run_kpack_llama_ab_box.sh
```

First all workers finish a 22-interface pilot. Numerical, route or setup
failure prevents starting the full phase. A measured performance regression
does not prevent measuring independent points. A fresh process follows any
failed device call. Already measured points remain on disk. Resume with the
same arguments plus `AB_RUN=/workspace/kpack-llama-ab-... AB_RESUME=1`.
PASS, REGRESSION and INCONCLUSIVE measurements are retained; resume is not a
way to select the fastest of repeated noisy runs. Changing the build, library,
SDK, source or physical-card mapping requires a new output directory.

## Meaning of the result

Both arms receive identical deterministic GGUF quantization, F32 activations
and resident expert IDs. The candidate uses F16 dense and BF16 indexed
compute; native arithmetic remains llama's own choice, including its Q8
activation quantization where applicable. A full finite-output and relative
L2 comparison plus independent raw-GGUF spot dots runs before timing.
This is a numerical smoke check, not a model-accuracy admission.

The timed interval is the **eager complete-operator event envelope**, on the
backend stream. It includes required input quantization/conversion, directory,
GEMM, reducer and output finish, and can include CPU launch starvation.
It is neither summed isolated kernel time nor model TPOT. No CUDA Graph,
Asys or profiler is used. CUDA-event and host wall samples are both retained.
The component sweep remains the tool for individually profiling a reducer or
prepass; this separate A/B gate deliberately tests the complete operation.

Warmup, loading, packing and selected-parent JIT are excluded. Each sample
traverses an entire rotating weight ring whose **active** compressed-weight
read set is at least 2.25 times L2, with a minimum of two copies. Both formats
use the same ring geometry. SF metadata by itself is not claimed cold. The
large E/topk ratio may need substantial memory; insufficient memory fails
explicitly instead of shrinking the ring.

Six ABBA rounds, three ring samples per block. All six round ratios must be
<=1 for PASS, with <=5% same-arm ABBA order drift. Stable slower points are
REGRESSION; overlapping/noisy points are INCONCLUSIVE. There is **no 5%
slowdown allowance** against native. This observed round envelope is not a
statistical confidence interval. Full coverage and all PASS are necessary
before claiming no measured operator regression in this scope. Missing,
unsupported, stale-route and numeric failures prevent admission.

`review-*.tgz` contains the exact workload/build/device identities, raw scalar
samples, correctness metrics, selected route lines and bounded failure logs.
It excludes weights, caches, JIT objects, binaries and profiler reports.
Model TPOT, fused gate/up/SwiGLU/down chains, TP/EP communication, model
activation distributions and other devices/SDKs still need their own A/B.
Local host/syntax checks do not imply that the PPU pilot has passed.
