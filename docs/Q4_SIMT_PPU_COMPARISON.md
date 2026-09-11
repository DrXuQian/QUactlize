# Q4 SIMT PPU comparison

This development-only package ports the six-family FP32 CUDA comparison to
native hgcc/PPU. It does not change production dispatch or offline bytes.
The prebuilt files are compile-checked, **not PPU device-admitted**; the box
runner performs that admission before timing each candidate.

## Scope

All cases use dense M=1, one expert, canonical Q4_K weights and FP16 A:

| N | K |
|---:|---:|
| 512 | 2048 |
| 1024 | 5120 |
| 4096 | 2048 |
| 4096 | 4096 |
| 5120 | 8192 |
| 8192 | 5120 |

The four arms are historical Xplane with FP32 dot accumulation (12 recipes),
current K-pack SIMT (24), optimized K-pack SIMT (108), and the current FQ
Tensor Core production selection (one selected parent/algorithm/Split-K).
The latter is not a new exhaustive Tensor Core sweep. Missing selected FQ
parents JIT on first use; compilation, uploads and warmup are outside timing.
SIMT and Xplane return F32; the production FQ API returns F16. All share the
same independent official-GGUF FP64 dot oracle and a conditioned 0.005 error
bound, with a zero-code negative. SIMT Split-K additionally checks the output
bitwise against the ordered FP32 sum of its device partials.

SIMT/Xplane accumulation precision matches, but this is not a layout-only
experiment: their half-affine dequantization expressions also differ. See
[the CUDA comparison](Q4_KPACK_FP32_COMPARISON.md).

Warm uses one resident weight; rotating uses at least two distinct allocations,
totalling at least 2.25 times the device-reported L2. Every timed graph traverses
whole rings, including across replay boundaries. If `DeviceProperties` reports
zero L2, the runner queries native `hggcDeviceGetAttribute(...,38,0)` separately.
The scalar attribute does not depend on the properties-structure layout. If
both interfaces leave capacity unavailable, use an explicit verified
`L2_BYTES`; no hardware size is guessed. For example, **if the board has 64 MiB**,
`L2_BYTES=67108864` selects that capacity and is recorded as `EXPLICIT_OVERRIDE`,
not an SDK-measured value. Both SDK observations remain in the receipt.
For a weight larger than L2, “warm” means repeatedly used, not fully L2-resident.

The [local SDK audit](PPU_SDK_L2_QUERY_AUDIT.md) confirms that the native
attribute-38 path is implemented, not an empty stub. The PPU result `0` from
the properties path alone does not prove that the scalar attribute is absent.

Screen all SIMT recipes with five samples. Confirm the top two plus the best
S1 when necessary, in six alternating-arm rounds of fifteen samples. Each
sample averages at least 32 complete calls via graph replay; Split-K includes
its reducer. First graph upload/launch is discarded. The result is the median
of the six round medians. New K-pack must be within 5% of FP32 Xplane separately
on all twelve shape/cache cases; a clean slower result remains `PARITY_OPEN`.

`summary.tsv` reports all four times and effective weight GB/s. This uses
one distinct low+metadata byte count per call, **not measured HBM traffic or
bandwidth utilization**. Warm-cache GB/s cannot be interpreted as HBM MBU.

## Run on the PPU box

Use an idle device. From the repository:

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 ACU=1 \
bash tools/run_q4_simt_ppu_box.sh
```

The script sources the SDK environment in a subshell. It needs Python with
NumPy, Torch and gguf, the LFS payloads under `q4-simt-ab-v1` and `kpack-jit-v2`,
and the SDK compiler for any missing FQ JIT parent. It never exits the caller's
Docker shell. Optional `RESULT_ROOT`, `PYTHON`, `JIT_CACHE`, and `ACU_BIN` select
existing local locations. `ACU=0` skips profiling without changing timing.

Results are written under `/workspace/q4-simt-ppu.XXXXXX/results`:

- `summary.tsv`: twelve comparison rows.
- `summary.json`: selected recipes, raw timing samples, per-arm failures.
- `authority.json`: source/package/fixture/runtime/device identities.
- `*.log`: exact per-batch numeric and timing records.
- `*.acurep` or `*.report`: selected new-K-pack and Xplane profiles.

The script prints and creates `RUN.results.tgz`; upload that archive. It omits
fixture arrays and the JIT build cache. Cases continue after a failed child,
but a failed candidate is never silently dropped to claim an arm passed.

Resume with `RESUME_RUN=/workspace/q4-simt-ppu.XXXXXX` and the same invocation.
Successful exact batches and ACU reports are reused only when their receipts
match; failed attempts remain preserved. A changed device/runtime/source/input
requires a new run directory.

### Repair a stale FQ JIT dispatcher without repeating SIMT

The original `q4-simt-ppu.0uCEmO` run completed the three SIMT arms but all twelve
FQ cases stopped at the JIT source-contract check. The small dispatcher had
not been rebuilt after kernel headers and the generator changed. This was not
a device arithmetic failure. The package now includes the rebuilt 599 KiB
dispatcher; the execution library and all three comparison kernels are unchanged.
Preflight checks the live checkout contract before starting a campaign.

After the previous run has finished, from the box repository:

```bash
(
  GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin develop &&
  git lfs pull --include="prebuilt/ppu0010/kpack-jit-v2/*.so" --exclude="" &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 ACU=1 \
  RESUME_RUN=/workspace/q4-simt-ppu.0uCEmO REFRESH_FQ=1 \
  bash tools/run_q4_simt_ppu_box.sh
)
```

`REFRESH_FQ=1` admits only the audited old orchestration (or the current one)
and a changed native package. The child runner, SIMT bundle, fixtures, device,
SDK runtime, L2 setting and sample counts must still match. Successful SIMT
batches are reused; FQ caches must carry the new native-manifest hash. Old
authority and failed logs remain saved. FQ measurements are explicitly marked
**supplemental**, not contemporaneous four-arm alternating rounds. JIT cold
compilation remains outside kernel timing; numerical device admission is still
required.

**ACU is not timing authority.** SDK 2.1.1 flushes even when its cache-control
setting says `none`; this runner explicitly uses `all` and profiles individual
nodes. ACU captures the warm-selected new/Xplane recipes for diagnosis, but
its durations must not replace the unprofiled warm/cold timings.

For the uploaded `5120×8192` case, both regimes selected Xplane `C2/W8/S1`
and new K-pack `C4/W8/S1`. The existing `n5120-k8192-{xplane,new}.acu.acurep`
reports therefore already cover both selected kernel specializations. See
[the first PPU ACU comparison](Q4_SIMT_PPU_ACU_20260911.md) for the counters
and the distinction between profiled cold calls and rotating-graph timings.

## Rebuild locally

```bash
python dev/gemv_ppu/build.py --sdk /path/to/PPU_SDK \
    --output /data/new-q4-ppu-build --jobs 6
```

The builder generates native hggc runtime includes, compiles PPU ISA, and
requires the probe and optimized specialization in device disassembly. It does
not link NVIDIA libraries. Generated sources, commands and hashes stay with
the build receipt. The small shared libraries are distributed with Git LFS.

Grouped, multi-token, non-Q4 and production selection changes remain outside
this diagnostic's admission scope.
