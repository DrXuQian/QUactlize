# GEMV / FQ / ScaleFirst operator comparison

Status: the FP32-affine SIMT experiment compiles locally with hgcc. Its
numerics and speed on PPU are pending; RTX 5090 results are not PPU admission.
No production kernel or heuristic is changed by this gate.

Local checks: the eight small compilation units plus link take 10.61 seconds;
the DSO is 1,070,896 bytes. The exact Q4 C16/W4 SIMT specialization has
`mma_en=0`, 60 vector registers and zero stack bytes in hgobjdump. Package
checks inspect the actual dynamic exports. Both real execution DSOs also
pass 240 host-only shape/config queries (plus ten invalid-config rejections)
under a compatible Ubuntu 24 runtime, without GPU initialization or launches.
The common-endpoint adapter bodies also pass six actual
RTX 5090 cases / eighteen mutable-permutation graph replays, including
broadcast/per-slot inputs, extent25,600 and a tail. That adapter test does
not execute or admit the PPU GEMM modules.

## Scope

| Case | Qtype | Logical call | Experts / active | Canonical active weight bytes |
| --- | --- | --- | --- | ---: |
| q4-up | Q4_K | eight 1×512×2048 expert GEMVs | 256 / 8 | 4.5 MiB |
| q5-down | Q5_K | eight 1×2048×512 expert GEMVs | 256 / 8 | 5.5 MiB |
| q4-dense-matched | Q4_K | 1×4096×2048 | 1 / 1 | 4.5 MiB |
| q4-dense-wide | Q4_K | 1×8192×5120 | 1 / 1 | 22.5 MiB |
| q4-dense-long-k | Q4_K | 1×5120×25600 | 1 / 1 | 70.3125 MiB |

The first three are small decode work units. The last two separate fixed
launch/scheduling overhead from large-weight reader efficiency. The dense
matched case has the same weight count as q4-up, but uses its own deterministic
GGUF fixture; it is not a concatenation-bit-identity claim.

The old pair reader and new FP32-affine/vector-load reader each scan 24 small
SIMT recipes (columns16/32, warps2/4/8, split1/2/4/8). This is an offline
diagnostic, not runtime model tuning. The pair baseline and SF prepass come
from `kpack-decode-sweep-v1/libquactlize_ppu_execution.so`; native-v1's older
execution DSO has only the scalar reader and cannot supply the pair API.
Only that small decode-sweep DSO is required, not its GEMM modules. FQ/SF use the **actual native C++
selector** and recorded parent/build key/split/grid, without a manual default
or silent fallback. The ten selected parents already exist in native-v1;
no GEMM module is compiled. The Q4/Q5 grouped FQ choices are TM8/S4 and TM8/S1.
The grouped SF choices retain their existing `DEVICE_BOUNDS` evidence label;
this test does not assert that they are globally optimal SF recipes.

## Comparable timing boundaries

All algorithms see the same logical GGUF weights, FP32 activation values and
GPU routing inputs, and return the same ordered FP32 output shape. Activations
are FP16-representable. No activation Q8 quantization is used.

- `pair`, `affine`: indexed/dense F32 GEMV including its Split-K reducer.
- `fq`: GPU gather/F32→F16 + selected native FQ call + F16→F32 scatter.
- `sf`: the same endpoints, with actual GPU-expanded metadata retained.
- `sf_with_prepass`: one all-expert GPU metadata prepass plus the entire SF
  endpoint pipeline in the same timed graph. This is a measured recomputation
  scenario, **not** the deployed immutable-weight cache behavior.

The input expert IDs, sorted-row permutation and cumulative offsets are
already on GPU. Creating those routing arrays is outside timing for every
arm. This is not full llama.cpp/model latency. Changing their contents tests
captured execution without re-preparing handles; there is no per-call CPU
router readback in the timed graph.

`core_median_us` separately times FQ/SF with resident FP16 A/output, including
native metadata/directory/producer/reducer but excluding endpoint adapters.
Do not rank this number against full F32 GEMV without naming that difference.
The SF planes consumed by GEMM come from the real PPU prepass, not host
oracle uploads. First-prepass event time, warmed prepass time, expanded bytes
and bit-exact scale/zero checks are recorded separately. The first event
interval can include first-launch initialization; ACU supplies kernel detail.

Screening uses three samples × eight graph calls. Final confirmation uses
four alternating-order rounds × eleven samples × sixteen calls. Every
screened recipe must pass the independent official-GGUF dot oracle. Winner
checks include eager/captured execution, three router profiles, guards,
fixed-order SIMT reduction, and independent FP32 FQ/SF partials where present.
Intermediate FP16 rounding differs across the algorithms; bit identity is
not the cross-algorithm accuracy contract.

The reported effective GB/s uses distinct active weight-plane bytes. It is
not measured DRAM bandwidth or proof of peak memory utilization. Warm graph
times and ACU replay times must not be mixed.

## Box: pull and execute, no compilation

```bash
(
  git pull --ff-only &&
  git lfs pull --include='prebuilt/ppu0010/kpack-native-v1/**,prebuilt/ppu0010/kpack-gemv-affine-v1/**,prebuilt/ppu0010/kpack-decode-sweep-v1/libquactlize_ppu_execution.so' &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
    bash tools/run_kpack_gemv_fq_sf_box.sh
)
```

Run only on an idle PPU. Default ACU location is
`$PPU_SDK/asight/bin/acu`; `--acu /absolute/path/to/acu` overrides it.
Use `--skip-acu` for timings alone, or `--only-cases q4-up q5-down` to restrict
the case set. Each case is a fresh process; failure preserves that case's
completed screening evidence and does not discard the other cases.

After a case passes, five separate native reports are captured:
`<case>-pair.acurep`, `-affine.acurep`, `-fq.acurep`, `-sf.acurep`, and
`-prepass.acurep`. Only one warmed graph call is inside each profiler range.
The GEMM reports include endpoint adapters, so inspect each component rather
than interpreting all kernels as GEMM. Full `--set full` ACU replay adds time
beyond warm timing. There are 25 reports for all five cases, not a full sweep.

The wrapper prints `summary.tsv`, the `acu/` directory and the results tarball.
Return that tarball. Keep the printed run directory if a retry is needed:

Timing and ACU status are separate columns: a failed report capture does not
invalidate an already-complete warm timing case.

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  RESUME_RUN=/workspace/kpack-gemv-fq-sf.REPLACE \
  bash tools/run_kpack_gemv_fq_sf_box.sh
```

Resume requires identical payload/source/device/count receipts. Complete
timing cases and hash-verified reports are reused; failed files are renamed
and preserved before retry. No old successful evidence is silently relabelled
as a measurement of a changed binary. The caller's Docker shell is preserved.

## Initial box launch failure and repair

`kpack-gemv-fq-sf.25YktY.results.tgz` contains five failures with empty
screen/winner lists: the runner requested `quactlize_kpack_gemv_pair_query_v1`
from native-v1's scalar-only execution library. No fixture, numeric test,
timing sample or ACU capture was reached. This was a runner package-selection
error, not evidence about PPU correctness or performance.

The repair selects the already-published decode-sweep pair DSO
(`07a7bbc02c8fbbe9a2fcb9e902dc85f8e2daba3b961976b16545b312b77b242e`),
checks the actual ELF exports and recipe inventory before GPU allocation,
and records both readers in result authority. It changes no binary or
production selection. All five cases must be run in a new results directory;
the failed archive remains preserved. Future failures print the case, phase,
exception and log path to the console instead of only an aggregate return code.
