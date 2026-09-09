# GEMV / FQ / ScaleFirst operator comparison

Status: the five-case Q4/Q5 PPU gate passes in `k5Tp2m`; 240 SIMT recipes,
25 confirmed endpoint arms and 25 ACU reports were reviewed. The FP32-affine
experiment improves on the old pair reader but does not generally beat FQ.
This is bounded operator evidence, not all-format or model admission. No
production kernel or heuristic is changed by this gate.

Subsequent user decision (2026-09-09): SIMT GEMV is sufficient for the
current milestone; further optimization is parked and automatic decode
stays on FQ. This is a delivery decision, not a rewrite of the measured
comparison or a claim of parity on every platform/shape. ScaleFirst is to
expand metadata for every call; use the `sf_with_prepass` full-call column
for that execution model. The `sf`/resident-core columns are diagnostic
components and cannot amortize expansion across requests. The current llama
adapter's cross-call `scale_ready` caching is a separate pending correction.

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

## Reviewed PPU results: k5Tp2m (2026-09-09)

Archive: `kpack-gemv-fq-sf.k5Tp2m.results.tgz`, SHA256
`4575ee6f969ceb31cf3b67504c048c67e0facc234378e5427f38a466a9f1ef8e`.
Its driver hashes match `ff1e488`; native/pair/affine manifest hashes match
the published packages. All 25 report and capture-receipt hashes verify.
The screen contains 48 distinct recipes per shape; confirmation contains
four rounds of eleven samples, each with sixteen calls. All metadata planes
match bitwise; the maximum confirmed conditioned dot error is 1.40e-4
(bound 5e-3), and the affine reader's maximum is 4.01e-8.

### Warm times, common F32 endpoints

Microseconds per call. FQ/SF include GPU input gather/cast and output
cast/scatter, as well as their selected directory/producer/reducer. Resident
SF excludes prepass. Expert IDs/order/offsets are ready GPU inputs for all
arms. No llama.cpp MMVQ/DMMV reference was executed in this gate.

| Case | Old pair | FP32 affine | Selected FQ | Resident SF |
| --- | ---: | ---: | ---: | ---: |
| q4-up | 18.419 | 17.385 | 18.025 | 17.602 |
| q5-down | 22.551 | 21.299 | 17.748 | 25.946 |
| q4-dense-matched | 18.262 | 17.099 | 13.001 | 12.907 |
| q4-dense-wide | 70.435 | 67.284 | 22.514 | 25.426 |
| q4-dense-long-k | 214.244 | 211.841 | 60.860 | 100.910 |

The affine reader gains 1.12–6.37% over the old pair reader. Its q4-up lead
over the full FQ endpoint pipeline is only 3.55%; it is 20.01% slower on
q5-down, 31.52% slower on the matched dense control, and 2.99x/3.48x the FQ
time on the larger dense controls. Small-input launch overhead therefore
cannot explain the entire SIMT gap.

The selected pair/affine recipes respectively are C16/W8/S1 for q4-up,
C16/W2/S1 and C32/W4/S1 for q5-down, C16/W8/S1 for both smaller dense cases,
and C16/W8/S8 for long-K. Split1/2/4/8 were all screened: the long-K S8
winner is not evidence that Split-K was omitted from SIMT selection.

### Resident core versus SF expansion

| Case | FQ core | SF core | Warm prepass alone | SF endpoint + prepass |
| --- | ---: | ---: | ---: | ---: |
| q4-up | 14.535 | 14.112 | 55.395 | 72.819 |
| q5-down | 14.265 | 22.460 | 55.402 | 82.101 |
| q4-dense-matched | 9.620 | 9.457 | 3.880 | 16.575 |
| q4-dense-wide | 18.997 | 21.943 | 11.173 | 36.268 |
| q4-dense-long-k | 56.500 | 94.888 | 28.417 | 130.609 |

Core times exclude endpoint adapters but include native scheduling and
reduction. For the equal-weight Q4 calls, the grouped/dense FQ core gap is
14.535 versus 9.620 us; this is an actual remaining scheduling/configuration
difference, not a weight-size mismatch. The fixtures have equal weight
counts, not identical source bytes, and their selected parents/splits differ.
The affine GEMVs are much closer: 17.385 versus 17.099 us.

Each grouped prepass expands **all 256 experts**, even though compute uses
eight: it reads 16 MiB of units and writes 32 MiB of scale/zero planes.
The measured ~55 us is not a negligible per-call cost. Immutable-weight
reuse amortizes it; `sf_with_prepass` explicitly measures recomputation,
not production cache behavior. First-event intervals range from 0.58 to
2.66 ms and can include first-launch initialization; they are not the warm
kernel latency. Independently profiled prepass kernels are ~59 us for the
two grouped cases. Do not multiply cold initialization by every token.

### ACU: SIMT producer, not bank conflicts

The following counters come from individual kernels under cache-cleared,
27-pass ACU replay. They are not warm graph endpoint times.

For q4-dense-wide:

| Main-kernel metric | Affine SIMT | FQ producer |
| --- | ---: | ---: |
| Duration (us) | 69.078 | 25.730 |
| DRAM throughput (% peak, `ppu__dram_throughput...elapsed`) | 12.33 | 32.31 |
| DRAM bytes | 23,661,888 | 23,753,856 |
| Executed instructions (`pu__inst_executed.sum`) | 36,368,896 | 5,717,760 |
| KVD bytes (`derived__kvd_bytes_total`) | 335,642,624 | 40,435,712 |
| Achieved occupancy (% active) | 86.86 | 21.32 |
| Shared bank conflicts | 0 | 221,184 |

The larger long-K control agrees: affine/FQ producer DRAM utilization is
12.71%/46.80%, while actual DRAM bytes are 74.03/74.37 MB. The affine
producer occupies 93.0% of active warp capacity yet executes 115.12 million
instructions versus FQ's 17.74 million. These counters argue against HBM
saturation, shared bank conflicts, or low occupancy as the primary large-
shape explanation. They support investigating scalar unpack/index arithmetic
and on-chip load/reuse cost. They do not by themselves prove one source
instruction is causal. KVD byte counts are not DRAM byte counts.

The long-K affine report separates a 211.508 us producer from a 3.222 us
reducer. The wide control's selected SIMT recipe is S1, with no reducer at
all. Producer work, not just Split-K postprocessing, must therefore be
addressed to close the large-shape gap.

The affine experiment does reduce q4-dense-wide global-memory instruction
count from 1,970,688 to 987,648 and KVD bytes from 503.41 to 335.64 MB, but
total executed instructions fall only 6.61%. A successful vector-load change
alone has not removed the much larger overall SIMT instruction workload.

The q5-down SF/FQ comparison also differs in scheduling: SF still launches
a rectangular `(1,32,256)` grid (8,192 CTAs), while FQ uses a compact 256-CTA
directory. Its current SF choice is `DEVICE_BOUNDS`, not a measured global
winner. SF's slower time here cannot be attributed solely to metadata format.

### Decision and remaining work

- Keep automatic decode on FQ. SIMT GEMV is provisionally accepted for the
  milestone and further optimization is parked; preserve its source/results.
- Track the equal-weight grouped/dense FQ gap and rectangular SF fallback.
  Per-call SF decisions must include GPU expansion and matching adapters.
- The same-device llama.cpp arm was not part of this PPU gate. Preserve that
  evidence boundary without making additional SIMT comparisons a blocker.
