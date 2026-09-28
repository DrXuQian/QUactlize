# H800 GPTQ comparison

Compare Quactlize's existing **GPTQ symmetric INT4** semantics against
Machete and Marlin, with independent CPU numerical checks. Do not adapt GGUF
groups or generate a restricted GGUF subset to fit the reference kernels.
The previous common-group64 GGUF experiment is superseded and its bundles
and results are explicitly rejected by this runner.

## Independent Machete gate

For **independent Machete acceptance**, pass `--family machete-streamk` to
both `cohort.py` and `reconfirm.py`. The compiled descriptors select only the
Machete reader/collective with Stream-K. Neither complete Marlin nor a
prepacked-Marlin hybrid can become the winner of this gate. `--family all`
retains the existing mixed-provider diagnostic; its result is a different
claim.

After a complete six-point screen and three-round confirmation, validate
the separate result with:

```bash
python3 tests/ppu17/reference/machete_parity.py \
  --screen /workspace/quactlize-ppu17-20260926/gptq-ab/results/cohort-machete-only-v1 \
  --confirmation /workspace/quactlize-ppu17-20260926/gptq-ab/results/confirm-machete-clock-matched-v1 \
  --output /workspace/quactlize-ppu17-20260926/gptq-ab/results/machete-replay
```

The output directory must be fresh. The gate verifies full screen inventory,
same-math controls, per-point code/scale identity, source/binary/device
authority, three independent rounds, matched selected clocks and complete
call scope. It picks one config per arm by median of rounds, not the fastest
round. `PARITY_WITHIN_5PCT` requires both the aggregate and every paired round
to meet5%; `strict_no_slower` is reported separately. Preserve clock-rejected
attempts and bind any retry assembly to its original receipts; never edit raw
results to turn a failed gate green.

## Fixed contract

- Group size128, desc_act=false, zero point8, all16 uint4 codes. The importer
  is the existing `tools/dump_real_weights.py:unpack_gptq_expert`; it supports
  the directly stored zero8 convention, not an inferred AutoGPTQ +1 variant.
- Original FP16 scales [K/128,N], FP16 A/C, FP32 accumulation. Weight decode
  is FP16((q-8)*scale). No requantization, merged groups, artificial GGUF units,
  metadata prepass, additive-zero tensor or activation quantization.
- Only lossless native weight/scale permutations differ. Quactlize uses
  signed int4 K-major storage for the Hopper ScaleOnly collective; Machete
  and Marlin use their native u4b8 packing. Original scale bits are checked.
- The default fixture is **synthetic GPTQ-format data**, with independent,
  arbitrary FP16 group scales. It is not a model checkpoint. An exact-shape
  NPZ of original qweight/qzeros/scales/group_size/desc_act can instead be
  supplied with `compare.py --weights FILE`. It is never cropped or re-quantized.
- Dense first: M1/8/128, N512/K2048 and N4096/K4096. Grouped GPTQ and other
  dtypes/group sizes require their own admission; this experiment does not
  claim those ports are already complete.

Quactlize's new `quactlize/backends/ppu17/gptq.cuh` ports the existing
FinegrainedScaleOnly contract to PPU CUTLASS4.3.0, with an isolated host
benchmark entry in `quactlize.cu`. The4.3 builder accepts the ordinary
cooperative schedule and detects mixed input from the element tuple, so the
old suffixed-schedule workaround is removed. This changes neither arithmetic
nor dependency files. No actlize, GGUF ABI, canonical
packing, production selector or PPU1.0/1.5 source is changed.

This is a prepared-kernel comparison, not a vLLM Python benchmark. Reference
device headers are unmodified vLLM
`11ba93f3646d4c5476c3b3fd56835589701f0fb1`, with their own CUTLASS4.4.2
`da5e086dab31d63815acafdac9a9c5893b1c69e2`.
Quactlize uses PPU CUTLASS4.3.0
`0eb6a3b218977892e2b6cd30d59880c440e54fd6`, content-locked to the supplied
archive by `quactlize/backends/ppu17/dependency.py`. Reference dependencies
are unchanged. Previous3.6 timing is not evidence for this new dependency.
The include graphs remain separate. Marlin scheduling reuses only actlize's
integer-only `MarlinStripeSchedulerCore` at
`021c69300864e03926108867d318572871a44d35`; its PPU device driver is not imported.
Initialize that submodule before building. No dependency sources are edited.

`build.py --reader prepacked` selects an additional experimental GPTQ provider:
the pinned Machete MMA-aligned packing/reader/collective is reused with **PPU
CUTLASS4.3**, paired with Stream-K or the adapted Marlin scheduler. This is
upstream implementation reuse, not a new algorithm. It preserves logical codes,
group128 and scale bits, but has different private bytes from the conventional
signed-int4 reader. Bundles are not interchangeable. Neither layout changes
canonical GGUF/K-pack storage or the existing GGUF-only q17 public ABI.

`--reader aligned` additionally builds `quactlize_marlin`: the complete
`gptq_marlin.cuh` backend provider, not just Marlin's scheduler around WGMMA.
It instantiates unmodified pinned upstream Marlin device templates on the
isolated PPU CUTLASS4.3 include graph and reuses Quactlize's original host-only
classic packing map. The reference remains a separately built DSO with its
own wrapper and CPU packing oracle. Report this as implementation reuse,
never as a new algorithm beating itself. Both arms retain FP32 reduction.
Scale permutation, workspace initialization and first-use exclusion match.
Serial same-stream workspace reuse is required; concurrent use is unsupported.

## Build and host checks

```bash
python3 tests/ppu17/reference/build.py \
  --reader aligned \
  --vllm /path/to/pinned-vllm \
  --cutlass /path/to/pinned-reference-cutlass \
  --ppu-cutlass /path/to/pinned-ppu-cutlass \
  --compiler /usr/local/cuda-12.8/bin/nvcc \
  --output /workspace/q17-gptq-build

python3 tests/ppu17/reference/test_fixture.py \
  --references /workspace/q17-gptq-build \
  --vllm /path/to/pinned-vllm

python3 tests/ppu17/reference/check_marlin_codegen.py \
  --bundle /workspace/q17-gptq-build \
  --cuobjdump /usr/local/cuda-12.8/bin/cuobjdump
```

Use fresh build/result directories. The host checks preserve original
quantization and reject unsupported zero conventions, group changes,
act-order, precision changes and old comparison receipts. Native signed-int4
packing is independently inverted; Marlin packing/scales are compared with
the pinned upstream CPU implementation. NumPy is needed for the importer;
Torch is needed for upstream CPU checks and device runs, not C++ compilation.

## Physical H800 measurement

```bash
python3 tests/ppu17/reference/cohort.py \
  --references /workspace/q17-gptq-build \
  --backend-source quactlize/backends/ppu17 \
  --expected-uuid GPU-REPLACE-WITH-VERIFIED-UUID \
  --device 0 --output /workspace/q17-gptq-screen

python3 tests/ppu17/reference/report.py \
  --cohort /workspace/q17-gptq-screen \
  --output /workspace/q17-gptq-report
```

The conventional six-point screen contains24 configs/point: nine pinned
Machete heuristic schedules, three native Marlin geometries, six conventional
Quactlize Stream-K tiles, two Marlin-scheduled tiles and four stage4/8 variants.
The optional prepacked screen contains31 configs/point: all nine upstream
Machete geometries plus three Marlin-scheduler controls and three self-reset
variants, four bounded stage4/8 challenges, and twelve reference configurations.
`aligned` adds the complete Marlin provider's three geometries, giving34 configs/point and204
across the six-point cohort. Its report selects the measured best among BOTH
Quactlize providers; this is not a production policy or an unmeasured cutoff.
Self-reset reuses the same
Marlin device body: initialize locks once in preparation, retain kernel-side
FP32 fixup/lock reset, and update host params after serial workspace sharing.
It is not enabled for Stream-K or concurrently shared workspaces.
Stage4/8 challenges use the same128x16x64 tile and unchanged packing/math.
Their controls are config0 (Stream-K) or config6 (self-reset Marlin), and
config6 itself retains config3 as a control. All controls must be present.
A screen is not optimality proof. Reconfirmation retains the two measured
leaders per arm plus any required same-math control; it must use a GPTQ screen,
never the retired GGUF-subset timings. Stage-count changes must preserve exact
output bits with their tile/scheduler control.
`reconfirm.py --points 0 1 3 4` explicitly confirms only the four decode
points; it may reuse their admitted receipts from an otherwise incomplete
cohort. Its completion flag applies to the printed requested subset, not the
omitted prefill points. Fresh-process failures are preserved while the other
requested points continue. The full default still requires a complete screen.

Device-free scheduler and layout checks can be compiled with CUDA12.8:

```bash
nvcc -std=c++17 -O2 --expt-relaxed-constexpr -arch=sm_90a \
  -I. -I/path/to/pinned-ppu-cutlass/include \
  tests/ppu17/marlin_scheduler.cu -o /workspace/q17-marlin-host
/workspace/q17-marlin-host

nvcc -std=c++17 -O2 --expt-relaxed-constexpr -arch=sm_90a \
  -I. -I/path/to/pinned-ppu-cutlass/include \
  -Itests/ppu17/reference/shim -I/path/to/pinned-vllm/csrc \
  tests/ppu17/prepacked_layout.cu -o /workspace/q17-prepacked-host
/workspace/q17-prepacked-host
```

The first covers256 work/peer cases and seven invalid inputs. The second checks
8192 coordinates against an independent scalar WGMMA lane/register formula,
256 aligned16-byte per-thread vectors and a stale-layout negative. These
executables do not launch a device kernel or prove PPU1.7 numerical admission.

A complete CPU FP64 dot of FP16 reconstructed weights checks every output,
guards and nonfinite values before timing. Allocation, packing, preparation,
JIT and graph upload are excluded; required workspace resets and full
reductions remain included. Sequential plans share workspace and each arm
rotates weights across a complete ring larger than2.25x queried L2.
Every config has >=250ms device warmup and11 event samples at stable clocks.
The process guard refuses pre-existing GPU workloads and stops only its own
child if another workload appears. No driver settings or other jobs change.

The goal remains to beat both measured references per shape, not merely an
old Quactlize implementation. No performance or native PPU1.7 admission is
implied by host compilation; physical H800 results are a separate tier.
