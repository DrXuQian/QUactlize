# H800 GPTQ comparison

Compare Quactlize's existing **GPTQ symmetric INT4** semantics against
Machete and Marlin, with independent CPU numerical checks. Do not adapt GGUF
groups or generate a restricted GGUF subset to fit the reference kernels.
The previous common-group64 GGUF experiment is superseded and its bundles
and results are explicitly rejected by this runner.

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
FinegrainedScaleOnly contract to PPU CUTLASS3.6.0, with an isolated host
benchmark entry in `quactlize.cu`. The tag adapter in
`mixed_input_policy.cuh` bridges that fork's suffixed builder policy to the
base schedule required by its existing mixed-input collective. It changes
neither arithmetic nor the dependency files. No actlize, GGUF ABI, canonical
packing, production selector or PPU1.0/1.5 source is changed.

This is a prepared-kernel comparison, not a vLLM Python benchmark. Reference
device headers are unmodified vLLM
`11ba93f3646d4c5476c3b3fd56835589701f0fb1`, with their own CUTLASS4.4.2
`da5e086dab31d63815acafdac9a9c5893b1c69e2`.
Quactlize uses PPU CUTLASS3.6.0
`023e82d03e80b4d5982f664925e15499033df314`.
The include graphs remain separate.

## Build and host checks

```bash
python3 tests/ppu17/reference/build.py \
  --vllm /path/to/pinned-vllm \
  --cutlass /path/to/pinned-reference-cutlass \
  --ppu-cutlass /path/to/pinned-ppu-cutlass \
  --compiler /usr/local/cuda-12.8/bin/nvcc \
  --output /workspace/q17-gptq-build

python3 tests/ppu17/reference/test_fixture.py \
  --references /workspace/q17-gptq-build \
  --vllm /path/to/pinned-vllm
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

The six points contain18 configs each: nine pinned Machete heuristic
schedules, three native Marlin geometries and six initial Quactlize Hopper
GPTQ configs. A screen is not optimality proof. Reconfirmation retains the
two measured leaders per arm; it must use a GPTQ screen, never the retired
GGUF-subset timings.

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
