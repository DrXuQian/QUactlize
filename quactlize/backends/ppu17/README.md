# PPU1.7 backend

This opt-in backend uses the PPU CUTLASS3.6 SM90 programming model. It does
not replace the PPU1.0/1.5 actlize backend, its build, ABI, or policy tables.
No change to the existing offline K-pack bytes is required.

## Contracts

- Q2_K/Q3_K/Q4_K/Q5_K/Q6_K and Q8_0; dense and compact grouped operations.
- Fully quantized (FQ) or separately expanded scale-first (SF) metadata.
- Direct FP16 or BF16 dequantization, FP32 accumulation; independently typed
  FP16/BF16/FP32 input/output. Finite FP32 values are not clipped into FP16.
- Canonical low/high plane maps, including Q5's high-plane permutation and
  Q3/Q6's paired metadata units, remain owned by the existing format headers.
- Grouped offsets stay on device. Empty experts and M/N tails are masked.
- Explicit SIMT/WGMMA/TMA-WGMMA configuration inventory. No ZW810 timing table is reused
  as a PPU1.7 heuristic. The structural inventory is not a claim of optimality.
- Prepared execution submits one target kernel, with no allocation, copy,
  prepass, reduction, or host synchronization. SF expansion is a separate API
  call. The one-shot convenience call performs host preparation, not GPU setup.

The WGMMA producer decodes B to swizzled shared memory. It is a quantized GEMM,
not a full-matrix dequantization followed by GEMM. Synchronous staging supplies
a baseline. A second producer uses two raw TMA stages for A and the canonical
B low/high planes; a stage is refilled only after its GMMA reads finish. TMA
currently requires A storage to match the FP16/BF16 compute type and 16-byte
global row alignment; other typed inputs use the synchronous/SIMT inventory.
Descriptors are built by `q17_prepare_v1`, outside graph capture. Stock SM90
mixed-input collectives require scale groups that do not match GGUF's 16/32
groups and cannot consume the canonical planes without an adapted producer.

Only C/POD types cross the backend boundary. A translation unit must never
include both the old actlize and this CUTLASS tree. Readers use portable format
traits and plane maps, not headers with optional legacy AIU re-exports. Host
contract tests check metadata decoding against the existing packed-unit owner.
Existing source hashes and cached dispatcher identities remain unchanged.

## Validation boundary

CUDA SM90a assembly checks prove Hopper source/codegen compatibility only.
They do not prove PPU1.7 native compilation, simulator correctness, physical
correctness, or speed. Native admission must check compiler-provided SM90A
features; defining `ACOMPUTE_VERSION=10700` on a PPU1.5 compiler is insufficient.

The backend is experimental until those gates pass. It is not enabled by the
existing dispatcher or exported by the shipping six-library bundle.

## Build and local validation

Use an external build directory, not the old `build.sh`/actlize include graph:

```bash
python3 quactlize/backends/ppu17/build.py \
  --cutlass /workspace/quactlize-ppu17-20260926/cutlass \
  --compiler /usr/local/cuda-12.8/bin/nvcc \
  --target cuda-source \
  --output /workspace/quactlize-ppu17-20260926/build --jobs 4
```

The output is a content-keyed directory with a manifest and
`libquactlize_ppu17.so`. `--target native` requires a genuine SM90a-capable
PPU compiler; the target probe runs before the six format compilations.
Compiler temporaries stay under the selected output directory.

`python3 quactlize/backends/ppu17/check.py --bundle DIR --output DIR` compiles/runs CPU contracts and the
single-launch application, verifies live device bodies, and rejects old
architecture/native-provenance negatives. It does not launch a GPU kernel.

Python imports are explicit and do not load actlize:

```python
from quactlize.backends.ppu17 import Backend, Call

backend = Backend(bundle)  # rejects a CUDA-source bundle by default
problem = backend.problem(qtype=12, operation=1, route=0, compute_type=1,
                          input_type=1, output_type=2,
                          m=75, n=256, k=512, experts=4, max_rows=65)
configs = backend.configs(problem)  # structural inventory, not a speed ranking
# Bind validated device pointers, strides, stream, device and an explicit config
# in Call. Keep device storage alive through stream completion.
call = Call(problem=problem, config=configs[chosen_index],
            a=a_ptr, low=low_ptr, high=None, units=units_ptr,
            offsets=offsets_ptr, output=output_ptr, stream=stream_ptr,
            a_stride=512, output_stride=256, device=device_ordinal)
with backend.prepare(call) as plan:
    plan.execute()
```

No existing exact or bucketed policy is edited. PPU1.7 timing, including SF's
separate expansion cost, must be measured before installing automatic routing.

## Single-launch application

The local validation builds `single-launch` in its output directory. These are
application commands for a compatible SM90 device/simulator, **not a simulator
CLI or PPU1.7 performance claim**:

```bash
# One quantized GEMM: Q4, BF16 A/B, FP32 C, ragged/empty experts, two TMA stages.
/workspace/quactlize-ppu17-20260926/validation-final/single-launch \
  --qtype 12 --compute 1 --input 1 --output 2 --config 5 --grouped

# One scale-expansion kernel; never mixed into the GEMM timing above.
/workspace/quactlize-ppu17-20260926/validation-final/single-launch \
  --qtype 12 --compute 1 --input 1 --output 2 --config 5 --grouped --expand
```

Qtypes are `8,10,11,12,13,14`; types are `0=FP16,1=BF16,2=FP32` (compute only
0/1). Configs 0/1 are SIMT with 8/16 N lanes, 2–4 are synchronous WGMMA,
5 is two-stage TMA WGMMA. Add `--sf` for pre-expanded metadata: the application
prepares its scale inputs on CPU, so each process still submits only one
target kernel. `--cpu-only` never initializes a GPU and is not a device pass.
Use `--m 1` through `--m 8` for dense decode and `--rows 0,1,1,0` through
`--rows 0,8,8,0` for grouped decode; `--rows` lists each expert's row count.
The default `--grouped` fixture uses rows `[0,1,9,65]` to cross tile boundaries.

This backend returns compact grouped results. MoE routing, gated activations,
Split-K, persistent grouped scheduling, full-dequant/library alternatives and
automatic production policy are not silently supplied by the new API. They
remain separate integration/admission work; the old implementations are intact.

## Physical Hopper checks and timing

`check_device.py` runs fresh one-launch numerical processes on a specified GPU
UUID. `--suite coverage` has 300 cases; `--suite extended` has 1,020, adding
M1–8, tile boundaries, N tails, K1024 phase wrap, ragged/empty experts and typed
I/O. A competing GPU job stops the campaign without killing that job. Resume
with `--resume-from OLD/summary.json` into a **fresh** output directory: the
application, library, UUID, case inventory and retained log hashes must match.

```bash
python3 quactlize/backends/ppu17/check_device.py \
  --bundle "$BUNDLE" --application "$VALIDATION/single-launch" \
  --device 0 --expected-uuid "$GPU_UUID" --suite extended \
  --output "$RESULTS/numerics"
```

For physical-device timing, `check.py` also builds a separate `benchmark`
application. Do **not** run this application in the one-launch simulator.
Its controller requires no other GPU jobs before starting and verifies the
benchmark's GPU PID before admitting kernel execution. All reported GPU
processes, clocks, temperature and power are sampled during the run. An
observed competing job or missing telemetry invalidates timing and stops only
our child process. The 100-ms process monitor is not a driver reservation:
schedule no other GPU jobs until this runner exits.

```bash
python3 quactlize/backends/ppu17/run_benchmark.py \
  --bundle "$BUNDLE" --application "$VALIDATION/benchmark" \
  --device 0 --expected-uuid "$GPU_UUID" --qtype 12 --compute 1 \
  --cache rotating --suite baseline --output "$RESULTS/hopper-baseline"
```

This bounded baseline uses six configs, FQ/SF, dense M1/M128 and grouped rows
`[0,1,8,65]`, N512/K2048. It checks complete CPU dots and output guards before
timing each weight copy. Prepared descriptors, allocation, transfers, initial
graph upload and three warmup graphs are excluded. Eleven device-event samples
measure graph-averaged kernel latency. Rotating weights cover more than twice
the **queried** device L2 capacity, counting only nonempty experts; warm-cache
timing is labeled separately. SF expansion is **not included**, so these
numbers cannot alone choose FQ vs SF or claim end-to-end speed. There is no
production policy update or PPU1.7 performance admission from this runner.

Keep profiler/sanitizer runs separate from timing. Preserve raw logs and the
summary's binary hashes, case identity and telemetry.

```bash
# 25 fresh-process memory, race and synchronization checks.
python3 quactlize/backends/ppu17/check_memory.py \
  --bundle "$BUNDLE" --application "$VALIDATION/single-launch" \
  --sanitizer "$CUDA_ROOT/bin/compute-sanitizer" \
  --device 0 --expected-uuid "$GPU_UUID" --output "$RESULTS/memory"

# 156 same-math baseline/candidate pairs. Compare full output bytes, not
# only checksums or a relaxed tolerance; each arm also checks the CPU oracle.
python3 quactlize/backends/ppu17/check_replay.py \
  --baseline "$BASELINE" --candidate "$BUNDLE" \
  --application "$VALIDATION/single-launch" \
  --device 0 --expected-uuid "$GPU_UUID" --output "$RESULTS/replay"

# Separate forced-cold single-launch profiles; never mix these times into
# the rotating benchmark. Host performance-counter permission is required.
python3 quactlize/backends/ppu17/profile.py \
  --bundle "$BUNDLE" --application "$VALIDATION/single-launch" \
  --ncu "$CUDA_ROOT/bin/ncu" \
  --device 0 --expected-uuid "$GPU_UUID" --output "$RESULTS/profile"
```

All result directories must be fresh. `single-launch --raw-output FILE`
exclusively creates a full-byte result after numerical validation, including
output guards; this option is for same-math replay, not timing. The memory
suite explicitly enables TMA bulk-copy and warpgroup-MMA checking.

Current H800 evidence and outstanding native gates are in
`docs/ppu17-integration.md`. Hardware-counter permission failures leave
profiling unadmitted; they do not justify changing host driver settings or
inferring missing counters from elapsed time.
