# Quactlize

Canonical K-pack weight-only GEMM and GEMV for PPU0010 / ZW810, built on
[actlize](third_party/actlize). Dense and grouped operators share the same
weight artifacts. The runtime includes measured configuration selection,
on-demand compilation, explicit FP16/BF16 compute, FP32 decode endpoints,
and MoE preparation and postprocessing.

This source distribution replaces the old build interface. It does **not**
provide the legacy `libquactlize_ppu_fmt*.so` libraries or the Torch extension.
Existing callers that require those libraries must keep using the `develop`
compatibility build until migrated. The PPU1.7 port also remains on `develop`.

## Formats and ownership

| GGUF type | Low / high bits | Codes per 16-bit word, low / high |
|---|---|---|
| Q2_K | 2 / 0 | 8 / — |
| Q3_K | 2 / 1 | 8 / 16 |
| Q4_K | 4 / 0 | 4 / — |
| Q5_K | 4 / 1 | 4 / 16 |
| Q6_K | 4 / 2 | 4 / 8 |
| Q8_0 | 8 / 0 | 2 / — |

These are versioned byte maps, not ordinary row-major bit packing. Use the
arrangement returned by [packing/api.h](quactlize/packing/api.h); do not
construct a descriptor from the bit widths alone. Layout, mapping, metadata,
capacity, and shape checks reject incompatible inputs. There is no Xplane
reader or fallback.

Low, high, and metadata planes each concatenate experts in the same order.
An absent high plane has size zero and accepts a null pointer. Artifacts do
not depend on a GEMM tile choice. Query allocation sizes before packing;
individual operators can have stricter shape requirements than the packer.

## Build from source

Requires Linux x86-64, Python 3.10+, a C++17 compiler, binutils, and the PPU
SDK. The current compiler baseline is SDK 2.1.1-a5c56e. `--sdk` is the root
containing `bin/hgcc`, `bin/hgobjdump`, `include`, and runtime libraries under
`lib`, **not** its `CUDA_SDK` child. No device is used during this build.

```bash
git submodule update --init --recursive
source /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK/envsetup.sh
python3 tools/build.py \
  --sdk /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  --output /workspace/quactlize-product-build \
  --jobs 192
```

The final `BUILD status=COMPLETE runtime=...` line identifies the runtime
directory. `build.json` records its relative path, source/SDK identity, and
completed stages. To continue an interrupted build, repeat the command with
`--resume`. Changed sources or SDK files require a new output directory.
No downloads, model files, external caller binaries, or Git LFS are involved.

The output contains five libraries:

| Library | Public entry points |
|---|---|
| `libquactlize_kpack_dispatch.so` | [dispatch/api.h](quactlize/dispatch/api.h): selection, prepared handles, JIT, MoE chains |
| `libquactlize_ppu_execution.so` | [execution/api.h](quactlize/execution/api.h): SIMT, metadata preparation, indexed execution |
| `pack/libquactlize_ppu_pack.so` | [packing/api.h](quactlize/packing/api.h): GGUF device input → canonical planes |
| `libquactlize_ppu_prefill.so` | [prefill/api.h](quactlize/prefill/api.h), [dequant/api.h](quactlize/dequant/api.h): per-call expansion and external BF16 composition |
| `libquactlize_ppu_gate_up.so` | [fusion/gate_up.h](quactlize/fusion/gate_up.h): paired gate/up packing and execution |

Keep the matching source checkout, submodule, SDK, and a writable JIT cache
available to the dispatcher. This is a source-based JIT distribution, not a
self-contained Python wheel. Verify a built directory without executing it:

```bash
python3 tools/verify_kpack_dispatch.py /path/to/runtime --sdk /path/to/PPU_SDK
```

Build and manifest verification are not device correctness or performance
admission.

## Selection and JIT

Production selection uses measured exact matches and constrained family
fallbacks. A policy miss is reported; the runtime does not invent a tactic.
The tables are specific to ZW810 and do not claim to be optimal for unmeasured
shapes or other devices. Recorded costs include measured reducers and
per-call expansion costs where applicable. Summed component estimates are
not presented as measured end-to-end latency.

Inspect the final small-M selection without a PPU:

```bash
python3 tools/kpack_jit.py plan-smallm \
  --request 12 0 4096 2048 1 1 1 1 0 \
  --request 12 2 512 2048 256 8 1 1 1 \
  --output /workspace/kpack-selection
```

Each request is `qtype mode N K experts topk channels tokens compute`.
Mode `0` is dense and `2` is indexed MoE; compute `0` is FP16 and `1` is BF16.
The plan records the chosen implementation and required TC specializations.
Prewarm those specializations, without tuning or GPU execution:

```bash
python3 tools/kpack_jit.py prewarm \
  --plan /workspace/kpack-selection/plan.json \
  --sdk /path/to/PPU_SDK --cache /workspace/kpack-jit --jobs 16
```

Call query/prepare outside graph capture. Prepared execution consumes
caller-owned device buffers, workspace, and a stream; it must not compile or
wait for CPU readback. Activation storage and compute precision are explicit
ABI fields. BF16 cannot be emulated by first narrowing to FP16. Scale-first
expansion is per call, not an implicit persistent scale cache.

Generic routing and MoE work live in `execution/` and `runtime/`.
`integrations/llama/` contains caller-specific indexed contracts. TP callers
must query each local shard and manage communication themselves; this library
does not provide a collective-communication backend.

## Scan a bounded set of configurations

Supply explicit candidates instead of a Cartesian-product sweep. The example
compares two SIMT recipes and two TC Split-K recipes; it is a starting set,
not a claim that these are the best configurations for every shape.

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/scan_configs.py \
  --sdk /path/to/PPU_SDK --bundle /path/to/runtime \
  --cache /workspace/kpack-jit \
  --candidates examples/q4-decode-configs.json \
  --qtype 12 --n 4096 --k 2048 --m 1 --samples 11 \
  --output /workspace/kpack-scan
```

The scanner needs NumPy and the `gguf` package. Each candidate runs in a
separate process, checks an independent GGUF dot-product oracle, then records
timings. Failures are retained and do not discard other results.

Timing is **warm resident-weight, complete-entry GPU event time**: it includes
Split-K reduction and host launch gaps, excludes compilation and SF metadata
preparation, and does not flush L2. It is not a cold-weight bandwidth test or
a model benchmark. `--experts` selects compact grouped rows; `--compute bf16`
is supported for SIMT and grouped TC. Dense BF16 TC uses the separate typed
decode ABI and is not covered by this scanner. Results never automatically
replace a production policy.

## Validation

On the host:

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest -q
```

Host tests cover byte maps and inverse conversion, descriptor validation,
measured selection decisions, and build orchestration. PPU tests are skipped
unless explicitly requested. On an idle PPU with a source-built runtime:

```bash
CUDA_VISIBLE_DEVICES=0 \
QUACTLIZE_SDK=/path/to/PPU_SDK \
QUACTLIZE_BUNDLE=/path/to/runtime \
QUACTLIZE_JIT_CACHE=/workspace/kpack-jit \
python3 -m pytest --device -q
```

Device tests compare packed bytes exactly and validate numerical TC/SIMT
outputs across six formats, both compute precisions, Split-K, dense FP32
endpoints, and grouped second-tile boundaries. They do not substitute for
full-model accuracy, TP communication, or performance regression tests.

For a source build followed by both suites in one recorded run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/validate.py \
  --sdk /path/to/PPU_SDK --output /workspace/quactlize-validation --jobs 192
```

Return `results/` from that output directory for review; it contains logs,
JUnit reports, source identity, and the runtime manifest, not compiled objects.
`--resume` reuses completed build stages but reruns the tests.

For source changes to grouped kernels, add `--baseline-source /path/to/previous/checkout`
to include 48 same-config comparisons across six formats, FP16/BF16, small and
larger groups, and S1/S2. Both checkouts need initialized submodules. The
comparison compiles 24 distinct modules, checks numerical outputs, then uses
four alternating-order rounds of 11 samples with 64 native repeats per
sample. Warm weights and SF preparation exclusion are explicit in each
result. A regression above 5% requires review; this is not a model TPOT gate.

## Portable GGUF reference

[reference/gguf_kpack.py](reference/gguf_kpack.py) is a standalone PyTorch
reference for Q2_K–Q6_K. It needs no PPU SDK or extension:

```bash
python3 reference/gguf_kpack.py pack MODEL.gguf MODEL.kpack-reference.gguf
python3 reference/gguf_kpack.py verify MODEL.kpack-reference.gguf --source MODEL.gguf
```

`pack` keeps original tensors and adds low/high/metadata companion tensors
with a versioned manifest. `verify` reconstructs the original GGUF blocks and
checks the bytes. This augmented GGUF is a **verification container**, not a
model that unmodified llama.cpp can execute. Repeat `--tensor NAME` to convert
a small subset. The reference deliberately rejects split and big-endian GGUF
files; it is a format specification, not the high-throughput loader.
