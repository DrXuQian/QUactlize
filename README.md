# quactlize

Low-bit weight-only GEMM kernels for the T-Head ZW810 PPU, implemented with
CuTe and the PPU CUTLASS-3 fork in `third_party/actlize`.

The product format is one K-packed family shared by dense, grouped/MoE,
decode, and prefill paths:

| GGUF format | Code planes in the offline artifact |
|---|---|
| Q2_K | low2 K-pack8 |
| Q3_K | low2 K-pack8 + high1 K-pack16 |
| Q4_K | low4 K-pack4 |
| Q5_K | low4 K-pack4 + high1 K-pack16 |
| Q6_K | low4 K-pack4 + high2 K-pack8 |

Each plane is stored in converter-native little-endian 16-bit transport
words. Scale/zero metadata remains in a separate packed-unit channel. An
artifact carries an explicit arrangement descriptor; consumers reject an
unknown or incompatible descriptor instead of guessing a layout.

## Repository layout

| Path | Purpose |
|---|---|
| `quactlize/include/` | PPU collectives, launchers, format policy, and offline-layout contracts |
| `quactlize/csrc/` | Python bindings, preprocessing, and the PPU device-library entry points |
| `quactlize-pack-gguf` | Installed GGUF-to-K-pack artifact converter |
| `tests/` | Correctness tests against independent host or format oracles |
| `benchmarks/` | Device correctness and timing harnesses |
| `tools/` | Real-shape plans, generated tactic sweeps, and result adjudication |
| `third_party/actlize/` | Pinned PPU CUTLASS-3 implementation |
| `dev/` | Develop-only derivations, negative controls, and experiments |

`dev/` is not product source and is deliberately excluded when product code
is rebuilt on `main`.

## PPU SDK

Device compilation is supported on Ubuntu 24.04 x86_64 with PPU SDK 2.1.1.
A PPU device is not required to compile or inspect a device image; executing a
kernel requires a compatible ZW810/ppu001 device and driver.

- SDK archive: <https://pkg.flytiger-eco.com/artifactory/generic-local/CUDA_SDK/v2.1.1/PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz>
- SHA-256: `63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd`

Verify the archive before extracting it:

```bash
printf '%s  %s\n' \
  63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd \
  PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz |
  sha256sum -c -

mkdir -p /path/to/ppu-sdk-2.1.1
tar -xzf PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz \
  -C /path/to/ppu-sdk-2.1.1 --strip-components=1
source /path/to/ppu-sdk-2.1.1/envsetup.sh
```

The published package targets Ubuntu 24.04. A private loader or userspace
runtime used on an older distribution is a local development workaround, not
a supported installation and not a product dependency.

## Building

Initialize the pinned dependencies first:

```bash
git submodule update --init --recursive
```

Install the PyTorch build used by the target environment first. Python and
host-side format code can then be installed without a PPU device:

```bash
python3 -m pip install --no-build-isolation -e '.[packer]'
```

Build one PPU target after sourcing the SDK environment:

```bash
PPU_DEFS='PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=0' \
TARGET=quactlize_ppu JOBS=16 bash build.sh
```

Build the complete runtime set in distinct directories. Five packed libraries
own K-pack placement and fully-quantized dense/grouped compute; a sixth Q4
ScaleFirst library owns the persistent `M>=64` prefill reader:

```bash
PPU_BUILD_DIR=build/ppu-q4-scalefirst \
PPU_DEFS='PPU_PACKED_SCALE=0 QUACTLIZE_DENSE_ONLY=12' \
TARGET=quactlize_ppu JOBS=16 bash build.sh

for fmt in 0 1 2 3 4; do
  PPU_BUILD_DIR="build/ppu-fmt${fmt}" \
  PPU_DEFS="PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=${fmt}" \
  TARGET=quactlize_ppu JOBS=16 bash build.sh
done
```

For an installable set, use the atomic bundle builder instead of copying those
six outputs by hand:

```bash
PPU_SDK=/path/to/ppu-sdk-2.1.1 \
PPU_SDK_ARCHIVE=/path/to/PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz \
JOBS=16 \
  bash tools/build_ppu_runtime_bundle.sh /opt/quactlize/ppu0010

quactlize-verify-ppu-bundle \
  /opt/quactlize/ppu0010 --ppu-sdk /path/to/ppu-sdk-2.1.1
```

The builder accepts only a clean tracked source identity and the admitted SDK
archive digest, cross-checks the installed compiler release against that
archive, builds each role in
an isolated directory, verifies its exports and embedded PPU image, and then
publishes the directory atomically. `manifest.json` binds the source commit,
submodule commits, compiler/SDK identity, exact compile definitions, filename,
size and SHA-256 for all six libraries. A format-selected library also exports
its compiled FMT identity; the runtime rejects a misplaced library before it
exposes any operator entry point.

`JOBS` is the compiler concurrency inside one role. `PPU_BUNDLE_JOBS` is the
number of isolated library roles built concurrently and defaults to one; it is
limited to six. Budget their product against the machine. For example, a
24-core local builder can use `JOBS=4 PPU_BUNDLE_JOBS=6` to compile all six
roles together. Parallel workers never write the shared stage or manifest;
the parent installs them in canonical role order after every build and source
authority check succeeds.

`build.sh` configures this repository directly and prints the exact output
binary. `PPU_BUILD_DIR` selects an out-of-tree build directory;
`PPU_BUILD_RESUME=1` resumes only when the recorded source identity is still
exact. `PPU_DEFS` carries the documented
`PPU_PACKED_SCALE`/`PPU_PACKED_FORMAT` library identity and deliberate
development A/B definitions. The build fails if a requested definition does
not reach that target's HGCC command.

## Offline conversion

The format-unified producer is explicit and writes the named
`quactlize.kquant-kpack.bundle` schema. Schema v3 stores one 128-byte-aligned,
byte-neutral resident region per tensor in a headerless `weights.bin`; the
manifest records its arrangement-v2 descriptor and low/high/units span hashes.
It also binds the bundle to `source={format,size_bytes,sha256}` and binds every record to its source GGUF tensor
ordinal, byte range and raw digest, so replacing a model at
the same path cannot silently reuse stale K-pack bytes. The strict bundle reader
rejects partial, extra or noncanonical contents. Installation provides one
packer entry point. The normal deployment form needs only the bundle root:

```bash
export QUACTLIZE_PPU_BUNDLE=/opt/quactlize/ppu0010
quactlize-pack-gguf MODEL.gguf OUT_DIR
```

`OUT_DIR` is the persistent sidecar. Build it once, retain it beside the model,
and on later loads validate and read it without converting again. Publication
uses a sibling staging directory plus one final rename, and an existing
`OUT_DIR` is never overwritten:

```python
from quactlize.pack_gguf import load_kpack_bundle

bundle = load_kpack_bundle("OUT_DIR", source="MODEL.gguf")
```

The `model` path in the manifest is a diagnostic hint; identical contents at a
different path still validate. A content mismatch is fatal. Validation hashes
the current GGUF and then uploads each saved resident region, but does not redo placement.
The original GGUF is never overwritten, and K-pack bytes are never labelled as
ordinary GGUF `Q*_K` tensors. Calling `load_kpack_bundle` without `source=` only
checks the sidecar itself and is insufficient to authorize a cache hit.

Individual overrides remain available for debugging or a nonstandard install:

```bash
QUACTLIZE_PPU_LIB_FMT0=/path/to/q4/libquactlize_ppu.so \
QUACTLIZE_PPU_LIB_FMT1=/path/to/q5/libquactlize_ppu.so \
QUACTLIZE_PPU_LIB_FMT2=/path/to/q2/libquactlize_ppu.so \
QUACTLIZE_PPU_LIB_FMT3=/path/to/q3/libquactlize_ppu.so \
QUACTLIZE_PPU_LIB_FMT4=/path/to/q6/libquactlize_ppu.so \
quactlize-pack-gguf MODEL.gguf OUT_DIR
```

The compile-time format IDs are `0=Q4_K`, `1=Q5_K`, `2=Q2_K`,
`3=Q3_K`, and `4=Q6_K`. Build and install one device library for every
format present in a model. The Python extension resolves the matching
`QUACTLIZE_PPU_LIB_FMT*` handle; it never sends a K-pack artifact to a library
built for another format.

The bundle supplies the default handle as well when running Q4 prefill. The packed FMT0 library
owns K-pack4 placement and fully-quantized compute. The default non-packed
library derives the hoisted metadata workspace and consumes it through the
persistent ScaleFirst v2 reader:

```bash
export QUACTLIZE_PPU_LIB=/path/to/q4-scalefirst/libquactlize_ppu.so
export QUACTLIZE_PPU_LIB_FMT0=/path/to/q4-packed/libquactlize_ppu.so
```

Use `--dry-run` to inspect tensor eligibility without writing an artifact.
The placement operation is host code exported by the format-selected PPU
library. A real conversion therefore requires that library and its shared
library dependencies to be loadable, but it does not launch a PPU kernel.

For a portable mapping oracle that needs no extension or PPU SDK, use the
standalone PyTorch reference. Its file mode reads a real GGUF and writes a
spec-valid augmented GGUF containing every original tensor plus I8
`low`/`high`/`units` companions and their arrangement-v2 manifest:

```bash
python reference/gguf_kpack.py pack MODEL.gguf MODEL.kpack-reference.gguf
python reference/gguf_kpack.py verify MODEL.kpack-reference.gguf --source MODEL.gguf
```

`--tensor NAME` may be repeated to convert a small subset while porting. The
augmented output is a byte-exact verification container, not a stock llama.cpp
runtime model: a consumer must first implement the manifest and companion
tensor contract. The verifier reconstructs official GGUF blocks from every
K-pack artifact, checks all original tensor hashes, and optionally proves that
all source tensors and metadata were preserved. The scalar implementation is
intended for small fixtures and mapping review, not full-model throughput; it
also deliberately rejects split or big-endian GGUF inputs and metadata arrays
whose empty value has lost its element type.

The supported Python compute surface is `quactlize.routes`:

- `prepare_fully_quantized_dense` and
  `prepare_fully_quantized_grouped` create canonical K-pack artifacts;
- `matmul_kpack_dense` selects the admitted dense implementation without
  changing resident weight bytes: Q4 uses fully-quantized compute for `M<64`
  and persistent ScaleFirst for `M>=64`; the other formats remain fully
  quantized for every M;
- `matmul_fully_quantized_grouped` consumes the exact grouped descriptor;
- `prepare_q4_kpack4_scale_workspace` explicitly hoists Q4 prefill metadata
  outside the hot path.

All compute entries validate the arrangement version, mapping ID, plane widths,
and qtype before calling a device operation.

## Validation and tuning

Local validation covers source policy, host round trips, CuTe layout algebra,
template instantiation, HGCC device compilation, embedded-image inspection,
ABI checks, and planted negative controls. These checks do not establish a
device result; development-only probes and negative controls are intentionally
not part of the installed product.

The production fully-quantized config policy is generated only from an
all-config K-pack measurement result and its authority file. Reproduce or
audit the checked-in table with:

```bash
python3 -B tools/generate_fq_kquant_measured_policy.py self-test
python3 -B tools/generate_fq_kquant_measured_policy.py generate --check \
  --summary "$OUT/results/summary.json" \
  --config-heuristic "$OUT/results/config-heuristic.json" \
  --authority "$OUT/results/result-authority.json" \
  --evidence-root "$OUT" \
  --output quactlize/include/ppu_kquant_measured_policy_data.inc
```

The evidence root must contain every relative path recorded by the authority;
generation hashes the actual files rather than trusting the JSON records. The
table is a pure dense lookup: explicit compiled config, exact measured point,
then the compiled default. It neither profiles nor interpolates at runtime.
Grouped measurements are evidence-checked but are not routed automatically
until the ABI carries the full expert-row distribution. See
`docs/WHEN_TO_TUNE.md` for that boundary and the evidence contract.

The box is required for:

- raw-bit and numeric correctness on PPU;
- asynchronous-copy, barrier, scoreboard, and cache behavior;
- resource counters and ACU reports;
- latency and performance admission.

### Prebuilt bundle device gate

The release correctness gate consumes an already-built six-library bundle. It
does not configure CMake, compile, or link anything on the box:

```bash
CUDA_VISIBLE_DEVICES=0 \
  python3 tools/run_prebuilt_ppu_box_gate.py /path/to/bundle \
    --ppu-sdk /path/to/ppu-sdk-2.1.1 \
    --q4-correctness-repeats 8192 \
    --output /workspace/quactlize-prebuilt-gate-result
```

The positional argument is the bundle directory containing `manifest.json`
and the six manifest-owned shared libraries. `--ppu-sdk` names the admitted SDK
root containing `hgobjdump` and `lib/libhggc_wrapper.so`. `--output` must name a
new directory whose parent already exists; the runner never overwrites prior
evidence. `CUDA_VISIBLE_DEVICES` must contain exactly one numeric device
ordinal; `0` above is an example, not a device-name or multi-device selector.
The box environment also needs NumPy and the official
`gguf==0.19.0` oracle required by the runner.

`--q4-correctness-repeats` repeats the exact fmt0 dense and grouped explicit
launches and requires one stable raw-bit output hash. Its default is `1` so
local runner tests remain fast; use `8192` for normal device admission or
`32768` for the stronger timing-sensitive closure.

Before launching a kernel, the runner verifies the manifest, every library
digest and embedded PPU image, the source/submodule authority, all six format
identities, and the selected-config ABI. It then checks host prepare/recover
and device dense/grouped correctness for Q2_K, Q3_K, Q4_K, Q5_K and Q6_K,
including an empty expert, null and explicit config selection, a wrong-mapping
negative, and a zeroed packed-unit fault. Successful evidence records
`device_library_builds=0` and `host_compilations=0`. This is a correctness and
ABI admission gate; it does not measure latency or replace a performance/ACU
board.

Representative real-shape entry points are:

```bash
# Q4 decode and prefill K-pack boards
INTERNAL_SWEEP_SPEC=/workspace/path/to/inventory.json \
OUT=/workspace/q4-kpack-decode JOBS=16 \
  bash tools/run_fq_q4k_kpack4_decode_real_shapes_box.sh
INTERNAL_SWEEP_SPEC=/workspace/path/to/inventory.json \
OUT=/workspace/q4-kpack-prefill JOBS=16 \
  bash tools/run_fq_q4k_kpack4_prefill_real_shapes_box.sh

```

Every device result is bound to the source SHA, generated manifest, binary,
shape, tactic, and artifact descriptor. A timing result from a numerically
incorrect row is invalid.

## Productization

`develop` contains historical baselines and diagnostic evidence. Product
changes are selectively rebuilt on the current `main`; `develop` is not
merged wholesale. The main branch admits only PPU product code, the canonical
K-pack descriptors, necessary tests, and concise operator-facing tooling.
