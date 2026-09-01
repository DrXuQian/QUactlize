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
| `tools/pack_gguf.py` | GGUF-to-K-pack artifact conversion |
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
python3 -m pip install --no-build-isolation -e .
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

`build.sh` configures this repository directly and prints the exact output
binary. `PPU_BUILD_DIR` selects an out-of-tree build directory;
`PPU_BUILD_RESUME=1` resumes only when the recorded source identity is still
exact. `PPU_DEFS` carries the documented
`PPU_PACKED_SCALE`/`PPU_PACKED_FORMAT` library identity and deliberate
development A/B definitions. The build fails if a requested definition does
not reach that target's HGCC command.

## Offline conversion

The format-unified producer is explicit and records the selected arrangement
in `manifest.json`. Run this command from a source checkout; installing a
dedicated packer console entry is still part of the selective main rebuild.

```bash
QUACTLIZE_PPU_LIB_FMT0=/path/to/q4/libquactlize_ppu.so \
QUACTLIZE_PPU_LIB_FMT1=/path/to/q5/libquactlize_ppu.so \
QUACTLIZE_PPU_LIB_FMT2=/path/to/q2/libquactlize_ppu.so \
QUACTLIZE_PPU_LIB_FMT3=/path/to/q3/libquactlize_ppu.so \
QUACTLIZE_PPU_LIB_FMT4=/path/to/q6/libquactlize_ppu.so \
python3 tools/pack_gguf.py MODEL.gguf OUT_DIR
```

The compile-time format IDs are `0=Q4_K`, `1=Q5_K`, `2=Q2_K`,
`3=Q3_K`, and `4=Q6_K`. Build and install one device library for every
format present in a model. The Python extension resolves the matching
`QUACTLIZE_PPU_LIB_FMT*` handle; it never sends a K-pack artifact to a library
built for another format.

Set the default handle as well when running Q4 prefill. The packed FMT0 library
derives its metadata workspace; the non-packed library consumes that workspace
through the persistent ScaleFirst v2 reader:

```bash
export QUACTLIZE_PPU_LIB=/path/to/q4-scalefirst/libquactlize_ppu.so
export QUACTLIZE_PPU_LIB_FMT0=/path/to/q4-packed/libquactlize_ppu.so
```

Use `--dry-run` to inspect tensor eligibility without writing an artifact.
The current placement operation is provided by the format-selected PPU device
library, so a real conversion run requires the device runtime.

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
device result.

The compile-only Q4 delivery gate is an example. It instantiates the real
AIU-plain writer and UniversalCopy reader, then checks their PPU ISA without
linking or launching a host executable:

```bash
PPU_SDK=/path/to/ppu-sdk-2.1.1 \
bash dev/fold_derivation/run_l247_q4_n16k64_delivery_codegen.sh
```

The box is required for:

- raw-bit and numeric correctness on PPU;
- asynchronous-copy, barrier, scoreboard, and cache behavior;
- resource counters and ACU reports;
- latency and performance admission.

Representative real-shape entry points are:

```bash
# Q4 decode and prefill K-pack boards
INTERNAL_SWEEP_SPEC=/workspace/path/to/inventory.json \
OUT=/workspace/q4-kpack-decode JOBS=16 \
  bash tools/run_fq_q4k_kpack4_decode_real_shapes_box.sh
INTERNAL_SWEEP_SPEC=/workspace/path/to/inventory.json \
OUT=/workspace/q4-kpack-prefill JOBS=16 \
  bash tools/run_fq_q4k_kpack4_prefill_real_shapes_box.sh

# Q2/Q3/Q5/Q6 dense+grouped and Q4 grouped K-pack/Xplane development A/B
OUT=/workspace/kquant-kpack-ab JOBS=16 \
  bash tools/run_fq_kquant_kpack_perf_box.sh
```

Every device result is bound to the source SHA, generated manifest, binary,
shape, tactic, and artifact descriptor. A timing result from a numerically
incorrect row is invalid.

## Productization

`develop` contains historical baselines and diagnostic evidence. Product
changes are selectively rebuilt on the current `main`; `develop` is not
merged wholesale. The main branch admits only PPU product code, the canonical
K-pack descriptors, necessary tests, and concise operator-facing tooling.
