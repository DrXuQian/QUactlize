# Quactlize

Weight-only quantized GEMM/GEMV for T-Head PPU. Use **develop** for current
integration, builds and tests; main is the separately curated product snapshot.
PPU1.0/1.5 use pinned actlize. The PPU1.7 backend is isolated under
[quactlize/backends/ppu17](quactlize/backends/ppu17/README.md).

## Build and test

Checkout the exact `third_party/actlize` gitlink before building. CI owns the
authenticated checkout; the public-library builder never fetches dependencies.

```bash
git switch develop
git pull --ff-only origin develop
git submodule update --init third_party/actlize

# Output directory must be new. PPU_SDK may also be supplied with --sdk.
PPU_SDK=/usr/local/PPU_SDK bash build.sh --library \
  --output /workspace/quactlize-package-RUN_ID --jobs 8

# Inventory / host-only checks.
bash run_tests.sh --suite all --list
PPU_SDK=/usr/local/PPU_SDK bash run_tests.sh --suite host

# Real PPU numerical tests against the complete built package.
CUDA_VISIBLE_DEVICES=0 PPU_SDK=/usr/local/PPU_SDK \
  bash run_tests.sh --suite accuracy --package /workspace/quactlize-package-RUN_ID
```

The package contains `libquactlize_ppu.so`, exported `include/`, a private
`quactlize-runtime/` closure (including JIT sources) and a hash manifest.
Install the whole package, not just the public .so. ncp_flash_lib builds and
installs it; selection, packing and execution remain in Quactlize.

See [CI.md](docs/CI.md) for dependencies, host/build/accuracy coverage, logs and
JUnit. `--suite all` runs the three product suites. Tooling, historical Torch
tests, TP/model benchmarks and PPU1.7 tests have separate prerequisites and are
not implied by product CI passing.

Legacy CMake/Torch targets remain available through explicit `TARGET=...`
build.sh invocations and setup.py. They are not the default integration API.
Do not use historical six-library/prebuilt commands as a fresh installation.

## Product boundaries

| Path | Responsibility |
|---|---|
| `quactlize/packing/` | Canonical GGUF-to-K-pack producer and arrangement ABI |
| `quactlize/dense/` | Public dense provider; dynamic input/output/workspace binding |
| `quactlize/dispatch/`, `policies/` | Measured selection tables, buckets and deterministic fallback |
| `quactlize/execution/`, `quactlize/decode/` | SIMT/GEMV, indexed execution and MoE helpers |
| `quactlize/prefill/`, `quactlize/dequant/` | Online SF/full dequant, provider composition and typed dense helpers |
| `quactlize/include/`, `quactlize/runtime/` | TC collectives, launchers, JIT and resident modules |
| `reference/`, `tests/`, `ci/` | Independent format oracles and regression tests |
| `dev/`, `benchmarks/`, `tools/` | Maintained experiments, profiling, sweeps, build internals and policy generation |

Canonical formats are Q8_0 K-pack2; Q2_K low2; Q3_K low2+high1;
Q4_K low4; Q5_K low4+high1; Q6_K low4+high2. Packed-unit metadata is
separate. Consumers validate the arrangement descriptor; unknown layouts
must not be guessed. Offline formats and rounding are independent of tuning.

Useful references:

- [Selection and prewarm](docs/KPACK_SELECTION.md)
- [Dynamic execution ABI](docs/KPACK_DYNAMIC_EXECUTION.md)
- [Measured dense prefill helpers and native comparison](docs/PPU_PREFILL_DENSE_PRODUCTION.md)
- [Branch/build policy](docs/BRANCHES.md)

## Development cleanup

Keep reusable tests and sweeps, not one runner per completed experiment.
Production translation units must instantiate only retained implementations;
unadmitted debug kernels must not accumulate in the installed runtime.

The retired dense-helper rounds R1–R5 are recoverable at commit
`8a004d42a91a23664e69762509d34c3b76bbf8e8`. Current production keeps typed
dequant IDs `0/8/20/21`, IO IDs `0..3`, and the existing general SF/full
routes. Retired dense IDs are rejected, never renumbered or silently aliased.
Production helper selection, arithmetic and configuration tables are unchanged.
Use the current model-specific Asys sweep for new comparisons, not a retired
round's command.
