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

# CI test job: host contracts + real PPU numerical tests; no package rebuild.
CUDA_VISIBLE_DEVICES=0 PPU_SDK=/usr/local/PPU_SDK \
  bash run_tests.sh --suite test --package /workspace/quactlize-package-RUN_ID \
  --output /workspace/quactlize-tests-RUN_ID

# Inventory / host-only checks.
bash run_tests.sh --suite test --list
PPU_SDK=/usr/local/PPU_SDK bash run_tests.sh --suite host
```

The package contains `libquactlize_ppu.so`, exported `include/`, a private
`quactlize-runtime/` closure (including JIT sources) and a hash manifest.
Install the whole package, not just the public .so. ncp_flash_lib builds and
installs it; selection, packing and execution remain in Quactlize.

See [CI.md](docs/CI.md) for dependencies, host/build/accuracy coverage, logs and
JUnit. `--suite test` only tests; the compatible `--suite all` also builds.
Both scripts honor `PYTHON=/path/to/python`. Tooling, historical Torch
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
`8a004d42a91a23664e69762509d34c3b76bbf8e8`. The later unselected Q8 experiments
are recoverable at `618298d6f8761b8eb4489ad2f1a217dd57f0bd47`.
Current production keeps typed dequant IDs `0/8/20/21/22/23/38/41`, IO IDs
`0..4`, compact widths `2/4/8`, and the general SF/full routes. IDs38/41 are
the measured F16-only Q8 helpers; BF16 retains direct BF16 reconstruction.
Retired IDs are rejected, never renumbered or silently aliased. Cleanup leaves
the selection tables, rounding and retained native kernel bodies unchanged.
Use the current model-specific Asys sweep for new comparisons, not a retired
round's command.

The old `benchmarks/` batch/format/shape wrappers and completed 5090 experiments
are retired (recoverable at `138ad8b`). Their dedicated checks and log parser
are retired with them; product tests and recorded measurement evidence remain.
The four remaining Python scripts have live consumers: `analyse.py` for
selection analysis, `fixtures.py` / `workloads.py` for shared shape definitions,
and `sweep_gemv_perf.py` for the legacy GEMV sweep entry. CMake benchmark targets,
their generated configuration inputs and the Q4 comparison fixture headers
remain because builds, oracles or current PPU comparisons still use them.

Current performance entrypoints are separate from CI:

- `tools/run_kpack_strategy_campaign_box.sh`: small-M selection campaign.
- `tools/run_kpack_strategy_prefill_box.sh`: prefill component campaign.
- `tools/run_prefill_model_asys_box.sh`: model-shape prefill Asys sweep.
- `tools/run_native_components_box.sh`: isolated native/candidate comparison.
- `tools/run_dense_dequant_box.sh`: retained typed helpers; `Q8=1` tests the
  Q8 product cohort, optionally against the existing native F16 converter.

The [final public-library/model validation](docs/PPU_DENSE_CONTIGUOUS_IO.md)
remains pending. Cleanup is not new GPU timing or a claim of model parity.

Use their declared device, SDK, input and measurement requirements; these are
not substitutes for `run_tests.sh` and do not run during product CI.
