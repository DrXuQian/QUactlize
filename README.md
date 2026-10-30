# Quactlize

Weight-only quantized GEMM/GEMV for T-Head PPU.

`develop` contains production code, admitted selection tables, package builders
and product correctness tests. `main` is the separately curated release line.
Debug kernels, benchmarks, Asys/ACU sweeps, historical results and their tests
are maintained on [`experiments/ppu-tuning`](https://github.com/DrXuQian/QUactlize/tree/experiments/ppu-tuning).
The complete pre-cleanup snapshot is `c436ce5e4dc14ec60149b973342749d16125dba4`.

## Build and test

Checkout the pinned `third_party/actlize` gitlink using your environment's Git
credentials. The public builder never fetches or modifies dependencies.

```bash
PPU_SDK=/usr/local/PPU_SDK bash build.sh --library \
  --output /workspace/quactlize-package-RUN_ID --jobs 8

CUDA_VISIBLE_DEVICES=0 PPU_SDK=/usr/local/PPU_SDK \
  bash run_tests.sh --suite test --package /workspace/quactlize-package-RUN_ID

# Host contracts / test inventory (no GPU execution).
PPU_SDK=/usr/local/PPU_SDK bash run_tests.sh --suite host
bash run_tests.sh --suite test --list
```

Build output must be a new directory. Tests without `--output` create a unique
temporary directory and print its path. Both entrypoints honor `PYTHON`.
See [CI integration](docs/CI.md) for prerequisites and result handling.

Install the complete package: `libncp_quactlize.so`, `include/`,
`quactlize-runtime/` and `manifest.json`. ncp_flash_lib only builds/installs it;
packing, config selection and execution remain in Quactlize. The public ABI is
declared in [packing/api.h](quactlize/packing/api.h) and
[dense/api.h](quactlize/dense/api.h).

## Source layout

| Directory | Responsibility |
|---|---|
| `quactlize/` | Formats, kernels, dispatch, JIT and public runtime |
| `policies/` | Admitted measurements, exact tables, buckets and required validation inputs |
| `tools/` | Package builders, deterministic table generators, GGUF metadata and JIT prewarm |
| `reference/` | Independent GGUF/K-pack format oracle |
| `ci/`, `tests/` | Product CI, reusable numerical fixtures and ABI/selection regressions |

PPU1.0/1.5 use pinned actlize; [PPU1.7](quactlize/backends/ppu17/README.md)
remains an isolated backend. Canonical formats are Q8_0 K-pack2 and
Q2_K–Q6_K per-plane K-pack. Consumers validate versioned arrangements; they
must not guess an unknown layout.

## Performance work

Use a separate worktree so profiling does not change the integration checkout:

```bash
git fetch origin
git worktree add --detach ../quactlize-tuning origin/experiments/ppu-tuning
```

Run sweeps and inspect their documentation there. Promote only validated
kernel changes, selection tables and focused regression tests to `develop`;
do not merge the experimental branch wholesale. Keep result archives and
compiled packages outside source branches. Repository cleanup is not a new
GPU performance admission.
