# Product CI

There are two entrypoints: `build.sh` and `run_tests.sh`. CI owns authenticated
checkout of the exact actlize gitlink. Neither entrypoint downloads dependencies
or changes Git state.

## Prerequisites

- Python 3.10+ and `python -m pip install -r ci/requirements.txt`.
- `git`, `gcc`, `g++`, pinned actlize and PPU SDK headers for host contracts.
- A complete PPU SDK (`PPU_SDK`) for package builds.
- A real supported PPU and the complete built package for accuracy tests.

```bash
# Build once; output must not already exist.
bash build.sh --library --sdk /usr/local/PPU_SDK \
  --output /workspace/quactlize-package-RUN_ID --jobs 8

# Test that exact package; do not rebuild it in the test stage.
CUDA_VISIBLE_DEVICES=0 PPU_SDK=/usr/local/PPU_SDK \
  bash run_tests.sh --suite test --package /workspace/quactlize-package-RUN_ID \
  --output /workspace/quactlize-test-RUN_ID
```

`--suite host` runs host ABI, format, numerical-oracle, dispatch, JIT-cache and
package checks. `--suite accuracy --package ...` runs PPU numerical tests.
`--suite test` runs both without rebuilding. `--suite all` additionally builds
a fresh package between host and accuracy phases. `--list` prints the exact
file inventory without loading a device or requiring SDK installation.

The device suite covers Q8_0 and Q2_K–Q6_K, F16/BF16 compute, dense and MoE,
dynamic addresses, graph replay where supported, and declared shape/boundary
cases. Host tests use explicitly identified doubles where no device is involved.
No skip, xfail, missing test or empty test run counts as a pass.

Each run writes `summary.json`, `inventory.json`, per-phase commands/logs and
JUnit XML. A nonzero exit is a failed gate. Omit `--output` for a unique
temporary results directory (printed in the final line); an explicit output
directory must be new. Failed logs are retained. `--timeout` bounds each phase.

The registered product tests live in `ci/suites.py`; unregistered test files
are rejected. Tests for sweep coordinators, historical experiment importers,
legacy Torch/six-library runners, model benchmarks and PPU1.7 development are
on `experiments/ppu-tuning`. They are not silently counted as passing product
CI, and product CI is not an exhaustive config sweep or model-performance gate.
