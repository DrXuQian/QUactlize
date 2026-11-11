# Product CI

There are two entrypoints: `build.sh` and `run_tests.sh`. CI owns authenticated
checkout of the exact actlize gitlink. Neither entrypoint downloads dependencies
or changes Git state.

For precompiled Python distribution use `build.sh --wheel`; see
[PPU_WHEEL.md](PPU_WHEEL.md) for installation outside the source tree and
installed-wheel host/device/Flare qualification. Host wheel tests use explicitly
labelled ELF fixtures, not PPU kernels.

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

For ZW890/ZW-M890P build with `build.sh --library --arch ppu_15` (or
`run_tests.sh --suite all --arch ppu_15`). Accuracy selects its test inventory
from the **package manifest**, not the runner default: PPU1.5 uses the 174-case
canonical dense/MoE/FFN cohort. The PPU1.0 inventory additionally tests
Flash-Next native16 Q5_1/Q2_0 and special Q8 shapes. The `serving` suite
requires PPU1.0 and `QUACTLIZE_SERVING_LIBRARY=/path/to/libncp_quactlize.so`.
PPU1.5 Flash-Next entrypoints reject query/prepare; they are not silently
counted as passing, skipped, or covered by the canonical PPU1.5 tests.

Each run writes `summary.json`, `inventory.json`, per-phase commands/logs and
JUnit XML. A nonzero exit is a failed gate. Omit `--output` for a unique
temporary results directory (printed in the final line); an explicit output
directory must be new. Failed logs are retained. `--timeout` bounds each phase.

The registered product tests live in `ci/suites.py`; unregistered test files
are rejected. Tests for sweep coordinators, historical experiment importers,
legacy Torch/six-library runners, model benchmarks and PPU1.7 development are
on `experiments/ppu-tuning`. They are not silently counted as passing product
CI, and product CI is not an exhaustive config sweep or model-performance gate.
