#!/usr/bin/env bash
# Run this script with bash. It never exits the caller's interactive shell.
(
  set -Eeuo pipefail
  ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
  : "${ROOT:?}" "${PPU_SDK:?set the SDK root containing envsetup.sh}" \
    "${QUACTLIZE_PPU_BUNDLE:?set the existing GEMM bundle path}"
  PYTHON=$(command -v "${PYTHON:-python3}")
  set +u
  source "$PPU_SDK/envsetup.sh"
  set -u
  export LD_LIBRARY_PATH="$PPU_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  PACKAGE="$ROOT/prebuilt/ppu0010/kpack-execution-v1"
  RESULT_ROOT=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
  RUN=$(mktemp -d "$RESULT_ROOT/kpack-execution.XXXXXX")
  test -d "$RUN" && test "$(dirname -- "$RUN")" = "$RESULT_ROOT"
  mkdir "$RUN/results"
  finish() {
    local rc=$?
    trap - EXIT
    printf 'runner_rc=%s\n' "$rc" > "$RUN/results/runner.rc"
    tar -czf "$RUN.results.tgz" -C "$RUN" results
    printf '\nrunner_rc=%s\nresults=%s.results.tgz\nCurrent Docker shell is preserved.\n' "$rc" "$RUN"
  }
  trap finish EXIT
  cd "$ROOT"
  test -s "$PACKAGE/manifest.json"
  test -s "$QUACTLIZE_PPU_BUNDLE/manifest.json"
  # Independent gates continue after a failure; each preserves completed rows.
  set +e
  "$PYTHON" tools/run_kpack_gemv_gate.py --sdk "$PPU_SDK" --bundle "$PACKAGE" \
    --gemm-bundle "$QUACTLIZE_PPU_BUNDLE" --output "$RUN/results/gemv" \
    --real-model-shapes --dense-real-shapes 2>&1 | tee "$RUN/results/gemv.log"
  GEMV_RC=${PIPESTATUS[0]}
  "$PYTHON" tools/run_kpack_grouped_device_gate.py --sdk "$PPU_SDK" \
    --bundle "$PACKAGE/grouped" --execution-bundle "$PACKAGE" \
    --output "$RUN/results/grouped" --real-model-shapes 2>&1 | tee "$RUN/results/grouped.log"
  GROUPED_RC=${PIPESTATUS[0]}
  printf 'gemv_rc=%s\ngrouped_rc=%s\n' "$GEMV_RC" "$GROUPED_RC" | tee "$RUN/results/gates.rc"
  test "$GEMV_RC" -eq 0 && test "$GROUPED_RC" -eq 0
)
