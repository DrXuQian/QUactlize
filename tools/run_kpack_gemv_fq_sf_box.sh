#!/usr/bin/env bash
# Run with bash; never source. The caller's Docker shell stays open on failure.
(
  set -Eeuo pipefail
  ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
  test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_gemv_fq_sf.py"
  SDK=${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}
  test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
  PYTHON=$(command -v "${PYTHON:-python3}")
  test -n "$PYTHON" && test -x "$PYTHON"
  export LD_LIBRARY_PATH="$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    printf 'FAIL: select exactly one PPU with CUDA_VISIBLE_DEVICES\n' >&2
    exit 2
  fi
  cd "$ROOT"
  if [[ -n ${RESUME_RUN:-} ]]; then
    RUN=$(realpath -e -- "$RESUME_RUN")
    test -n "$RUN" && test -d "$RUN/results"
    RESUME=(--resume)
  else
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    RUN=$(mktemp -d "$RESULT_DIR/kpack-gemv-fq-sf.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    RESUME=()
  fi
  printf 'GEMV_FQ_SF_BOX PREBUILT_ONLY default_cases=5 simt_recipes=24/reader device=%s run=%s\n' "$CUDA_VISIBLE_DEVICES" "$RUN"
  rc=0
  "$PYTHON" -u tools/run_kpack_gemv_fq_sf.py --collect --sdk "$SDK" \
    --output "$RUN/results" "${RESUME[@]}" "$@" 2>&1 | tee -a "$RUN/console.log" || rc=$?
  if test -d "$RUN/results"; then
    tar -czf "$RUN.results.tgz" -C "$RUN" results console.log
  else
    tar -czf "$RUN.results.tgz" -C "$RUN" console.log
  fi
  printf '\nsummary=%s/results/summary.tsv\nacu=%s/results/acu\nresults=%s.results.tgz\nrunner_rc=%s\nCurrent Docker shell is preserved.\n' "$RUN" "$RUN" "$RUN" "$rc"
  exit "$rc"
)
