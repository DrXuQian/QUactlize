#!/usr/bin/env bash
# Execute with bash, not source. Failures return to the caller's Docker shell.
set -euo pipefail
ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_gpu_compact.py"
cd "$ROOT"
SDK=${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}
test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
export LD_LIBRARY_PATH="$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    printf 'FAIL: select exactly one device ordinal with CUDA_VISIBLE_DEVICES\n' >&2
    exit 2
fi
if [[ -n "${RESUME_RUN:-}" ]]; then
    RUN=$(realpath -e -- "$RESUME_RUN")
    test -n "$RUN" && test -f "$RUN/results/authority.json"
    RESUME=(--resume)
else
    RUN=$(mktemp -d /workspace/kpack-compact.XXXXXX)
    test -n "$RUN" && test -d "$RUN"
    RESUME=()
fi
case "$RUN" in
    /workspace/kpack-compact.*) ;;
    *) printf 'FAIL: result directory must be /workspace/kpack-compact.*\n' >&2; exit 2 ;;
esac
printf 'GPU_COMPACT_BOX PREBUILT_ONLY jobs=11 expected_cells=204 device=%s run=%s\n' "$CUDA_VISIBLE_DEVICES" "$RUN"
rc=0
python3 -u tools/run_kpack_gpu_compact.py --sdk "$SDK" \
    --output "$RUN/results" "${RESUME[@]}" 2>&1 | tee "$RUN/console.log" || rc=$?
if test -d "$RUN/results"; then
    tar -czf "$RUN.results.tgz" -C "$RUN" results console.log
else
    tar -czf "$RUN.results.tgz" -C "$RUN" console.log
fi
printf 'results=%s.results.tgz\n' "$RUN"
printf 'runner_rc=%s run=%s\n' "$rc" "$RUN"
exit "$rc"
