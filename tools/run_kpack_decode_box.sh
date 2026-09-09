#!/usr/bin/env bash
# Prebuilt, bounded decode experiment. Run with bash, never source this file.
set -euo pipefail
ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_decode_sweep.py"
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
    RUN=$(mktemp -d /workspace/kpack-decode.XXXXXX)
    test -n "$RUN" && test -d "$RUN"
    RESUME=()
fi
case "$RUN" in
    /workspace/kpack-decode.*) ;;
    *) printf 'FAIL: result directory must be /workspace/kpack-decode.*\n' >&2; exit 2 ;;
esac
printf 'KPACK_DECODE_BOX PREBUILT_ONLY jobs=16 expected_cells=260 device=%s run=%s\n' "$CUDA_VISIBLE_DEVICES" "$RUN"
rc=0
python3 -u tools/run_kpack_decode_sweep.py --sdk "$SDK" \
    --output "$RUN/results" "${RESUME[@]}" 2>&1 | tee "$RUN/console.log" || rc=$?
if test -d "$RUN/results"; then
    tar -czf "$RUN.results.tgz" -C "$RUN" results console.log
else
    tar -czf "$RUN.results.tgz" -C "$RUN" console.log
fi
printf 'results=%s.results.tgz\n' "$RUN"
printf 'runner_rc=%s run=%s\n' "$rc" "$RUN"
exit "$rc"
