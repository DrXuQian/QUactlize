#!/usr/bin/env bash
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run with bash, not source.\n' >&2
    return 2
fi
set -euo pipefail
ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_grouped_postops.py"
cd "$ROOT"
SDK=${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}
test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
export LD_LIBRARY_PATH="$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=$(mktemp -d /workspace/kpack-grouped-postops.XXXXXX)
test -n "$RUN" && test -d "$RUN"
rc=0
printf 'GROUPED_POSTOPS_BOX PREBUILT_ONLY run=%s\n' "$RUN"
python3 -u tools/run_kpack_grouped_postops.py --sdk "$SDK" --output "$RUN/results" "$@" \
    2>&1 | tee "$RUN/console.log" || rc=$?
tar -czf "$RUN.results.tgz" -C "$RUN" .
printf 'results=%s.results.tgz\nrunner_rc=%s\n' "$RUN" "$rc"
exit "$rc"
