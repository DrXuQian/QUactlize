#!/usr/bin/env bash
# Run with bash, not source. All kernel modules are already compiled.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run with bash, not source.\n' >&2
    return 2
fi
set -euo pipefail
ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_dense_grouped_ab.py"
cd "$ROOT"
SDK=${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}
test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
export LD_LIBRARY_PATH="$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    printf 'FAIL: select exactly one PPU with CUDA_VISIBLE_DEVICES\n' >&2
    exit 2
fi
RUN=$(mktemp -d /workspace/q4-dense-grouped.XXXXXX)
test -n "$RUN" && test -d "$RUN"
printf 'Q4_DENSE_GROUPED_BOX PREBUILT_ONLY arms=5 device=%s run=%s\n' "$CUDA_VISIBLE_DEVICES" "$RUN"
rc=0
python3 -u tools/run_kpack_dense_grouped_ab.py "$@" --sdk "$SDK" \
    --output "$RUN/results" 2>&1 | tee "$RUN/console.log" || rc=$?
if test -d "$RUN/results"; then
    tar -czf "$RUN.results.tgz" -C "$RUN" results console.log
else
    tar -czf "$RUN.results.tgz" -C "$RUN" console.log
fi
printf 'results=%s.results.tgz\nrunner_rc=%s\n' "$RUN" "$rc"
exit "$rc"
