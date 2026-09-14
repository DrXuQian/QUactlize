#!/usr/bin/env bash
# Execute as a script. Never exit the user's enclosing Docker shell.
(
    set -Eeuo pipefail
    RUN=""
    finish() {
        rc=$?
        trap - EXIT
        if [[ -n "$RUN" && -d "$RUN/results" ]]; then
            tar -czf "$RUN.results.tgz" -C "$RUN" results || rc=1
            printf '\nresults=%s.results.tgz\n' "$RUN"
        fi
        printf 'runner_rc=%s; current Docker shell is preserved.\n' "$rc"
        exit "$rc"
    }
    trap finish EXIT
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_cost_parallel.py"
    cd "$ROOT"
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/envsetup.sh"
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    # Reuse the bundle compiled for the interrupted single-card campaign.
    # Do not call the builder or invalidate 134 already compiled parents.
    BASE=$(git rev-parse --short=12 7571b46)
    [[ "$BASE" =~ ^[0-9a-f]{12}$ ]]
    BUILD=$(realpath -e -- "${BUILD_DIR:-$RESULT_DIR/kpack-cost-build-$BASE}")
    test -n "$BUILD" && test -f "$BUILD/manifest.json"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    read -r -a cards <<< "${DEVICES:-0 1 2 3 4 5 6 7}"
    for card in "${cards[@]}"; do [[ "$card" =~ ^[0-9]+$ ]]; done
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -f "$RUN/results/authority.json"
    else
        RUN=$(mktemp -d "$RESULT_DIR/kpack-cost-multi.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    printf 'COST_MULTI bundle=%s run=%s devices=%s build=SKIPPED\n' "$BUILD" "$RUN" "${cards[*]}"
    "$PYTHON" -u tools/run_kpack_cost_parallel.py --sdk "$SDK" --bundle "$BUILD" \
        --output "$RUN/results" --devices "${cards[@]}" 2>&1 | tee -a "$RUN/results/console.log"
)
