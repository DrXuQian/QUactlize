#!/usr/bin/env bash
# Launch as a script: errors do not exit the enclosing Docker shell.
(
    set -Eeuo pipefail
    RUN=""
    stage=precheck
    finish() {
        rc=$?
        trap - EXIT
        if [[ -n "$RUN" && -d "$RUN/results" ]]; then
            if tar -czf "$RUN.results.tgz" -C "$RUN" results; then
                printf '\nresults=%s.results.tgz\n' "$RUN"
            else
                printf 'Archive failed; results remain in %s/results\n' "$RUN" >&2
                rc=1
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "COST_SUPPLEMENT FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    PHASE=${1:-dense-mid}
    case "$PHASE" in all|dense-mid|dense-rest|formats|grouped-mid|sparse) ;; *) printf 'Unknown phase: %s\n' "$PHASE"; false;; esac
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_cost_supplement.py"
    cd "$ROOT"
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -x "$SDK/bin/hgcc" && test -f "$SDK/lib/libhggc_wrapper.so"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    "$PYTHON" -c 'import numpy, gguf, torch'
    if [[ ${FETCH_PAYLOADS:-1} == 1 ]]; then
        stage=fetch
        # Only the two existing component carriers; never fetch the old giant sweep.
        git lfs pull --include="prebuilt/ppu0010/kpack-dequant-v4/**,prebuilt/ppu0010/kpack-prefill-measure-v1/**" --exclude=""
    fi
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    SOURCE=$(git rev-parse --short=12 HEAD)
    [[ "$SOURCE" =~ ^[0-9a-f]{12}$ ]]
    BUILD=${BUILD_DIR:-"$RESULT_DIR/kpack-cost-build-$SOURCE"}
    CACHE=${COMPILE_CACHE:-"$RESULT_DIR/kpack-cost-compile-cache"}
    test -n "$BUILD" && test -n "$CACHE"
    JOBS=${JOBS:-192}
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -d "$RUN/results" && test -f "$RUN/results/authority.json"
    else
        RUN=$(mktemp -d "$RESULT_DIR/kpack-cost-supplement.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    printf 'COST_SUPPLEMENT phase=%s run=%s build=%s jobs=%s\n' "$PHASE" "$RUN" "$BUILD" "$JOBS"
    printf 'Use the same idle PPU. Missing parent images build on box; first JIT/upload is untimed.\n'
    stage=build
    "$PYTHON" -u tools/build_kpack_cost_supplement.py --sdk "$SDK" --cache "$CACHE" \
        --output "$BUILD" --jobs "$JOBS" 2>&1 | tee -a "$RUN/results/build.log"
    stage=measure
    "$PYTHON" -u tools/run_kpack_cost_supplement.py --sdk "$SDK" --bundle "$BUILD" \
        --output "$RUN/results" --phase "$PHASE" 2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
