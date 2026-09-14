#!/usr/bin/env bash
# A script/subshell failure must not exit the enclosing Docker shell.
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
                printf 'Archive failed; results remain at %s/results\n' "$RUN" >&2
                rc=1
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "DECODE_REDUCER FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_decode_reducer.py"
    cd "$ROOT"
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    export PPU_SDK="$SDK"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    "$PYTHON" -c 'import numpy, gguf'
    if [[ ${FETCH_PAYLOADS:-1} == 1 ]]; then
        stage=fetch
        git lfs pull --include="prebuilt/ppu0010/kpack-prefill-measure-v1/**" --exclude=""
    fi
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -d "$RUN/results" && test -f "$RUN/results/authority.json"
    else
        RUN=$(mktemp -d "$RESULT_DIR/kpack-decode-reducer.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    printf 'DECODE_REDUCER run=%s compile=NONE GEMM=NONE FULL_DEQUANT=NONE\n' "$RUN"
    printf 'Use the same idle PPU. Reused small partials, not cold HBM or a fused reduce/scatter measurement.\n'
    stage=reducer
    "$PYTHON" -u tools/run_kpack_decode_reducer.py --sdk "$SDK" --output "$RUN/results" \
        2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
