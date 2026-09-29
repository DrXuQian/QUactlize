#!/usr/bin/env bash
# One-device replay of both rank shards; no model, collective or large rebuild.
(
    set -Eeuo pipefail
    RUN= stage=precheck
    finish() {
        local rc=$?
        trap - EXIT ERR
        if [[ -n $RUN && -d $RUN ]]; then
            printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/runner-status.txt"
            tar --exclude='*.o' --exclude='fixtures/*.bin' --exclude='diagnostic/fixtures/*.bin' \
                --exclude='diagnostic/build/fresh' --exclude='diagnostic/build/shipped' \
                -czf "$RUN.results.tgz" -C "$RUN" . || rc=1
            printf 'results=%s.results.tgz\n' "$RUN"
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "Q4_TP2_SIMT FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    cd "$ROOT"
    PYTHON=$(command -v "${PYTHON:-python3}")
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -x "$SDK/bin/hgcc" && test -r "$SDK/envsetup.sh"
    test -r "$SDK/include/hggc_runtime_api.h"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK" CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    export LD_LIBRARY_PATH="$SDK/targets/x86_64-linux/lib:$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PATH="$SDK/bin:$PATH" LC_ALL=C
    JOBS=${JOBS:-192}
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    RUN=$(mktemp -d "$RESULT_DIR/q4-tp2-simt.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    git rev-parse HEAD > "$RUN/source.txt"
    ARGS=()
    case ${Q4_TP2_FIELD_AB:-0} in
        0) ;;
        1) ARGS+=(--field-ab) ;;
        *) printf 'Q4_TP2_FIELD_AB must be 0 or 1\n' >&2; false ;;
    esac
    if [[ -n ${KPACK_BUNDLE:-} ]]; then
        ARGS+=(--bundle "$KPACK_BUNDLE")
    elif [[ -n ${PREVIOUS_RUN:-} ]]; then
        ARGS+=(--previous "$PREVIOUS_RUN")
    else
        printf 'Set PREVIOUS_RUN to the Q4 local diagnostic, or set KPACK_BUNDLE.\n' >&2
        false
    fi
    stage=host-tests
    "$PYTHON" -m unittest discover -s tests -p test_kpack_tp2_simt.py -v \
        2>&1 | tee "$RUN/host-tests.log"
    stage=local-replay
    "$PYTHON" -u tools/run_kpack_tp2_simt.py --sdk "$SDK" --output "$RUN/diagnostic" \
        --jobs "$JOBS" "${ARGS[@]}" 2>&1 | tee "$RUN/console.log"
    stage=diagnostic-complete
)
