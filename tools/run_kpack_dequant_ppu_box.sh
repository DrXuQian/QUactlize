#!/usr/bin/env bash
# Invoke with bash. The inner subshell preserves the surrounding Docker shell.
(
    set -Eeuo pipefail
    stage=precheck
    RUN=""
    finish() {
        rc=$?
        trap - EXIT
        if [[ -n "$RUN" && -d "$RUN/results" ]]; then
            if tar -czf "$RUN.results.tgz" -C "$RUN" results; then
                printf '\nresults=%s.results.tgz\n' "$RUN"
            else
                printf 'Archive failed; raw results remain at %s/results\n' "$RUN" >&2
                if [[ $rc == 0 ]]; then rc=1; fi
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "KPACK_DEQUANT FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_dequant_gate.py"
    cd "$ROOT"
    test "$#" -le 1
    INVENTORY=${1:-all}
    case "$INVENTORY" in
        all) BUNDLE_REL=prebuilt/ppu0010/kpack-dequant-v2; RUN_PREFIX=kpack-dequant-ppu ;;
        full-reader) BUNDLE_REL=prebuilt/ppu0010/kpack-dequant-v3; RUN_PREFIX=kpack-full-reader-ppu ;;
        *) printf 'Expected no argument or full-reader\n' >&2; false ;;
    esac
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    export PPU_SDK="$SDK" PATH="$SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    if [[ ${FETCH_PAYLOADS:-1} == 1 ]]; then
        stage=fetch
        git lfs pull --include="$BUNDLE_REL/*.so" --exclude=""
    fi
    EXTRA=()
    [[ ${ACU:-1} == 0 || ${ACU:-1} == 1 ]]
    if [[ ${ACU:-1} == 1 ]]; then
        ACU_BIN=${ACU_BIN:-$SDK/asight/bin/acu}
        test -x "$ACU_BIN"
        EXTRA+=(--acu "$ACU_BIN")
    fi
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -f "$RUN/results/authority.json"
    else
        RUN=$(mktemp -d "$RESULT_DIR/$RUN_PREFIX.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    stage=dequant-only
    printf 'KPACK_DEQUANT run=%s inventory=%s compile=NONE jit=NONE gemm_calls=0\n' "$RUN" "$INVENTORY"
    printf 'Use an idle PPU; SF and full BF16 expansion are separate measurements.\n'
    "$PYTHON" -u tools/run_kpack_dequant_gate.py --sdk "$SDK" --output "$RUN/results" \
        --bundle "$ROOT/$BUNDLE_REL" --inventory "$INVENTORY" \
        --qtypes "${QTYPES:-12,13}" --peak-gbps "${PEAK_GBPS:-2700}" "${EXTRA[@]}" \
        2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
