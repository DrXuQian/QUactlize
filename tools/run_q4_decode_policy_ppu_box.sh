#!/usr/bin/env bash
# Run as a child/subshell; never exit the invoking Docker shell.
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
    trap 'printf "Q4_DECODE_POLICY FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_q4_decode_policy_gate.py"
    cd "$ROOT"
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
        git lfs pull --include="prebuilt/ppu0010/q4-decode-policy-v1/*.so,prebuilt/ppu0010/q4-decode-policy-v1/modules/*/kernel.so,prebuilt/ppu0010/q4-decode-sweep-v1/*.so" --exclude=""
    fi
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -f "$RUN/results/authority.json"
    else
        RUN=$(mktemp -d "$RESULT_DIR/q4-decode-policy-ppu.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    stage=selected-gate
    printf 'Q4_DECODE_POLICY run=%s compile=NONE jit=NONE candidates=ONE_PER_CASE\n' "$RUN"
    printf 'Use one idle PPU. Policy boundaries/ID replay; first launch and setup excluded.\n'
    "$PYTHON" -u tools/run_q4_decode_policy_gate.py --sdk "$SDK" --output "$RUN/results" \
        2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
