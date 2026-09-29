#!/usr/bin/env bash
# Deliberate subshell: neither failure nor SDK setup closes the Docker shell.
(
    set -Eeuo pipefail
    stage=precheck
    task_run=
    finish() {
        local rc=$?
        trap - EXIT ERR
        if [[ -n $task_run && -d $task_run ]]; then
            printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$task_run/runner-status.txt"
            # ACU reports are small and included; no model, source tree or .so.
            local files=(runner-status.txt)
            if [[ -d $task_run/results ]]; then files+=(results); fi
            if [[ -f $task_run/console.log ]]; then files+=(console.log); fi
            tar -czf "$task_run.results.tgz" -C "$task_run" "${files[@]}" || rc=1
            printf 'results=%s.results.tgz\n' "$task_run"
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "PAIRED_CHAIN_BOX FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    TASK_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$TASK_ROOT" && test -f "$TASK_ROOT/dev/tp2_chain/run.py"
    cd "$TASK_ROOT"
    TASK_PYTHON=$(command -v "${PYTHON:-python3}")
    TASK_SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    TASK_RESULTS=$(realpath -e -- "${RESULT_ROOT:-/sim/eec/shared/junfu.qx}")
    test -n "$TASK_SDK" && test -x "$TASK_SDK/asight/bin/acu"
    test -n "$TASK_RESULTS" && test -d "$TASK_RESULTS" && test -w "$TASK_RESULTS"
    export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export LD_LIBRARY_PATH="$TASK_SDK/lib:$TASK_SDK/targets/x86_64-linux/lib:$TASK_SDK/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$TASK_PYTHON" -c 'import numpy, gguf'
    ARGS=()
    if [[ -n ${PAIRED_CHAIN_RUN:-} ]]; then
        task_run=$(realpath -e -- "$PAIRED_CHAIN_RUN")
        test -n "$task_run" && test -d "$task_run/results"
        ARGS+=(--resume)
    else
        task_run=$(mktemp -d "$TASK_RESULTS/paired-chain.XXXXXX")
        test -n "$task_run" && test -d "$task_run"
    fi
    printf 'PAIRED_CHAIN_BOX run=%s devices=%s\n' "$task_run" "${PAIRED_CHAIN_DEVICES:-6,7}"
    stage=paired-chain
    "$TASK_PYTHON" -u tools/run_paired_chain_box.py --sdk "$TASK_SDK" --result-root "$TASK_RESULTS" \
        --output "$task_run/results" --devices "${PAIRED_CHAIN_DEVICES:-6,7}" \
        --l2-bytes "${L2_BYTES:-67108864}" "${ARGS[@]}" 2>&1 | tee -a "$task_run/console.log"
    stage=complete
)
