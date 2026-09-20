#!/usr/bin/env bash
# All work stays in this subshell. Keep the Docker shell on failure.
(
    set -Eeuo pipefail
    TASK_RUN= stage=precheck
    finish() {
        local rc=$?
        trap - EXIT ERR
        if [[ -n $TASK_RUN && -d $TASK_RUN/results ]]; then
            printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$TASK_RUN/results/runner-status.txt"
            tar --exclude='*.asysrep' --exclude='*.sqlite*' --exclude='*.tgz' \
                -czf "$TASK_RUN.results.tgz" -C "$TASK_RUN" results || rc=1
            printf 'results=%s.results.tgz\nAsys: %s/results/*-trace-*/**/proof.asysrep\n' "$TASK_RUN" "$TASK_RUN"
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "PAIRED_MODEL_BOX FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    TASK_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    cd "$TASK_ROOT"
    TASK_SDK=$(realpath -e -- "${PPU_SDK:-/sim/eec/shared/junfu.qx/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    TASK_RESULTS=$(realpath -e -- "${RESULT_ROOT:-/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9}")
    TASK_PYTHON=$(command -v "${PYTHON:-python3}")
    test -r "$TASK_SDK/envsetup.sh" && test -x "$TASK_SDK/bin/hgcc" && test -d "$TASK_RESULTS"
    set +u
    source "$TASK_SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$TASK_ROOT"
    export PPU_SDK="$TASK_SDK" CUDA_HOME="$TASK_SDK/CUDA_SDK" PPU_SDK_HOME="$TASK_SDK/CUDA_SDK"
    export DG_JIT_HGCC_COMPILER="$TASK_SDK/bin/hgcc"
    export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-$TASK_RESULTS/kpack-tp2-deepgemm-jit}
    export LD_LIBRARY_PATH="$TASK_SDK/lib:$TASK_SDK/targets/x86_64-linux/lib:$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export QUACTLIZE_PPU_BUNDLE=${QUACTLIZE_PPU_BUNDLE:-$TASK_RESULTS/kpack-q4-smallk.5DyMwt/update/bundle}
    EXTRA=()
    if [[ -n ${PAIRED_SELECTION_RUN:-} ]]; then
        TASK_RUN=$(realpath -e -- "$PAIRED_SELECTION_RUN");test -d "$TASK_RUN/results";EXTRA+=(--resume)
    else
        TASK_RUN=$(mktemp -d "$TASK_RESULTS/paired-selection.XXXXXX")
    fi
    test -n "$TASK_RUN" && test -d "$TASK_RUN"
    if [[ ${VERIFY_ONLY:-0} == 1 ]]; then EXTRA+=(--preflight-only); fi
    stage=paired-integration
    "$TASK_PYTHON" -u tools/run_paired_selection_box.py --sdk "$TASK_SDK" --output "$TASK_RUN" \
        --result-root "$TASK_RESULTS" --devices "${PAIRED_DEVICES:-6,7}" --jobs "${JOBS:-192}" \
        --llama "${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp}" \
        --ncp "${NCP_LIB_DIR:-/sim/eec/shared/junfu.qx/ncp_flash_lib}" \
        --cache "${CACHE_ROOT:-$TASK_RESULTS/kpack-tp2-model-cache}" \
        --jit-cache "${JIT_CACHE:-$TASK_RESULTS/kpack-model-jit-cache}" \
        --model-root "${MODEL_ROOT:-/sim/eec/shared/AI_workspace/llm-models}" \
        --phases "${PAIRED_PHASES:-all}" "${EXTRA[@]}" 2>&1 | tee -a "$TASK_RUN/console.log"
    stage=complete
)
