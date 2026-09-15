#!/usr/bin/env bash
# Reuse the failing model, corpus, runtime and caller build. Do not run a sweep.
(
    set -Eeuo pipefail
    RUN= stage=precheck
    finish() {
        local rc=$?
        trap - EXIT ERR
        if [[ -n $RUN && -d $RUN/diagnostic/results ]]; then
            printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/diagnostic/results/runner-status.txt"
            cp "$RUN/console.log" "$RUN/diagnostic/results/console.log"
            tar -czf "$RUN.results.tgz" -C "$RUN/diagnostic" results || rc=1
            printf '\nresults=%s.results.tgz\n' "$RUN"
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "KPACK_FIRST_NONFINITE FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_first_nonfinite.py"
    PREVIOUS=$(realpath -e -- "${PREVIOUS_RUN:?Set PREVIOUS_RUN to the failing run directory}")
    test -n "$PREVIOUS" && test -s "$PREVIOUS/results/caller-ci-build.json"
    LLAMA=$(realpath -e -- "${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp}")
    test -n "$LLAMA" && test -f "$LLAMA/.aoneci/scripts/build.sh"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -r "$SDK/envsetup.sh"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    export PPU_SDK="$SDK" LC_ALL=C CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    JOBS=${JOBS:-192}
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]
    RESULT_BASE=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_BASE" && test -d "$RESULT_BASE"
    RUN=$(mktemp -d "$RESULT_BASE/kpack-first-nonfinite.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    stage=diagnose
    cd "$ROOT"
    OPTIONS=()
    DEFAULT_ARMS='native-logits reference-tensors native-tensors'
    if [[ -n ${SNAPSHOT_UNTIL:-} ]]; then
        OPTIONS+=(--snapshot-until "$SNAPSHOT_UNTIL")
        DEFAULT_ARMS='reference-tensors native-tensors'
    fi
    read -r -a ARMS <<< "${DIAGNOSTIC_ARMS:-$DEFAULT_ARMS}"
    # The parent exists; the Python runner exclusively creates its output.
    "$PYTHON" -u tools/run_kpack_first_nonfinite.py --previous "$PREVIOUS" --llama "$LLAMA" \
        --sdk "$SDK" --jobs "$JOBS" --arms "${ARMS[@]}" "${OPTIONS[@]}" --output "$RUN/diagnostic" 2>&1 | tee "$RUN/console.log"
    stage=complete
)
