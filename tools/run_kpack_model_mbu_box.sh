#!/usr/bin/env bash
# Run with bash. Failure never exits the caller's Docker shell.
(
    set -Eeuo pipefail
    stage=precheck
    RUN=""
    finish() {
        rc=$?
        trap - EXIT
        if [[ -n "$RUN" && -d "$RUN/results" ]]; then
            if tar -czf "$RUN.results.tgz" -C "$RUN" results console.log; then
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
    trap 'printf "MODEL_MBU FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_model_mbu.py"
    cd "$ROOT"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    PYTHON_BIN=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON_BIN" && test -x "$PYTHON_BIN"
    export PPU_SDK="$SDK"
    export PATH="$SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    PREVIOUS=$(realpath -e -- "${PREVIOUS_RUN:-/workspace/kpack-q4-resume.5k41x3_y}")
    test -n "$PREVIOUS" && test -f "$PREVIOUS/results/trace.command.json"
    LLAMA=$(realpath -e -- "${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp}")
    test -n "$LLAMA" && test -f "$LLAMA/tests/quactlize_native.py"
    ACU_BIN=${ACU_BIN:-$SDK/asight/bin/acu}
    test -x "$ACU_BIN"
    "$PYTHON_BIN" -c 'import numpy, torch, gguf'
    CANDIDATE="$ROOT/prebuilt/ppu0010/model-simt-followup-v1"
    test -f "$CANDIDATE/manifest.json"
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    RUN=$(mktemp -d "$RESULT_DIR/kpack-model-mbu.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    stage=diagnostic
    printf 'MODEL_MBU run=%s tc_profiles=3 simt_points=5 cold_rounds=6 samples=15\n' "$RUN"
    printf 'MODEL_MBU candidate=PREBUILT production=UNCHANGED TC_JIT=REUSE_EXISTING_CACHE\n'
    "$PYTHON_BIN" -u tools/run_kpack_model_mbu.py --previous "$PREVIOUS" \
        --sdk "$SDK" --llama "$LLAMA" --candidate "$CANDIDATE" --acu "$ACU_BIN" \
        --l2-bytes "${L2_BYTES:-67108864}" --output "$RUN/results" 2>&1 | tee "$RUN/console.log"
    stage=complete
)
