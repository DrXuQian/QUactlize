#!/usr/bin/env bash
# Run with bash, not source. The subshell preserves the calling Docker shell.
(
    set -Eeuo pipefail
    stage=precheck
    RUN=""
    finish() {
        rc=$?
        trap - EXIT
        if [[ -n "$RUN" && -d "$RUN/results" ]]; then
            if tar -czf "$RUN.results.tgz" -C "$RUN" results; then
                printf '\nresults=%s.results.tgz\nsummary=%s/results/summary.tsv\n' "$RUN" "$RUN"
            else
                printf 'Archive failed; results remain in %s/results\n' "$RUN" >&2
                if [[ $rc == 0 ]]; then rc=1; fi
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "Q4_ASTAGE_BOX FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/dev/gemv_ppu/run_astage.py"
    cd "$ROOT"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    export PPU_SDK="$SDK"
    export PATH="$SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    "$PYTHON" -c 'import numpy, torch, gguf'
    "$PYTHON" -c 'from pathlib import Path; from dev.gemv_ppu.build_astage import verify; verify(Path("prebuilt/ppu0010/q4-astage-v1"),Path("prebuilt/ppu0010/q4-simt-ab-v1")); print("Q4_ASTAGE_PACKAGE PASS controls=UNCHANGED production=UNCHANGED")'
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -d "$RUN/results"
    else
        RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
        test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
        RUN=$(mktemp -d "$RESULT_DIR/q4-astage-ppu.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    if [[ -n ${FIXTURES:-} ]]; then
        FIXTURE_DIR=$(realpath -e -- "$FIXTURES")
        test -n "$FIXTURE_DIR" && test -d "$FIXTURE_DIR"
    elif [[ -d "$RUN/fixtures" ]]; then
        FIXTURE_DIR="$RUN/fixtures"
    else
        stage=fixtures
        "$PYTHON" -u dev/gemv_cuda/prepare_fixtures.py --q4-dense-wide --output "$RUN/fixtures" \
            2>&1 | tee "$RUN/results/fixtures.log"
        FIXTURE_DIR="$RUN/fixtures"
    fi
    EXTRA=()
    [[ ${ACU:-1} == 0 || ${ACU:-1} == 1 ]]
    [[ ${ANCHOR_ONLY:-0} == 0 || ${ANCHOR_ONLY:-0} == 1 ]]
    if [[ ${ACU:-1} == 0 ]]; then
        EXTRA+=(--skip-acu)
    else
        ACU_BIN=${ACU_BIN:-$SDK/asight/bin/acu}
        test -x "$ACU_BIN"
        EXTRA+=(--acu "$ACU_BIN")
    fi
    if [[ ${ANCHOR_ONLY:-0} == 1 ]]; then EXTRA+=(--anchor-only); fi
    if [[ -n ${L2_BYTES:-} ]]; then EXTRA+=(--l2-bytes "$L2_BYTES"); fi
    stage=compare
    printf 'Q4_ASTAGE_BOX run=%s arms=xplane,baseline,shared-a cache=warm+rotating\n' "$RUN"
    printf 'PREBUILT=1 compile=NONE JIT=NONE sweep=NONE first_launch=EXCLUDED rounds=6 samples=15\n'
    "$PYTHON" -u dev/gemv_ppu/run_astage.py --sdk "$SDK" --fixtures "$FIXTURE_DIR" \
        --output "$RUN/results" "${EXTRA[@]}" 2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
