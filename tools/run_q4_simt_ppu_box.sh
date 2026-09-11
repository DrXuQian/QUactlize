#!/usr/bin/env bash
# Execute with bash, never source. Failure does not terminate the Docker shell.
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
                printf 'Archive failed; uncompressed results remain in %s/results\n' "$RUN" >&2
                if [[ $rc == 0 ]]; then rc=1; fi
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "Q4_PPU_BOX FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/dev/gemv_ppu/run.py"
    cd "$ROOT"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so" && test -x "$SDK/bin/hgcc"
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
    BUNDLE="$ROOT/prebuilt/ppu0010/q4-simt-ab-v1"
    NATIVE=${NATIVE_BUNDLE:-$ROOT/prebuilt/ppu0010/kpack-jit-v2}
    test -s "$BUNDLE/manifest.json" && test -s "$NATIVE/manifest.json"
    "$PYTHON" -c 'import sys; from pathlib import Path; from dev.gemv_ppu.run import verify_bundle; from tools.verify_kpack_dispatch import verify; verify_bundle(Path(sys.argv[1])); verify(Path(sys.argv[2])); print("Q4_PPU_PACKAGE PASS")' "$BUNDLE" "$NATIVE"
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -d "$RUN/fixtures" && test -d "$RUN/results"
    else
        RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
        test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
        RUN=$(mktemp -d "$RESULT_DIR/q4-simt-ppu.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
        stage=fixtures
        "$PYTHON" -u dev/gemv_cuda/prepare_fixtures.py --q4-dense-wide --output "$RUN/fixtures" 2>&1 | tee "$RUN/results/fixtures.log"
    fi
    EXTRA=()
    if [[ ${ACU:-1} == 0 ]]; then
        EXTRA+=(--skip-acu)
    else
        ACU_BIN=${ACU_BIN:-$SDK/asight/bin/acu}
        test -x "$ACU_BIN"
        EXTRA+=(--acu "$ACU_BIN")
    fi
    if [[ -n ${L2_BYTES:-} ]]; then EXTRA+=(--l2-bytes "$L2_BYTES"); fi
    stage=compare
    printf 'Q4_PPU_BOX run=%s device=%s arms=4 shapes=6 cache=warm+rotating\n' "$RUN" "$CUDA_VISIBLE_DEVICES"
    printf 'SIMT=PREBUILT FQ=PRODUCTION_SELECTOR_JIT_MISS_ONLY all_first_launch_costs=EXCLUDED\n'
    "$PYTHON" -u dev/gemv_ppu/run.py --sdk "$SDK" --bundle "$BUNDLE" --native-bundle "$NATIVE" \
        --jit-cache "${JIT_CACHE:-$RUN/jit-cache}" --fixtures "$RUN/fixtures" --output "$RUN/results" \
        "${EXTRA[@]}" 2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
