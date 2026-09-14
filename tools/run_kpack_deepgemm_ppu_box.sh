#!/usr/bin/env bash
# Supplement only the installed Python-JIT BF16 MoE provider; preserve old costs.
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
                printf 'Archive failed; raw results remain at %s/results\n' "$RUN" >&2
                rc=1
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "KPACK_DEEPGEMM FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_bf16_gate.py"
    cd "$ROOT"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
    if [[ -z ${DEQUANT_RESULTS:-} ]]; then
        printf 'Set DEQUANT_RESULTS to the previous v2 results/dequant directory. No dequant sweep is rerun.\n' >&2
        false
    fi
    DEQUANT=$(realpath -e -- "$DEQUANT_RESULTS")
    test -n "$DEQUANT" && test -f "$DEQUANT/authority.json" && test -f "$DEQUANT/result.json"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    export PPU_SDK="$SDK"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    BUNDLE="$ROOT/prebuilt/ppu0010/kpack-dequant-v2"
    if [[ ${FETCH_PAYLOADS:-1} == 1 ]]; then
        stage=fetch
        git lfs pull --include="prebuilt/ppu0010/kpack-dequant-v2/*.so" --exclude=""
    fi
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -d "$RUN/results/bf16"
    else
        RUN=$(mktemp -d "$RESULT_DIR/kpack-deepgemm-python.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir -p "$RUN/results/bf16"
    fi
    EXTRA=()
    [[ ${ACU:-1} == 0 || ${ACU:-1} == 1 ]]
    if [[ ${ACU:-1} == 1 ]]; then
        ACU_BIN=${ACU_BIN:-$SDK/asight/bin/acu}
        test -x "$ACU_BIN"
        EXTRA+=(--acu "$ACU_BIN")
    fi
    printf 'KPACK_DEEPGEMM run=%s provider=PYTHON_JIT dequant=v2-reused cublas=NOT_RERUN\n' "$RUN"
    printf 'Use an idle PPU. Installed DeepGEMM JIT and first-use warmup are excluded.\n'
    stage=deepgemm-only
    "$PYTHON" -u tools/run_kpack_bf16_gate.py --sdk "$SDK" --bundle "$BUNDLE" \
        --dequant-results "$DEQUANT" --output "$RUN/results/bf16" --provider deepgemm --gate-first \
        --qtypes "${QTYPES:-12,13}" --ms "${MS:-2048,4096}" "${EXTRA[@]}" \
        2>&1 | tee -a "$RUN/results/bf16/console.log"
    stage=complete
    printf 'KPACK_DEEPGEMM summary=%s/results/bf16/summary.tsv\n' "$RUN"
)
