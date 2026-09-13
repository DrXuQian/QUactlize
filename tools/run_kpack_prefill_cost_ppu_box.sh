#!/usr/bin/env bash
# Run with bash; failure never exits the enclosing Docker shell.
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
                printf 'Archive failed; raw results are retained at %s/results\n' "$RUN" >&2
                rc=1
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "KPACK_PREFILL_COST FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_bf16_gate.py"
    cd "$ROOT"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
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
    if [[ ${FETCH_PAYLOADS:-1} == 1 ]]; then
        stage=fetch
        git lfs pull --include="prebuilt/ppu0010/kpack-dequant-v2/*.so" --exclude=""
    fi
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -d "$RUN/results/dequant"
    else
        RUN=$(mktemp -d "$RESULT_DIR/kpack-prefill-cost.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir -p "$RUN/results/dequant" "$RUN/results/bf16"
    fi
    EXTRA=()
    [[ ${ACU:-1} == 0 || ${ACU:-1} == 1 ]]
    if [[ ${ACU:-1} == 1 ]]; then
        ACU_BIN=${ACU_BIN:-$SDK/asight/bin/acu}
        test -x "$ACU_BIN"
        EXTRA+=(--acu "$ACU_BIN")
    fi
    printf 'KPACK_PREFILL_COST run=%s local_payload_compile=NONE provider_first_use=EXCLUDED\n' "$RUN"
    printf 'Use an idle PPU. BF16 providers must already be installed; DeepGEMM may JIT during untimed setup.\n'
    failed=0
    stage=dequant-only
    if "$PYTHON" -u tools/run_kpack_dequant_gate.py --sdk "$SDK" --output "$RUN/results/dequant" \
        --qtypes "${QTYPES:-12,13}" --peak-gbps "${PEAK_GBPS:-2700}" "${EXTRA[@]}" \
        2>&1 | tee -a "$RUN/results/dequant/console.log"; then
        printf 'KPACK_PREFILL_COST dequant=PASS\n'
    else
        failed=1
        printf 'KPACK_PREFILL_COST dequant=INCOMPLETE valid independent families remain usable\n'
    fi
    stage=bf16-provider-only
    if "$PYTHON" -u tools/run_kpack_bf16_gate.py --sdk "$SDK" --dequant-results "$RUN/results/dequant" \
        --output "$RUN/results/bf16" --qtypes "${QTYPES:-12,13}" --ms "${MS:-2048,4096}" "${EXTRA[@]}" \
        2>&1 | tee -a "$RUN/results/bf16/console.log"; then
        printf 'KPACK_PREFILL_COST bf16=PASS\n'
    else
        failed=1
        printf 'KPACK_PREFILL_COST bf16=INCOMPLETE see per-family logs\n'
    fi
    stage=complete
    printf 'KPACK_PREFILL_COST summary=%s/results/bf16/summary.tsv\n' "$RUN"
    test "$failed" == 0
)
