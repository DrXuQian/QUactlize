#!/usr/bin/env bash
# Invoke with bash. Keep the parent Docker shell and successful cases intact.
(
    set -Eeuo pipefail
    RUN= stage=precheck
    finish() {
        rc=$?
        trap - EXIT ERR
        if [[ -n $RUN && -d $RUN/results ]]; then
            tar -czf "$RUN.results.tgz" -C "$RUN" results || rc=1
            printf '\nresults=%s.results.tgz\nsummary=%s/results/summary.tsv\n' "$RUN" "$RUN"
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "SIMT_FORMATS_BOX FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_simt_formats.py"
    cd "$ROOT"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -r "$SDK/envsetup.sh"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    export PPU_SDK="$SDK" LC_ALL=C
    export PATH="$SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    "$PYTHON" -c 'import numpy,torch,gguf'
    read -r -a GPUS <<< "${DEVICES:-0}"
    read -r -a TOKENS_ARG <<< "${TOKENS:-1 2 3 4 5 6 7 8}"
    read -r -a QTYPES_ARG <<< "${QTYPES:-10 11 13 14 8}"
    BUNDLE="$ROOT/prebuilt/ppu0010/simt-register-reuse-v1"
    if [[ ${FETCH_PAYLOADS:-1} == 1 ]]; then
        stage=fetch
        git lfs pull --include='prebuilt/ppu0010/simt-register-reuse-v1/*.so' --exclude=''
    fi
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -f "$RUN/results/authority.json"
    else
        RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
        test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
        RUN=$(mktemp -d "$RESULT_DIR/simt-formats-ppu.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    stage=sweep
    printf 'SIMT_FORMATS_BOX run=%s compile=NONE jit=NONE cache=ROTATING_ACTIVE_WEIGHTS\n' "$RUN"
    printf 'Use idle devices. Timings include Split-K reducers, not TC or model adapters.\n'
    "$PYTHON" -u tools/run_simt_formats.py --sdk "$SDK" --bundle "$BUNDLE" --output "$RUN/results" \
        --devices "${GPUS[@]}" --tokens "${TOKENS_ARG[@]}" --qtypes "${QTYPES_ARG[@]}" \
        --l2-bytes "${L2_BYTES:?Set the verified PPU L2 bytes}" --rounds "${ROUNDS:-4}" --samples "${SAMPLES:-11}" \
        2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
