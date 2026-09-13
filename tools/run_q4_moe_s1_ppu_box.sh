#!/usr/bin/env bash
# Run with bash, not source. Never close the caller's Docker shell.
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
                printf 'Archive failed; raw results retained in %s/results\n' "$RUN" >&2
                if [[ $rc == 0 ]]; then rc=1; fi
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "Q4_MOE_S1_BOX FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/dev/gemv_ppu/run_moe_s1.py"
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
    [[ ${FETCH_PAYLOADS:-1} == 0 || ${FETCH_PAYLOADS:-1} == 1 ]]
    if [[ ${FETCH_PAYLOADS:-1} == 1 ]]; then
        stage=fetch-payloads
        git lfs pull --include="prebuilt/ppu0010/q4-moe-s1-v1/*.so,prebuilt/ppu0010/q4-smallm-v1/libq4_smallm_n512_k2048.so" --exclude=""
    fi
    stage=verify
    "$PYTHON" - <<'PY'
from pathlib import Path
from dev.gemv_ppu.moe_s1 import verify
verify(Path('prebuilt/ppu0010/q4-moe-s1-v1'))
print('Q4_MOE_S1_PACKAGE VERIFIED shapes=6 cases=108 recipes=16 inputs=F16+F32 compile=NONE JIT=NONE')
PY
    if [[ -n ${RESUME_RUN:-} ]]; then
        CANDIDATE_RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$CANDIDATE_RUN" && test -f "$CANDIDATE_RUN/results/authority.json"
        "$PYTHON" - "$CANDIDATE_RUN/results/authority.json" <<'PY'
import json,sys
from pathlib import Path
from dev.gemv_ppu.moe_s1 import SCHEMA
if json.loads(Path(sys.argv[1]).read_text()).get('schema') != SCHEMA:
    raise SystemExit('Resume directory is not a Q4 MoE S1 campaign')
PY
        RUN="$CANDIDATE_RUN"
    else
        RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
        test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
        RUN=$(mktemp -d "$RESULT_DIR/q4-moe-s1-ppu.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    EXTRA=()
    [[ ${ACU:-1} == 0 || ${ACU:-1} == 1 ]]
    if [[ ${ACU:-1} == 1 ]]; then
        ACU_BIN=${ACU_BIN:-$SDK/asight/bin/acu}
        test -x "$ACU_BIN"
        EXTRA+=(--acu "$ACU_BIN")
    fi
    if [[ -n ${L2_BYTES:-} ]]; then EXTRA+=(--l2-bytes "$L2_BYTES"); fi
    stage=gate
    printf 'Q4_MOE_S1_BOX run=%s tokens=1..8 experts=256 topk=8 A=F32 output=F32\n' "$RUN"
    printf 'One visible PPU must be idle; resident timing excludes setup/JIT/graph upload.\n'
    printf 'Kernel S1 only; production policy and existing fused MoE chain are unchanged.\n'
    "$PYTHON" -u dev/gemv_ppu/run_moe_s1.py --sdk "$SDK" --output "$RUN/results" "${EXTRA[@]}" \
        2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
