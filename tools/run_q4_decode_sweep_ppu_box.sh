#!/usr/bin/env bash
# The subshell preserves the invoking Docker shell on errors and interruption.
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
                printf 'Archive failed; raw results remain at %s/results\n' "$RUN" >&2
                if [[ $rc == 0 ]]; then rc=1; fi
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "Q4_DECODE_BOX FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/dev/gemv_ppu/run_decode_sweep.py"
    cd "$ROOT"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    export PPU_SDK="$SDK" PATH="$SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    "$PYTHON" -c 'import numpy, torch, gguf'
    [[ ${FETCH_PAYLOADS:-1} == 0 || ${FETCH_PAYLOADS:-1} == 1 ]]
    if [[ ${FETCH_PAYLOADS:-1} == 1 ]]; then
        stage=fetch-payloads
        git lfs pull --include="prebuilt/ppu0010/q4-decode-sweep-v1/*.so,prebuilt/ppu0010/q4-decode-sweep-v1/modules/*/kernel.so,prebuilt/ppu0010/q4-moe-s1-v2/*.so,prebuilt/ppu0010/q4-smallm-v1/libq4_smallm_n*.so" --exclude=""
    fi
    stage=verify
    "$PYTHON" - <<'PY'
from dev.gemv_ppu import decode_sweep as spec
from dev.gemv_ppu.run_decode_sweep import verify_inventory
m=spec.verify()
verify_inventory(m)
pool=[c for w in spec.workloads() for c in spec.catalog(m,w)]
print(f'Q4_DECODE_PACKAGE VERIFIED dense=96 grouped=276 TC_parents={len(m["modules"])} '
      f'SIMT_cells={sum(c["arm"]=="simt" for c in pool)} TC_cells={sum(c["arm"]=="tc" for c in pool)} '
      'compile=NONE JIT=NONE device_admission=PENDING')
PY
    if [[ -n ${RESUME_RUN:-} ]]; then
        CANDIDATE_RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$CANDIDATE_RUN" && test -f "$CANDIDATE_RUN/results/authority.json"
        "$PYTHON" - "$CANDIDATE_RUN/results/authority.json" <<'PY'
import json,sys
from pathlib import Path
from dev.gemv_ppu.decode_sweep import SCHEMA
if json.loads(Path(sys.argv[1]).read_text()).get('schema')!=SCHEMA:
    raise SystemExit('Resume requires a Q4 decode sweep, not the earlier MoE comparison')
PY
        RUN="$CANDIDATE_RUN"
    else
        RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
        test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
        RUN=$(mktemp -d "$RESULT_DIR/q4-decode-sweep-ppu.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    EXTRA=(--retry-failed --operators "${OPERATORS:-all}")
    [[ ${ACU:-1} == 0 || ${ACU:-1} == 1 ]]
    if [[ ${ACU:-1} == 1 ]]; then
        ACU_BIN=${ACU_BIN:-$SDK/asight/bin/acu}
        test -x "$ACU_BIN"
        EXTRA+=(--acu "$ACU_BIN")
    fi
    if [[ -n ${L2_BYTES:-} ]]; then EXTRA+=(--l2-bytes "$L2_BYTES"); fi
    stage=sweep
    printf 'Q4_DECODE_BOX run=%s operators=%s default_cases=372\n' "$RUN" "${OPERATORS:-all}"
    printf 'One visible PPU must be idle. Dense M1..8; MoE tokens1..8/top8/E256. No prefill sweep.\n'
    printf 'F32 endpoints: TC casts, GPU routing, directory, reducer and scatter INCLUDED.\n'
    printf 'Screen5; best2 per arm + current policy; alternating 6x15 confirmation. First launch excluded.\n'
    printf 'Default ACU: 18 representative cases, both winners. Forced-cold counters are not rotating event timings.\n'
    printf 'Failures preserve valid cells; resume retries failed cells in fresh processes. Production policy unchanged.\n'
    "$PYTHON" -u dev/gemv_ppu/run_decode_sweep.py --sdk "$SDK" --output "$RUN/results" "${EXTRA[@]}" \
        2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
