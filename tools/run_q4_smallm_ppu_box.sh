#!/usr/bin/env bash
# Run with bash, not source. Failures never close the caller's Docker shell.
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
                printf 'Archive failed; raw results remain in %s/results\n' "$RUN" >&2
                if [[ $rc == 0 ]]; then rc=1; fi
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "Q4_SMALLM_BOX FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/dev/gemv_ppu/run_smallm.py"
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
        git lfs pull --include="prebuilt/ppu0010/q4-smallm-v1/*.so,prebuilt/ppu0010/q4-smallm-v1/modules/*/kernel.so,prebuilt/ppu0010/q4-medium-refine-v1/*.so,prebuilt/ppu0010/q4-small-latency-v1/*.so,prebuilt/ppu0010/q4-reader-followup-v1/*.so,prebuilt/ppu0010/q4-reader-reuse-v1/*.so,prebuilt/ppu0010/q4-config-sweep-v1/*.so,prebuilt/ppu0010/q4-cold-shapes-v1/*.so,prebuilt/ppu0010/q4-h800-port-v1/*.so,prebuilt/ppu0010/q4-simt-ab-v1/*.so" --exclude=""
    fi
    stage=verify
    "$PYTHON" - <<'PY'
from pathlib import Path
from dev.gemv_ppu import smallm as spec
m=spec.verify(Path('prebuilt/ppu0010/q4-smallm-v1'))
spec.medium_refine.verify(*(Path('prebuilt/ppu0010')/p for p in spec.PACKAGES))
print(f'Q4_SMALLM_PACKAGE PASS M=2,3,4,5,6,7,8 shapes=6 cases=42 SIMT_contexts=52 TC_parents={len(m["modules"])} compile=NONE JIT=NONE')
PY
    if [[ -n ${RESUME_RUN:-} ]]; then
        CANDIDATE_RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$CANDIDATE_RUN" && test -f "$CANDIDATE_RUN/results/authority.json"
        "$PYTHON" - "$CANDIDATE_RUN/results/authority.json" <<'PY'
import json,sys
from pathlib import Path
from dev.gemv_ppu.smallm import SCHEMA
if json.loads(Path(sys.argv[1]).read_text()).get('schema') != SCHEMA:
    raise SystemExit('Resume directory is not a Q4 small-M campaign')
PY
        RUN="$CANDIDATE_RUN"
    else
        RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
        test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
        RUN=$(mktemp -d "$RESULT_DIR/q4-smallm-ppu.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    if [[ -n ${FIXTURES:-} ]]; then
        FIXTURE_DIR=$(realpath -e -- "$FIXTURES")
        test -n "$FIXTURE_DIR" && test -d "$FIXTURE_DIR"
    else
        stage=fixtures
        if [[ ! -e "$RUN/fixtures" ]]; then mkdir "$RUN/fixtures"; fi
        test -d "$RUN/fixtures" && test ! -L "$RUN/fixtures"
        FIXTURE_DIR="$RUN/fixtures"
        "$PYTHON" - "$FIXTURE_DIR" <<'PY' 2>&1 | tee -a "$RUN/results/fixture.log"
from pathlib import Path
import sys
from dev.gemv_cuda.prepare_fixtures import export
from dev.gemv_ppu.smallm import SHAPES
from dev.gemv_ppu.run import read_fixture
out=Path(sys.argv[1])
for n,k in SHAPES:
    path=out/f'q12-n{n}-k{k}-e1-c1.npz'
    if path.exists():
        read_fixture(path)
        print(f'Q4_SMALLM_FIXTURE reuse={path.name}',flush=True)
    else:
        export(out,12,n,k,1,[0],1)
PY
    fi
    EXTRA=()
    [[ ${ACU:-1} == 0 || ${ACU:-1} == 1 ]]
    if [[ ${ACU:-1} == 0 ]]; then
        EXTRA+=(--skip-acu)
    else
        ACU_BIN=${ACU_BIN:-$SDK/asight/bin/acu}
        test -x "$ACU_BIN"
        EXTRA+=(--acu "$ACU_BIN")
    fi
    if [[ -n ${L2_BYTES:-} ]]; then EXTRA+=(--l2-bytes "$L2_BYTES"); fi
    stage=compare
    printf 'Q4_SMALLM_BOX run=%s cases=42 cache=ROTATING_ONLY first_launch=EXCLUDED production=UNCHANGED\n' "$RUN"
    printf 'TC=current-policy-and-five-parent-scan split=1,2,4,8 reducer=INCLUDED SIMT=single-multirow-launch\n'
    "$PYTHON" -u dev/gemv_ppu/run_smallm.py --sdk "$SDK" --fixtures "$FIXTURE_DIR" \
        --output "$RUN/results" "${EXTRA[@]}" 2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
