#!/usr/bin/env bash
# Small prebuilt Q8 experiment; does not rebuild or modify the model runtime.
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
    trap 'printf "Q8_TOPOLOGY FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/dev/gemv_simt/run_q8_topology.py"
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
    SHIPPING=$(realpath -e -- "${SHIPPING_BUNDLE:-/workspace/quactlize-model-artifact-61cbcc4344/prebuilt/ppu0010/kpack-model-runtime-v1}")
    test -n "$SHIPPING" && test -f "$SHIPPING/libquactlize_ppu_execution.so"
    CANDIDATE="$ROOT/prebuilt/ppu0010/q8-topology-v1"
    test -f "$CANDIDATE/manifest.json"
    ACU_BIN=${ACU_BIN:-$SDK/asight/bin/acu}
    test -x "$ACU_BIN"
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    RUN=$(mktemp -d "$RESULT_DIR/q8-topology.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    mkdir "$RUN/results"
    : > "$RUN/console.log"
    stage=host-tests
    "$PYTHON_BIN" -m unittest discover -s "$ROOT/tests" -p test_q8_topology.py -v 2>&1 |
        tee -a "$RUN/results/host-tests.log" "$RUN/console.log"
    stage=verify
    "$PYTHON_BIN" - "$CANDIDATE" "$SHIPPING" <<'PY' 2>&1 | tee -a "$RUN/results/verify.log" "$RUN/console.log"
from pathlib import Path
import sys
from dev.gemv_simt.run_q8_topology import verify
m=verify(Path(sys.argv[1]),Path(sys.argv[2]))
print('Q8_TOPOLOGY VERIFIED candidates=26 shapes=3 cells=290 shipping=IMMUTABLE production=UNCHANGED')
PY
    stage=device
    printf 'Q8_TOPOLOGY START run=%s shapes=3 cells=290 confirm=6x15 profiles=shipping+winner\n' "$RUN"
    printf 'Q8_TOPOLOGY FULL_CALL split=1,2,4,8 reducer=INCLUDED no_compilation=1\n'
    "$PYTHON_BIN" -u dev/gemv_simt/run_q8_topology.py --bundle "$CANDIDATE" \
        --shipping "$SHIPPING" --sdk "$SDK" --output "$RUN/results/sweep" \
        --l2-bytes "${L2_BYTES:-0}" --acu "$ACU_BIN" 2>&1 | tee -a "$RUN/console.log"
    stage=complete
)
