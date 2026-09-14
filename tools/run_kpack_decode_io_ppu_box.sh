#!/usr/bin/env bash
# Execute precompiled endpoint proofs. The invoking Docker shell is retained.
(
    set -Eeuo pipefail
    phase=precheck
    RUN=""
    finish() {
        rc=$?
        trap - EXIT
        if [[ -n "$RUN" && -d "$RUN/results" ]]; then
            if tar -czf "$RUN.results.tgz" -C "$RUN" results; then
                printf '\nresults=%s.results.tgz\n' "$RUN"
            else
                printf 'Archive failed; results remain at %s/results\n' "$RUN" >&2
                rc=1
            fi
        fi
        printf 'runner_rc=%s phase=%s\nCurrent Docker shell is preserved.\n' "$rc" "$phase"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "KPACK_DECODE_IO_BOX FAIL phase=%s line=%s rc=%s\n" "$phase" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_decode_io.py"
    cd "$ROOT"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK" PATH="$SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    if [[ ${FETCH_PAYLOADS:-1} == 1 ]]; then
        phase=fetch
        git lfs pull --include="prebuilt/ppu0010/kpack-decode-io-v1/**,prebuilt/ppu0010/kpack-fusion-v1/libquactlize_ppu_pack.so" --exclude=""
    fi
    BUNDLE=$(realpath -e -- "${BUNDLE:-$ROOT/prebuilt/ppu0010/kpack-decode-io-v1}")
    test -n "$BUNDLE" && test -f "$BUNDLE/manifest.json"
    PACK=$(realpath -e -- "${PACK_LIBRARY:-$ROOT/prebuilt/ppu0010/kpack-fusion-v1/libquactlize_ppu_pack.so}")
    test -n "$PACK" && test -f "$PACK"
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    RUN=$(mktemp -d "$RESULT_DIR/kpack-decode-io.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    mkdir "$RUN/results"
    phase=verify
    "$PYTHON" tools/verify_kpack_dispatch.py "$BUNDLE" --sdk "$SDK" | tee "$RUN/results/verify.log"
    "$PYTHON" - "$BUNDLE" <<'PY'
import json, pathlib, sys
from quactlize.runtime.compiler import sha
root=pathlib.Path(sys.argv[1]).resolve(strict=True)
manifest=json.loads((root/'manifest.json').read_text())
for item in manifest['decode_io_gate']['simt_binaries']:
    path=(root/item['path']).resolve(strict=True)
    if path.parent != root or sha(path)!=item['sha256']:
        raise ValueError('SIMT proof payload differs')
print('KPACK_DECODE_IO_PAYLOAD PASS compile=NONE jit=NONE')
PY
    phase=pack-library
    "$PYTHON" - "$PACK" 2>&1 <<'PY' | tee "$RUN/results/pack-library.log"
from pathlib import Path
import sys
from tools.run_kpack_moe_gate import load_pack_library
load_pack_library(Path(sys.argv[1]))
PY
    printf 'Use one idle PPU; do not overlap the eight-card cost campaign. run=%s\n' "$RUN"
    failed=0
    phase=indexed-stages
    if "$BUNDLE/kpack_indexed_cuda" 2>&1 | tee "$RUN/results/indexed.log"; then :; else failed=1; fi
    phase=moe-stages
    if "$BUNDLE/kpack_moe_chain_cuda" --multi-token 2>&1 | tee "$RUN/results/moe-stages.log"; then :; else failed=1; fi
    phase=dense-and-moe
    if "$PYTHON" -u tools/run_kpack_decode_io.py --sdk "$SDK" --bundle "$BUNDLE" --output "$RUN/results" \
        --pack-library "$PACK" 2>&1 | tee "$RUN/results/device.log"; then :; else failed=1; fi
    phase=complete
    test "$failed" == 0
)
