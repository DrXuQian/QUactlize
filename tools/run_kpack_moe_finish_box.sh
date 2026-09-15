#!/usr/bin/env bash
# Run a precompiled helper gate; never exit the invoking Docker shell.
(
    set -Eeuo pipefail
    phase=precheck
    RUN=""
    finish() {
        rc=$?
        trap - EXIT
        if [[ -n "$RUN" && -d "$RUN/results" ]]; then
            tar -czf "$RUN.results.tgz" -C "$RUN" results || rc=1
            printf '\nresults=%s.results.tgz\n' "$RUN"
        fi
        printf 'runner_rc=%s phase=%s\nCurrent Docker shell is preserved.\n' "$rc" "$phase"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "MOE_HELPER FAIL phase=%s line=%s rc=%s\n" "$phase" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_moe_finish_box.sh"
    cd "$ROOT"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/envsetup.sh"
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    if [[ ${FETCH_PAYLOADS:-1} == 1 ]]; then
        phase=fetch
        git lfs pull --include='prebuilt/ppu0010/kpack-moe-finish-v1/moe-helper-proof' --exclude=''
    fi
    BUNDLE=$(realpath -e -- "${BUNDLE:-$ROOT/prebuilt/ppu0010/kpack-moe-finish-v1}")
    test -n "$BUNDLE" && test -f "$BUNDLE/manifest.json"
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    RUN=$(mktemp -d "$RESULT_DIR/kpack-moe-finish.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    mkdir "$RUN/results"
    phase=verify
    "$PYTHON" - "$BUNDLE" "$ROOT" <<'PY' | tee "$RUN/results/verify.log"
import hashlib,json,pathlib,sys
bundle=pathlib.Path(sys.argv[1]).resolve(strict=True);root=pathlib.Path(sys.argv[2]).resolve(strict=True)
m=json.loads((bundle/'manifest.json').read_text())
if m['schema']!='quactlize.moe-helper-proof.v1': raise ValueError('unsupported proof manifest')
p=(bundle/m['binary']).resolve(strict=True)
if p.parent!=bundle or hashlib.sha256(p.read_bytes()).hexdigest()!=m['sha256']:
    raise ValueError('proof binary differs from manifest')
for path,want in m['source_hashes'].items():
    source=(root/path).resolve(strict=True)
    if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest()!=want:
        raise ValueError('proof source differs: '+path)
print('MOE_HELPER_PAYLOAD PASS sha256='+m['sha256']+' compile=NONE jit=NONE scope=HELPERS_NOT_MODEL')
PY
    cp "$BUNDLE/manifest.json" "$RUN/results/manifest.json"
    failed=0
    phase=correctness
    if "$BUNDLE/moe-helper-proof" --mixed 2>&1 | tee "$RUN/results/correctness.log"; then :; else failed=1; fi
    if [[ "$failed" == 0 ]]; then
        phase=timing
        if "$BUNDLE/moe-helper-proof" --mixed --benchmark 2>&1 | tee "$RUN/results/timing.log"; then :; else failed=1; fi
    fi
    phase=complete
    printf 'MOE_HELPER_DONE failed=%s scope=HELPERS_ONLY_NO_MODEL_SPEEDUP_CLAIM\n' "$failed"
    test "$failed" == 0
)
