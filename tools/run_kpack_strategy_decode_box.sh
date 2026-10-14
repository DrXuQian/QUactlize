#!/usr/bin/env bash
# Experimental five-point cold screen; no installed library/production writes.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'Run with bash, not source.' >&2
    return 2
fi
set -euo pipefail
decode_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
decode_python=$(command -v "${PYTHON:-python3}")
: "${PROBE_RUN:?Set the completed resource probe directory}"
decode_probe=$(realpath -e -- "$PROBE_RUN")
test -f "$decode_probe/resources/summary.json"
decode_root=$(realpath -e -- "${RUN_ROOT:-/workspace}")
test -d "${decode_root:?}"
decode_resume=()
if [[ -n "${RESUME_DIR:-}" ]]; then
    decode_run=$(realpath -e -- "$RESUME_DIR")
    test -f "$decode_run/decode/plan.json"
    decode_resume=(--resume)
else
    decode_run=$(mktemp -d "${decode_root:?}/kpack-rule-decode.XXXXXXXX")
fi
stage=precheck
finish() {
    decode_rc=$?
    trap - EXIT
    "$decode_python" - "$decode_run" "$decode_rc" "$stage" <<'PY'
from pathlib import Path
import io,json,sys,tarfile,time
root=Path(sys.argv[1]).resolve(strict=True)
archive=Path(str(root)+'.'+str(time.time_ns())+'.results.tgz')
with tarfile.open(archive,'x:gz') as out:
    data=json.dumps(dict(runner_rc=int(sys.argv[2]),stage=sys.argv[3])).encode()
    entry=tarfile.TarInfo('runner-status.json');entry.size=len(data);out.addfile(entry,io.BytesIO(data))
    for path in sorted(root.rglob('*')):
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root) and path.suffix in ('.json','.log','.txt'):
            out.add(path,arcname=path.relative_to(root),recursive=False)
print('results='+str(archive),flush=True)
PY
    printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$decode_rc" "$stage"
    exit "$decode_rc"
}
trap finish EXIT
cd -- "${decode_source:?}"
decode_args=(--resource-run "$decode_probe" --output "$decode_run/decode" "${decode_resume[@]}")
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    stage=plan-only
    "$decode_python" -u tools/run_kpack_strategy_decode.py "${decode_args[@]}" --plan-only
else
    : "${CUDA_VISIBLE_DEVICES:?Select the same single card as the resource probe}"
    : "${GATE_RUNS:?Colon-separated TC pilot run directories, including the successful retry}"
    if [[ "${IDLE_CONFIRMED:-0}" != 1 ]]; then
        printf '%s\n' 'Confirm the card is idle, then set IDLE_CONFIRMED=1.' >&2
        exit 2
    fi
    IFS=: read -r -a decode_gates <<< "$GATE_RUNS"
    for decode_gate in "${decode_gates[@]}"; do
        decode_gate=$(realpath -e -- "$decode_gate")
        if [[ -f "$decode_gate/pilot/plan.json" ]]; then decode_gate="$decode_gate/pilot"; fi
        test -f "$decode_gate/plan.json"
        decode_args+=(--execution-run "$decode_gate")
    done
    decode_sdk=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
    export LD_LIBRARY_PATH="$decode_sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    stage=cold-smallm-screen
    "$decode_python" -u tools/run_kpack_strategy_decode.py "${decode_args[@]}" --sdk "$decode_sdk" \
        --jobs "${JOBS:-16}" --l2-bytes "${L2_BYTES:-0}" --idle-confirmed \
        2>&1 | tee -a "$decode_run/console.log"
fi
stage=complete
printf 'STRATEGY_DECODE_RESULTS %s/decode/summary.json\n' "$decode_run"
