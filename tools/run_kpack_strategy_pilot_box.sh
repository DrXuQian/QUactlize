#!/usr/bin/env bash
# Reuse a resource probe. No library build, installation or production change.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'Run with bash, not source.' >&2
    return 2
fi
set -euo pipefail

pilot_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
pilot_python=$(command -v "${PYTHON:-python3}")
: "${PROBE_RUN:?Set PROBE_RUN to the completed kpack-rule-probe directory}"
pilot_probe=$(realpath -e -- "$PROBE_RUN")
test -f "$pilot_probe/resources/summary.json"
pilot_root=$(realpath -e -- "${RUN_ROOT:-/workspace}")
test -d "${pilot_root:?}"
if [[ -n "${RESUME_DIR:-}" ]]; then
    pilot_run=$(realpath -e -- "$RESUME_DIR")
    test -d "${pilot_run:?}" && test -f "$pilot_run/pilot/plan.json"
    pilot_resume=(--resume)
else
    pilot_run=$(mktemp -d "${pilot_root:?}/kpack-rule-exec.XXXXXXXX")
    pilot_resume=()
fi
stage=precheck
finish() {
    pilot_rc=$?
    trap - EXIT
    "$pilot_python" - "$pilot_run" "$pilot_rc" "$stage" <<'PY'
from pathlib import Path
import io,json,sys,tarfile,time
run=Path(sys.argv[1]).resolve(strict=True)
archive=Path(str(run)+'.'+str(time.time_ns())+'.results.tgz')
with tarfile.open(archive,'x:gz') as out:
    status=json.dumps(dict(runner_rc=int(sys.argv[2]),stage=sys.argv[3])).encode()
    entry=tarfile.TarInfo('runner-status.json');entry.size=len(status)
    out.addfile(entry,io.BytesIO(status))
    for p in sorted(run.rglob('*')):
        if (p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(run)
                and p.suffix in ('.json','.log','.txt')):
            out.add(p,arcname=p.relative_to(run),recursive=False)
print('results='+str(archive),flush=True)
PY
    printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$pilot_rc" "$stage"
    exit "$pilot_rc"
}
trap finish EXIT
cd -- "${pilot_source:?}"
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    stage=plan-only
    "$pilot_python" -u tools/run_kpack_strategy_pilot.py --resource-run "$pilot_probe" \
        --output "$pilot_run/pilot" --plan-only
else
    : "${CUDA_VISIBLE_DEVICES:?Select the same single card as the resource probe}"
    if [[ "${IDLE_CONFIRMED:-0}" != 1 ]]; then
        printf '%s\n' 'Confirm the card is idle, then set IDLE_CONFIRMED=1.' >&2
        exit 2
    fi
    pilot_sdk=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
    export LD_LIBRARY_PATH="$pilot_sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    stage=execution-and-isolated-timer
    "$pilot_python" -u tools/run_kpack_strategy_pilot.py --resource-run "$pilot_probe" \
        --sdk "$pilot_sdk" --output "$pilot_run/pilot" --idle-confirmed \
        "${pilot_resume[@]}" 2>&1 | tee -a "$pilot_run/console.log"
fi
stage=complete
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    printf 'STRATEGY_EXEC_PLAN_ONLY %s/pilot/plan.json\n' "$pilot_run"
else
    printf 'STRATEGY_EXEC_RESULTS %s/pilot/summary.json\n' "$pilot_run"
fi
