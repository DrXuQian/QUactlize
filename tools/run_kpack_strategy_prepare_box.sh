#!/usr/bin/env bash
# Independent preparatory work on another machine, NOT a full timing campaign.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'Run with bash, not source.' >&2
    return 2
fi
set -euo pipefail
prep_task=${1:-}
case "$prep_task" in
    prefill-resources) prep_cohort=prefill; prep_max=160 ;;
    device-pilot) prep_cohort=pilot; prep_max=64 ;;
    *) printf 'Usage: bash %s {prefill-resources|device-pilot}\n' "$0" >&2; exit 2 ;;
esac
[[ $# == 1 ]]
prep_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
prep_python=$(command -v "${PYTHON:-python3}")
prep_root=$(realpath -e -- "${RUN_ROOT:-/workspace}")
test -d "$prep_root"
if [[ -n "${RESUME_DIR:-}" ]]; then
    prep_run=$(realpath -e -- "$RESUME_DIR")
    if [[ ! -f "$prep_run/task.json" ]]; then
        printf 'MISSING %s/task.json; RESUME_DIR must name a preparation run.\n' "$prep_run" >&2
        exit 2
    fi
else
    prep_run=$(mktemp -d "${prep_root:?}/kpack-rule-prepare.XXXXXXXX")
fi
stage=precheck
finish() {
    prep_rc=$?
    trap - EXIT
    "$prep_python" - "$prep_run" "$prep_rc" "$stage" <<'PY'
from pathlib import Path
import io,json,sys,tarfile,time
root=Path(sys.argv[1]).resolve(strict=True)
archive=Path(str(root)+'.'+str(time.time_ns())+'.results.tgz')
with tarfile.open(archive,'x:gz') as out:
    payload=json.dumps(dict(runner_rc=int(sys.argv[2]),stage=sys.argv[3])).encode()
    entry=tarfile.TarInfo('runner-status.json');entry.size=len(payload)
    out.addfile(entry,io.BytesIO(payload))
    for p in sorted(root.rglob('*')):
        if (p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) and
                p.suffix in ('.json','.log','.txt')):
            out.add(p,arcname=p.relative_to(root),recursive=False)
print('results='+str(archive),flush=True)
PY
    printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$prep_rc" "$stage"
    exit "$prep_rc"
}
trap finish EXIT
cd -- "${prep_source:?}"
"$prep_python" - "$prep_run" "$prep_task" "$prep_cohort" <<'PY'
from pathlib import Path
import json,sys
from dev.strategy.journal import atomic_json
path=Path(sys.argv[1])/'task.json'
record=dict(task=sys.argv[2],cohort=sys.argv[3],production_changed=False,performance_admitted=False)
if path.exists() and json.loads(path.read_text())!=record:
    raise ValueError('resume task differs; use a new output directory')
if not path.exists():atomic_json(path,record)
PY
printf 'STRATEGY_PREPARE_START task=%s run=%s scope=PREPARATION_NOT_FULL_TIMING\n' "$prep_task" "$prep_run"
if [[ "${PLAN_ONLY:-0}" != 1 ]]; then
    : "${CUDA_VISIBLE_DEVICES:?Select exactly one PPU card}"
    if [[ "$prep_task" == device-pilot && "${IDLE_CONFIRMED:-0}" != 1 ]]; then
        printf '%s\n' 'Device execution requires an idle card; set IDLE_CONFIRMED=1.' >&2
        exit 2
    fi
    prep_sdk=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
    for entry in bin/hgcc bin/hgobjdump include/hggc_runtime_api.h \
                 lib/libhggc_wrapper.so lib/libhggcrt1.so lib/libhggc.so lib/libhg_wrapper.so; do
        if [[ ! -f "$prep_sdk/$entry" ]]; then
            printf 'MISSING %s\n' "$prep_sdk/$entry" >&2
            exit 1
        fi
    done
    export LD_LIBRARY_PATH="$prep_sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
stage=blind-plan
if [[ ! -f "$prep_run/plan/plan.json" ]]; then
    if "$prep_python" -u tools/audit_kpack_rules.py --cohort "$prep_cohort" \
        --output "$prep_run/plan" 2>&1 | tee -a "$prep_run/plan.log"; then
        :
    else
        prep_audit_rc=$?
        # A measured-rule recall miss is retained as evidence. It is not an
        # infrastructure exception to hide or a performance win.
        [[ "$prep_audit_rc" == 2 && -s "$prep_run/plan/audit.json" ]] || exit "$prep_audit_rc"
    fi
fi
prep_args=(--plan "$prep_run/plan/plan.json" --output "$prep_run/resources"
    --cache "$prep_run/cache" --jobs "${JOBS:-192}" --max-modules "$prep_max")
if [[ -d "$prep_run/resources" ]]; then prep_args+=(--resume); fi
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    stage=plan-only
    "$prep_python" -u tools/probe_kpack_strategy.py "${prep_args[@]}" --plan-only \
        2>&1 | tee -a "$prep_run/resources.log"
    printf 'STRATEGY_PREPARE_PLAN_ONLY %s/resources/inventory.json\n' "$prep_run"
else
    stage=compile-and-resource-query
    "$prep_python" -u tools/probe_kpack_strategy.py "${prep_args[@]}" --sdk "$prep_sdk" \
        2>&1 | tee -a "$prep_run/resources.log"
    if [[ "$prep_task" == device-pilot ]]; then
        stage=tc-execution-smoke
        prep_pilot_resume=()
        if [[ -d "$prep_run/pilot" ]]; then prep_pilot_resume=(--resume); fi
        "$prep_python" -u tools/run_kpack_strategy_pilot.py --resource-run "$prep_run" \
            --output "$prep_run/pilot" --sdk "$prep_sdk" --idle-confirmed "${prep_pilot_resume[@]}" \
            2>&1 | tee -a "$prep_run/pilot.log"
        printf '%s\n' 'STRATEGY_PREPARE_DONE scope=WARM_TC_MECHANISM_ONLY_NOT_COLD_SELECTION'
    else
        printf '%s\n' 'STRATEGY_PREPARE_DONE scope=PREFILL_TC_RESOURCE_ONLY kernel_launches=0 performance_proven=0'
    fi
    printf 'RESOURCE_RUN=%s\nRESOURCE_CACHE=%s/cache\n' "$prep_run" "$prep_run"
fi
stage=complete
