#!/usr/bin/env bash
# Known small-M cohort only. Large-M/full-dequant and model A/B remain separate.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'Run with bash, not source.' >&2
    return 2
fi
set -euo pipefail
campaign_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
campaign_python=$(command -v "${PYTHON:-python3}")
: "${SCREEN_RUN:?Set the completed five-point screen directory}"
: "${RESOURCE_CACHE:?Set the existing resource probe cache directory}"
campaign_screen=$(realpath -e -- "$SCREEN_RUN")
campaign_cache=$(realpath -e -- "$RESOURCE_CACHE")
campaign_root=$(realpath -e -- "${RUN_ROOT:-/workspace}")
test -d "$campaign_screen"
test -d "$campaign_cache"
test -d "$campaign_root"
campaign_resume=()
if [[ -n "${RESUME_DIR:-}" ]]; then
    campaign_run=$(realpath -e -- "$RESUME_DIR")
    test -f "$campaign_run/measurement/plan/plan.json"
    campaign_resume=(--resume)
else
    campaign_run=$(mktemp -d "${campaign_root:?}/kpack-rule-campaign.XXXXXXXX")
fi
stage=precheck
finish() {
    campaign_rc=$?
    trap - EXIT
    "$campaign_python" - "$campaign_run" "$campaign_rc" "$stage" <<'PY'
from pathlib import Path
import io,json,sys,tarfile,time
root=Path(sys.argv[1]).resolve(strict=True)
archive=Path(str(root)+'.'+str(time.time_ns())+'.results.tgz')
with tarfile.open(archive,'x:gz') as out:
    data=json.dumps(dict(runner_rc=int(sys.argv[2]),stage=sys.argv[3])).encode()
    entry=tarfile.TarInfo('runner-status.json');entry.size=len(data);out.addfile(entry,io.BytesIO(data))
    # Keep all raw evidence on box. The transfer contains summaries, manifests,
    # components and selected/failed receipts, not tens of thousands of duplicate
    # successful current/attempt snapshots or device binaries.
    for path in sorted(root.rglob('*')):
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root) or
                path.suffix not in ('.json','.log','.txt')):continue
        relative=path.relative_to(root)
        if 'arms' in relative.parts:
            if path.name!='current.json':continue
            row=json.loads(path.read_text())
            if row.get('status')=='PASS' and row.get('identity',{}).get('arm',{}).get('roles')!=['confirmation']:
                continue
        if path.name.startswith('probe-') and path.stat().st_size==0:continue
        out.add(path,arcname=relative,recursive=False)
print('results='+str(archive),flush=True)
PY
    printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$campaign_rc" "$stage"
    exit "$campaign_rc"
}
trap finish EXIT
cd -- "${campaign_source:?}"
campaign_args=(--output "$campaign_run/measurement" --previous-screen "$campaign_screen"
    --resource-cache "$campaign_cache" --jobs "${JOBS:-192}" --hours "${HOURS:-10}"
    "${campaign_resume[@]}")
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    stage=plan-only
    "$campaign_python" -u tools/run_kpack_strategy_campaign.py "${campaign_args[@]}" --plan-only
else
    : "${CUDA_VISIBLE_DEVICES:?Select the same single idle card as the prior screen}"
    if [[ "${IDLE_CONFIRMED:-0}" != 1 ]]; then
        printf '%s\n' 'Confirm no inference/profiler workload on this card; set IDLE_CONFIRMED=1.' >&2
        exit 2
    fi
    campaign_sdk=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
    test -x "$campaign_sdk/bin/hgcc"
    export LD_LIBRARY_PATH="$campaign_sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    stage=smallm-screen-and-confirm
    "$campaign_python" -u tools/run_kpack_strategy_campaign.py "${campaign_args[@]}" --sdk "$campaign_sdk" \
        --l2-bytes "${L2_BYTES:-0}" --idle-confirmed 2>&1 | tee -a "$campaign_run/console.log"
fi
stage=complete
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    printf 'STRATEGY_CAMPAIGN_RESULTS %s/measurement/counts.json\n' "$campaign_run"
else
    printf 'STRATEGY_CAMPAIGN_RESULTS %s/measurement/summary.json\n' "$campaign_run"
fi
