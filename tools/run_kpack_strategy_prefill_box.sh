#!/usr/bin/env bash
# Full registered large-M component screen. Each machine/card owns whole families.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'Run with bash, not source.' >&2
    return 2
fi
set -euo pipefail
prefill_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
prefill_python=$(command -v "${PYTHON:-python3}")
prefill_root=$(realpath -e -- "${RUN_ROOT:-/workspace}")
test -d "$prefill_root"
if [[ -n "${RESUME_DIR:-}" ]]; then
    prefill_run=$(realpath -e -- "$RESUME_DIR")
    test -f "$prefill_run/measurement/plan/plan.json"
    prefill_resume=(--resume)
else
    prefill_run=$(mktemp -d "${prefill_root:?}/kpack-rule-prefill.XXXXXXXX")
    prefill_resume=()
fi
stage=precheck
finish() {
    prefill_rc=$?
    trap - EXIT
    if [[ -f "$prefill_run/measurement/summary.json" && -f "$prefill_run/measurement/authority.json" ]]; then
        "$prefill_python" "$prefill_source/tools/pack_kpack_strategy_review.py" "$prefill_run"
    else
    "$prefill_python" - "$prefill_run" "$prefill_rc" "$stage" <<'PY'
from pathlib import Path
import io,json,sys,tarfile,time
root=Path(sys.argv[1]).resolve(strict=True)
archive=Path(str(root)+'.'+str(time.time_ns())+'.setup-errors.tgz')
with tarfile.open(archive,'x:gz') as out:
    data=json.dumps(dict(runner_rc=int(sys.argv[2]),stage=sys.argv[3])).encode()
    entry=tarfile.TarInfo('runner-status.json');entry.size=len(data)
    out.addfile(entry,io.BytesIO(data))
    paths=[root/'console.log',root/'measurement/counts.json']
    paths+=sorted((root/'measurement').glob('*.log'))[-32:]
    paths+=sorted((root/'measurement/build-attempts').glob('*/*.log'))[-32:]
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):continue
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-65536));data=stream.read(65536)
        name=str(path.relative_to(root))+('.tail.txt' if path.suffix=='.log' else '')
        entry=tarfile.TarInfo(name);entry.size=len(data);out.addfile(entry,io.BytesIO(data))
print('results='+str(archive),flush=True)
PY
    fi
    printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$prefill_rc" "$stage"
    exit "$prefill_rc"
}
trap finish EXIT
cd -- "${prefill_source:?}"
prefill_args=(--output "$prefill_run/measurement" --jobs "${JOBS:-192}" --hours "${HOURS:-10}"
    --shard "${SHARD:-0}" --shards "${SHARDS:-1}" --l2-bytes "${L2_BYTES:-0}" "${prefill_resume[@]}")
if [[ -n "${PREVIOUS_RUN:-}" ]]; then
    prefill_previous=$(realpath -e -- "$PREVIOUS_RUN")
    prefill_args+=(--previous-output "$prefill_previous")
fi
if [[ "${CUBLAS_EVENT_UPGRADE:-0}" == 1 ]]; then
    prefill_args+=(--upgrade-cublas-events)
fi
if [[ -n "${RESOURCE_CACHE:-}" ]]; then
    prefill_cache=$(realpath -e -- "$RESOURCE_CACHE")
    test -d "$prefill_cache"
    prefill_args+=(--cache "$prefill_cache")
fi
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    stage=plan-only
    prefill_args+=(--plan-only)
else
    : "${CUDA_VISIBLE_DEVICES:?Select exactly one idle physical PPU card}"
    [[ "${IDLE_CONFIRMED:-0}" == 1 ]] || { printf '%s\n' 'Set IDLE_CONFIRMED=1 only on an idle card.' >&2; exit 2; }
    prefill_sdk=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
    for path in bin/hgcc bin/hgobjdump include/hggc_runtime_api.h lib/libhggc_wrapper.so \
                lib/libhggcrt1.so lib/libhggc.so lib/libhg_wrapper.so; do
        if [[ ! -f "$prefill_sdk/$path" ]]; then printf 'MISSING %s\n' "$prefill_sdk/$path" >&2; exit 1; fi
    done
    export LD_LIBRARY_PATH="$prefill_sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    prefill_args+=(--sdk "$prefill_sdk" --idle-confirmed)
    stage=large-m-components
fi
printf 'STRATEGY_PREFILL_START run=%s shard=%s/%s model_admission=0\n' "$prefill_run" "${SHARD:-0}" "${SHARDS:-1}"
"$prefill_python" -u tools/run_kpack_strategy_prefill.py "${prefill_args[@]}" 2>&1 | tee -a "$prefill_run/console.log"
stage=complete
