#!/usr/bin/env bash
# A resource-only six-point pilot, NOT the full sweep or a performance gate.
# Invoke with bash, not source; no installed library/model/cache is replaced.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'Run this script with bash; do not source it.' >&2
    return 2
fi
set -euo pipefail

strategy_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
strategy_python=$(command -v "${PYTHON:-python3}")
strategy_output_root=$(realpath -e -- "${RUN_ROOT:-/workspace}")
test -d "${strategy_output_root:?}"
strategy_run=$(mktemp -d "${strategy_output_root:?}/kpack-rule-probe.XXXXXXXX")
test -d "${strategy_run:?}" && test ! -L "${strategy_run:?}"
stage=precheck

finish() {
    strategy_rc=$?
    trap - EXIT
    printf 'runner_rc=%s stage=%s\n' "$strategy_rc" "$stage" > "${strategy_run:?}/runner-status.txt"
    # Keep diagnostics and receipts, not compiled DSOs/objects or any weights.
    "$strategy_python" - "$strategy_run" <<'PY'
from pathlib import Path
import sys,tarfile
run=Path(sys.argv[1]).resolve(strict=True)
archive=Path(str(run)+'.results.tgz')
with tarfile.open(archive,'x:gz') as out:
    for p in sorted(run.rglob('*')):
        if (p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(run)
                and p.suffix in ('.json','.log','.txt','.cu')):
            out.add(p,arcname=p.relative_to(run),recursive=False)
print('results='+str(archive),flush=True)
PY
    printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$strategy_rc" "$stage"
    exit "$strategy_rc"
}
trap finish EXIT

cd -- "${strategy_root:?}"
if [[ "${PLAN_ONLY:-0}" != 1 ]]; then
    : "${CUDA_VISIBLE_DEVICES:?Select exactly one PPU card explicitly}"
    strategy_sdk=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
    for entry in bin/hgcc bin/hgobjdump include/hggc_runtime_api.h \
                 lib/libhggc_wrapper.so lib/libhggcrt1.so lib/libhggc.so lib/libhg_wrapper.so; do
        if [[ ! -f "$strategy_sdk/$entry" ]]; then
            printf 'MISSING %s\n' "$strategy_sdk/$entry" >&2
            exit 1
        fi
    done
    stage=sdk-header-abi
    "$strategy_python" - "$strategy_sdk" <<'PY'
import json,sys
from dev.strategy.isolated import check_sdk_graph_abi
print('STRATEGY_GRAPH_HOST_ABI '+json.dumps(check_sdk_graph_abi(sys.argv[1])),flush=True)
PY
fi

stage=blind-pilot-plan
if "$strategy_python" -u tools/audit_kpack_rules.py --cohort pilot --output "$strategy_run/plan" \
        2>&1 | tee "$strategy_run/plan.log"; then
    :
else
    strategy_audit_rc=$?
    # An explicit recorded RULE_MISS is expected development evidence, not
    # permission to ignore a traceback, missing plan or malformed arguments.
    if [[ "$strategy_audit_rc" != 2 || ! -s "$strategy_run/plan/audit.json" ]]; then
        exit "$strategy_audit_rc"
    fi
fi
stage=resource-plan
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    "$strategy_python" -u tools/probe_kpack_strategy.py --plan "$strategy_run/plan/plan.json" \
        --output "$strategy_run/resources" --max-modules 64 --plan-only 2>&1 | tee "$strategy_run/resources.log"
else
    stage=compile-and-resource-query
    "$strategy_python" -u tools/probe_kpack_strategy.py --plan "$strategy_run/plan/plan.json" \
        --sdk "$strategy_sdk" --output "$strategy_run/resources" --cache "$strategy_run/cache" \
        --jobs "${JOBS:-192}" --max-modules 64 2>&1 | tee "$strategy_run/resources.log"
fi
stage=complete
printf '%s\n' 'STRATEGY_PROBE_COMPLETE scope=RESOURCE_ONLY kernel_launches=0 performance_proven=0 production_changed=0'
