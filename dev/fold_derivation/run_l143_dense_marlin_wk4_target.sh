#!/usr/bin/env bash
# Historical name retained because CI and the box runner invoke it.  The
# authority is now the standalone Marlin stack, not the retired generic WK4
# compatibility path that L140-L155 used to authorize.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

run_and_tail() {
  local label="$1"; shift
  local log
  log="$(mktemp "${TMPDIR:-/tmp}/quactlize-${label}.XXXXXX")"
  if ! "$@" >"${log}" 2>&1; then
    cat "${log}" >&2
    rm -f "${log}"
    echo "[l143] FAIL: ${label}" >&2
    exit 1
  fi
  tail -n 1 "${log}"
  rm -f "${log}"
}

run_and_tail l167 bash "$repo/dev/fold_derivation/run_l167_classic_marlin_format.sh"
run_and_tail l168 bash "$repo/dev/fold_derivation/run_l168_marlin_pipeline_trace.sh"
run_and_tail l169 bash "$repo/dev/fold_derivation/run_l169_standalone_marlin_unit.sh"
run_and_tail l170 bash "$repo/dev/fold_derivation/run_l170_standalone_marlin_scheduler.sh"
run_and_tail contract python3 "$repo/ci/check_dense_marlin_wk4_target.py"
run_and_tail profile python3 "$repo/ci/check_classic_marlin_156_profile.py"
bash -n "$repo/tools/run_dense_marlin_wk4_box.sh"

echo '[l143] PASS: standalone Marlin format + cadence + generated type + scheduler lifecycle; generic WK4 compatibility is absent; no device result claimed'
