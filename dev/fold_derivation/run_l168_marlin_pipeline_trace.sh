#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
marlin_root="$(cd "${repo_root}/.." && pwd)"
out="${QUACTLIZE_L168_OUT:-/tmp/quactlize_l168}"
cxx="${CXX:-g++}"

mkdir -p "${out}"
"${cxx}" -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  "${repo_root}/dev/fold_derivation/l168_marlin_pipeline_trace.cpp" \
  -o "${out}/l168_marlin_pipeline_trace"

"${out}/l168_marlin_pipeline_trace" --marlin-root="${marlin_root}"

for plant in occupancy-grid missing-stage-attempt flat-reduction; do
  log="${out}/${plant}.log"
  set +e
  "${out}/l168_marlin_pipeline_trace" \
    --marlin-root="${marlin_root}" --plant="${plant}" >"${log}" 2>&1
  status=$?
  set -e

  if [[ ${status} -ne 1 ]]; then
    cat "${log}"
    echo "[l168:runner] ${plant}: expected causal RED exit 1, got ${status}" >&2
    exit 1
  fi
  if ! rg -Fq "[l168:red] plant=${plant} caught=1 result=RED" "${log}"; then
    cat "${log}"
    echo "[l168:runner] ${plant}: missing named causal RED witness" >&2
    exit 1
  fi
  cat "${log}"
done

echo "[l168:runner] positive=PASS negative_controls=3/3_RED result=PASS"
