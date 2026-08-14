#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
oracle="${repo}/dev/fold_derivation/l174_marlin_compute_contract.py"
out="${QUACTLIZE_L174_OUT:-/workspace/quactlize-l174}"
mkdir -p "${out}"

python3 "${oracle}"

plants=(
  runtime-dispatch wrong-dequant-constant wrong-nblock-order wrong-helper-order
  m8-x4-fallback m8-discarded-destinations m8-padded-a
  m8-broadens-m
)
for plant in "${plants[@]}"; do
  log="${out}/${plant}.log"
  set +e
  python3 "${oracle}" --plant="${plant}" >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 1 ]] ||
     ! grep -Fq "[l174:red] plant=${plant} caught=1" "${log}"; then
    cat "${log}" >&2
    echo "[l174] FAIL: ${plant} did not produce its named causal RED" >&2
    exit 1
  fi
done

echo "[l174:runner] positive=PASS negative_controls=${#plants[@]}/${#plants[@]}_RED result=PASS"
