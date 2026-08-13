#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L177_OUT:-/tmp/quactlize_l177}"
mkdir -p "${out}"

flags=(-std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr -D__HGGCCC__
       -I "${repo}/dev/fold_derivation/stub_inc"
       -I "${repo}/third_party/actlize/include"
       -I "${repo}/third_party/actlize/tools/util/include"
       -I "${repo}/quactlize/include")
nvcc "${flags[@]}" -o "${out}/l177" \
  "${repo}/dev/fold_derivation/l177_marlin_handoff_lifecycle.cpp"
"${out}/l177"

for plant in local-q early-reset skip-handoff; do
  log="${out}/${plant}.log"
  set +e
  "${out}/l177" --plant="${plant}" >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 1 ]] ||
     ! grep -Fq "[l177:red] plant=${plant} caught=1" "${log}"; then
    cat "${log}" >&2
    echo "[l177] FAIL: ${plant} did not produce its named causal RED" >&2
    exit 1
  fi
done

echo '[l177:runner] positive=PASS negative_controls=3/3_RED result=PASS'
