#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L140_OUT:-/tmp/quactlize_l140_warpk_tactic_axis}"
mkdir -p "${out}"
src="${repo}/dev/fold_derivation/l140_warpk_tactic_axis.cu"
flags=(-std=c++17 -arch=sm_80 -w --expt-relaxed-constexpr -D__HGGCCC__
       -I "${repo}/dev/fold_derivation/stub_inc"
       -I "${repo}/third_party/actlize/include"
       -I "${repo}/third_party/actlize/tools/util/include"
       -I "${repo}/quactlize/include"
       -I "${repo}/tests" -I "${repo}/benchmarks")

nvcc "${flags[@]}" -o "${out}/positive" "${src}"
"${out}/positive"

for plant in L140_DROP_K_COHORT_PLANT L140_ACCEPT_FOLDED_WK_PLANT; do
  log="${out}/${plant}.log"
  set +e
  nvcc "${flags[@]}" -D"${plant}"=1 -o "${out}/${plant}" "${src}" >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -eq 0 ]] || ! grep -q 'L140 deliberate regression' "${log}"; then
    echo "[l140] FAIL: ${plant} did not turn red" >&2
    sed -n '1,24p' "${log}" >&2
    exit 1
  fi
done

echo '[l140] PASS: old Cfg exact; 2N x 4K is 256 threads; missing-K and folded-WK plants red'
