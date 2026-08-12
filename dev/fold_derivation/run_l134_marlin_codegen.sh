#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
src="${repo}/dev/fold_derivation/l134_marlin_codegen.cu"
out="${QUACTLIZE_L134_OUT:-/tmp/quactlize_l134}"
mkdir -p "${out}"
flags=(-std=c++17 -arch=sm_80 -w --expt-relaxed-constexpr
       -I "${repo}/dev/fold_derivation/stub_inc"
       -I "${repo}/third_party/actlize/include"
       -I "${repo}/third_party/actlize/tools/util/include"
       -I "${repo}/quactlize/include"
       -I "${repo}/tests" -I "${repo}/benchmarks")

nvcc "${flags[@]}" -ptx -o "${out}/l134.ptx" "${src}"
test -s "${out}/l134.ptx"

wrong="${out}/wrong.log"
set +e
nvcc "${flags[@]}" -DL134_WRONG_EXPECTATION=1 -ptx \
  -o "${out}/wrong.ptx" "${src}" >"${wrong}" 2>&1
wrong_rc=$?
set -e
if [[ ${wrong_rc} -eq 0 ]] ||
   ! grep -q 'deliberate wrong expectation' "${wrong}" ||
   ! grep -Eq '13.*12|12.*13' "${wrong}"; then
  echo "[l134] FAIL: wrong constexpr expectation did not expose actual I=13" >&2
  sed -n '1,20p' "${wrong}" >&2
  exit 1
fi

raw="${out}/raw.log"
set +e
nvcc "${flags[@]}" -DL134_RAW_CORE_PLANT=1 -ptx \
  -o "${out}/raw.ptx" "${src}" >"${raw}" 2>&1
raw_rc=$?
set -e
if [[ ${raw_rc} -eq 0 ]] ||
   ! grep -q 'may not bypass production raw-shape lowering' "${raw}"; then
  echo "[l134] FAIL: raw-core substitution did not fail the production-Cfg binding" >&2
  sed -n '1,20p' "${raw}" >&2
  exit 1
fi

bpc="${out}/bpc.log"
set +e
nvcc "${flags[@]}" -DL134_IGNORE_BLOCKS_PER_CU_PLANT=1 -ptx \
  -o "${out}/bpc.ptx" "${src}" >"${bpc}" 2>&1
bpc_rc=$?
set -e
if [[ ${bpc_rc} -eq 0 ]] ||
   ! grep -q 'must lower the explicit B=2 launch cohort' "${bpc}"; then
  echo "[l134] FAIL: hardcoded-B1 plant did not fail the production-Cfg binding" >&2
  sed -n '1,20p' "${bpc}" >&2
  exit 1
fi

python3 "${repo}/ci/check_dense_marlin_codegen.py" "${out}/l134.ptx"
echo '[l134] PASS: real dense Cfg constexpr/static_assert + runtime PTX probe; wrong-value, raw-core and hardcoded-B1 controls red'
