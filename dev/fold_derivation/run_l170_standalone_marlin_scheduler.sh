#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L170_OUT:-/tmp/quactlize_l170}"
mkdir -p "${out}"

flags=(-std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr -D__HGGCCC__
       -I "${repo}/dev/fold_derivation/stub_inc"
       -I "${repo}/third_party/actlize/include"
       -I "${repo}/third_party/actlize/tools/util/include"
       -I "${repo}/quactlize/include")
nvcc "${flags[@]}" -o "${out}/l170" \
  "${repo}/dev/fold_derivation/l170_standalone_marlin_scheduler.cpp"
"${out}/l170"

for compile_plant in LEGACY_44_DESCRIPTOR RECOMPUTED_PREDICATE; do
  log="${out}/compile-${compile_plant}.log"
  set +e
  nvcc "${flags[@]}" -c -DL170_PLANT_${compile_plant}=1 \
    -o "${out}/compile-${compile_plant}.o" \
    "${repo}/dev/fold_derivation/l170_standalone_marlin_scheduler.cpp" \
    >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    echo "[l170] FAIL: compile plant ${compile_plant} was accepted" >&2
    exit 1
  fi
done
grep -Fq 'L170 rejects the legacy 44-byte work descriptor' \
  "${out}/compile-LEGACY_44_DESCRIPTOR.log" || {
    cat "${out}/compile-LEGACY_44_DESCRIPTOR.log" >&2
    echo '[l170] FAIL: legacy descriptor plant missed its ABI assertion' >&2
    exit 1
  }
grep -Fq 'L170 requires the cached split predicate, not recomputation' \
  "${out}/compile-RECOMPUTED_PREDICATE.log" || {
    cat "${out}/compile-RECOMPUTED_PREDICATE.log" >&2
    echo '[l170] FAIL: recomputed-predicate plant missed its cache assertion' >&2
    exit 1
  }

for plant in cell-hole forward-q wrong-slice aliased-lock no-reset early-reset inactive-valid; do
  log="${out}/${plant}.log"
  set +e
  "${out}/l170" --plant="${plant}" >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 1 ]] ||
     ! grep -Fq "[l170:red] plant=${plant} caught=1" "${log}"; then
    cat "${log}" >&2
    echo "[l170] FAIL: ${plant} did not produce its named causal RED" >&2
    exit 1
  fi
done

echo '[l170:descriptor] hot=20B legacy-44B=RED local-lock=RED recomputed-predicate=RED'
echo '[l170:runner] positive=PASS negative_controls=7/7_RED result=PASS'
