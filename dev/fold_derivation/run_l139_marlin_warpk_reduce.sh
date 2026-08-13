#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L139_OUT:-/tmp/quactlize_l139_marlin_warpk_reduce}"
mkdir -p "${out}"

inc=(-I "${repo}/dev/fold_derivation/stub_inc"
     -I "${repo}/third_party/actlize/include"
     -I "${repo}/third_party/actlize/tools/util/include"
     -I "${repo}/quactlize/include")
base=(nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}")
src="${repo}/dev/fold_derivation/l139_marlin_warpk_reduce.cu"

"${base[@]}" -D__HGGCCC__ --expt-relaxed-constexpr -DL139_TYPE_ONLY=1 \
  -o "${out}/types" "${src}"
"${out}/types" | tee "${out}/types.out"
grep -q "L139 type PASS:" "${out}/types.out"

"${base[@]}" -o "${out}/positive" "${src}"
"${out}/positive" | tee "${out}/positive.out"
grep -q "L139 PASS:" "${out}/positive.out"

for fault in 1 2 3 4 5 6; do
  "${base[@]}" -DL139_FAULT="${fault}" -o "${out}/fault_${fault}" "${src}"
  set +e
  "${out}/fault_${fault}" >"${out}/fault_${fault}.out" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    echo "L139 fault ${fault} unexpectedly stayed green" >&2
    cat "${out}/fault_${fault}.out" >&2
    exit 1
  fi
  grep -q "L139 EXPECTED-RED fault=${fault}" "${out}/fault_${fault}.out"
done

grep -q 'output-owners fault=4 cta_threads=256 declared=256 selected=256 stripes=32 tile=2048 arithmetic_mismatch=1' \
  "${out}/fault_4.out"
grep -Eq 'output-owners fault=5 cta_threads=256 declared=64 selected=64 stripes=32 tile=2048 arithmetic_mismatch=0 .*coverage_holes=[1-9][0-9]* coverage_duplicates=[1-9][0-9]*' \
  "${out}/fault_5.out"
grep -q 'output-owners fault=6 cta_threads=256 declared=64 selected=64 stripes=32 tile=2048 arithmetic_mismatch=0 lane_holes=0 lane_duplicates=0 coverage_holes=0 coverage_duplicates=0 association_mismatches=1536' \
  "${out}/fault_6.out"

echo "L139 negative controls: wrong compact map / missing K cohort / non-survivor output / CTA-owner regression / owner alias / generic fragment-order substitution all red PASS"
