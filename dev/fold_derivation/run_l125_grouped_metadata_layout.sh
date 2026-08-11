#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L125_OUT:-/tmp/quactlize_l125}"
mkdir -p "${out}"

inc=(-I "${repo}/dev/fold_derivation/stub_inc"
     -I "${repo}/third_party/actlize/include"
     -I "${repo}/third_party/actlize/tools/util/include"
     -I "${repo}/quactlize/include" -I "${repo}/tests" -I "${repo}/benchmarks")
base=(nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}")
src="${repo}/dev/fold_derivation/l125_grouped_metadata_layout.cu"

"${base[@]}" -D__HGGCCC__ --expt-relaxed-constexpr -DL125_TYPE_ONLY=1 \
  -o "${out}/types" "${src}"
"${base[@]}" -o "${out}/layout" "${src}"

if "${base[@]}" -D__HGGCCC__ --expt-relaxed-constexpr -DL125_TYPE_ONLY=1 \
     -DL125_SELECTED_WN=16 -o "${out}/wrong_type" "${src}" \
     >"${out}/wrong_type.log" 2>&1; then
  echo "L125 negative: wrong WN16 G5 type unexpectedly compiled" >&2
  exit 1
fi
grep -q "L125 selected policy is not the shipping G5 metadata type" \
  "${out}/wrong_type.log"

"${out}/types"
"${out}/layout"
echo "L125 compiled negative: wrong legal WN16 instance rejected by the shipping-type assertion PASS"
