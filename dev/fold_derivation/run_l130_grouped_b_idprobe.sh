#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L130_OUT:-/tmp/quactlize_l130}"
mkdir -p "${out}"

inc=(-I "${repo}/dev/fold_derivation/stub_inc"
     -I "${repo}/third_party/actlize/include"
     -I "${repo}/third_party/actlize/tools/util/include"
     -I "${repo}/quactlize/include"
     # legacy_pipeline.hpp intentionally retains its deleted production
     # relative include; this child makes ../unfused_weight_dequantize.hpp
     # resolve to quactlize/include without adding a gate-only shim.
     -I "${repo}/quactlize/include/gemv_lowbit"
     -I "${repo}/tests" -I "${repo}/benchmarks")
base=(nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}")
src="${repo}/dev/fold_derivation/l130_grouped_b_idprobe.cu"

"${base[@]}" -D__HGGCCC__ --expt-relaxed-constexpr -DL130_TYPE_ONLY=1 \
  -o "${out}/types" "${src}"
"${base[@]}" -o "${out}/layout" "${src}"

if "${base[@]}" -D__HGGCCC__ --expt-relaxed-constexpr -DL130_TYPE_ONLY=1 \
     -DL130_SELECTED_WN=16 -o "${out}/wrong_type" "${src}" \
     >"${out}/wrong_type.log" 2>&1; then
  echo "L130 negative: wrong WN16 G5 type unexpectedly compiled" >&2
  exit 1
fi
grep -q "L130 selected policy is not the shipping G5 B type" \
  "${out}/wrong_type.log"

"${out}/types"
"${out}/layout"
echo "L130 compiled negative: wrong legal WN16 instance rejected by the shipping-type assertion PASS"
