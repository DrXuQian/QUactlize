#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L138_OUT:-/tmp/quactlize_l138}"
mkdir -p "${out}"

# The independent baseline anchor is L123's permanent assertion: its ordinary
# run requires `shipping-anchor=0`.  Keep L138 independent (and fast) while
# making the exact prerequisite explicit in this output; the full local tier
# runs L123 itself.
grep -q 'diff(w1,b)==0' "${repo}/dev/fold_derivation/l123_warp_nk_topology.cu"
grep -q 'shipping-anchor=%zu' "${repo}/dev/fold_derivation/l123_warp_nk_topology.cu"
echo "L138 prerequisite: L123 owns the executable shipping WK1 xplane anchor=0"

inc=(-I "${repo}/dev/fold_derivation/stub_inc"
     -I "${repo}/third_party/actlize/include"
     -I "${repo}/third_party/actlize/tools/util/include"
     -I "${repo}/quactlize/include"
     -I "${repo}/tests"
     -I "${repo}/benchmarks")

nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}" \
  -o "${out}/l138_wk_shadow_delivery" \
  "${repo}/dev/fold_derivation/l138_wk_shadow_delivery.cu"

"${out}/l138_wk_shadow_delivery" | tee "${out}/l138_wk_shadow_delivery.out"
grep -q "shipping-xplane-WK1 anchor=L123+WK4-BIJECTIVE old-K4=EXPECTED-RED wrong-pair=EXPECTED-RED result=PASS" \
  "${out}/l138_wk_shadow_delivery.out"
