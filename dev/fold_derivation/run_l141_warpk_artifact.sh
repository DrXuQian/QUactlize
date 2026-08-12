#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L141_OUT:-/tmp/quactlize_l141}"
mkdir -p "${out}"

# L142/L143 own the independent production-layout proof that the two-source
# consumer can read the shipping bytes directly.  L141 pins the explicit
# writer/reader API to that result and keeps unproved formats fail-closed.
grep -q 'derive_two_source_map' "${repo}/dev/fold_derivation/l138_wk_shadow_delivery.cu"
grep -q 'two-source-availability' "${repo}/dev/fold_derivation/l138_wk_shadow_delivery.cu"
grep -q 'diff(w1,b)==0' "${repo}/dev/fold_derivation/l123_warp_nk_topology.cu"
grep -q 'DirectShippingPairs' "${repo}/dev/fold_derivation/l142_production_destination_map.cu"
grep -q 'DirectResult direct_pair_scatter' "${repo}/dev/fold_derivation/l143_wk4_production_delivery.cu"

inc=(-I "${repo}/dev/fold_derivation/stub_inc"
     -I "${repo}/third_party/actlize/include"
     -I "${repo}/third_party/actlize/tools/util/include"
     -I "${repo}/quactlize/include"
     -I "${repo}/tests"
     -I "${repo}/benchmarks")

nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}" \
  -o "${out}/l141_warpk_artifact" \
  "${repo}/dev/fold_derivation/l141_warpk_artifact.cpp"

"${out}/l141_warpk_artifact" | tee "${out}/l141_warpk_artifact.out"
grep -q 'WK1=SHIPPING-BYTE-IDENTICAL WK4-CONSUMER=SHIPPING-BYTE-IDENTICAL+ROUNDTRIP result=PASS' \
  "${out}/l141_warpk_artifact.out"

if nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}" \
    -DL141_NEGATIVE_UNPROVED_FORMAT=1 -c \
    -o "${out}/l141_unproved_format.o" \
    "${repo}/dev/fold_derivation/l141_warpk_artifact.cpp" \
    >"${out}/l141_unproved_format.log" 2>&1; then
  echo "L141 FAIL: an unproved int2/WK4 consumer topology compiled" >&2
  exit 1
fi
grep -q 'non-default WarpK consumer mapping is proved only for the shipping-map ordinary-int4 2N x 4K target' \
  "${out}/l141_unproved_format.log"
echo "L141 unproved int2/WK4 consumer topology: EXPECTED-FAIL"
