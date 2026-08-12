#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L141_OUT:-/tmp/quactlize_l141}"
mkdir -p "${out}"

# L138 owns the independent two-source derivation; L141 turns that exact map
# into a writer/reader API and pins its observed hash. Require that derivation
# and L123's shipping anchor to remain in-tree before accepting this narrower
# format gate.
grep -q 'derive_two_source_map' "${repo}/dev/fold_derivation/l138_wk_shadow_delivery.cu"
grep -q 'two-source-WK4' "${repo}/dev/fold_derivation/l138_wk_shadow_delivery.cu"
grep -q 'diff(w1,b)==0' "${repo}/dev/fold_derivation/l123_warp_nk_topology.cu"

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
grep -q 'WK1=SHIPPING-BYTE-IDENTICAL WK4=BIJECTIVE+ROUNDTRIP stale-WK1=EXPECTED-RED result=PASS' \
  "${out}/l141_warpk_artifact.out"

if nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}" \
    -DL141_NEGATIVE_UNPROVED_FORMAT=1 -c \
    -o "${out}/l141_unproved_format.o" \
    "${repo}/dev/fold_derivation/l141_warpk_artifact.cpp" \
    >"${out}/l141_unproved_format.log" 2>&1; then
  echo "L141 FAIL: an unproved int2/WK4 artifact compiled" >&2
  exit 1
fi
grep -q 'WarpK artifacts are first enabled only for ordinary single-plane int4 F1' \
  "${out}/l141_unproved_format.log"
echo "L141 unproved int2/WK4 artifact: EXPECTED-FAIL"
