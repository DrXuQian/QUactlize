#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L123_OUT:-/tmp/quactlize_l123}"
mkdir -p "${out}"
inc=(-I "${repo}/dev/fold_derivation/stub_inc"
     -I "${repo}/third_party/actlize/include"
     -I "${repo}/third_party/actlize/tools/util/include"
     -I "${repo}/quactlize/include" -I "${repo}/tests" -I "${repo}/benchmarks")
base=(nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}")
src="${repo}/dev/fold_derivation/l123_warp_nk_topology.cu"
"${base[@]}" -D__HGGCCC__ --expt-relaxed-constexpr -DL123_TYPE_ONLY=1 \
  -o "${out}/types" "${src}"
if "${base[@]}" -D__HGGCCC__ --expt-relaxed-constexpr -DL123_TYPE_ONLY=1 \
    -DL123_BREAK_PERMK=1 -o "${out}/bad_permk" "${src}" >"${out}/bad_permk.log" 2>&1; then
  echo "L123 negative: stale PermutationK unexpectedly compiled" >&2; exit 1
fi
grep -q "warp-K and PermutationK must change together" "${out}/bad_permk.log"
"${base[@]}" -o "${out}/topology" "${src}"
"${out}/types"
echo "L123 negative: AtomLayout.K changed without PermutationK was rejected PASS"
"${out}/topology"
