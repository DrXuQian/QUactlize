#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L155_OUT:-/tmp/quactlize_l155_wk4_indexed_converter}"
mkdir -p "$out"

inc=(-I "$repo/dev/fold_derivation/stub_inc"
     -I "$repo/third_party/actlize/include"
     -I "$repo/third_party/actlize/tools/util/include"
     -I "$repo/quactlize/include")
base=(nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}")
src="$repo/dev/fold_derivation/l155_wk4_indexed_converter.cu"

"${base[@]}" -o "$out/positive" "$src"
"$out/positive" | tee "$out/positive.out"
grep -qx 'L155 indexed-vs-template outputs=128 bad=0 select-word-bad=0' "$out/positive.out"
grep -qx 'L155 planted high-bit=64 phase-bit=64 destination=64 EXPECTED-RED' "$out/positive.out"

for spec in \
  'HIGH_BIT:wk low bit selected the vreg column' \
  'PHASE_BIT:wk high bit selected the byte phase' \
  'DESTINATION:vi/ti destination axes were transposed'; do
  name="${spec%%:*}"
  reason="${spec#*:}"
  "${base[@]}" -D"L155_BAD_${name}=1" -o "$out/bad_${name}" "$src"
  set +e
  "$out/bad_${name}" >"$out/bad_${name}.out" 2>&1
  rc=$?
  set -e
  test "$rc" -eq 1
  grep -qx "L155 EXPECTED-FAIL: $reason" "$out/bad_${name}.out"
done

echo 'L155 negative controls: high-bit/phase-bit/destination each exact-red PASS'
