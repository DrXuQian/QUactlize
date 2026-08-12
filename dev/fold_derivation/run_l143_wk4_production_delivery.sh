#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L143_OUT:-/tmp/quactlize_l143}"
mkdir -p "$out"
inc=(-I "$repo/dev/fold_derivation/stub_inc"
     -I "$repo/third_party/actlize/include"
     -I "$repo/third_party/actlize/tools/util/include"
     -I "$repo/quactlize/include")
nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}" -o "$out/l143" \
  "$repo/dev/fold_derivation/l143_wk4_production_delivery.cu"
"$out/l143" | tee "$out/l143.out"
grep -q 'direct-pair pairs=8192/8192 codes=16384/16384 destinations=8192/8192 bad-pairs=0 formula-mismatch=0 bad-fragments=0 map-diff=0' "$out/l143.out"
grep -q 'production-order hash=ea96e6b4155759c3 .* EXPECTED-RED' "$out/l143.out"
grep -q 'compact-order hash=17dfe6248fc38143 .* EXPECTED-RED' "$out/l143.out"
grep -q 'adjacent-nibble .* EXPECTED-RED' "$out/l143.out"
grep -q 'swapped-sources .* EXPECTED-RED' "$out/l143.out"
grep -q 'WK1 shipping map-diff=0 byte-diff=0 result=BIT-IDENTICAL' "$out/l143.out"
grep -q 'production-vs-compact-source-fragment-diff=0' "$out/l143.out"
grep -q 'shipping-pair-scatter=EXACT artifact-order=RED compact-order=RED first32=RED wrong-pair=RED source-swap=RED WK1-BYTES=UNCHANGED result=PASS' "$out/l143.out"
