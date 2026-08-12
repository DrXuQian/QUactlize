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
grep -q 'real-production-destination=BIJECTIVE compact-negative=RED first32-negative=RED result=PASS' "$out/l143.out"
