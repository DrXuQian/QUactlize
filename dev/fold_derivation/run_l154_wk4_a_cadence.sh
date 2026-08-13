#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L154_OUT:-/tmp/quactlize_l154_wk4_a_cadence}"
mkdir -p "$out"

inc=(-I "$repo/dev/fold_derivation/stub_inc"
     -I "$repo/third_party/actlize/include"
     -I "$repo/third_party/actlize/tools/util/include"
     -I "$repo/quactlize/include")
base=(nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}")
src="$repo/dev/fold_derivation/l154_wk4_a_cadence.cu"

"${base[@]}" -o "$out/positive" "$src"
"$out/positive" | tee "$out/positive.out"
grep -q 'A_K_BLOCKS=2 B_K_BLOCKS=1 K_ATOM_PER_COPY=2' "$out/positive.out"
grep -q 'old-lower64=167,122,141,144,155,166,137,148, device=EXACT' "$out/positive.out"
grep -q 'fixed-all=277,328,283,286,321,292,303,306, golden=EXACT' "$out/positive.out"

"${base[@]}" -DL154_OLD_A_CADENCE=1 -o "$out/old" "$src"
set +e
"$out/old" >"$out/old.out" 2>&1
rc=$?
set -e
test "$rc" -eq 1
grep -q 'L154 EXPECTED-RED: one B block loaded only A block zero' "$out/old.out"

echo 'L154 negative control: old single-A-block cadence reproduces device values and is red PASS'
