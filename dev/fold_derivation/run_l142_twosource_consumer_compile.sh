#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
stub="$root/dev/fold_derivation/stub_inc"
act="$root/third_party/actlize"
src="$root/dev/fold_derivation/l142_twosource_consumer_compile.cu"
map_src="$root/dev/fold_derivation/l142_production_destination_map.cu"
out="${QUACTLIZE_L142_OUT:-/tmp/quactlize_l142}"
mkdir -p "$out"

nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr \
  -I"$stub" -I"$act/include" -I"$root/quactlize/include" \
  -I"$root/dev/fold_derivation" -x cu "$map_src" -o "$out/map"
"$out/map"

flags=(
  -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__
  -include "$stub/ppu_arch_shim.h" -Xcudafe --error_limit=100000
  -I"$stub" -I"$act/include" -I"$act/tools/util/include"
  -I"$root/tests" -I"$root/benchmarks" -I"$root/quactlize/include" -I"$root/dev"
  -cuda -x cu -Wno-deprecated-gpu-targets
)

nvcc "${flags[@]}" -o "$out/wk4.cpp" "$src" >"$out/wk4.log" 2>&1 || true
wk4_new_errors="$(grep ': error:' "$out/wk4.log" | grep -v 'asm operand type size' || true)"
if [[ -n "$wk4_new_errors" ]]; then
  echo "L142 FAIL: WK4 production body has non-baseline compile errors" >&2
  printf '%s\n' "$wk4_new_errors" >&2
  exit 1
fi

nvcc "${flags[@]}" -DL142_UNPROVED_WK2=1 -o "$out/wk2.cpp" "$src" \
    >"$out/wk2.log" 2>&1 || true
if ! grep -q 'not yet proved for this K-cohort count' "$out/wk2.log"; then
  echo "L142 FAIL: unproved two-K-cohort consumer compiled" >&2
  exit 1
fi

echo "L142 production Cfg WK4=COMPILED WK2=NOT-YET-PROVED/EXPECTED-FAIL result=PASS"
