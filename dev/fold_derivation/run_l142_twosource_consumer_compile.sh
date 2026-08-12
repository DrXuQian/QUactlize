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
  -include "$stub/ppu0010_arch_shim.h" -Xcudafe --error_limit=100000
  -I"$stub" -I"$act/include" -I"$act/tools/util/include"
  -I"$root/tests" -I"$root/benchmarks" -I"$root/quactlize/include" -I"$root/dev"
  -cuda -x cu -Wno-deprecated-gpu-targets
)

set +e
nvcc "${flags[@]}" -o "$out/wk4.cpp" "$src" >"$out/wk4.log" 2>&1
wk4_rc=$?
set -e
wk4_asm_signature="$(
  grep ': error: asm operand type size' "$out/wk4.log" |
    sed -nE 's#^.*/(mma_ppu0010\.hpp|copy_ppu0010_aiu\.hpp)\(([0-9]+)\):.*#\1:\2#p' |
    sort | uniq -c | awk '{print $2 "=" $1}'
)"
expected_asm_signature=$'copy_ppu0010_aiu.hpp:273=1\ncopy_ppu0010_aiu.hpp:282=1\ncopy_ppu0010_aiu.hpp:67=1\ncopy_ppu0010_aiu.hpp:76=1\nmma_ppu0010.hpp:264=8\nmma_ppu0010.hpp:266=8\nmma_ppu0010.hpp:292=8\nmma_ppu0010.hpp:295=8\nmma_ppu0010.hpp:322=8\nmma_ppu0010.hpp:325=8\nmma_ppu0010.hpp:352=8\nmma_ppu0010.hpp:355=8\nmma_ppu0010.hpp:382=8\nmma_ppu0010.hpp:385=8'
wk4_all_errors="$(grep -c ': error:' "$out/wk4.log" || true)"
if [[ "$wk4_asm_signature" != "$expected_asm_signature" || "$wk4_all_errors" -ne 84 ]]; then
  echo "L142 FAIL: WK4 production body diagnostic baseline changed" >&2
  echo "expected vendor signature:" >&2
  printf '%s\n' "$expected_asm_signature" >&2
  echo "actual vendor signature (all errors=$wk4_all_errors):" >&2
  printf '%s\n' "$wk4_asm_signature" >&2
  grep ': error:' "$out/wk4.log" >&2 || true
  exit 1
fi
wk4_baseline_errors=84
if [[ "$wk4_rc" -eq 0 ]]; then
  [[ -s "$out/wk4.cpp" ]] || {
    echo "L142 FAIL: compiler returned success without a generated WK4 body" >&2
    exit 1
  }
else
  [[ "$wk4_baseline_errors" -gt 0 ]] &&
  grep -q 'Stages=4' "$out/wk4.log" &&
  grep -q 'FrgTensorC=cute::Tensor<cute::ArrayEngine<float, 32UL>' "$out/wk4.log" &&
  grep -q 'MarlinMixedInputKernel' "$out/wk4.log" &&
  grep -q 'mma_ppu0010\|copy_ppu0010_aiu' "$out/wk4.log" &&
  ! grep -q 'mma_ppu0015\|copy_ppu0015_aiu' "$out/wk4.log" || {
    echo "L142 FAIL: filtered diagnostics do not prove the real WK4 kernel body was instantiated" >&2
    exit 1
  }
fi

nvcc "${flags[@]}" -DL142_UNPROVED_WK2=1 -o "$out/wk2.cpp" "$src" \
    >"$out/wk2.log" 2>&1 || true
if ! grep -q 'not yet proved for this K-cohort count' "$out/wk2.log"; then
  echo "L142 FAIL: unproved two-K-cohort consumer compiled" >&2
  exit 1
fi

echo "L142 production Cfg arch=PPU0010 WK4=FULL-BODY-REACHED baseline-errors=$wk4_baseline_errors nonbaseline=0 WK2=NOT-YET-PROVED/EXPECTED-FAIL result=PASS"
