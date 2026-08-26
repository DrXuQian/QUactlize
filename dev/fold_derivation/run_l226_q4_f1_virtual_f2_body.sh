#!/usr/bin/env bash
set -uo pipefail

main() {
  local root out log rc errors nonvendor
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  out="${QUACTLIZE_L226_OUT:-/workspace/quactlize-l226-q4-f1-virtual-f2-body}/run-$$"
  case "$out" in /workspace/*) ;; *) echo "[l226] FAIL: output outside /workspace: $out" >&2; return 2 ;; esac
  mkdir -p "$out" || return 2
  log="$out/compile.log"
  "$root/dev/fold_derivation/nvidia_nvcc_or_skip.sh" "$(command -v nvcc 2>/dev/null || true)" l226-runner || return $?
  set +e
  nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr \
    -D__HGGCCC__ -include "$root/dev/fold_derivation/stub_inc/ppu0010_arch_shim.h" \
    -Xcudafe --error_limit=100000 \
    -I"$root/dev/fold_derivation/stub_inc" \
    -I"$root/third_party/actlize/include" \
    -I"$root/third_party/actlize/tools/util/include" \
    -I"$root/third_party/actlize/examples/common" \
    -I"$root/tests" -I"$root/benchmarks" -I"$root/quactlize/include" -I"$root/dev" \
    -cuda -x cu -Wno-deprecated-gpu-targets \
    -o "$out/l226.cpp" "$root/dev/fold_derivation/l226_q4_f1_virtual_f2_body.cu" \
    >"$log" 2>&1
  rc=$?
  set -e
  errors="$(grep -c ': error:' "$log" || true)"
  nonvendor="$(grep ': error:' "$log" | grep -vcE 'mma_ppu0010.hpp|copy_ppu0010_aiu.hpp' || true)"
  if [[ $rc -eq 0 || $errors -le 0 || $nonvendor -ne 0 ]]; then
    echo "[l226] FAIL: exact body rc=$rc errors=$errors nonvendor=$nonvendor" >&2
    grep ': error:' "$log" >&2 || true
    return 1
  fi
  echo "[l226] PASS exact=q12/A64/64x128x128-w64x64-s3 virtual=F2 body=REACHED vendor_asm_errors=$errors nonvendor=0 artifacts=$out"
}

main "$@"
