#!/usr/bin/env bash
set -uo pipefail

main() {
  local root out log rc errors nonvendor
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  out="${QUACTLIZE_L214_OUT:-/workspace/quactlize-l214-q4-a32-exact}"
  case "$out" in
    /workspace/*) ;;
    *) printf 'L214 FAIL: output must be below /workspace: %s\n' "$out" >&2; return 2 ;;
  esac
  mkdir -p "$out" || return 2
  log="$out/compile.log"

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
    -o "$out/l214.cpp" "$root/dev/fold_derivation/l214_q4_a32_exact_type.cu" \
    >"$log" 2>&1
  rc=$?
  set -e
  errors="$(grep -c ': error:' "$log" || true)"
  nonvendor="$(grep ': error:' "$log" | \
    grep -vcE 'mma_ppu0010.hpp|copy_ppu0010_aiu.hpp' || true)"
  if [ "$rc" -eq 0 ] || [ "$errors" -ne 84 ] || [ "$nonvendor" -ne 0 ]; then
    printf 'L214 FAIL: exact body baseline rc=%s errors=%s nonvendor=%s\n' \
      "$rc" "$errors" "$nonvendor" >&2
    grep ': error:' "$log" >&2 || true
    return 1
  fi
  printf 'L214 exact q12/A32/64x64x128-w16x32-s8 body=REACHED '
  printf 'vendor-asm-baseline=%s nonvendor=0 result=PASS\n' "$errors"
}

main "$@"
