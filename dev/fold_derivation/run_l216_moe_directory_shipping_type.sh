#!/usr/bin/env bash
set -uo pipefail

main() {
  local root out log rc errors nonvendor
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  out="${QUACTLIZE_L216_OUT:-/workspace/quactlize-l216-moe-directory-type}"
  case "$out" in
    /workspace/*) ;;
    *) printf '[l216] FAIL: output must be below /workspace: %s\n' "$out" >&2; return 2 ;;
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
    -o "$out/l216.cpp" "$root/dev/fold_derivation/l216_moe_directory_shipping_type.cu" \
    >"$log" 2>&1
  rc=$?
  set -e

  errors="$(grep -c ': error:' "$log" || true)"
  nonvendor="$(grep ': error:' "$log" | \
    grep -vcE 'mma_ppu0010.hpp|copy_ppu0010_aiu.hpp' || true)"
  if [ "$nonvendor" -ne 0 ]; then
    printf '[l216] FAIL: persistent shipping type has %s non-vendor diagnostic(s)\n' \
      "$nonvendor" >&2
    grep ': error:' "$log" >&2 || true
    return 1
  fi
  if [ "$rc" -eq 0 ]; then
    test -s "$out/l216.cpp" || {
      printf '[l216] FAIL: compiler returned success without generated device body\n' >&2
      return 1
    }
  else
    if [ "$errors" -eq 0 ] ||
       ! grep -q 'GroupPersistentMixedInputKernel' "$log" ||
       ! grep -qE 'mma_ppu0010.hpp|copy_ppu0010_aiu.hpp' "$log"; then
      printf '[l216] FAIL: diagnostics do not prove the exact persistent body was reached\n' >&2
      tail -120 "$log" >&2
      return 1
    fi
  fi
  printf '[l216] PASS: Q4 ScaleOnly 64x64x64-w64x32-s3 uses exact shipping collective '
  printf 'through GroupPersistentMixedInputKernel; vendor-asm=%s nonvendor=0\n' "$errors"
}

main "$@"
