#!/usr/bin/env bash
set -uo pipefail

main() {
  local root out qtype name rc errors nonvendor passes
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  out="${QUACTLIZE_L216_OUT:-/workspace/quactlize-l216-moe-directory-type}"
  case "$out" in
    /workspace/*) ;;
    *) printf '[l216] FAIL: output must be below /workspace: %s\n' "$out" >&2; return 2 ;;
  esac
  mkdir -p "$out" || return 2
  passes=0

  for spec in 10:Q2_K 11:Q3_K 12:Q4_K 13:Q5_K 14:Q6_K; do
    qtype="${spec%%:*}"
    name="${spec##*:}"
    local arm="$out/q${qtype}" log="$out/q${qtype}/compile.log"
    mkdir -p "$arm" || return 2

    # Keep the expected vendor-inline-asm rejection observable even when the
    # caller exported errexit through SHELLOPTS.  An `if` condition is exempt
    # from errexit; a bare nvcc command could terminate the five-format census
    # after the first expected nonzero return.
    if nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr \
        -D__HGGCCC__ -DL216_QTYPE="$qtype" \
        -include "$root/dev/fold_derivation/stub_inc/ppu0010_arch_shim.h" \
        -Xcudafe --error_limit=100000 \
        -I"$root/dev/fold_derivation/stub_inc" \
        -I"$root/third_party/actlize/include" \
        -I"$root/third_party/actlize/tools/util/include" \
        -I"$root/third_party/actlize/examples/common" \
        -I"$root/tests" -I"$root/benchmarks" -I"$root/quactlize/include" -I"$root/dev" \
        -cuda -x cu -Wno-deprecated-gpu-targets \
        -o "$arm/l216.cpp" "$root/dev/fold_derivation/l216_moe_directory_shipping_type.cu" \
        >"$log" 2>&1; then
      rc=0
    else
      rc=$?
    fi

    errors="$(grep -c ': error:' "$log" || true)"
    nonvendor="$(grep ': error:' "$log" | \
      grep -vcE 'mma_ppu0010.hpp|copy_ppu0010_aiu.hpp' || true)"
    if [ "$nonvendor" -ne 0 ]; then
      printf '[l216] FAIL: %s persistent shipping type has %s non-vendor diagnostic(s)\n' \
        "$name" "$nonvendor" >&2
      grep ': error:' "$log" >&2 || true
      return 1
    fi
    if [ "$rc" -eq 0 ]; then
      if [ ! -s "$arm/l216.cpp" ]; then
        printf '[l216] FAIL: %s compiler returned success without generated device body\n' "$name" >&2
        return 1
      fi
    else
      if [ "$errors" -eq 0 ] ||
         ! grep -q 'GroupPersistentMixedInputKernel' "$log" ||
         ! grep -qE 'mma_ppu0010.hpp|copy_ppu0010_aiu.hpp' "$log"; then
        printf '[l216] FAIL: %s diagnostics do not prove the exact persistent body was reached\n' "$name" >&2
        tail -120 "$log" >&2
        return 1
      fi
    fi
    printf '[l216:format] PASS qtype=%s format=%s metadata=ScaleFirst driver=GroupPersistentMixedInputKernel vendor-asm=%s nonvendor=0\n' \
      "$qtype" "$name" "$errors"
    passes=$((passes + 1))
  done

  if [ "$passes" -ne 5 ]; then
    printf '[l216] FAIL: expected five format proofs, got %s\n' "$passes" >&2
    return 1
  fi
  printf '[l216] PASS: Q2_K/Q3_K/Q4_K/Q5_K/Q6_K exact ScaleFirst collectives reach GroupPersistentMixedInputKernel; formats=5 nonvendor=0\n'
}

main "$@"
