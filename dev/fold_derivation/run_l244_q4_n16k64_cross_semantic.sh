#!/usr/bin/env bash
set -euo pipefail

main() {
  local root out compiler source
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  out="${QUACTLIZE_L244_OUT:-/tmp/quactlize-l244-q4-n16k64-cross-semantic}"
  mkdir -p "$out"
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l244-runner] SKIP: nvcc is unavailable\n'
    return 0
  fi

  source="$root/dev/fold_derivation/l244_q4_n16k64_cross_semantic.cu"
  local -a common=(
    -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
    -I "$root/dev/fold_derivation/stub_inc"
    -I "$root/third_party/actlize/include"
    -I "$root/quactlize/include"
  )
  if ! "$compiler" "${common[@]}" -DL244_COMPILER_PROBE=1 "$source" \
       -o "$out/compiler-probe" >"$out/compiler-probe.log" 2>&1; then
    printf '[l244-runner] SKIP: nvcc cannot compile the CUDA host-oracle probe\n'
    return 0
  fi

  "$compiler" "${common[@]}" "$source" -o "$out/l244" \
    >"$out/build.log" 2>&1 || {
      if grep -F 'hggc_fp8.h' "$out/build.log" >/dev/null; then
        printf '[l244-runner] SKIP: nvcc delegates to the PPU frontend; use committed host evidence\n'
        return 0
      fi
      printf '[l244-runner] FAIL: cross-semantic oracle did not build\n' >&2
      tail -n 180 "$out/build.log" >&2
      return 2
    }
  "$out/l244" | tee "$out/run.log"
  grep -E '^L244 (CROSS|Q4_N16K64_CROSS_SEMANTIC)' "$out/run.log" \
    >"$out/canonical.log"
  diff -u \
    "$root/dev/fold_derivation/l244_q4_n16k64_cross_semantic.expected.txt" \
    "$out/canonical.log"

  local macro label
  while read -r macro label; do
    "$compiler" "${common[@]}" -D"$macro"=1 "$source" \
      -o "$out/red-$label" >"$out/red-$label.build.log" 2>&1
    if "$out/red-$label" >"$out/red-$label.run.log" 2>&1; then
      printf '[l244-runner] FAIL: %s negative escaped\n' "$label" >&2
      return 1
    fi
    grep -E '^L244 Q4_N16K64_CROSS_SEMANTIC FAIL ' \
      "$out/red-$label.run.log" >/dev/null
    printf '[l244-red] PASS plant=%s result=RED\n' "$label"
  done <<'EOF'
L244_PLANT_WN_PITCH wn-pitch
L244_PLANT_WARP_BASE warp-base
EOF
  printf '[l244-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
