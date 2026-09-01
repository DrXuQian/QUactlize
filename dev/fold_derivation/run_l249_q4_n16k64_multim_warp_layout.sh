#!/usr/bin/env bash
set -euo pipefail

main() {
  local root out compiler source
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  out="${QUACTLIZE_L249_OUT:-/tmp/quactlize-l249-q4-n16k64-multim-warp}"
  mkdir -p "$out"
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l249-runner] SKIP: nvcc is unavailable\n'
    return 0
  fi

  source="$root/dev/fold_derivation/l249_q4_n16k64_multim_warp_layout.cu"
  local -a common=(
    -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
    -I "$root/dev/fold_derivation/stub_inc"
    -I "$root/third_party/actlize/include"
    -I "$root/quactlize/include"
  )
  if ! "$compiler" "${common[@]}" -DL249_COMPILER_PROBE=1 "$source" \
       -o "$out/compiler-probe" >"$out/compiler-probe.log" 2>&1; then
    printf '[l249-runner] SKIP: nvcc cannot compile the CUDA host-oracle probe\n'
    return 0
  fi

  "$compiler" "${common[@]}" "$source" -o "$out/l249" \
    >"$out/build.log" 2>&1 || {
      if grep -F 'hggc_fp8.h' "$out/build.log" >/dev/null; then
        printf '[l249-runner] SKIP: nvcc delegates to the PPU frontend; use committed host evidence\n'
        return 0
      fi
      printf '[l249-runner] FAIL: multi-M-warp oracle did not build\n' >&2
      tail -n 180 "$out/build.log" >&2
      return 2
    }
  "$out/l249" | tee "$out/run.log"
  grep -E '^L249 (MULTIM|Q4_N16K64_MULTIM_WARP)' "$out/run.log" \
    >"$out/canonical.log"
  diff -u \
    "$root/dev/fold_derivation/l249_q4_n16k64_multim_warp_layout.expected.txt" \
    "$out/canonical.log"

  local macro label
  while read -r macro label; do
    "$compiler" "${common[@]}" -D"$macro"=1 "$source" \
      -o "$out/red-$label" >"$out/red-$label.build.log" 2>&1
    if "$out/red-$label" >"$out/red-$label.run.log" 2>&1; then
      printf '[l249-runner] FAIL: %s negative escaped\n' "$label" >&2
      return 1
    fi
    grep -E '^L249 Q4_N16K64_MULTIM_WARP FAIL ' \
      "$out/red-$label.run.log" >/dev/null
    case "$label" in
      warp-n-pitch)
        grep -E '^L249 MULTIM WOM=2 WN=16 .*physical_bad=[1-9][0-9]* vmnk_bad=0 result=FAIL$' \
          "$out/red-$label.run.log" >/dev/null
        grep -E '^L249 MULTIM WOM=2 WN=32 .*physical_bad=[1-9][0-9]* vmnk_bad=0 result=FAIL$' \
          "$out/red-$label.run.log" >/dev/null
        ;;
      physical-warp-n)
        grep -E '^L249 MULTIM WOM=2 WN=16 .*vmnk_bad=[1-9][0-9]* result=FAIL$' \
          "$out/red-$label.run.log" >/dev/null
        grep -E '^L249 MULTIM WOM=2 WN=32 .*vmnk_bad=[1-9][0-9]* result=FAIL$' \
          "$out/red-$label.run.log" >/dev/null
        ;;
      converter-destination)
        # WN16 has one N-rest cohort and is the intentional identity control;
        # WN32 makes the borrowed physical rest stride observably wrong.
        grep -E '^L249 MULTIM WOM=2 WN=16 .*result=PASS$' \
          "$out/red-$label.run.log" >/dev/null
        grep -E '^L249 MULTIM WOM=2 WN=32 .*destination_bad=[1-9][0-9]* .*result=FAIL$' \
          "$out/red-$label.run.log" >/dev/null
        ;;
    esac
    printf '[l249-red] PASS plant=%s result=RED\n' "$label"
  done <<'EOF'
L249_PLANT_WARP_N_PITCH warp-n-pitch
L249_PLANT_PHYSICAL_WARP_N physical-warp-n
L249_PLANT_INPUT_OWNED_DESTINATION converter-destination
EOF

  if "$compiler" "${common[@]}" -DL249_PLANT_RVALUE_OWNER=1 "$source" \
       -o "$out/red-rvalue-owner" >"$out/red-rvalue-owner.build.log" 2>&1; then
    printf '[l249-runner] FAIL: rvalue-owner negative compiled\n' >&2
    return 1
  fi
  grep -F 'make_copy_view' "$out/red-rvalue-owner.build.log" >/dev/null
  grep -F 'argument #1 does not match parameter' \
    "$out/red-rvalue-owner.build.log" >/dev/null
  printf '[l249-red] PASS plant=rvalue-owner result=RED\n'
  printf '[l249-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
