#!/usr/bin/env bash
set -euo pipefail

main() {
  local root out compiler source
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  out="${QUACTLIZE_L238_OUT:-/tmp/quactlize-l238-provider-scaffold}"
  mkdir -p "$out"
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l238-runner] SKIP: nvcc is unavailable\n'
    return 0
  fi

  source="$root/dev/fold_derivation/l238_provider_scaffold_compile_oracle.cu"
  local -a common=(
    -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
    -I "$root/dev/fold_derivation/stub_inc"
    -I "$root/third_party/actlize/include"
    -I "$root/quactlize/include"
  )

  "$compiler" "${common[@]}" "$source" -o "$out/l238" \
    >"$out/build.log" 2>&1 || {
      if grep -F 'hggc_fp8.h' "$out/build.log" >/dev/null; then
        printf '[l238-runner] SKIP: nvcc delegates to the PPU frontend; use the SDK target include path\n'
        return 0
      fi
      printf '[l238-runner] FAIL: compile oracle did not build\n' >&2
      tail -n 120 "$out/build.log" >&2
      return 2
    }

  "$out/l238" | tee "$out/run.log"
  grep -Fqx 'L238 PROVIDER_SCAFFOLD_COMPILE_ORACLE PASS providers=3 shared-contracts=2 reds=6' \
    "$out/run.log"
  diff -u \
    "$root/dev/fold_derivation/l238_provider_scaffold_compile_oracle.expected.txt" \
    "$out/run.log"

  local macro label sentinel
  while read -r macro label sentinel; do
    if "$compiler" "${common[@]}" -D"$macro"=1 "$source" \
         -o "$out/red-$label" >"$out/red-$label.build.log" 2>&1; then
      printf '[l238-runner] FAIL: %s negative escaped\n' "$label" >&2
      return 1
    fi
    grep -F "$sentinel" "$out/red-$label.build.log" >/dev/null
    printf '[l238-red] PASS plant=%s result=RED\n' "$label"
  done <<'EOF'
L238_PLANT_LAYOUT_MISMATCH layout-mismatch QUACTLIZE_B_DELIVERY_LAYOUT_MISMATCH
L238_PLANT_ELEMENT_MISMATCH element-mismatch QUACTLIZE_B_DELIVERY_ELEMENT_MISMATCH
L238_PLANT_SHARED_BYTES_MISMATCH shared-bytes-mismatch QUACTLIZE_B_DELIVERY_STAGE_BYTES_MISMATCH
L238_PLANT_N_ATOM_MISMATCH n-atom-mismatch QUACTLIZE_B_DELIVERY_N_ATOM_MISMATCH
L238_PLANT_K_ATOM_MISMATCH k-atom-mismatch QUACTLIZE_B_DELIVERY_K_ATOM_MISMATCH
L238_PLANT_TAG_MISMATCH tag-mismatch B writer and reader must name the same shared encoding
EOF
  printf '[l238-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
