#!/usr/bin/env bash
set -euo pipefail

main() {
  local root out compiler source
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  out="${QUACTLIZE_L236_OUT:-/tmp/quactlize-l236-kquant-kpack}"
  mkdir -p "$out"
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l236-runner] SKIP: nvcc is unavailable\n'
    return 0
  fi
  source="$root/dev/fold_derivation/l236_kquant_kpack_production_fragment.cu"
  local -a common=(
    -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
    -I "$root/dev/fold_derivation/stub_inc"
    -I "$root/third_party/actlize/include"
    -I "$root/quactlize/include"
  )

  "$compiler" "${common[@]}" "$source" -o "$out/l236" \
    >"$out/build.log" 2>&1 || {
      if grep -F 'hggc_fp8.h' "$out/build.log" >/dev/null; then
        printf '[l236-runner] SKIP: nvcc delegates to the PPU frontend; use the SDK target include path\n'
        return 0
      fi
      tail -n 120 "$out/build.log" >&2
      return 2
    }
  "$out/l236" | tee "$out/run.log"
  grep -Fqx 'L236 KQUANT_KPACK_PRODUCTION_FRAGMENT PASS planes=55 pairs=33' \
    "$out/run.log"

  local macro label
  while read -r macro label; do
    "$compiler" "${common[@]}" -D"$macro"=1 "$source" \
      -o "$out/red-$label" >"$out/red-$label.build.log" 2>&1
    if "$out/red-$label" >"$out/red-$label.run.log" 2>&1; then
      printf '[l236-runner] FAIL: %s negative escaped\n' "$label" >&2
      return 1
    fi
    grep -Fqx 'L236 KQUANT_KPACK_PRODUCTION_FRAGMENT FAIL planes=55 pairs=33' \
      "$out/red-$label.run.log"
    printf '[l236-red] PASS plant=%s result=RED\n' "$label"
  done <<'EOF'
L236_ROTATE_DESTINATION rotated-destination
L236_LEGACY_LOADER_STRIDE legacy-loader-stride
L236_SHIFT_HIGH_SOURCE shifted-high-source
EOF
  printf '[l236-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
