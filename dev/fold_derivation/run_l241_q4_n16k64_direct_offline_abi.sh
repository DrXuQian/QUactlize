#!/usr/bin/env bash
set -euo pipefail

main() {
  local root out compiler source
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  out="${QUACTLIZE_L241_OUT:-/tmp/quactlize-l241-q4-n16k64-direct-offline}"
  mkdir -p "$out"
  compiler="${CXX:-$(command -v c++ 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l241-runner] FAIL: C++ compiler is unavailable\n' >&2
    return 2
  fi
  source="$root/dev/fold_derivation/l241_q4_n16k64_direct_offline_abi.cu"
  local -a common=(
    -std=c++17 -O2 -Wall -Wextra -Werror -x c++
    -I "$root/quactlize/include"
  )

  "$compiler" "${common[@]}" "$source" -o "$out/l241" \
    >"$out/build.log" 2>&1
  "$out/l241" | tee "$out/run.log"
  grep -E '^L241 (OFFLINE|Q4_N16K64_DIRECT_OFFLINE)' \
    "$out/run.log" >"$out/canonical.log"
  diff -u \
    "$root/dev/fold_derivation/l241_q4_n16k64_direct_offline_abi.expected.txt" \
    "$out/canonical.log"

  local macro label
  while read -r macro label; do
    "$compiler" "${common[@]}" -D"$macro"=1 "$source" \
      -o "$out/red-$label" >"$out/red-$label.build.log" 2>&1
    if "$out/red-$label" >"$out/red-$label.run.log" 2>&1; then
      printf '[l241-runner] FAIL: %s negative escaped\n' "$label" >&2
      return 1
    fi
    grep -E '^L241 Q4_N16K64_DIRECT_OFFLINE FAIL ' \
      "$out/red-$label.run.log" >/dev/null
    printf '[l241-red] PASS plant=%s result=RED\n' "$label"
  done <<'EOF'
L241_PLANT_WRONG_BITPERM wrong-bitperm
L241_PLANT_OLD_LAYOUT1 old-layout1
EOF
  printf '[l241-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
