#!/usr/bin/env bash
set -euo pipefail

main() {
  local root out compiler source
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  out="${QUACTLIZE_L246_OUT:-/tmp/quactlize-l246-coord-iterator-lifetime}"
  mkdir -p "$out"
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l246-runner] SKIP: nvcc is unavailable\n'
    return 0
  fi
  source="$root/dev/fold_derivation/l246_coord_iterator_lifetime.cpp"

  "$compiler" \
    -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w \
    -I "$root/dev/fold_derivation/stub_inc" \
    -I "$root/third_party/actlize/include" \
    "$source" -o "$out/l246" >"$out/build.log" 2>&1 || {
      printf '[l246-runner] FAIL: coordinate-iterator lifetime gate did not build\n' >&2
      tail -n 120 "$out/build.log" >&2
      return 2
    }

  "$out/l246" | tee "$out/run.log"
  diff -u \
    "$root/dev/fold_derivation/l246_coord_iterator_lifetime.expected.txt" \
    "$out/run.log"
  printf '[l246-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
