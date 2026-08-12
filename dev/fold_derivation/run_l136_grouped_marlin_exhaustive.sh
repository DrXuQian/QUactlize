#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
manifest="${1:?usage: run_l136_grouped_marlin_exhaustive.sh manifest.tsv}"
expected="$(awk '{n += $10} END {print n+0}' "$manifest")"
out="${QUACTLIZE_L136_OUT:-$(mktemp -d)}"
mkdir -p "$out"
include_override=()
if [ -n "${L136_INCLUDE_OVERRIDE:-}" ]; then
  IFS=: read -ra override_roots <<< "$L136_INCLUDE_OVERRIDE"
  for root in "${override_roots[@]}"; do include_override+=(-I "$root"); done
fi
nvcc -std=c++17 -x cu -arch=sm_80 -w \
  "${include_override[@]}" \
  -I"$repo/dev/fold_derivation/stub_inc" \
  -I"$repo/third_party/actlize/include" \
  -I"$repo/benchmarks" -I"$repo/quactlize/include" \
  -I"$repo/third_party/actlize/tools/util/include" \
  "$repo/dev/fold_derivation/l136_grouped_marlin_exhaustive.cu" \
  -o "$out/l136"
"$out/l136" "$manifest" "$expected"
