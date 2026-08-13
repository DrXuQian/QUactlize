#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
out="${QUACTLIZE_L172_OUT:-/tmp/quactlize_l172}"
mkdir -p "$out"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"$repo/quactlize/include" \
  "$repo/dev/fold_derivation/l172_standalone_marlin_tactic_space.cpp" \
  -o "$out/l172"
"$out/l172"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"$repo/quactlize/include" \
  "$repo/dev/fold_derivation/emit_marlin_tactic_space.cpp" \
  -o "$out/emit_marlin_tactic_space"
"$out/emit_marlin_tactic_space" >"$out/census.txt"

plants=(drop-load-axis admit-stage3 collapse-warp-k broaden-classic-warp-k)
for plant in "${plants[@]}"; do
  log="$out/plant-${plant}.log"
  set +e
  "$out/l172" "--plant=${plant}" >"$log" 2>&1
  rc=$?
  set -e
  if [[ $rc -eq 0 ]] || ! grep -Fq "[l172:red] plant=${plant} caught=1" "$log"; then
    echo "[l172] FAIL: plant ${plant} did not produce its named RED" >&2
    sed -n '1,20p' "$log" >&2
    exit 1
  fi
done

echo '[l172:runner] positive=PASS negative_controls=4/4_RED emitter=PASS result=PASS'
