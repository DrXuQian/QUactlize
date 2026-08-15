#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L196_OUT:-/workspace/quactlize-l196-fixed-splitk-last-arriver}"
mkdir -p "$out"

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$root/quactlize/include" \
  "$root/dev/fold_derivation/l196_fixed_splitk_last_arriver.cpp" \
  -o "$out/l196_fixed_splitk_last_arriver"
"$out/l196_fixed_splitk_last_arriver"
python3 "$root/ci/check_fixed_splitk_last_arriver.py"
