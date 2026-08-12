#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

${CXX:-c++} -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$repo/quactlize/include" \
  "$repo/dev/fold_derivation/l138_dense_arrangement_abi.cpp" \
  -o "$tmp/l138"
"$tmp/l138"

if ${CXX:-c++} -std=c++17 -O2 -I"$repo/quactlize/include" \
     "$repo/dev/fold_derivation/l138_dense_arrangement_negative.cpp" \
     -o "$tmp/negative" >"$tmp/negative.log" 2>&1; then
  echo "L138 FAIL: an F=2 artifact compiled as an F=1 reader" >&2
  exit 1
fi
if ! grep -q 'L138_EXPECTED_F2_TO_F1_REJECTION' "$tmp/negative.log"; then
  echo "L138 FAIL: negative compile failed for an unrelated reason" >&2
  sed -n '1,8p' "$tmp/negative.log" >&2
  exit 1
fi
echo "L138 compiled negative: F2 artifact -> F1 reader EXPECTED_FAIL/PASS"
