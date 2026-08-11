#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp_dir="$(mktemp -d /tmp/quactlize-l132.XXXXXX)"
case "$tmp_dir" in
  /tmp/quactlize-l132.*) ;;
  *) echo "L132 unsafe temporary path: $tmp_dir" >&2; exit 2 ;;
esac
cleanup() { rm -rf -- "$tmp_dir"; }
trap cleanup EXIT

"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -pedantic \
  "$repo_root/dev/fold_derivation/l132_g5_harness_slot_map.cpp" \
  -o "$tmp_dir/l132"
"$tmp_dir/l132"
