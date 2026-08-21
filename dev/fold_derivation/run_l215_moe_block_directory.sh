#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$ROOT/build/l215-moe-block-directory"
mkdir -p "$BUILD"

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$ROOT/quactlize/include" \
  -I"$ROOT/third_party/actlize/include" \
  "$ROOT/dev/fold_derivation/l215_moe_block_directory.cpp" \
  -o "$BUILD/l215_moe_block_directory"

"$BUILD/l215_moe_block_directory"
