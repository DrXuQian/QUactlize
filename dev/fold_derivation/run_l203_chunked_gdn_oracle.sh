#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_ROOT=/workspace/quactlize-l203-chunked-gdn
mkdir -p "$BUILD_ROOT"

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$ROOT/quactlize/include" \
  "$ROOT/dev/fold_derivation/l203_chunked_gdn_oracle.cpp" \
  -o "$BUILD_ROOT/l203_chunked_gdn_oracle"

"$BUILD_ROOT/l203_chunked_gdn_oracle"
