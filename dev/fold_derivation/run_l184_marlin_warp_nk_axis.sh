#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
src="${repo}/dev/fold_derivation/l184_marlin_warp_nk_axis.cpp"
build_root="${repo}/build/l184-marlin-warp-nk-axis"
mkdir -p "${build_root}"

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"${repo}/quactlize/include" \
  "${src}" -o "${build_root}/l184_marlin_warp_nk_axis"

"${build_root}/l184_marlin_warp_nk_axis"
