#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${QZ_L188_BUILD_DIR:-/workspace/quactlize-l188-fixed-splitk}"
compiler="${CXX:-g++}"

mkdir -p "${build_dir}"

"${compiler}" \
  -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -I"${repo_root}/quactlize/include" \
  "${repo_root}/dev/fold_derivation/l188_fixed_splitk_partition.cpp" \
  -o "${build_dir}/l188_fixed_splitk_partition"

"${build_dir}/l188_fixed_splitk_partition"
