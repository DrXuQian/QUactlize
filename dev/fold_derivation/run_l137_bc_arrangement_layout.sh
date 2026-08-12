#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
out=${TMPDIR:-/tmp}/quactlize-l137-bc-arrangement
"${NVCC:-/usr/local/cuda/bin/nvcc}" -std=c++17 -O2 -x cu -arch=sm_80 -w \
  -I "$repo_dir/dev/fold_derivation/stub_inc" \
  -I "$repo_dir/quactlize/include" \
  -I "$repo_dir/third_party/actlize/include" \
  "$repo_dir/dev/fold_derivation/l137_bc_arrangement_layout.cu" -o "$out"
cases=(
  q2-a32-low q2-a64-low q2-a128-low q2-a256-low
  q3-a64-low q3-a128-low q3-a256-low q3-a64-high q3-a128-high q3-a256-high
  q4-a32-low q4-a64-low q4-a128-low q4-a256-low
  q5-a64-low q5-a128-low q5-a256-low q5-a64-high q5-a128-high q5-a256-high
  q6-a32-low q6-a64-low q6-a128-low q6-a32-high q6-a64-high q6-a128-high
  controls
)
for case_name in "${cases[@]}"; do
  "$out" "$case_name"
done
