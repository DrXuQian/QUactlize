#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L225_OUT:-/workspace/quactlize-l225-q4-f1-virtual-f2-type}/run-$$"
mkdir -p "$out"
compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
if [[ -z "$compiler" ]]; then
  echo '[l225-runner] FAIL: nvcc unavailable' >&2
  exit 2
fi
flags=(-std=c++17 -arch=sm_80 -w --expt-relaxed-constexpr -D__HGGCCC__ -DPPU_FORCE_INSTANTIATE=1
       -I "$repo/dev/fold_derivation/stub_inc"
       -I "$repo/third_party/actlize/include"
       -I "$repo/third_party/actlize/tools/util/include"
       -I "$repo/quactlize/include")
src="$repo/dev/fold_derivation/l225_q4_f1_virtual_f2_type.cu"
"$compiler" "${flags[@]}" "$src" -o "$out/positive" >"$out/positive-build.log" 2>&1 || {
  echo '[l225-runner] FAIL: positive type closure did not compile' >&2
  tail -n 120 "$out/positive-build.log" >&2
  exit 2
}
"$out/positive" | tee "$out/positive.log"
grep -Fqx \
  'L225_Q4_F1_VIRTUAL_F2_TYPE PASS default=UNCHANGED physical=F1 compute=F2 t64=TYPE_IDENTICAL t128_t256=MMA_ONLY smem_delta=0 runtime_branch_delta=0' \
  "$out/positive.log"

set +e
"$compiler" "${flags[@]}" -DL225_NEG_T32=1 "$src" -o "$out/t32" >"$out/t32-build.log" 2>&1
t32_rc=$?
set -e
if [[ $t32_rc -eq 0 ]] || ! grep -Fq 'ArtifactTileK must completely tile TacticTileK' "$out/t32-build.log"; then
  echo "[l225-runner] FAIL: unproved T32 path was not rejected exactly (rc=$t32_rc)" >&2
  exit 1
fi
echo "[l225-runner] PASS: exact default/physical invariants, T64 identity, T128/T256 MMA-only, T32 RED; artifacts=$out"
