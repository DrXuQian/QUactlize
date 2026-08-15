#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L191_OUT:-/workspace/quactlize-l191-dense-splitk-sweep-type}"
mkdir -p "${out}"

command -v nvcc >/dev/null 2>&1 || {
  echo '[l191] FAIL: nvcc is required for the prepared sweep type gate' >&2
  exit 1
}

flags=(
  -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include"
  -Wno-deprecated-gpu-targets
)
source="${repo}/dev/fold_derivation/l191_dense_splitk_sweep_type.cu"
binary="${out}/l191"
nvcc "${flags[@]}" "${source}" -o "${binary}" >"${out}/build.log" 2>&1 || {
  echo '[l191] FAIL: exact production M1/APack prepared type did not compile' >&2
  sed -n '1,120p' "${out}/build.log" >&2
  exit 1
}
"${binary}" | tee "${out}/run.log"
grep -Fq 'artifact_tk=64 packed_rows=1 atom_m=8 -> PASS' "${out}/run.log" || {
  echo '[l191] FAIL: compiled type did not publish the fixed authority' >&2
  exit 1
}

# One changed variable: an ArtifactTK128 row cannot satisfy this sweep's TK64
# authority.  The planted compile must fail at the production static_assert;
# otherwise the positive compile proves only that some packed-A type exists.
plant="${out}/artifact128.cu"
sed 's/cutlass::int4b_t, 64>/cutlass::int4b_t, 128>/' "${source}" >"${plant}"
if nvcc "${flags[@]}" "${plant}" -o "${out}/artifact128" \
    >"${out}/artifact128.log" 2>&1; then
  echo '[l191] FAIL: ArtifactTK128 negative control compiled' >&2
  exit 1
fi
grep -Fq 'Shipping::MainloopPolicy::ArtifactTileK == 64' \
  "${out}/artifact128.log" || {
  echo '[l191] FAIL: ArtifactTK128 failed for an unrelated reason' >&2
  sed -n '1,100p' "${out}/artifact128.log" >&2
  exit 1
}

echo "[l191] PASS: prepared production type is M1/APack/m8/TK64; ArtifactTK128 plant rejected; artifacts=${out}"
