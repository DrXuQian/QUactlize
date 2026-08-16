#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L200_OUT:-/workspace/quactlize-l200-dense-w4-splitk-production}"
mkdir -p "${out}"

command -v nvcc >/dev/null 2>&1 || {
  echo '[l200:runner] FAIL: nvcc is required for the production type/call-edge gate' >&2
  exit 1
}

source="${repo}/dev/fold_derivation/l200_dense_w4_splitk_production.cu"
base_flags=(
  -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include"
  -Wno-deprecated-gpu-targets
)

# The real backend is compiled as one of five QUACTLIZE_DENSE_ONLY format
# islands.  Preprocess the shipping TU itself (not a reconstructed condition):
# Q4/default owns the W4 type, while a Q2 island retains ABI symbols without
# even seeing the production namespace/type body.
backend="${repo}/quactlize/csrc/device/ppu_dense_backend.cu"
nvcc "${base_flags[@]}" -E -DQUACTLIZE_DENSE_ONLY=12 "${backend}" \
  -o "${out}/backend-q12.ii" >"${out}/backend-q12-preprocess.log" 2>&1 || {
    echo '[l200:runner] FAIL: Q4 format-island preprocessing failed' >&2
    sed -n '1,100p' "${out}/backend-q12-preprocess.log" >&2
    exit 1
  }
for token in \
  'namespace ppu_dense_w4_splitk' \
  'using ProductionShipping' \
  'ppu_dense_w4_splitk::prepare_selected<>' \
  'quactlize_ppu_dense_w4_splitk_dev_v1'; do
  grep -Fq "${token}" "${out}/backend-q12.ii" || {
    echo "[l200:runner] FAIL: Q4 island lost production token: ${token}" >&2
    exit 1
  }
done

nvcc "${base_flags[@]}" -E -DQUACTLIZE_DENSE_ONLY=10 "${backend}" \
  -o "${out}/backend-q10.ii" >"${out}/backend-q10-preprocess.log" 2>&1 || {
    echo '[l200:runner] FAIL: Q2 format-island preprocessing failed' >&2
    sed -n '1,100p' "${out}/backend-q10-preprocess.log" >&2
    exit 1
  }
for token in \
  'quactlize_ppu_dense_w4_splitk_workspace_bytes_v1' \
  'quactlize_ppu_dense_w4_splitk_dev_v1'; do
  grep -Fq "${token}" "${out}/backend-q10.ii" || {
    echo "[l200:runner] FAIL: Q2 island lost fail-closed ABI symbol: ${token}" >&2
    exit 1
  }
done
if grep -Fq 'namespace ppu_dense_w4_splitk' "${out}/backend-q10.ii" || \
   grep -Fq 'ppu_dense_w4_splitk::' "${out}/backend-q10.ii"; then
  echo '[l200:runner] FAIL: Q2 island instantiated or referenced the W4 production type' >&2
  exit 1
fi

# The public profile must remain a real C ABI, independent of CUDA/C++ types.
c_probe="${out}/profile-abi.c"
python3 - "${c_probe}" <<'PY'
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(r'''
#include "quactlize_ppu_device.h"
#include <stddef.h>

_Static_assert(sizeof(quactlize_ppu_dense_w4_splitk_key_v1) == 88,
               "W4 Split-K key ABI drifted");
_Static_assert(sizeof(quactlize_ppu_dense_w4_splitk_profile_v1) == 96,
               "W4 Split-K profile ABI drifted");
_Static_assert(offsetof(quactlize_ppu_dense_w4_splitk_profile_v1, key) == 4,
               "W4 Split-K key offset drifted");
_Static_assert(offsetof(quactlize_ppu_dense_w4_splitk_profile_v1, selected_s) == 92,
               "W4 Split-K selected_s offset drifted");

int main(void) {
  quactlize_ppu_dense_w4_splitk_profile_v1 profile = {0};
  return profile.selected_s;
}
''', encoding="utf-8")
PY
cc -std=c11 -I "${repo}/quactlize/include" "${c_probe}" \
  -o "${out}/profile-abi" >"${out}/profile-abi-build.log" 2>&1 || {
    echo '[l200:runner] FAIL: public W4 Split-K profile is not a stable C ABI' >&2
    sed -n '1,100p' "${out}/profile-abi-build.log" >&2
    exit 1
  }
"${out}/profile-abi"

# Instantiate the exact production prepare_selected body against a dependent
# marker in PreparedOnePlaneLauncher::initialize.  Both headers are copied so
# the production launch header's quoted include resolves to the overlay.
overlay="${out}/overlay"
mkdir -p "${overlay}"
cp "${repo}/quactlize/include/ppu_dense_w4_splitk_launch.cuh" "${overlay}/"
cp "${repo}/quactlize/include/dense_splitk_parallel_ppu.cuh" "${overlay}/"
python3 - "${overlay}/dense_splitk_parallel_ppu.cuh" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = '''      hggcStream_t stream = nullptr) {
    initialized_ = false;'''
if text.count(needle) != 1:
    raise SystemExit("L200 PreparedOnePlaneLauncher::initialize seam is not unique")
plant = needle.replace(
    "    initialized_ = false;",
    '''    static_assert(sizeof(ShippingTypes) == 0,
                  "L200_PRODUCTION_PREPARE_REACHED_PREPARED_INITIALIZE");
    initialized_ = false;''')
path.write_text(text.replace(needle, plant, 1), encoding="utf-8")
PY

device_flags=(
  -I "${overlay}"
  "${base_flags[@]}"
  -DL200_FORCE_PRODUCTION_PREPARE=1
  -Xcudafe --error_limit=100000
  -cuda -x cu
)

set +e
nvcc "${device_flags[@]}" "${source}" \
  -o "${out}/production-positive.cu.cpp" \
  >"${out}/production-positive.log" 2>&1
positive_rc=$?
set -e
marker='L200_PRODUCTION_PREPARE_REACHED_PREPARED_INITIALIZE'
if [[ "${positive_rc}" -eq 0 ]] || \
    [[ "$(grep -Fc "${marker}" "${out}/production-positive.log" || true)" -ne 1 ]]; then
  echo '[l200:runner] FAIL: real production prepare edge did not reach Prepared::initialize exactly once' >&2
  sed -n '1,150p' "${out}/production-positive.log" >&2
  exit 1
fi
unexpected="${out}/production-positive-unexpected.log"
grep -E ': (error|fatal error|catastrophic error):' \
  "${out}/production-positive.log" \
  | grep -Fv "${marker}" >"${unexpected}" || true
if [[ -s "${unexpected}" ]]; then
  echo '[l200:runner] FAIL: production call-edge witness carried an unrelated compiler error' >&2
  sed -n '1,100p' "${unexpected}" >&2
  exit 1
fi
for token in \
  'ppu_dense_w4_splitk::prepare_selected' \
  'PreparedOnePlaneLauncher' \
  'MainloopQuactlizeMixedInput<2' \
  'cutlass::int4b_t'; do
  grep -Fq "${token}" "${out}/production-positive.log" || {
    echo "[l200:runner] FAIL: exact production instantiation chain lost: ${token}" >&2
    sed -n '1,180p' "${out}/production-positive.log" >&2
    exit 1
  }
done

# Same exact source/header overlay, changing only the production prepare edge.
# The marker must disappear and front-end lowering must be clean.
nvcc "${device_flags[@]}" -DQUACTLIZE_W4_SPLITK_SEVER_PREPARE_EDGE=1 \
  "${source}" -o "${out}/production-severed.cu.cpp" \
  >"${out}/production-severed.log" 2>&1 || {
    echo '[l200:runner] FAIL: severed production prepare control did not compile cleanly' >&2
    sed -n '1,150p' "${out}/production-severed.log" >&2
    exit 1
  }
if grep -Fq "${marker}" "${out}/production-severed.log"; then
  echo '[l200:runner] FAIL: Prepared::initialize marker survived prepare-edge severing' >&2
  exit 1
fi

binary="${out}/l200-dense-w4-splitk-production"
nvcc "${base_flags[@]}" "${source}" -o "${binary}" \
  >"${out}/host-build.log" 2>&1 || {
    echo '[l200:runner] FAIL: production profile/workspace oracle did not compile' >&2
    sed -n '1,140p' "${out}/host-build.log" >&2
    exit 1
  }
"${binary}" | tee "${out}/host-run.log"

test "$(grep -Fc '[l200:case]' "${out}/host-run.log")" -eq 35 || {
  echo '[l200:runner] FAIL: profile/workspace denominator is not 35 controls' >&2
  exit 1
}
test "$(grep -Fc 'route=fixed-splitk' "${out}/host-run.log")" -eq 3 || {
  echo '[l200:runner] FAIL: exact S2/S4/S8 profiles did not exclusively select parallel' >&2
  exit 1
}
test "$(grep -Fc 'route=literal-shipping-s1' "${out}/host-run.log")" -eq 32 || {
  echo '[l200:runner] FAIL: at least one negative escaped literal shipping S1' >&2
  exit 1
}
grep -Fq \
  '[l200:type] s1_exact=1 same_mainloop=1 mode=ScaleOnly gs=128 artifact=xplane-tk64 tactic=8x64x128/8x16/s2 profile_bytes=96 -> PASS' \
  "${out}/host-run.log" || {
    echo '[l200:runner] FAIL: exact measured production type verdict missing' >&2
    exit 1
  }
grep -Fq \
  '[l200] PASS controls=35 shipping_calls=32 parallel_calls=3 full_key_fields=22 profile_axis={1,2,4,8}' \
  "${out}/host-run.log" || {
    echo '[l200:runner] FAIL: final production profile verdict missing' >&2
    exit 1
  }

echo "[l200:runner] PASS abi=C-v1 production=backend-W4-ScaleOnly-gs128-tacticTK128-artifactTK64 controls=35/35 call-edge=instantiated/severed artifacts=${out}"
