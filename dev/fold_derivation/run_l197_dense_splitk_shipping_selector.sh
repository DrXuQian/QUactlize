#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L197_OUT:-/workspace/quactlize-l197-dense-splitk-shipping-selector}"
mkdir -p "${out}"

command -v nvcc >/dev/null 2>&1 || {
  echo '[l197:runner] FAIL: nvcc is required for the production type binding' >&2
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
source="${repo}/dev/fold_derivation/l197_dense_splitk_shipping_selector.cu"
binary="${out}/l197_dense_splitk_shipping_selector"

# Force the production function-template body through both concrete
# PreparedOnePlaneLauncher::initialize call sites.  The temporary overlay's
# dependent marker avoids treating successful type formation as call-edge
# coverage, while keeping all generated probe artifacts under /workspace.
overlay="${out}/overlay"
prepared_header="${overlay}/dense_splitk_parallel_ppu.cuh"
mkdir -p "${overlay}"
cp "${repo}/quactlize/include/dense_splitk_parallel_ppu.cuh" \
  "${prepared_header}"
python3 - "${prepared_header}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = '''      hggcStream_t stream = nullptr) {
    initialized_ = false;'''
if text.count(needle) != 1:
    raise SystemExit("L197 PreparedOnePlaneLauncher::initialize seam is not unique")
plant = needle.replace(
    "    initialized_ = false;",
    '''    static_assert(sizeof(ShippingTypes) == 0,
                  "L197_PREPARED_INITIALIZE_EDGE_INSTANTIATED");
    initialized_ = false;''')
text = text.replace(needle, plant, 1)
path.write_text(text, encoding="utf-8")
PY

device_flags=(
  -I "${overlay}"
  "${flags[@]}"
  -DL197_FORCE_PRODUCTION_EDGE=1
  -Xcudafe --error_limit=100000
  -cuda -x cu
)

set +e
nvcc "${device_flags[@]}" "${source}" \
  -o "${out}/production-edge-positive.cu.cpp" \
  >"${out}/production-edge-positive.log" 2>&1
positive_rc=$?
set -e
marker='L197_PREPARED_INITIALIZE_EDGE_INSTANTIATED'
if [[ "${positive_rc}" -eq 0 ]] || \
    [[ "$(grep -Fc "${marker}" "${out}/production-edge-positive.log" || true)" -ne 1 ]]; then
  echo '[l197:runner] FAIL: forced production edge did not instantiate Prepared::initialize exactly once' >&2
  sed -n '1,140p' "${out}/production-edge-positive.log" >&2
  exit 1
fi
unexpected="${out}/production-edge-positive-unexpected.log"
grep -E ': (error|fatal error|catastrophic error):' \
  "${out}/production-edge-positive.log" \
  | grep -Fv "${marker}" >"${unexpected}" || true
if [[ -s "${unexpected}" ]]; then
  echo '[l197:runner] FAIL: production-edge witness carried an unrelated compiler error' >&2
  sed -n '1,100p' "${unexpected}" >&2
  exit 1
fi
for token in \
  'prepare_selected_production' \
  'PreparedOnePlaneLauncher' \
  'MainloopQuactlizeMixedInput<2' \
  'KernelAiuPackedA<1, Schedule>' \
  'cutlass::int4b_t'; do
  grep -Fq "${token}" "${out}/production-edge-positive.log" || {
    echo "[l197:runner] FAIL: production-edge instantiation chain lost: ${token}" >&2
    sed -n '1,160p' "${out}/production-edge-positive.log" >&2
    exit 1
  }
done

# Same address formation and planted header, with the only production call
# edge severed.  The marker must disappear and CUDA front-end lowering must be
# clean; this establishes causality for the positive diagnostic.
nvcc "${device_flags[@]}" -DL197_SEVER_PRODUCTION_EDGE=1 "${source}" \
  -o "${out}/production-edge-severed.cu.cpp" \
  >"${out}/production-edge-severed.log" 2>&1 || {
    echo '[l197:runner] FAIL: route-severed production-edge control did not compile cleanly' >&2
    sed -n '1,140p' "${out}/production-edge-severed.log" >&2
    exit 1
  }
if grep -Fq "${marker}" "${out}/production-edge-severed.log"; then
  echo '[l197:runner] FAIL: Prepared::initialize marker survived route severing' >&2
  exit 1
fi

# The executable may report production_edge=1 only after both compiler arms
# above have established and isolated the real call-edge witness.
nvcc "${flags[@]}" -DL197_PRODUCTION_EDGE_WITNESSED=1 \
  "${source}" -o "${binary}" \
  >"${out}/build.log" 2>&1 || {
    echo '[l197:runner] FAIL: production selector/type oracle did not compile' >&2
    tail -n 160 "${out}/build.log" >&2
    exit 1
  }

"${binary}" | tee "${out}/run.log"

test "$(grep -Fc '[l197:case]' "${out}/run.log")" -eq 24 || {
  echo '[l197:runner] FAIL: selector denominator is not 24 controls' >&2
  exit 1
}
test "$(grep -Fc 'route=fixed-splitk -> PASS' "${out}/run.log")" -eq 3 || {
  echo '[l197:runner] FAIL: profile S2/S4/S8 did not exclusively select fixed Split-K' >&2
  exit 1
}
test "$(grep -Fc 'route=shipping-s1 -> PASS' "${out}/run.log")" -eq 21 || {
  echo '[l197:runner] FAIL: at least one fallback escaped the shipping S1 edge' >&2
  exit 1
}
grep -Fq \
  '[l197:type] s1_exact=1 same_mainloop=1 mode=ScaleOnly bits=4+0 gs=128 artifact=xplane-tk64 packed_a_rows=1 bc=0 production_edge=1 -> PASS' \
  "${out}/run.log" || {
    echo '[l197:runner] FAIL: exact compiled production type/call edge was not proved' >&2
    exit 1
  }
grep -Fq \
  '[l197] PASS controls=24 shipping_calls=21 parallel_calls=3 default=shipping-s1 profile_axis={1,2,4,8}' \
  "${out}/run.log" || {
    echo '[l197:runner] FAIL: final selector verdict missing' >&2
    exit 1
  }

echo "[l197:runner] PASS type=production-W4-ScaleOnly-gs128-xplane64 controls=24/24 profile-parallel=3/3 fallback-s1=21/21 artifacts=${out}"
