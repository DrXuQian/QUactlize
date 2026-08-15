#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L190_OUT:-/workspace/quactlize-l190-dense-splitk-parallel-type}"
mkdir -p "${out}"

command -v nvcc >/dev/null 2>&1 || {
  echo '[l190] FAIL: nvcc is required for the type/device-body gate' >&2
  exit 1
}

source="${repo}/dev/fold_derivation/l190_dense_splitk_parallel_type.cu"
base_flags=(
  -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include"
  -Wno-deprecated-gpu-targets
)

# Arm 1: compile and execute all type/lowering contracts with the sole device
# call edge severed.  This is a clean executable proof, not evidence inferred
# from a device compiler diagnostic.
host_binary="${out}/l190-host-contract"
nvcc "${base_flags[@]}" -DL190_SEVER_DEVICE_BODY=1 \
  "${source}" -o "${host_binary}" >"${out}/host-build.log" 2>&1 || {
  echo '[l190] FAIL: host type/grid/workspace contract did not compile' >&2
  sed -n '1,120p' "${out}/host-build.log" >&2
  exit 1
}
"${host_binary}" | tee "${out}/host-run.log"
grep -Fq \
  '[l190:host] PASS: exact shipping mainloop retained; S8 grid/workspace exact; S1 authority unchanged' \
  "${out}/host-run.log" || {
  echo '[l190] FAIL: host executable returned without the complete contract' >&2
  exit 1
}

# Arm 2: put one dependent marker at the entrance to the exact production
# Split-K operator in a temporary include overlay.  The source tree is not
# edited.  Reaching this marker proves that the concrete wrapper instantiated
# GemmUniversalMixedInputSplitKParallel::operator(), not merely KernelTypes.
overlay="${out}/overlay"
kernel_rel='quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp'
kernel_probe="${overlay}/${kernel_rel}"
mkdir -p "$(dirname "${kernel_probe}")"
cp "${repo}/quactlize/include/dense_splitk_parallel_ppu.cuh" \
  "${overlay}/dense_splitk_parallel_ppu.cuh"
cp "${repo}/quactlize/include/${kernel_rel}" "${kernel_probe}"
python3 - "${kernel_probe}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = '''  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {'''
if text.count(needle) != 1:
    raise SystemExit("L190 production Split-K operator seam is not unique")
plant = needle + '''
    static_assert(sizeof(CollectiveMainloop) == 0,
                  "L190_SPLITK_DEVICE_BODY_INSTANTIATED");'''
text = text.replace(needle, plant, 1)
path.write_text(text, encoding="utf-8")
PY

# Exercise SplitKernel::can_implement itself before the device-body marker.
# Each negative differs from the accepted row by one partial-ABI field and
# must remain rejected.
admission_flags=(
  -I "${overlay}"
  "${base_flags[@]}"
  -DL190_SEVER_DEVICE_BODY=1 -DL190_ADMISSION_PROBE=1
)
for admission_case in valid bad-stride unaligned; do
  case_defs=()
  case "${admission_case}" in
    valid) ;;
    bad-stride) case_defs=(-DL190_PLANT_BAD_STRIDE=1) ;;
    unaligned) case_defs=(-DL190_PLANT_UNALIGNED=1) ;;
  esac
  admission_binary="${out}/admission-${admission_case}"
  nvcc "${admission_flags[@]}" "${case_defs[@]}" "${source}" \
    -o "${admission_binary}" >"${out}/admission-${admission_case}-build.log" 2>&1 || {
    echo "[l190] FAIL: admission case=${admission_case} did not compile" >&2
    sed -n '1,100p' "${out}/admission-${admission_case}-build.log" >&2
    exit 1
  }
  "${admission_binary}" | tee "${out}/admission-${admission_case}-run.log"
  grep -Fq '[l190:admission] PASS' "${out}/admission-${admission_case}-run.log" || {
    echo "[l190] FAIL: admission case=${admission_case} did not reach its fixed verdict" >&2
    exit 1
  }
done

device_flags=(
  -I "${overlay}"
  "${base_flags[@]}"
  -Xcudafe --error_limit=100000
  -cuda -x cu
)

compile_device_probe() {
  local name="$1"
  shift
  set +e
  nvcc "${device_flags[@]}" "$@" "${source}" \
    -o "${out}/${name}.cu.cpp" >"${out}/${name}.log" 2>&1
  local rc=$?
  set -e
  printf '%s' "${rc}"
}

positive_rc="$(compile_device_probe device-positive)"
marker='error: static assertion failed with "L190_SPLITK_DEVICE_BODY_INSTANTIATED"'
if [[ "${positive_rc}" -eq 0 ]] || \
    [[ "$(grep -Fc "${marker}" "${out}/device-positive.log" || true)" -ne 1 ]]; then
  echo '[l190] FAIL: concrete Split-K wrapper did not reach its exact production device body once' >&2
  sed -n '1,100p' "${out}/device-positive.log" >&2
  exit 1
fi
unexpected="${out}/device-positive-unexpected.log"
grep -E ': (error|fatal error|catastrophic error):' "${out}/device-positive.log" \
  | grep -Fv "${marker}" >"${unexpected}" || true
if [[ -s "${unexpected}" ]]; then
  echo '[l190] FAIL: device-body witness carried an unrelated compiler error' >&2
  sed -n '1,80p' "${unexpected}" >&2
  exit 1
fi

# Bind the witness to the exact shipping row and the owned FP32 partial
# epilogue.  These strings come from the compiler's instantiation stack, not a
# parallel type-name model maintained by this script.
for token in \
  'GemmUniversalMixedInputSplitKParallel<ProblemShape_, CollectiveMainloop_, CollectivePartialEpilogue_>::operator()' \
  'Stages=3, kContinous=cute::C<256>' \
  'PPU0010_8x16x16_F32F16F16F32_TN' \
  'ElementBOptionalTuple=cute::tuple<cutlass::int4b_t, cutlass::half_t, cutlass::half_t>' \
  'CollectivePartialEpilogue_=dense_splitk_parallel_ppu::AdapterVisiblePartialEpilogue' \
  'AcConvert<float, 1, float'; do
  grep -Fq "${token}" "${out}/device-positive.log" || {
    echo "[l190] FAIL: exact device-body instantiation chain lost: ${token}" >&2
    sed -n '1,120p' "${out}/device-positive.log" >&2
    exit 1
  }
done

# Same source, same planted production header, one changed variable: sever the
# wrapper's only SplitKernel::operator() edge.  The compiler must finish and
# the marker must disappear.  This prevents parsing the header or forming the
# type from masquerading as device-body coverage.
severed_rc="$(compile_device_probe device-route-severed -DL190_SEVER_DEVICE_BODY=1)"
if [[ "${severed_rc}" -ne 0 ]] || [[ ! -s "${out}/device-route-severed.cu.cpp" ]]; then
  echo "[l190] FAIL: route-severed device control did not compile cleanly (nvcc rc=${severed_rc})" >&2
  sed -n '1,100p' "${out}/device-route-severed.log" >&2
  exit 1
fi
if grep -Fq 'L190_SPLITK_DEVICE_BODY_INSTANTIATED' "${out}/device-route-severed.log"; then
  echo '[l190] FAIL: production device-body marker survived route severing' >&2
  exit 1
fi

echo "[l190] PASS: host type/grid/workspace exact; production S8 device body reached once; route-severed control clean; artifacts=${out}"
