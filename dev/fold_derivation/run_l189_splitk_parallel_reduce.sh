#!/usr/bin/env bash
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L189_OUT:-/workspace/quactlize-l189-splitk-parallel-reduce}"
mkdir -p "${out}"

nvcc="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
nvidia_smi="${NVIDIA_SMI:-$(command -v nvidia-smi 2>/dev/null || true)}"
if [[ -z "${nvcc}" || -z "${nvidia_smi}" ]]; then
  echo "[l189] SKIP: nvcc and a visible NVIDIA device are required for the live two-kernel gate"
  exit 3
fi
if ! "${nvidia_smi}" -L >"${out}/devices.log" 2>&1 ||
   [[ ! -s "${out}/devices.log" ]]; then
  echo "[l189] SKIP: no NVIDIA device is visible for the live two-kernel gate"
  exit 3
fi

arch="${L189_CUDA_ARCH:-}"
if [[ -z "${arch}" ]]; then
  capability="$(${nvidia_smi} --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  if [[ ! "${capability}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "[l189] SKIP: could not derive the visible device compute capability"
    exit 3
  fi
  arch="sm_${capability/./}"
fi

source="${repo}/dev/fold_derivation/l189_splitk_parallel_reduce.cu"
binary="${out}/l189_splitk_parallel_reduce"
flags=(
  -std=c++17 -O3 -arch="${arch}" --expt-relaxed-constexpr -D__HGGCCC__
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/quactlize/include"
  -I "${repo}/third_party/actlize/include"
)

"${nvcc}" "${flags[@]}" "${source}" -o "${binary}" \
  >"${out}/build.log" 2>&1
build_rc=$?
if [[ ${build_rc} -ne 0 ]]; then
  echo "[l189] FAIL: exact reduction/device body did not compile for ${arch}"
  tail -n 40 "${out}/build.log"
  exit 1
fi

"${binary}" | tee "${out}/run.log"
run_rc=${PIPESTATUS[0]}
if [[ ${run_rc} -ne 0 ]]; then
  echo "[l189] FAIL: live two-kernel reduction returned rc=${run_rc}"
  exit 1
fi
if ! grep -Fq '[l189] PASS: live reduction and all admission controls' "${out}/run.log"; then
  echo "[l189] FAIL: executable returned zero without the complete PASS contract"
  exit 1
fi

sanitizer="$(command -v compute-sanitizer 2>/dev/null || true)"
if [[ -n "${sanitizer}" ]]; then
  "${sanitizer}" --tool memcheck --error-exitcode=99 "${binary}" \
    >"${out}/memcheck.log" 2>&1
  sanitizer_rc=$?
  if [[ ${sanitizer_rc} -ne 0 ]] ||
     ! grep -Fq 'ERROR SUMMARY: 0 errors' "${out}/memcheck.log"; then
    echo "[l189] FAIL: reducer memory-safety gate returned rc=${sanitizer_rc}"
    tail -n 60 "${out}/memcheck.log"
    exit 1
  fi
  echo "[l189] memcheck PASS: 0 errors"
else
  echo "[l189] memcheck SKIP: compute-sanitizer is unavailable"
fi

echo "[l189] PASS: ${arch} compiled and ran; artifacts=${out}"
