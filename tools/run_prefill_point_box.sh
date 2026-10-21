#!/usr/bin/env bash
# One Q4 KV point only. Freeze source and local actlize; no submodule fetch.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
GPU=${GPU:-1}
[[ ${GPU} =~ ^[0-9]+$ ]] || { echo 'ERROR: GPU must be one device ordinal.' >&2; exit 1; }
[[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN; this diagnostic uses a new directory.' >&2; exit 1; }
args=()
if [[ ${PLAN_ONLY:-0} == 1 ]]; then
    args+=(--plan-only)
else
    [[ ${IDLE_CONFIRMED:-0} == 1 ]] || {
        echo 'ERROR: check the assigned card is idle, then set IDLE_CONFIRMED=1.' >&2; exit 1;
    }
    PPU_SDK=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
    LLAMA_CI_DIR=$(realpath -e -- "${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp.ncp}")
    LLAMA_CI_BUILD_DIR=$(realpath -e -- "${LLAMA_CI_BUILD_DIR:-${LLAMA_CI_DIR}/build-ci-32F-cublas}")
    test -x "${PPU_SDK}/bin/hgcc"
    test -f "${LLAMA_CI_BUILD_DIR}/bin/libggml-cuda.so"
    args+=(--confirm-idle --sdk "${PPU_SDK}" --llama-source "${LLAMA_CI_DIR}" --llama-build "${LLAMA_CI_BUILD_DIR}")
    [[ -z ${PPU_SMI:-} ]] || args+=(--smi "${PPU_SMI}")
    export PATH="${PPU_SDK}/bin:${PATH}"
    export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/targets/x86_64-linux/lib:${PPU_SDK}/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    "${PYTHON}" -c 'import numpy,torch;from gguf.quants import dequantize'
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -d "${OUTPUT_ROOT:-/workspace}"
REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
[[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
if ! git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"; then
    echo "ERROR: local actlize lacks pinned ${ACTLIZE_REV}; no network fetch attempted." >&2; exit 1
fi
RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-prefill-point.XXXXXXXX")
[[ -n ${RUN} && -d ${RUN} && ! -L ${RUN} ]] || exit 1
mkdir "${RUN}/source"
git -C "${ROOT}" archive "${REV}" | tar -x -C "${RUN}/source"
mkdir -p "${RUN}/source/third_party/actlize"
git -C "${ROOT}/third_party/actlize" archive "${ACTLIZE_REV}" | tar -x -C "${RUN}/source/third_party/actlize"
test -f "${RUN}/source/tools/run_prefill_point.py"
export PREFILL_POINT_SOURCE_SHA="${REV}" PREFILL_POINT_ACTLIZE_SHA="${ACTLIZE_REV}"
echo "PREFILL_POINT_BOX run=${RUN} gpu=${GPU} shape=2048x1024x5120 clock_MHz=1700 actual_clock=ADVISORY sweep=0 cuBLAS_graph=0 asys=0"
set +e
"${PYTHON}" -u "${RUN}/source/tools/run_prefill_point.py" \
    --output "${RUN}/results" --device "${GPU}" --l2-bytes "${L2_BYTES:-67108864}" \
    --rounds 3 --samples 17 "${args[@]}" 2>&1 | tee "${RUN}/console.log"
rc=${PIPESTATUS[0]}
set -e
echo "runner_rc=${rc} results=${RUN}/results/review.tgz"
exit "${rc}"
