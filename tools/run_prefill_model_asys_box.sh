#!/usr/bin/env bash
# Qwen3-32B PP2048 only. Source/actlize frozen locally; no checkout/fetch on Box.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
GPU=${GPU:-1}
[[ ${GPU} =~ ^[0-9]+$ ]] || { echo 'ERROR: one GPU ordinal required.' >&2; exit 1; }
[[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN; use RESUME_RUN for an existing frozen sweep.' >&2; exit 1; }
MODEL=${MODEL:-/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo/Qwen3-32B-Q4_K_M-GGUF/Qwen3-32B-Q4_K_M.gguf}
test -f "${MODEL}"
args=()
if [[ ${PLAN_ONLY:-0} == 1 ]]; then
    args+=(--plan-only)
else
    [[ ${IDLE_CONFIRMED:-0} == 1 ]] || {
        echo 'ERROR: confirm this physical GPU is idle, then set IDLE_CONFIRMED=1.' >&2; exit 1;
    }
    PPU_SDK=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
    LLAMA_CI_DIR=$(realpath -e -- "${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp.ncp}")
    LLAMA_CI_BUILD_DIR=$(realpath -e -- "${LLAMA_CI_BUILD_DIR:-${LLAMA_CI_DIR}/build-ci-32F-cublas}")
    test -x "${PPU_SDK}/bin/hgcc"
    test -x "${PPU_SDK}/asight/bin/asys"
    test -f "${LLAMA_CI_BUILD_DIR}/bin/libggml-cuda.so"
    args+=(--confirm-idle --sdk "${PPU_SDK}" --llama-source "${LLAMA_CI_DIR}" --llama-build "${LLAMA_CI_BUILD_DIR}")
    [[ -z ${PPU_SMI:-} ]] || args+=(--smi "${PPU_SMI}")
    export PATH="${PPU_SDK}/bin:${PATH}"
    export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/targets/x86_64-linux/lib:${PPU_SDK}/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    "${PYTHON}" -c 'import numpy,torch;from gguf.quants import dequantize'
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -n ${RESUME_RUN:-} ]]; then
    RUN=$(realpath -e -- "${RESUME_RUN}")
    test -f "${RUN}/source/tools/run_prefill_model_asys.py"
    test -f "${RUN}/results/plan.json"
    args+=(--resume)
else
    test -d "${OUTPUT_ROOT:-/workspace}"
    REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
    ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
    [[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
    if ! git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"; then
        echo "ERROR: local actlize lacks pinned ${ACTLIZE_REV}; no network fetch attempted." >&2; exit 1
    fi
    RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/qwen32-prefill-asys.XXXXXXXX")
    [[ -n ${RUN} && -d ${RUN} && ! -L ${RUN} ]] || exit 1
    mkdir "${RUN}/source"
    git -C "${ROOT}" archive "${REV}" | tar -x -C "${RUN}/source"
    mkdir -p "${RUN}/source/third_party/actlize"
    git -C "${ROOT}/third_party/actlize" archive "${ACTLIZE_REV}" | tar -x -C "${RUN}/source/third_party/actlize"
fi
# Reuse is optional. The run's authority remembers its payload path on resume.
if [[ -n ${PREVIOUS_RUN:-} ]]; then
    PREVIOUS_RUN=$(realpath -e -- "${PREVIOUS_RUN}")
    if [[ -f ${PREVIOUS_RUN}/results/build/receipt.json ]]; then
        args+=(--reuse-build "${PREVIOUS_RUN}/results/build")
    else
        test -f "${PREVIOUS_RUN}/build/receipt.json"
        args+=(--reuse-build "${PREVIOUS_RUN}/build")
    fi
fi
[[ -z ${TC_CACHE:-} ]] || args+=(--cache "${TC_CACHE}")
echo "PREFILL_ASYS_BOX run=${RUN} gpu=${GPU} M=2048 shape_scope=QWEN3_32B_QUANTIZED_DENSE clock_MHz=1700 actual_clock=ADVISORY kernel_changes=0"
set +e
"${PYTHON}" -u "${RUN}/source/tools/run_prefill_model_asys.py" \
    --output "${RUN}/results" --model "${MODEL}" --device "${GPU}" \
    --l2-bytes "${L2_BYTES:-67108864}" --jobs "${JOBS:-8}" \
    --configs-per-batch "${CONFIGS_PER_BATCH:-16}" "${args[@]}" 2>&1 | tee "${RUN}/console-$(date +%s).log"
rc=${PIPESTATUS[0]}
set -e
echo "runner_rc=${rc} review=${RUN}/results/review-*.tgz"
echo "Asys=${RUN}/results/batches/*/attempt-*/trace.asysrep"
echo "Resume with RESUME_RUN=${RUN} and the same GPU/environment. Do not delete its source/cache/results."
exit "${rc}"
