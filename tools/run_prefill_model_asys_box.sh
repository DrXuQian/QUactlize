#!/usr/bin/env bash
# Dense prefill; legacy Qwen3-32B PP2048 or a reviewed model cohort.
# Source/actlize frozen locally; no checkout/fetch on Box.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
GPU=${GPU:-1}
[[ ${GPU} =~ ^[0-9]+$ ]] || { echo 'ERROR: one GPU ordinal required.' >&2; exit 1; }
[[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN; use RESUME_RUN for an existing frozen sweep.' >&2; exit 1; }
args=()
NATIVE_REFERENCE=${NATIVE_REFERENCE:-1}
[[ ${NATIVE_REFERENCE} == 0 || ${NATIVE_REFERENCE} == 1 ]] || { echo 'ERROR: NATIVE_REFERENCE must be 0 or 1.' >&2; exit 1; }
[[ ${NATIVE_REFERENCE} != 0 ]] || args+=(--no-native)
if [[ -n ${FINALIST_PLAN:-} ]]; then
    [[ -z ${COHORT:-} && -z ${MODEL:-} ]] || { echo 'ERROR: FINALIST_PLAN cannot mix with COHORT/MODEL.' >&2; exit 1; }
    FINALIST_PLAN=$(realpath -e -- "${FINALIST_PLAN}")
    args+=(--finalist-plan "${FINALIST_PLAN}" --shard "${SHARD:-0}" --shards "${SHARDS:-1}"
           --max-modules "${MAX_MODULES:-96}")
    [[ -z ${POINT_ID:-} ]] || args+=(--point "${POINT_ID}")
elif [[ -n ${COHORT:-} ]]; then
    COHORT=$(realpath -e -- "${COHORT}")
    args+=(--cohort "${COHORT}" --shard "${SHARD:-0}" --shards "${SHARDS:-1}"
           --max-cases "${MAX_CASES:-96}" --max-modules "${MAX_MODULES:-256}")
    [[ -z ${POINT_ID:-} ]] || args+=(--point "${POINT_ID}")
else
    MODEL=${MODEL:-/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo/Qwen3-32B-Q4_K_M-GGUF/Qwen3-32B-Q4_K_M.gguf}
    test -f "${MODEL}"
    args+=(--model "${MODEL}")
fi
if [[ ${PLAN_ONLY:-0} == 1 ]]; then
    args+=(--plan-only)
else
    [[ ${IDLE_CONFIRMED:-0} == 1 ]] || {
        echo 'ERROR: confirm this physical GPU is idle, then set IDLE_CONFIRMED=1.' >&2; exit 1;
    }
    PPU_SDK=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
    test -x "${PPU_SDK}/bin/hgcc"
    test -x "${PPU_SDK}/asight/bin/asys"
    args+=(--confirm-idle --sdk "${PPU_SDK}")
    if [[ ${NATIVE_REFERENCE} == 1 ]]; then
        LLAMA_CI_DIR=$(realpath -e -- "${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp.ncp}")
        LLAMA_CI_BUILD_DIR=$(realpath -e -- "${LLAMA_CI_BUILD_DIR:-${LLAMA_CI_DIR}/build-ci-32F-cublas}")
        test -f "${LLAMA_CI_BUILD_DIR}/bin/libggml-cuda.so"
        args+=(--llama-source "${LLAMA_CI_DIR}" --llama-build "${LLAMA_CI_BUILD_DIR}")
    fi
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
    REV=$(git -C "${ROOT}" rev-parse --verify "${SOURCE_REV:-HEAD}^{commit}")
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
echo "PREFILL_ASYS_BOX run=${RUN} gpu=${GPU} cohort=${FINALIST_PLAN:-${COHORT:-QWEN3_32B_PP2048}} shard=${SHARD:-0}/${SHARDS:-1} jobs=${JOBS:-8} configs_per_batch=${CONFIGS_PER_BATCH:-16} clock_MHz=1700 actual_clock=ADVISORY native_reference=${NATIVE_REFERENCE} accumulator=FP32"
set +e
"${PYTHON}" -u "${RUN}/source/tools/run_prefill_model_asys.py" \
    --output "${RUN}/results" --device "${GPU}" \
    --l2-bytes "${L2_BYTES:-67108864}" --jobs "${JOBS:-8}" \
    --configs-per-batch "${CONFIGS_PER_BATCH:-16}" "${args[@]}" 2>&1 | tee "${RUN}/console-$(date +%s).log"
rc=${PIPESTATUS[0]}
set -e
echo "runner_rc=${rc} review=${RUN}/results/review-*"
echo "Asys=${RUN}/results/batches/*/attempt-*/trace.asysrep"
echo "Resume with RESUME_RUN=${RUN} and the same GPU/environment. Do not delete its source/cache/results."
exit "${rc}"
