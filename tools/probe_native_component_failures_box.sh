#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
: "${PREVIOUS_RUN:?set the completed native component run directory}"
test -f "${PREVIOUS_RUN}/summary.json"
[[ ${IDLE_CONFIRMED:-0} == 1 ]] || { echo 'Confirm the selected GPU is idle first.' >&2; exit 1; }
LLAMA_CI_DIR=${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp.ncp}
LLAMA_CI_BUILD_DIR=${LLAMA_CI_BUILD_DIR:-${LLAMA_CI_DIR}/build-ci}
PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
[[ ${CUDA_VISIBLE_DEVICES} =~ ^[0-9]+$ ]] || { echo 'Use one physical GPU ordinal.' >&2; exit 1; }
test -d "${OUTPUT_ROOT:-/workspace}"
RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-native-diagnostic.XXXXXXXX")
echo "RUN=${RUN}"
"${PYTHON:-python3}" -u "${ROOT}/tools/probe_native_component_failures.py" \
    --previous "${PREVIOUS_RUN}" --output "${RUN}" --llama-source "${LLAMA_CI_DIR}" \
    --llama-build "${LLAMA_CI_BUILD_DIR}" --sdk "${PPU_SDK}" \
    --l2-bytes "${L2_BYTES:-0}" --confirm-idle
