#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
LLAMA_CI_DIR=${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp.ncp}
LLAMA_CI_BUILD_DIR=${LLAMA_CI_BUILD_DIR:-${LLAMA_CI_DIR}/build-ci}
PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
GPUS=${GPUS:-1,2,3,4,5,6,7}
if [[ ${IDLE_CONFIRMED:-0} != 1 ]]; then
    echo 'ERROR: confirm all requested GPUs are idle, then set IDLE_CONFIRMED=1' >&2
    exit 1
fi
for required in "${LLAMA_CI_BUILD_DIR}/CMakeCache.txt" "${LLAMA_CI_BUILD_DIR}/bin/libggml-cuda.so" \
                "${LLAMA_CI_BUILD_DIR}/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make"; do
    if [[ ! -f ${required} ]]; then
        echo "ERROR: missing native PPU build input: ${required}" >&2
        echo 'Use the matching existing llama .aoneci/scripts/build.sh build, or set LLAMA_CI_BUILD_DIR.' >&2
        exit 1
    fi
done
if [[ ! -d ${PPU_SDK} ]]; then
    echo "ERROR: missing SDK root: ${PPU_SDK}" >&2
    exit 1
fi
"${PYTHON:-python3}" -c 'import numpy; from gguf.quants import dequantize'
if [[ -z ${RUN:-} ]]; then
    test -d "${OUTPUT_ROOT:-/workspace}"
    RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-native-components.XXXXXXXX")
else
    RUN=$(realpath -e -- "${RUN}")
    test -f "${RUN}/run.json"
fi
echo "RUN=${RUN}"
echo 'Scope: all 1544 known workloads; isolated components + small helper supplement, no config sweep or caller edits.'
"${PYTHON:-python3}" -u "${ROOT}/tools/run_native_components_multi.py" \
    --devices "${GPUS}" --output "${RUN}" --llama-source "${LLAMA_CI_DIR}" \
    --llama-build "${LLAMA_CI_BUILD_DIR}" --sdk "${PPU_SDK}" \
    --l2-bytes "${L2_BYTES:-0}" --phase "${PHASE:-all}" --confirm-idle
