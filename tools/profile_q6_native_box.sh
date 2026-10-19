#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
: "${PREVIOUS_RUN:?set the completed native component run directory}"
[[ ${IDLE_CONFIRMED:-0} == 1 ]] || { echo 'ERROR: confirm one otherwise idle GPU first.' >&2; exit 1; }
PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
LLAMA_CI_DIR=${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp.ncp}
LLAMA_CI_BUILD_DIR=${LLAMA_CI_BUILD_DIR:-${LLAMA_CI_DIR}/build-ci}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
[[ ${CUDA_VISIBLE_DEVICES} =~ ^[0-9]+$ ]] || { echo 'ERROR: select exactly one GPU ordinal.' >&2; exit 1; }
for required in "${PREVIOUS_RUN}/summary.json" "${LLAMA_CI_BUILD_DIR}/CMakeCache.txt" \
    "${LLAMA_CI_BUILD_DIR}/bin/libggml-cuda.so" "${PPU_SDK}/bin/hgcc" \
    "${ACU:-${PPU_SDK}/asight/bin/acu}"; do
    [[ -f ${required} ]] || { echo "ERROR: missing ${required}" >&2; exit 1; }
done
"${PYTHON:-python3}" -c 'import numpy; from gguf.quants import dequantize'
[[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN; this focused experiment uses a fresh directory.' >&2; exit 1; }
test -d "${OUTPUT_ROOT:-/workspace}"
RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-q6-acu.XXXXXXXX")
[[ -d ${RUN} && ! -L ${RUN} ]] || exit 1
SOURCE=${RUN}/source
mkdir "${SOURCE}"
# Freeze source once, without a checkout, nested network fetch, or modifying
# another running campaign. actlize is taken from the pinned local git object.
REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
[[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"
git -C "${ROOT}" archive "${REV}" | tar -x -C "${SOURCE}"
mkdir -p "${SOURCE}/third_party/actlize"
git -C "${ROOT}/third_party/actlize" archive "${ACTLIZE_REV}" | tar -x -C "${SOURCE}/third_party/actlize"
echo "Q6_ACU_START run=${RUN} source=${REV} gpu=${CUDA_VISIBLE_DEVICES}"
echo 'One Q6 MoE point: tokens=8 topk=8 N=512 K=2048 E=256; SIMT S1 versus unchanged native MMVQ.'
echo 'Fresh ABBA component events, then 3 forced-cold ACU reports. No llama rebuild or config sweep.'
set +e
"${PYTHON:-python3}" -u "${SOURCE}/tools/profile_q6_native.py" \
    --previous "${PREVIOUS_RUN}" --output "${RUN}/results" \
    --llama-source "${LLAMA_CI_DIR}" --llama-build "${LLAMA_CI_BUILD_DIR}" \
    --sdk "${PPU_SDK}" --acu "${ACU:-${PPU_SDK}/asight/bin/acu}" \
    --l2-bytes "${L2_BYTES:-67108864}" --confirm-idle 2>&1 | tee "${RUN}/console.log"
rc=${PIPESTATUS[0]}
set -e
echo "results=${RUN}/results/review.tgz"
echo "runner_rc=${rc}"
exit "${rc}"
