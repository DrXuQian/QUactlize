#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
: "${PREVIOUS_RUN:?set the original complete raw run directory}"
: "${DIAGNOSTIC_REVIEW:?set the successful 17-job diagnostic review.tgz}"
test -f "${PREVIOUS_RUN}/summary.json"
test -f "${DIAGNOSTIC_REVIEW}"
[[ ${IDLE_CONFIRMED:-0} == 1 ]] || { echo 'Confirm all requested GPUs are idle first.' >&2; exit 1; }
LLAMA_CI_DIR=${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp.ncp}
LLAMA_CI_BUILD_DIR=${LLAMA_CI_BUILD_DIR:-${LLAMA_CI_DIR}/build-ci}
PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
if [[ -z ${RUN:-} ]]; then
    test -d "${OUTPUT_ROOT:-/workspace}"
    RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-native-retry.XXXXXXXX")
else
    RUN=$(realpath -e -- "${RUN}")
    test -f "${RUN}/run.json"
fi
echo "RUN=${RUN}"
echo '268 failed points only; preserve 1178 measurements and 36 helpers; defer 98 native FP16 accuracy issues.'
"${PYTHON:-python3}" -u "${ROOT}/tools/retry_native_components.py" \
    --previous "${PREVIOUS_RUN}" --diagnostic "${DIAGNOSTIC_REVIEW}" --output "${RUN}" \
    --devices "${GPUS:-1,2,3,4,5,7}" --llama-source "${LLAMA_CI_DIR}" \
    --llama-build "${LLAMA_CI_BUILD_DIR}" --sdk "${PPU_SDK}" \
    --l2-bytes "${L2_BYTES:-67108864}" --confirm-idle
