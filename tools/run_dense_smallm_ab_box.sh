#!/usr/bin/env bash
# Frozen 360-point small-M selected/native A/B; no caller rebuild or config sweep.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
GPUS=${GPUS:-0,1,2,3}
PPU_SDK=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
LLAMA_CI_DIR=$(realpath -e -- "${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp.ncp}")
LLAMA_CI_BUILD_DIR=$(realpath -e -- "${LLAMA_CI_BUILD_DIR:-${LLAMA_CI_DIR}/build-ci}")
[[ ${IDLE_CONFIRMED:-0} == 1 || ${PLAN_ONLY:-0} == 1 ]] || {
    echo 'ERROR: confirm the requested physical GPU ordinals are idle; set IDLE_CONFIRMED=1.' >&2
    exit 1
}
args=(--sdk "$PPU_SDK" --llama-source "$LLAMA_CI_DIR" --llama-build "$LLAMA_CI_BUILD_DIR"
      --devices "$GPUS" --jobs "${JOBS:-8}" --l2-bytes "${L2_BYTES:-67108864}")
if [[ ${PLAN_ONLY:-0} == 1 ]]; then args+=(--plan-only); else args+=(--confirm-idle); fi
if [[ -n ${RESUME_RUN:-} ]]; then
    SMALLM_RUN=$(realpath -e -- "$RESUME_RUN")
    test -f "$SMALLM_RUN/run/run.json"
    test -f "$SMALLM_RUN/source/tools/run_dense_smallm_ab.py"
    args+=(--resume)
else
    test -d "${OUTPUT_ROOT:-/workspace}"
    REV=$(git -C "$ROOT" rev-parse --verify 'HEAD^{commit}')
    ACTLIZE_REV=$(git -C "$ROOT" rev-parse "$REV:third_party/actlize")
    [[ $REV =~ ^[0-9a-f]{40}$ && $ACTLIZE_REV =~ ^[0-9a-f]{40}$ ]] || exit 1
    git -C "$ROOT/third_party/actlize" cat-file -e "$ACTLIZE_REV^{commit}"
    SMALLM_RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-smallm-native.XXXXXXXX")
    [[ -n $SMALLM_RUN && -d $SMALLM_RUN && ! -L $SMALLM_RUN ]] || exit 1
    mkdir "$SMALLM_RUN/source"
    git -C "$ROOT" archive "$REV" | tar -x -C "$SMALLM_RUN/source"
    mkdir -p "$SMALLM_RUN/source/third_party/actlize"
    git -C "$ROOT/third_party/actlize" archive "$ACTLIZE_REV" | tar -x -C "$SMALLM_RUN/source/third_party/actlize"
fi
export PATH="$PPU_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$LLAMA_CI_BUILD_DIR/bin:$PPU_SDK/lib:$PPU_SDK/targets/x86_64-linux/lib:$PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "SMALLM_RUN=$SMALLM_RUN"
echo "SMALLM_NATIVE_BOX devices=$GPUS M=1..8 shapes=360 configs=CURRENT_SELECTED_ONLY native=UNCHANGED clock_MHz=1700"
set +e
"$PYTHON" -u "$SMALLM_RUN/source/tools/run_dense_smallm_ab.py" \
    --output "$SMALLM_RUN/run" "${args[@]}" 2>&1 | tee "$SMALLM_RUN/console-$(date +%s).log"
rc=${PIPESTATUS[0]}
set -e
echo "runner_rc=$rc results=$SMALLM_RUN/run/review-lite-*.tar.xz"
echo "Resume with RESUME_RUN=$SMALLM_RUN and the same GPUS, SDK and caller build."
exit "$rc"
