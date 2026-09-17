#!/usr/bin/env bash
# Reuse completed component/numerical evidence; no build or bundle fetch.
(
    set -Eeuo pipefail
    trap 'printf "KPACK_MODEL_CONTINUE FAIL line=%s rc=%s; Docker shell preserved\n" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/resume_kpack_q4_model.py"
    cd "$ROOT"
    PREVIOUS=$(realpath -e -- "${PREVIOUS_RUN:?Set PREVIOUS_RUN to the stopped model run}")
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    LLAMA=$(realpath -e -- "${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp}")
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PREVIOUS" && test -d "$PREVIOUS/results" && test -n "$SDK" && test -n "$LLAMA"
    test -x "$PYTHON" && test -r "$SDK/envsetup.sh"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export QUACTLIZE_PPU_BUNDLE=${QUACTLIZE_PPU_BUNDLE:-/workspace/quactlize-runtime-artifact-2826cf1-46fc3096e1a1/prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1/bundle}
    test -s "$QUACTLIZE_PPU_BUNDLE/manifest.json"
    EXTRA=()
    if [[ ${PERFORMANCE_ONLY:-0} == 1 ]]; then
        EXTRA+=(--performance-only)
        if [[ ${MODEL_ACU:-0} == 1 ]]; then EXTRA+=(--profile-acu); fi
    fi
    if [[ ${REPAIR_PREFILL:-0} == 1 ]]; then
        mapfile -t PIN < <("$PYTHON" -c 'import json; m=json.load(open("tools/kpack_q4_model_artifact.json")); print(m["branch"]); print(m["commit"]); print(m["path"])')
        [[ ${#PIN[@]} == 3 && ( ${PIN[0]} == artifacts/kpack-model-runtime-v1 || ${PIN[0]} == artifacts/kpack-model-paired-n4-v1 ) && ${PIN[1]} =~ ^[0-9a-f]{40}$ && ${PIN[2]} == prebuilt/ppu0010/kpack-model-runtime-v1 ]]
        ART="$(dirname "$PREVIOUS")/quactlize-model-artifact-${PIN[1]:0:10}"
        git fetch origin "${PIN[0]}"
        git cat-file -e "${PIN[1]}^{commit}"
        if [[ -e $ART ]]; then
            test -d "$ART" && test "$(git -C "$ART" rev-parse HEAD)" == "${PIN[1]}"
        else
            GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$ART" "${PIN[1]}"
        fi
        git -C "$ART" lfs pull origin --include="${PIN[2]}/**" --exclude=""
        EXTRA+=(--repair-prefill)
    fi
    "$PYTHON" -u tools/resume_kpack_q4_model.py --previous "$PREVIOUS" --llama "$LLAMA" --sdk "$SDK" \
        --device "${CUDA_VISIBLE_DEVICES:-0}" \
        "${EXTRA[@]}" \
        --corpus "${GSM8K_FILE:-/sim/eec/shared/AI_workspace/llm-models/datasets/gsm8k/main/test-00000-of-000001.parquet}"
)
