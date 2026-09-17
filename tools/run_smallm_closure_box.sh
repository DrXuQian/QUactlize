#!/usr/bin/env bash
# Box-only compilation/timing; invoking with bash preserves the Docker shell.
(
    set -Eeuo pipefail
    RUN= stage=precheck
    finish() {
        local rc=$?
        trap - EXIT ERR
        if [[ -n $RUN && -d $RUN/results ]]; then
            printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/results/runner-status.txt"
            tar -czf "$RUN.results.tgz" -C "$RUN" results || rc=1
            printf '\nresults=%s.results.tgz\nresume: RESUME_RUN=%s bash tools/run_smallm_closure_box.sh\n' "$RUN" "$RUN"
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "SMALLM_CLOSURE_BOX FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_smallm_closure.py"
    cd "$ROOT"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/envsetup.sh" && test -x "$SDK/bin/hgcc"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    export PPU_SDK="$SDK" LC_ALL=C
    export PATH="$SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    "$PYTHON" -c 'import numpy, gguf, torch'
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -f "$RUN/results/plan.json"
    else
        RUN=$(mktemp -d "$RESULT_DIR/smallm-closure.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    if [[ -z ${BUNDLE:-} ]]; then
        stage=fetch
        mapfile -t INFO < <("$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["branch"]); print(m["commit"]); print(m["path"]); print(m["manifest_sha256"])' tools/kpack_q4_model_artifact.json)
        [[ ${#INFO[@]} == 4 && ( ${INFO[0]} == artifacts/kpack-model-runtime-v1 || ${INFO[0]} == artifacts/kpack-model-paired-n4-v1 ) && ${INFO[1]} =~ ^[0-9a-f]{40}$ && ${INFO[2]} == prebuilt/ppu0010/kpack-model-runtime-v1 && ${INFO[3]} =~ ^[0-9a-f]{64}$ ]]
        ART="$RESULT_DIR/quactlize-model-artifact-${INFO[1]:0:10}"
        git fetch origin "${INFO[0]}"
        git cat-file -e "${INFO[1]}^{commit}"
        if [[ -e "$ART" ]]; then
            test -d "$ART" && test "$(git -C "$ART" rev-parse HEAD)" == "${INFO[1]}"
        else
            GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$ART" "${INFO[1]}"
        fi
        # Only the small-M execution image is reused. No caller binaries or
        # old sweeping bundles are downloaded for this campaign.
        git -C "$ART" lfs pull origin --include="${INFO[2]}/libquactlize_ppu_execution.so" --exclude=""
        BUNDLE="$ART/${INFO[2]}"
        "$PYTHON" -c 'import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2],"manifest differs"' "$BUNDLE/manifest.json" "${INFO[3]}"
    fi
    BUNDLE=$(realpath -e -- "$BUNDLE")
    test -n "$BUNDLE" && test -f "$BUNDLE/manifest.json"
    "$PYTHON" -c 'import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text()); assert hashlib.sha256((p/"libquactlize_ppu_execution.so").read_bytes()).hexdigest()==m["execution_sha256"],"execution image missing/changed/LFS pointer"' "$BUNDLE"
    read -r -a GPUS <<< "${DEVICES:-0 1 2 3 4 5 6 7}"
    read -r -a COMPUTE <<< "${COMPUTE_TYPES:-f16 bf16}"
    EXTRA=()
    [[ ${PLAN_ONLY:-0} == 1 ]] && EXTRA+=(--plan-only)
    [[ ${BUILD_ONLY:-0} == 1 ]] && EXTRA+=(--build-only)
    stage=compile-and-sweep
    printf 'SMALLM_CLOSURE_BOX run=%s jobs=%s devices=%s\n' "$RUN" "${JOBS:-192}" "${GPUS[*]}"
    printf 'Only idle devices. Candidates are extracted from prior results; no full Cartesian sweep.\n'
    "$PYTHON" -u tools/run_smallm_closure.py --sdk "$SDK" --bundle "$BUNDLE" --output "$RUN" \
        --model-plan "${MODEL_PLAN:-$ROOT/tools/kpack_batched_int4_2048.json}" \
        --model-root "${MODEL_ROOT:-/sim/eec/shared/AI_workspace/llm-models}" \
        --devices "${GPUS[@]}" --compute "${COMPUTE[@]}" --jobs "${JOBS:-192}" \
        --l2-bytes "${L2_BYTES:?Set verified L2 capacity in bytes}" "${EXTRA[@]}" \
        2>&1 | tee -a "$RUN/results/console.log"
    stage=complete
)
